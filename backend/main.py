from flask import Flask, request, send_file, send_from_directory
from ultralytics import YOLO
import numpy as np
import cv2
import io
import os
import torch

app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")

MODEL_PATH = os.environ.get("MODEL_PATH", "model/best.pt")
INFER_SIZE = int(os.environ.get("INFER_SIZE", 320))
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.25))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", 80))
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", 1))

torch.set_num_threads(TORCH_THREADS)

# Load and warm the model once so the first request is faster.
model = YOLO(MODEL_PATH)
_warmup_frame = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)
model.predict(
    source=_warmup_frame,
    imgsz=INFER_SIZE,
    conf=CONF_THRESHOLD,
    device="cpu",
    verbose=False
)


# -----------------------------
# Serve frontend
# -----------------------------
@app.route("/")
def serve_frontend():
    return send_from_directory("../frontend", "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory("../frontend", path)


# -----------------------------
# Upload + YOLO inference
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if file is None:
        return {"error": "No file uploaded"}, 400

    # Convert uploaded file to an OpenCV image.
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image file"}, 400

    # Resize before inference to keep CPU work predictable on Render's free tier.
    img = cv2.resize(img, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_AREA)

    results = model.predict(
        source=img,
        imgsz=INFER_SIZE,
        conf=CONF_THRESHOLD,
        device="cpu",
        verbose=False
    )[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
        cls = int(box.cls)
        conf = float(box.conf)

        label = f"{results.names[cls]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    # Lower JPEG quality slightly to reduce encode time and response size.
    _, buffer = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )

    return send_file(
        io.BytesIO(buffer),
        mimetype="image/jpeg"
    )


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
