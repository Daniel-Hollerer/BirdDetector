from flask import Flask, request, send_file, send_from_directory
from ultralytics import YOLO
import numpy as np
import cv2
import io
import os

app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")

# Load model once
model = YOLO("model/best.pt")


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
    file = request.files["file"]

    # Convert uploaded file → OpenCV image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # resize
    img = cv2.resize(img, (320,320))

    # Run YOLO
    results = model(
    img,
    imgsz=320,
    verbose=False,
    device="cpu"
    )[0]

    # Draw bounding boxes
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

    # Convert image → bytes
    _, buffer = cv2.imencode(".jpg", img)

    return send_file(
        io.BytesIO(buffer),
        mimetype="image/jpeg"
    )


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port = port)