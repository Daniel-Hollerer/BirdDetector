from flask import Flask, request, send_file
from ultralytics import YOLO
import numpy as np
import cv2
import os
import torch
import base64
import shutil
import subprocess
import tempfile
import time
import uuid

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "model/best.pt")
INFER_SIZE = int(os.environ.get("INFER_SIZE", 640))
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.25))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", 0.45))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", 100))
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", max(1, (os.cpu_count() or 1) - 1)))
USE_AUGMENT = os.environ.get("USE_AUGMENT", "false").lower() == "true"
MAX_DETECTIONS = int(os.environ.get("MAX_DETECTIONS", 20))
VIDEO_OUTPUT_DIR = os.environ.get("VIDEO_OUTPUT_DIR", "/tmp/bird-detection-output")
PROCESSED_FILE_TTL_SECONDS = int(os.environ.get("PROCESSED_FILE_TTL_SECONDS", 1800))
VIDEO_INFERENCE_STRIDE = max(1, int(os.environ.get("VIDEO_INFERENCE_STRIDE", 1)))
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "https://bird-detector.vercel.app")

torch.set_num_threads(TORCH_THREADS)
torch.set_grad_enabled(False)

os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
processed_media = {}


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Range"
    return response


def predict_image(image):
    with torch.inference_mode():
        return model.predict(
            source=image,
            imgsz=INFER_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            augment=USE_AUGMENT,
            device="cpu",
            verbose=False
        )[0]


def draw_detection(image, x1, y1, x2, y2, label):
    thickness = max(2, round(min(image.shape[:2]) / 320))
    font_scale = max(0.5, min(image.shape[:2]) / 900)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness)
    cv2.putText(
        image,
        label,
        (x1, max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 255),
        thickness
    )


def cleanup_processed_media():
    now = time.time()
    expired_ids = [
        media_id
        for media_id, media in processed_media.items()
        if now - media["created_at"] > PROCESSED_FILE_TTL_SECONDS
    ]

    for media_id in expired_ids:
        media = processed_media.pop(media_id, None)
        if media and os.path.exists(media["path"]):
            os.remove(media["path"])


def register_processed_media(path, mimetype):
    cleanup_processed_media()
    media_id = uuid.uuid4().hex
    processed_media[media_id] = {
        "path": path,
        "mimetype": mimetype,
        "created_at": time.time()
    }
    return media_id


def transcode_video_for_web(input_path):
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return input_path, False, "Browser playback may fail because ffmpeg is not installed."

    output_path = os.path.splitext(input_path)[0] + "-web.mp4"
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        output_path
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return input_path, False, "Video was processed, but web transcoding failed."

    os.remove(input_path)
    return output_path, True, None


def extract_detections(results, annotated_img):
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
        cls = int(box.cls)
        conf = float(box.conf)
        name = results.names[cls]
        detections.append({
            "name": name,
            "confidence": round(conf, 4)
        })

        label = f"{name} {conf:.2f}"
        draw_detection(annotated_img, x1, y1, x2, y2, label)

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections

# Load and warm the model once so the first request is faster.
model = YOLO(MODEL_PATH)
_warmup_frame = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)
predict_image(_warmup_frame)


@app.route("/", methods=["GET"])
def root():
    return {"service": "bird-detection-backend", "status": "ok"}


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


@app.route("/upload", methods=["OPTIONS"])
@app.route("/upload-video", methods=["OPTIONS"])
@app.route("/processed/<media_id>", methods=["OPTIONS"])
def options(media_id=None):
    return ("", 204)



def process_upload(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None

    annotated_img = img.copy()

    results = predict_image(img)
    detections = extract_detections(results, annotated_img)

    _, buffer = cv2.imencode(
        ".jpg",
        annotated_img,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )

    best_detection = max(detections, key=lambda item: item["confidence"], default=None)
    return {
        "filename": file_storage.filename or "uploaded-image",
        "image": base64.b64encode(buffer.tobytes()).decode("ascii"),
        "top_detection": best_detection,
        "detections": detections
    }


def process_video_upload(file_storage):
    cleanup_processed_media()

    suffix = os.path.splitext(file_storage.filename or "upload.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as temp_input:
        file_storage.save(temp_input.name)
        input_path = temp_input.name

    output_path = os.path.join(VIDEO_OUTPUT_DIR, f"{uuid.uuid4().hex}.mp4")
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        os.remove(input_path)
        return None

    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    fps = fps if fps > 0 else 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if width <= 0 or height <= 0:
        capture.release()
        os.remove(input_path)
        return None

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )
    if not writer.isOpened():
        capture.release()
        os.remove(input_path)
        return None

    detections_by_name = {}
    processed_frames = 0
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        annotated_frame = frame.copy()
        if frame_index % VIDEO_INFERENCE_STRIDE == 0:
            results = predict_image(frame)
            frame_detections = extract_detections(results, annotated_frame)

            for detection in frame_detections:
                existing = detections_by_name.get(detection["name"])
                if existing is None or detection["confidence"] > existing["confidence"]:
                    detections_by_name[detection["name"]] = detection

        writer.write(annotated_frame)
        processed_frames += 1
        frame_index += 1

    capture.release()
    writer.release()
    os.remove(input_path)

    transcoded_output_path, web_playable, warning = transcode_video_for_web(output_path)

    detections = sorted(
        detections_by_name.values(),
        key=lambda item: item["confidence"],
        reverse=True
    )
    best_detection = detections[0] if detections else None
    media_id = register_processed_media(transcoded_output_path, "video/mp4")

    return {
        "filename": file_storage.filename or "uploaded-video",
        "media_type": "video",
        "video_url": f"/processed/{media_id}",
        "download_url": f"/processed/{media_id}",
        "web_playable": web_playable,
        "warning": warning,
        "top_detection": best_detection,
        "detections": detections,
        "frame_count": processed_frames or frame_count,
        "fps": round(fps, 2),
        "duration_seconds": round((processed_frames or frame_count) / fps, 2) if fps else None
    }


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files:
        single_file = request.files.get("file")
        if single_file is not None:
            files = [single_file]

    if not files:
        return {"error": "No file uploaded"}, 400

    processed_results = []
    for file_storage in files:
        result = process_upload(file_storage)
        if result is None:
            return {"error": f"Invalid image file: {file_storage.filename or 'upload'}"}, 400
        processed_results.append(result)

    return {"results": processed_results}


@app.route("/upload-video", methods=["POST"])
def upload_video():
    file_storage = request.files.get("video")
    if file_storage is None:
        return {"error": "No video uploaded"}, 400

    result = process_video_upload(file_storage)
    if result is None:
        return {"error": "Invalid video file"}, 400

    return result


@app.route("/processed/<media_id>", methods=["GET"])
def get_processed_media(media_id):
    cleanup_processed_media()
    media = processed_media.get(media_id)
    if media is None or not os.path.exists(media["path"]):
        return {"error": "Processed file not found or expired"}, 404

    return send_file(media["path"], mimetype=media["mimetype"], conditional=True)


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
