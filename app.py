"""
app.py — Flask web server for PlateVision
==========================================
Serves the HTML frontend and provides three REST endpoints:

  GET  /              → renders templates/index.html
  GET  /health        → JSON model readiness probe
  POST /predict       → image inference (returns annotated image + detections)
  POST /predict/video → video inference (returns annotated MP4)
"""

import io
import os
import tempfile
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── Model path resolution ─────────────────────────────────────────────────────
# Prefer a trained best.pt; fall back to the pretrained yolov8n checkpoint.
_TRAINED_MODEL  = Path("artifacts/model_trainer/car_plate_detector/weights/best.pt")
_FALLBACK_MODEL = Path("yolov8n.pt")

def _resolve_model() -> tuple[str, bool]:
    """Return (model_path, is_fallback). Re-checks disk each call so dvc pull is picked up."""
    if _TRAINED_MODEL.exists():
        return str(_TRAINED_MODEL), False
    return str(_FALLBACK_MODEL), True

MODEL_PATH, _IS_FALLBACK = _resolve_model()

# ── Lazy-load YOLO via the prediction pipeline ───────────────────────────────
from src.carPlateDetection.pipeline.prediction_pipeline import (
    get_model,
    predict_image,
)

# Pre-warm the model at startup so /health returns instantly
try:
    get_model(MODEL_PATH)
    _MODEL_READY = True
except Exception as exc:
    print(f"[WARNING] Model failed to load at startup: {exc}")
    _MODEL_READY = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Simple readiness probe consumed by the JS nav-bar indicator."""
    global _MODEL_READY, MODEL_PATH, _IS_FALLBACK
    # Re-resolve every time so a freshly pulled best.pt is noticed
    MODEL_PATH, _IS_FALLBACK = _resolve_model()
    if not _MODEL_READY or _IS_FALLBACK:
        try:
            get_model(MODEL_PATH)
            _MODEL_READY = True
        except Exception:
            _MODEL_READY = False
    return jsonify({
        "model_ready":  _MODEL_READY,
        "model_path":   MODEL_PATH,
        "model_name":   Path(MODEL_PATH).name,
        "is_fallback":  _IS_FALLBACK,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Image inference endpoint.

    Expects multipart/form-data with key 'file' containing an image.
    Returns JSON:
      {
        "image_b64":   "<base64 JPEG string>",
        "detections":  [{"class_name", "confidence", "bbox", "plate_text"}, ...],
        "total":       <int>,
        "latency_ms":  <int>
      }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a file under the key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        result = predict_image(
            image_bytes=image_bytes,
            model_path=MODEL_PATH,
            conf=0.25,
            iou=0.45,
            run_ocr=True,
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/predict/video", methods=["POST"])
def predict_video():
    """
    Video inference endpoint.

    Accepts an MP4/AVI/MOV/MKV upload and returns an annotated MP4 video.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a file under the key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    suffix = Path(file.filename).suffix.lower() or ".mp4"

    # Save upload to a temp file, run frame-by-frame inference, return result
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        file.save(tmp_in.name)
        input_path = tmp_in.name

    output_path = input_path.replace(suffix, "_out.mp4")

    try:
        import cv2
        from ultralytics import YOLO

        model = get_model(MODEL_PATH)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video file."}), 400

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=0.25, iou=0.45,
                                    save=False, verbose=False)
            annotated = results[0].plot()  # BGR numpy array
            out.write(annotated)

        cap.release()
        out.release()

        return send_file(
            output_path,
            mimetype="video/mp4",
            as_attachment=False,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass
        # output_path is sent via send_file — Flask handles cleanup after response


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
