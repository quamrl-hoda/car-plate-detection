"""
app.py — Production Flask API
==============================
Routes
------
GET  /              Serve UI (templates/index.html)
GET  /health        Model readiness probe (Docker / K8s)
POST /predict       Image inference  — multipart file OR base64 JSON
POST /predict/video Video inference  — multipart file → annotated MP4
GET  /train         Re-run full DVC pipeline, reset model cache
GET  /debug         Diagnose label counts + metrics without retraining

Environment variables
---------------------
YOLO_CONF   float  confidence threshold  (default 0.25)
YOLO_IOU    float  NMS IoU threshold     (default 0.45)
YOLO_OCR    1|0    enable Tesseract OCR  (default 1)
PORT        int    listen port           (default 8080)
"""
import io
import json
import os
import sys
import uuid
import base64
import subprocess
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS, cross_origin

# ── Windows CP-1252 console fix (YOLO logs contain emoji) ────────────
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,  encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer,  encoding="utf-8", errors="replace")
os.putenv("LANG",   "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH    = Path("artifacts/model_trainer/car_plate_detector/weights/best.pt")
UPLOAD_DIR    = Path("artifacts/uploads")
CONF          = float(os.getenv("YOLO_CONF", "0.10"))
IOU           = float(os.getenv("YOLO_IOU",  "0.40"))
RUN_OCR       = os.getenv("YOLO_OCR", "1") == "1"
PORT          = int(os.getenv("PORT", "8080"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────

def _tmp(suffix: str = ".jpg") -> Path:
    return UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"


def _read_file_bytes() -> tuple[bytes, str]:
    """
    Accept both multipart/form-data (key='file') and application/json
    (key='image' as base64 string, data-URI prefix stripped).
    Returns (raw_bytes, file_suffix).
    """
    if request.files.get("file"):
        f   = request.files["file"]
        ext = Path(f.filename or "img.jpg").suffix or ".jpg"
        return f.read(), ext
    body = request.get_json(silent=True) or {}
    b64  = body.get("image", "")
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    if not b64:
        raise ValueError("No file uploaded and no 'image' key in JSON body.")
    return base64.b64decode(b64), ".jpg"


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Used by Docker HEALTHCHECK and load-balancer probes."""
    return jsonify({
        "status":      "ok",
        "model_ready": MODEL_PATH.exists(),
        "model_path":  str(MODEL_PATH),
        "conf":        CONF,
        "iou":         IOU,
        "ocr_enabled": RUN_OCR,
    })


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    """
    Image inference.

    Accepts:
      • multipart/form-data  key='file'
      • application/json     key='image' (base64, data-URI ok)

    Returns:
      {
        image_b64   : annotated JPEG as base64,
        detections  : [{class_name, confidence, bbox, plate_text}],
        total       : int,
        latency_ms  : int
      }
    """
    try:
        raw, _ = _read_file_bytes()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        from carPlateDetection.pipeline.prediction_pipeline import predict_image
        result = predict_image(
            image_bytes = raw,
            model_path  = str(MODEL_PATH.resolve()),
            conf        = CONF,
            iou         = IOU,
            run_ocr     = RUN_OCR,
        )
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        app.logger.exception("predict error")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/video", methods=["POST"])
@cross_origin()
def predict_video():
    """
    Video inference.
    Accepts multipart file (mp4/avi/mov/mkv).
    Returns annotated MP4 as streaming download.
    Requires opencv-python.
    """
    try:
        import cv2
    except ImportError:
        return jsonify({"error": "opencv-python required for video. Run: pip install opencv-python"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file in request."}), 400

    f      = request.files["file"]
    suffix = Path(f.filename or "vid.mp4").suffix or ".mp4"
    tmp_in = _tmp(f"_in{suffix}")
    tmp_out = _tmp("_out.mp4")
    f.save(str(tmp_in))

    try:
        from carPlateDetection.pipeline.prediction_pipeline import get_model
        model = get_model(str(MODEL_PATH.resolve()))

        cap = cv2.VideoCapture(str(tmp_in))
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video file."}), 400

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        out = cv2.VideoWriter(
            str(tmp_out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h),
        )
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=CONF, iou=IOU, save=False, verbose=False)
            out.write(results[0].plot())
        cap.release()
        out.release()

        return send_file(
            str(tmp_out),
            mimetype="video/mp4",
            as_attachment=False,
            download_name="annotated.mp4",
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        app.logger.exception("video predict error")
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_in.exists():
            tmp_in.unlink()


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def train():
    """
    Trigger full DVC pipeline rerun via main.py.
    Resets the in-memory model cache so the new weights are loaded on
    the very next /predict call without restarting the server.
    """
    from carPlateDetection.pipeline.prediction_pipeline import reset_model
    try:
        r = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(Path(__file__).parent),
        )
        if r.returncode == 0:
            reset_model()
            return jsonify({"status": "success", "stdout": r.stdout[-4000:]})
        return jsonify({"status": "error", "stderr": r.stderr[-4000:]}), 500
    except Exception as e:
        app.logger.exception("train trigger error")
        return jsonify({"error": str(e)}), 500


@app.route("/debug")
def debug():
    """
    Diagnostic endpoint — call this to verify the pipeline without retraining.
    Shows label counts, a sample label line, and the latest eval metrics.
    """
    import glob as _glob
    import random as _rand

    info: dict = {}
    for split in ["train", "valid", "test"]:
        txts = _glob.glob(f"artifacts/data_transformation/{split}/labels/*.txt")
        xmls = _glob.glob(f"artifacts/data_ingestion/{split}/labels/*.xml")
        imgs = _glob.glob(f"artifacts/data_transformation/{split}/images/*")
        ne   = [f for f in txts if Path(f).read_text().strip()]
        info[split] = {
            "images":        len(imgs),
            "txt_labels":    len(txts),
            "xml_source":    len(xmls),
            "non_empty_txt": len(ne),
            "sample_label":  Path(_rand.choice(ne)).read_text()[:120] if ne else "EMPTY",
        }

    info["model_ready"] = MODEL_PATH.exists()
    m = Path("artifacts/model_evaluation/metrics.json")
    info["metrics"] = json.loads(m.read_text()) if m.exists() else "not found"
    return jsonify(info)


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  PlateVision — YOLOv8 Licence Plate API")
    print(f"  Model  : {MODEL_PATH}  (exists={MODEL_PATH.exists()})")
    print(f"  CONF   : {CONF}   IOU : {IOU}   OCR : {RUN_OCR}")
    print(f"  URL    : http://localhost:{PORT}")
    print("=" * 55)
    app.run(host="0.0.0.0", port=PORT, debug=False)
