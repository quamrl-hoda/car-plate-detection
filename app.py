import os
import io
import uuid
import base64
import sys
import time
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

# ── Encoding fix for Windows CP-1252 terminals ────────────────────────
if hasattr(sys.stdout, "buffer"):
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH    = Path("artifacts/model_trainer/car_plate_detector/weights/best.pt")
UPLOAD_FOLDER = Path("artifacts/uploads")
CONF          = 0.25
IOU           = 0.45
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ── Lazy model loader ─────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python main.py` or `uv run dvc repro` first."
            )
        from ultralytics import YOLO
        _model = YOLO(str(MODEL_PATH))
    return _model


def numpy_to_base64_jpeg(np_array) -> str:
    """Convert a numpy BGR image array to a base64 JPEG string.

    Uses PIL instead of cv2 so opencv is NOT required at runtime.
    YOLO's .plot() returns a BGR numpy array — we flip to RGB for PIL.
    """
    from PIL import Image
    import numpy as np

    # YOLO returns BGR — convert to RGB
    rgb = np_array[:, :, ::-1]
    pil_img = Image.fromarray(rgb.astype("uint8"))

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_upload(data_bytes: bytes, suffix: str = ".jpg") -> Path:
    """Save raw bytes to a temp file and return its path."""
    tmp = UPLOAD_FOLDER / f"{uuid.uuid4().hex}{suffix}"
    tmp.write_bytes(data_bytes)
    return tmp


# ════════════════════════════════════════════════════════════════════════
#  Routes
# ════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
@cross_origin()
def index():
    return render_template("index.html")


# ── Health ────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
@cross_origin()
def health():
    return jsonify({
        "status":      "ok",
        "model_ready": MODEL_PATH.exists(),
        "model_path":  str(MODEL_PATH),
    })


# ── Predict (base64 JSON) ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    """Run YOLOv8 inference on a base64-encoded image.

    Request  : POST /predict
               Content-Type: application/json
               Body: { "image": "<base64 string>" }

    Response : {
                 "image_b64":  "<annotated image as base64 JPEG>",
                 "detections": [{"class_name": str, "confidence": float, "bbox": [x1,y1,x2,y2]}],
                 "total":      int,
                 "latency_ms": int
               }
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Send JSON with key 'image' (base64 string)."}), 400

    b64_str = data["image"]
    if "," in b64_str:                        # strip data-URI prefix
        b64_str = b64_str.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(b64_str)
    except Exception as e:
        return jsonify({"error": f"Invalid base64: {e}"}), 400

    tmp_path = save_upload(img_bytes, ".jpg")

    try:
        model = get_model()
        t0    = time.perf_counter()
        results = model.predict(
            source=str(tmp_path),
            conf=CONF, iou=IOU,
            save=False, verbose=False,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000)

        # annotated image → base64 (PIL, no cv2 needed)
        img_b64 = numpy_to_base64_jpeg(results[0].plot())

        detections = [
            {
                "class_name": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "bbox":       [round(v, 1) for v in box.xyxy[0].tolist()],
            }
            for box in results[0].boxes
        ]

        return jsonify({
            "image_b64":  img_b64,
            "detections": detections,
            "total":      len(detections),
            "latency_ms": latency_ms,
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        app.logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ── Predict (multipart file upload) ───────────────────────────────────
@app.route("/predict/upload", methods=["POST"])
@cross_origin()
def predict_upload():
    """Alternative endpoint for curl / Postman file uploads.

    curl -X POST http://localhost:8080/predict/upload -F "file=@plate.jpg"
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected."}), 400

    suffix   = Path(f.filename).suffix or ".jpg"
    tmp_path = save_upload(f.read(), suffix)

    try:
        model  = get_model()
        t0     = time.perf_counter()
        results = model.predict(
            source=str(tmp_path),
            conf=CONF, iou=IOU,
            save=False, verbose=False,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000)

        img_b64 = numpy_to_base64_jpeg(results[0].plot())

        detections = [
            {
                "class_name": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "bbox":       [round(v, 1) for v in box.xyxy[0].tolist()],
            }
            for box in results[0].boxes
        ]

        return jsonify({
            "image_b64":  img_b64,
            "detections": detections,
            "total":      len(detections),
            "latency_ms": latency_ms,
        })

    except Exception as e:
        app.logger.exception("Upload prediction error")
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ── Retrain ───────────────────────────────────────────────────────────
@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def train():
    """Trigger a full pipeline run via main.py."""
    global _model
    try:
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(Path(__file__).parent),
        )
        if result.returncode == 0:
            _model = None          # force model reload on next predict
            return jsonify({
                "status":  "success",
                "message": "Training complete. Model reloaded.",
                "stdout":  result.stdout[-2000:],
            })
        else:
            return jsonify({
                "status":  "error",
                "message": "Training failed.",
                "stderr":  result.stderr[-2000:],
            }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  NeuralScan — Car Plate Detection API")
    print(f"  Model : {MODEL_PATH}")
    print(f"  Ready : {MODEL_PATH.exists()}")
    print("  URL   : http://localhost:8080")
    print("=" * 60)
    app.run(host="0.0.0.0", port=8080, debug=False)