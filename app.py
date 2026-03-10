"""
app.py
======
Flask API — direct translation of the Streamlit app.

Model path: artifacts/model_trainer/car_plate_detector/weights/best.pt
            (same as /Users/.../best.pt in the Streamlit app)

Routes:
  GET  /              Web UI
  GET  /health        Model status
  POST /predict       Image inference  (multipart file or base64 JSON)
  POST /predict/video Video inference  (multipart file → MP4)
  GET  /train         Re-run full DVC training pipeline
  GET  /debug         Diagnose labels + metrics
"""
import io, json, os, sys, uuid, base64, subprocess
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS, cross_origin

# ── Tesseract path (Windows) ──────────────────────────────────────
# Change this path if you installed Tesseract somewhere else
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
except ImportError:
    pass  # pytesseract not installed — OCR will be disabled

# Windows UTF-8 fix (YOLO logs contain emoji)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)

# ── Settings ──────────────────────────────────────────────────────
CONF    = float(os.getenv("YOLO_CONF", "0.25"))
IOU     = float(os.getenv("YOLO_IOU",  "0.45"))
RUN_OCR = os.getenv("YOLO_OCR", "1") == "1"
PORT    = int(os.getenv("PORT", "8080"))
TEMP    = Path("temp"); TEMP.mkdir(exist_ok=True)   # mirrors Streamlit's "temp" folder


def _tmp(suffix=".jpg"):
    return TEMP / f"{uuid.uuid4().hex}{suffix}"


def _parse_request():
    """Accept multipart file OR base64 JSON. Returns (raw_bytes, suffix)."""
    if request.files.get("file"):
        f = request.files["file"]
        return f.read(), Path(f.filename or "img.jpg").suffix or ".jpg"
    body = request.get_json(silent=True) or {}
    b64  = body.get("image", "")
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    if not b64:
        raise ValueError("Send multipart key='file' OR JSON {'image': base64_string}")
    return base64.b64decode(b64), ".jpg"


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    from carPlateDetection.pipeline.prediction_pipeline import active_model_path, model_ready
    m_path = Path("artifacts/model_evaluation/metrics.json")
    metrics = json.loads(m_path.read_text()) if m_path.exists() else {}
    return jsonify({
        "status":       "ok",
        "model_ready":  model_ready(),
        "active_model": active_model_path(),
        "metrics":      metrics,
        "conf":         CONF,
        "iou":          IOU,
        "ocr_enabled":  RUN_OCR,
    })


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    """Image inference — mirrors predict_and_save_image() exactly."""
    try:
        raw, _ = _parse_request()
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    try:
        from carPlateDetection.pipeline.prediction_pipeline import predict_image
        return jsonify(predict_image(raw, conf=CONF, iou=IOU, run_ocr=RUN_OCR))
    except Exception as e:
        app.logger.exception("predict error")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/video", methods=["POST"])
@cross_origin()
def predict_video():
    """Video inference — mirrors predict_and_plot_video() exactly."""
    try:
        import cv2
    except ImportError:
        return jsonify({"error": "pip install opencv-python"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f     = request.files["file"]
    ext   = Path(f.filename or "vid.mp4").suffix or ".mp4"
    in_p  = _tmp(f"_in{ext}")
    out_p = _tmp("_out.mp4")
    f.save(str(in_p))

    try:
        from carPlateDetection.pipeline.prediction_pipeline import get_model
        model = get_model()

        cap = cv2.VideoCapture(str(in_p))
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video file"}), 400

        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        out = cv2.VideoWriter(str(out_p),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = model.predict(rgb_frame, device="cpu",
                                      save=False, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence      = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{float(confidence)*100:.2f}%",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255, 0, 0), 2)
            out.write(frame)
        cap.release()
        out.release()
        return send_file(str(out_p), mimetype="video/mp4",
                         as_attachment=False, download_name="annotated.mp4")
    except Exception as e:
        app.logger.exception("video error")
        return jsonify({"error": str(e)}), 500
    finally:
        if in_p.exists(): in_p.unlink()


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def train():
    from carPlateDetection.pipeline.prediction_pipeline import reset_model
    try:
        r = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=str(Path(__file__).parent),
        )
        if r.returncode == 0:
            reset_model()
            return jsonify({"status": "success", "stdout": r.stdout[-4000:]})
        return jsonify({"status": "error", "stderr": r.stderr[-4000:]}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug")
def debug():
    import glob as g, random as rnd
    info = {}
    for split in ["train", "valid", "test"]:
        txts = g.glob(f"artifacts/data_transformation/{split}/labels/*.txt")
        xmls = g.glob(f"artifacts/data_ingestion/{split}/labels/*.xml")
        imgs = g.glob(f"artifacts/data_transformation/{split}/images/*")
        ne   = [f for f in txts if Path(f).read_text().strip()]
        info[split] = {
            "images":        len(imgs),
            "txt_labels":    len(txts),
            "xml_source":    len(xmls),
            "non_empty_txt": len(ne),
            "sample":        Path(rnd.choice(ne)).read_text()[:80] if ne else "EMPTY",
        }
    m = Path("artifacts/model_evaluation/metrics.json")
    info["metrics"] = json.loads(m.read_text()) if m.exists() else "not found"
    from carPlateDetection.pipeline.prediction_pipeline import active_model_path
    info["active_model"] = active_model_path()
    return jsonify(info)


if __name__ == "__main__":
    from carPlateDetection.pipeline.prediction_pipeline import active_model_path, model_ready
    print("=" * 55)
    print("  PlateVision — YOLOv8 Licence Plate Detection")
    print(f"  Model  : {active_model_path()}")
    print(f"  Ready  : {model_ready()}")
    print(f"  CONF:{CONF}  IOU:{IOU}  OCR:{RUN_OCR}  PORT:{PORT}")
    print(f"  URL    : http://localhost:{PORT}")
    print("=" * 55)
    app.run(host="0.0.0.0", port=PORT, debug=False)