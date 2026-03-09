import os
import io
import base64
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #
MODEL_PATH = "artifacts/model_trainer/car_plate_detector/weights/best.pt"
UPLOAD_FOLDER = "artifacts/uploads"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------------------------------------------ #
#  Load model (lazy – loaded on first request)
# ------------------------------------------------------------------ #
_model = None


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
    return _model


# ------------------------------------------------------------------ #
#  HTML template (minimal single-page UI)
# ------------------------------------------------------------------ #
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Car Plate Detection</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px;
               margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
        h1   { color: #333; }
        .card { background: white; padding: 24px; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 24px; }
        input[type=file] { margin: 12px 0; }
        button { background: #007bff; color: white; padding: 10px 20px;
                 border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        #result-img { max-width: 100%; margin-top: 16px; border-radius: 4px; }
        #detections { margin-top: 16px; }
        .det-item { background: #e8f4fd; padding: 8px 12px;
                    border-radius: 4px; margin: 4px 0; font-size: 14px; }
    </style>
</head>
<body>
    <h1>🚗 Car Plate Detection</h1>
    <div class="card">
        <h2>Upload an Image</h2>
        <input type="file" id="imageInput" accept="image/*" />
        <br/>
        <button onclick="detect()">Detect Plates</button>
    </div>
    <div class="card" id="result-card" style="display:none">
        <h2>Detection Results</h2>
        <img id="result-img" src="" alt="Result"/>
        <div id="detections"></div>
    </div>

    <script>
    async function detect() {
        const input = document.getElementById('imageInput');
        if (!input.files.length) { alert('Please select an image.'); return; }

        const formData = new FormData();
        formData.append('file', input.files[0]);

        const res = await fetch('/predict', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.error) { alert('Error: ' + data.error); return; }

        document.getElementById('result-card').style.display = 'block';
        document.getElementById('result-img').src =
            'data:image/jpeg;base64,' + data.image_b64;

        const div = document.getElementById('detections');
        div.innerHTML = '';
        data.detections.forEach(d => {
            div.innerHTML +=
                `<div class="det-item">
                    <b>${d.class_name}</b> &nbsp;|&nbsp;
                    Confidence: ${(d.confidence * 100).toFixed(1)}%  &nbsp;|&nbsp;
                    Box: [${d.bbox.map(v => v.toFixed(0)).join(', ')}]
                 </div>`;
        });
        if (!data.detections.length) {
            div.innerHTML = '<p>No plates detected.</p>';
        }
    }
    </script>
</body>
</html>
"""


# ------------------------------------------------------------------ #
#  Routes
# ------------------------------------------------------------------ #
@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    """Accept an uploaded image, run inference, return annotated image + detections."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save temp file
    ext = Path(file.filename).suffix or ".jpg"
    temp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
    file.save(temp_path)

    try:
        import cv2
        import numpy as np

        model = get_model()
        results = model.predict(
            source=temp_path,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            save=False,
            verbose=False,
        )

        # Annotated image → base64
        annotated = results[0].plot()   # BGR numpy array
        _, buffer = cv2.imencode(".jpg", annotated)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        # Detection list
        detections = []
        for box in results[0].boxes:
            detections.append({
                "class_name": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist(),
            })

        return jsonify({
            "image_b64": img_b64,
            "detections": detections,
            "total": len(detections),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/health", methods=["GET"])
def health():
    model_ready = os.path.exists(MODEL_PATH)
    return jsonify({
        "status": "ok",
        "model_ready": model_ready,
        "model_path": MODEL_PATH,
    })


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
