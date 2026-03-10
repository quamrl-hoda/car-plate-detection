"""
prediction_pipeline.py
======================
Mirrors the Streamlit app's predict functions exactly.

Model is loaded from:
  artifacts/model_trainer/car_plate_detector/weights/best.pt
  (same file as /Users/.../best.pt in the Streamlit app)

If best.pt doesn't exist yet (training not run), falls back to
yolov8n.pt (pretrained) so the app always works.
"""
import io, re, base64, threading, time
from pathlib import Path
from carPlateDetection import logger

# ── Model singleton (thread-safe) ─────────────────────────────────
_model             = None
_model_path_loaded = None
_lock              = threading.Lock()

# Path mirrors the Streamlit app's hardcoded /Users/.../best.pt
BEST_PT   = Path("artifacts/model_trainer/car_plate_detector/weights/best.pt")
PRETRAIN  = "yolov8n.pt"   # fallback — auto-downloaded by ultralytics


def _pick_model() -> str:
    """
    Use best.pt if it exists (was trained).
    Fall back to pretrained yolov8n.pt so the app works immediately.
    """
    if BEST_PT.exists():
        logger.info(f"Loading custom model: {BEST_PT}")
        return str(BEST_PT)
    logger.warning(f"best.pt not found at {BEST_PT}. Using pretrained yolov8n.pt")
    return PRETRAIN


def model_ready() -> bool:
    return BEST_PT.exists()


def active_model_path() -> str:
    return _model_path_loaded or _pick_model()


def get_model(path: str = None):
    global _model, _model_path_loaded
    target = path or _pick_model()
    if _model is None or _model_path_loaded != target:
        with _lock:
            if _model is None or _model_path_loaded != target:
                from ultralytics import YOLO
                _model             = YOLO(target)
                _model_path_loaded = target
                logger.info(f"Model loaded: {target}")
    return _model


def reset_model():
    global _model, _model_path_loaded
    with _lock:
        _model = None
        _model_path_loaded = None
    logger.info("Model cache cleared — will reload on next inference.")


# ── OCR — pre-import pytesseract once so errors are visible ───────
import os as _os

_pytesseract = None
try:
    import pytesseract as _pytesseract_mod
    # Point to the Windows binary (UB-Mannheim default install path)
    if _os.name == "nt":
        for _tess_path in [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]:
            if _os.path.isfile(_tess_path):
                _pytesseract_mod.pytesseract.tesseract_cmd = _tess_path
                break
    _pytesseract = _pytesseract_mod
    logger.info("pytesseract loaded — OCR enabled")
except Exception as _e:
    logger.warning(
        f"pytesseract import failed — OCR disabled. "
        f"Fix: uv pip install --upgrade pyarrow   Error: {_e}"
    )


def ocr_plate(bgr_crop) -> str:
    """
    Tesseract OCR on a plate crop (BGR numpy array).
    Returns "" silently if pytesseract/tesseract is not available.
    """
    if bgr_crop is None or bgr_crop.size == 0 or _pytesseract is None:
        return ""
    try:
        import cv2
        gray  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        h, w  = gray.shape
        scale = max(3.0, 100.0 / h) if h > 0 else 3.0
        gray  = cv2.resize(gray, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)
        gray  = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k     = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        th    = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
        cfg   = ("--psm 7 --oem 3 "
                 "-c tessedit_char_whitelist="
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        text  = _pytesseract.image_to_string(th, config=cfg).strip()
        return re.sub(r"[^A-Z0-9 ]", "", text.upper()).strip()
    except Exception:
        return ""


# ── Image prediction ──────────────────────────────────────────────

def _bgr_to_b64jpeg(arr) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr[:, :, ::-1].astype("uint8")).save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def predict_image(
    image_bytes: bytes,
    model_path: str = None,
    conf: float     = 0.25,
    iou: float      = 0.45,
    run_ocr: bool   = True,
) -> dict:
    """
    Detect plates in a single image.
    Mirrors predict_and_save_image() from the Streamlit app.

    Returns:
        image_b64   : annotated JPEG as base64 string
        detections  : [{class_name, confidence, bbox, plate_text}]
        total       : int
        latency_ms  : int
        model_used  : str
    """
    import numpy as np
    from PIL import Image as PILImage

    model   = get_model(model_path)
    img_pil = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_rgb = np.array(img_pil)
    img_bgr = img_rgb[:, :, ::-1].copy()   # matches cv2.cvtColor BGR→RGB in Streamlit

    t0      = time.perf_counter()
    results = model.predict(img_bgr, conf=conf, iou=iou,
                            device="cpu", save=False, verbose=False)
    ms      = round((time.perf_counter() - t0) * 1000)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        conf_val   = round(float(box.conf), 4)
        cls_name   = model.names[int(box.cls)]
        crop       = img_bgr[y1:y2, x1:x2]
        plate_text = ocr_plate(crop) if run_ocr else ""
        detections.append({
            "class_name": cls_name,
            "confidence": conf_val,
            "bbox":       [x1, y1, x2, y2],
            "plate_text": plate_text,
        })

    # ── Draw annotations — notebook style ────────────────────────────
    # Green box + large red confidence % + white label box with plate text
    annotated = img_bgr.copy()
    try:
        import cv2

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            pct_text  = f"{det['confidence'] * 100:.2f}%"
            plate_txt = det["plate_text"]
            font      = cv2.FONT_HERSHEY_SIMPLEX
            box_w     = max(x2 - x1, 1)

            # 1. Thick green bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 2. Large bold RED confidence % above the box
            conf_scale = max(1.4, box_w / 80)
            conf_thick = max(3, int(conf_scale * 2))
            (_, ch), _ = cv2.getTextSize(pct_text, font, conf_scale, conf_thick)
            cy = max(y1 - 8, ch + 4)
            cv2.putText(annotated, pct_text,
                        (x1, cy), font, conf_scale,
                        (0, 0, 255), conf_thick, cv2.LINE_AA)

            # 3. Solid white label box with green border + dark plate text
            if plate_txt:
                lbl_scale = max(0.9, box_w / 120)
                lbl_thick = max(2, int(lbl_scale * 2))
                (pw, ph), lb = cv2.getTextSize(plate_txt, font, lbl_scale, lbl_thick)
                px, py = 8, 6
                lx1, ly1 = x1, y2 - ph - lb - py * 2
                lx2, ly2 = x1 + pw + px * 2, y2
                cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), (255, 255, 255), -1)
                cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
                cv2.putText(annotated, plate_txt,
                            (lx1 + px, ly2 - lb - py),
                            font, lbl_scale, (20, 20, 20),
                            lbl_thick, cv2.LINE_AA)

    except ImportError:
        annotated = results[0].plot()   # fallback if cv2 missing

    return {
        "image_b64":  _bgr_to_b64jpeg(annotated),
        "detections": detections,
        "total":      len(detections),
        "latency_ms": ms,
        "img_shape":  list(img_bgr.shape[:2]),
        "speed": {
            "preprocess_ms":  round(results[0].speed.get("preprocess",  0), 1),
            "inference_ms":   round(results[0].speed.get("inference",   ms), 1),
            "postprocess_ms": round(results[0].speed.get("postprocess", 0), 1),
        },
        "model_used": _model_path_loaded or "unknown",
    }
