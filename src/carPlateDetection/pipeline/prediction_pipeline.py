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

# ── Tesseract path (Windows) ──────────────────────────────────────
try:
    import pytesseract as _pt
    _TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if Path(_TESS).exists():
        _pt.pytesseract.tesseract_cmd = _TESS
except ImportError:
    pass

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


# ── OCR ───────────────────────────────────────────────────────────

def ocr_plate(bgr_crop) -> str:
    """
    Tesseract OCR on a plate crop (BGR numpy array).
    Returns "" if tesseract not installed — never crashes.

    Preprocessing:
      1. Grayscale
      2. Upscale to min 100px height (Tesseract needs large text)
      3. CLAHE adaptive contrast
      4. Gaussian blur (denoise)
      5. Otsu threshold
      6. Morphological close (reconnect broken letter strokes)
      7. psm 7 = single text line + alphanumeric whitelist
      8. Strip non-alphanumeric output
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return ""
    try:
        import cv2, pytesseract
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
        text  = pytesseract.image_to_string(th, config=cfg).strip()
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

    return {
        "image_b64":  _bgr_to_b64jpeg(results[0].plot()),
        "detections": detections,
        "total":      len(detections),
        "latency_ms": ms,
        "model_used": _model_path_loaded or "unknown",
    }