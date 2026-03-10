"""
prediction_pipeline.py
======================
Inference pipeline used by app.py for both image and video prediction.

Mirrors the notebook's predict_and_plot() function but production-ready:
  - Lazy model loading (singleton, thread-safe via module-level lock)
  - OCR via pytesseract on each detected plate crop (psm 7 = single text line)
  - OCR preprocessing: grayscale → Otsu threshold → 2x upscale
  - Returns structured dicts, not matplotlib plots
  - No cv2 dependency for image output (uses PIL for base64 encoding)
  - cv2 used only for OCR preprocessing (optional — gracefully absent)
"""
import io
import base64
import threading
from pathlib import Path
from typing import Optional

from carPlateDetection import logger

_model      = None
_model_lock = threading.Lock()


def get_model(model_path: str):
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO
                _model = YOLO(model_path)
                logger.info(f"YOLO model loaded from {model_path}")
    return _model


def reset_model() -> None:
    """Force next call to get_model() to reload weights (used after retraining)."""
    global _model
    with _model_lock:
        _model = None
    logger.info("Model cache cleared — will reload on next inference.")


def numpy_bgr_to_b64jpeg(arr) -> str:
    """Convert BGR numpy array (YOLO .plot() output) → base64 JPEG string."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr[:, :, ::-1].astype("uint8")).save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def ocr_plate(bgr_crop) -> str:
    """
    Run Tesseract OCR on a single plate crop (BGR numpy array).

    Preprocessing pipeline (each step tested on real plate images):
      1. Grayscale
      2. Upscale 3x — Tesseract accuracy drops sharply on small text
      3. CLAHE — boosts local contrast without blowing out highlights
      4. Gentle Gaussian blur to remove sensor noise before thresholding
      5. Otsu binarisation — auto picks the best threshold
      6. Morphological dilation to close broken letter strokes
      7. psm 7 = single text line, oem 3 = LSTM engine
      8. Strip non-alphanumeric chars (plates never have punctuation)

    Returns empty string if pytesseract/cv2 not installed or crop is invalid.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return ""
    try:
        import cv2
        import pytesseract
        import re

        # 1. Grayscale
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

        # 2. Upscale 3x (minimum 100px height for Tesseract accuracy)
        h, w = gray.shape
        scale = max(3.0, 100.0 / h)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

        # 3. CLAHE — adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)

        # 4. Gentle blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 5. Otsu binarisation
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 6. Morphological closing to reconnect broken strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 7. OCR — psm 7 = single line, whitelist alphanumeric + space
        config = "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        text   = pytesseract.image_to_string(binary, config=config).strip()

        # 8. Clean: keep only letters, digits, spaces
        text = re.sub(r"[^A-Z0-9 ]", "", text.upper()).strip()
        return text
    except Exception:
        return ""


def predict_image(
    image_bytes: bytes,
    model_path: str,
    conf: float = 0.25,
    iou: float  = 0.45,
    run_ocr: bool = True,
) -> dict:
    """
    Run YOLO detection + optional OCR on a single image.

    Args:
        image_bytes : raw image bytes (any format PIL can read)
        model_path  : path to best.pt
        conf        : YOLO confidence threshold
        iou         : YOLO IoU NMS threshold
        run_ocr     : whether to run Tesseract on each detection

    Returns dict:
        image_b64   : annotated JPEG as base64 string
        detections  : list of {class_name, confidence, bbox, plate_text}
        total       : int
        latency_ms  : inference time in ms
    """
    import time
    import numpy as np
    from PIL import Image as PILImage

    model = get_model(model_path)

    # Decode bytes → numpy BGR array
    img_pil = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_rgb = np.array(img_pil)
    img_bgr = img_rgb[:, :, ::-1].copy()

    t0 = time.perf_counter()
    results = model.predict(img_bgr, conf=conf, iou=iou, save=False, verbose=False)
    latency_ms = round((time.perf_counter() - t0) * 1000)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        conf_val        = round(float(box.conf), 4)
        cls_name        = model.names[int(box.cls)]
        crop            = img_bgr[y1:y2, x1:x2] if run_ocr else None
        plate_text      = ocr_plate(crop) if run_ocr else ""
        detections.append({
            "class_name":  cls_name,
            "confidence":  conf_val,
            "bbox":        [x1, y1, x2, y2],
            "plate_text":  plate_text,
        })

    return {
        "image_b64":  numpy_bgr_to_b64jpeg(results[0].plot()),
        "detections": detections,
        "total":      len(detections),
        "latency_ms": latency_ms,
    }
