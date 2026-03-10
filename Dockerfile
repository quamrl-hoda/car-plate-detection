# ══════════════════════════════════════════════════════════════════════
# Dockerfile — PlateVision YOLOv8 Licence Plate Detection
# ══════════════════════════════════════════════════════════════════════
# Multi-stage build:
#   builder  — installs Python dependencies into a venv
#   runtime  — lean image, copies only the venv + app code
#
# The trained model weights (best.pt) are NOT baked into the image.
# They are mounted at runtime via Docker volume:
#   -v /host/path/to/model_trainer:/app/artifacts/model_trainer
# ══════════════════════════════════════════════════════════════════════

# ── Stage 1: builder ─────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps needed to build wheels (cv2, Pillow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only what's needed to install deps (layer-cache friendly)
COPY requirements.txt setup.py ./
COPY src/ ./src/

# Install into an isolated venv
RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt && \
    /venv/bin/pip install --no-cache-dir -e .


# ── Stage 2: runtime ─────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

# Tesseract OCR (optional — remove if you don't need plate text reading)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy venv and application code
COPY --from=builder /venv     /venv
COPY --from=builder /app/src  ./src
COPY config/      ./config/
COPY params.yaml  main.py app.py ./
COPY templates/   ./templates/
COPY static/      ./static/

# Create writable dirs
RUN mkdir -p artifacts/uploads logs

# Set env
ENV PATH="/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV YOLO_CONF=0.25
ENV YOLO_IOU=0.45
ENV YOLO_OCR=1
ENV PORT=8080

EXPOSE 8080

# Health check — used by Docker and Kubernetes
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["python", "app.py"]
