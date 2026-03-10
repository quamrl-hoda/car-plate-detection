# PlateVision — YOLOv8 Licence Plate Detection

> End-to-end ML pipeline: data ingestion → XML label conversion → YOLOv8 fine-tuning → evaluation → Flask REST API with web UI and Tesseract OCR.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange) ![Flask](https://img.shields.io/badge/Flask-3.0-green) ![Tesseract](https://img.shields.io/badge/OCR-Tesseract_5.x-yellow) ![DVC](https://img.shields.io/badge/Pipeline-DVC-purple)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Tesseract Setup (Windows)](#tesseract-ocr-setup-windows)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [API Endpoints](#api-endpoints)
- [Environment Variables](#environment-variables)
- [Diagnosing Zero Detections](#diagnosing-zero-detections)
- [Docker](#docker)
- [CI/CD](#cicd-pipeline)
- [Training Hyperparameters](#training-hyperparameters)
- [Common Errors](#common-errors--fixes)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e .
pip install -r requirements.txt

# 2. Run full pipeline (download data → convert labels → train → evaluate)
python main.py

# 3. Start the web app
uv run app.py
# → http://localhost:8080
```

---

## Tesseract OCR Setup (Windows)

> ⚠️ Tesseract must be installed separately — it is **not** a Python package.

**Step 1** — Download the installer:
👉 https://github.com/UB-Mannheim/tesseract/wiki

Click `tesseract-ocr-w64-setup-5.x.x.exe` (64-bit)

**Step 2** — Run the installer, keep the default path:
```
C:\Program Files\Tesseract-OCR\
```

**Step 3** — Verify installation (open a **new** terminal):
```bash
where tesseract
# Expected: C:\Program Files\Tesseract-OCR\tesseract.exe
```

**Step 4** — Test from Python:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
print(pytesseract.get_tesseract_version())  # Should print: 5.x.x
```

> ✅ The path `C:\Program Files\Tesseract-OCR\tesseract.exe` is already hardcoded in `prediction_pipeline.py` — no PATH changes needed.

---

## Project Structure

```
car-plate-detection/
├── app.py                          Flask API (image + video inference + OCR)
├── main.py                         Run all 5 pipeline stages sequentially
├── params.yaml                     All training hyperparameters
├── config/
│   └── config.yaml                 Artifact paths + Google Drive dataset URL
├── dvc.yaml                        DVC pipeline definition (5 stages)
├── requirements.txt
├── setup.py
├── verify_labels.py                Diagnose label conversion before training
├── temp/                           Upload temp folder (auto-created)
├── artifacts/
│   ├── data_ingestion/             Downloaded + extracted dataset
│   ├── data_transformation/        YOLO-format .txt labels
│   ├── model_trainer/
│   │   └── car_plate_detector/
│   │       └── weights/
│   │           ├── best.pt         ← trained model used by the app
│   │           └── last.pt
│   └── model_evaluation/
│       └── metrics.json            mAP50, precision, recall
├── src/carPlateDetection/
│   ├── components/
│   │   ├── data_ingestion.py       Download + extract + XML→YOLO split
│   │   ├── data_validation.py      Verify split folders exist + non-empty
│   │   ├── data_transformation.py  Copy + convert XML→YOLO to output dir
│   │   ├── model_trainer.py        YOLOv8 fine-tune
│   │   └── model_evaluation.py     YOLO val() → metrics.json
│   ├── pipeline/
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_validation.py
│   │   ├── stage_03_data_transformation.py
│   │   ├── stage_04_model_trainer.py
│   │   ├── stage_05_model_evaluation.py
│   │   └── prediction_pipeline.py  Inference + OCR (used by app.py)
│   ├── config/configuration.py
│   ├── entity/config_entity.py
│   ├── constants/__init__.py
│   └── utils/common.py
├── templates/index.html            Web UI
├── static/css/style.css
├── static/js/main.js
├── tests/
│   ├── test_data_ingestion.py
│   ├── test_data_transformation.py
│   └── test_app.py
├── .github/workflows/ci-cd.yml     Lint → Test → Docker → Deploy
└── Dockerfile                      Multi-stage build
```

---

## Pipeline Stages

| Stage | Component | What it does |
|-------|-----------|--------------|
| 1 — Data Ingestion | `data_ingestion.py` | Downloads zip from Google Drive, extracts, auto-detects layout (4 types supported), splits 70/20/10 |
| 2 — Data Validation | `data_validation.py` | Verifies train/valid/test splits exist with non-empty images and labels folders |
| 3 — Data Transformation | `data_transformation.py` | Converts Pascal VOC XML → YOLO `.txt`, writes `data.yaml` with absolute paths |
| 4 — Model Trainer | `model_trainer.py` | Fine-tunes `yolov8n.pt`, saves `best.pt` and `last.pt` |
| 5 — Model Evaluation | `model_evaluation.py` | Runs `model.val()`, saves mAP50/precision/recall to `metrics.json` |

```bash
# Run with DVC (skips unchanged stages)
dvc repro

# Force retrain from a specific stage
uv run dvc repro --force data_transformation model_trainer model_evaluation
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Model status, active model path, metrics |
| POST | `/predict` | Image inference — multipart file or base64 JSON |
| POST | `/predict/video` | Video inference — returns annotated MP4 |
| GET | `/train` | Re-run full training pipeline, reload model |
| GET | `/debug` | Label counts + metrics diagnostics |

### `/predict` Request

```bash
# Multipart file
curl -X POST http://localhost:8080/predict -F "file=@Cars8.png"

# Base64 JSON
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_string>"}'
```

### `/predict` Response

```json
{
  "image_b64":  "<annotated JPEG as base64>",
  "detections": [
    {
      "class_name":  "license_plate",
      "confidence":  0.91,
      "bbox":        [120, 200, 280, 250],
      "plate_text":  "MH14 GN 9239"
    }
  ],
  "total":      1,
  "latency_ms": 169,
  "model_used": "artifacts/model_trainer/car_plate_detector/weights/best.pt"
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_CONF` | `0.25` | Confidence threshold — lower catches more plates |
| `YOLO_IOU` | `0.45` | NMS IoU threshold |
| `YOLO_OCR` | `1` | Enable Tesseract OCR (`0` to disable) |
| `PORT` | `8080` | Flask listen port |

---

## Diagnosing Zero Detections

> If `metrics.json` shows `mAP50: 0.0` the model trained on empty labels.

```bash
# Step 1 — Check label conversion
python verify_labels.py
# Expected: OK: 432 label files ready for training.

# Step 2 — Force retrain if labels were empty
uv run dvc repro --force data_transformation model_trainer model_evaluation

# Step 3 — Check via API
curl http://localhost:8080/debug

# Step 4 — Confirm metrics after training
cat artifacts/model_evaluation/metrics.json
# Expected: { "mAP50": 0.85, "precision": 0.89, "recall": 0.81 }
```

---

## Docker

```bash
# Build
docker build -t platevision .

# Run (mount trained weights)
docker run -p 8080:8080 \
  -v $(pwd)/artifacts/model_trainer:/app/artifacts/model_trainer \
  platevision

# With env overrides
docker run -p 8080:8080 \
  -e YOLO_CONF=0.15 \
  -e YOLO_OCR=1 \
  -v $(pwd)/artifacts/model_trainer:/app/artifacts/model_trainer \
  platevision
```

---

## CI/CD Pipeline

| Job | Trigger | What it does |
|-----|---------|--------------|
| `lint` | Every push | `flake8` + `isort --profile=black` |
| `test` | After lint | `pytest tests/ --cov=src` |
| `build-docker` | After tests (push only) | Docker Hub build + push |
| `deploy-staging` | `develop` branch | SSH deploy on port `8080` |
| `deploy-prod` | `main` + manual approval | SSH deploy on port `80` |

**Required GitHub Secrets:**
`DOCKERHUB_USERNAME` `DOCKERHUB_TOKEN` `STAGING_HOST` `STAGING_USER` `STAGING_SSH_KEY` `PROD_HOST` `PROD_USER` `PROD_SSH_KEY`

---

## Training Hyperparameters

All in `params.yaml`:

| Parameter | Value | Note |
|-----------|-------|------|
| `epochs` | `100` | Use 100+ — 50 gives `mAP50=0.0` |
| `image_size` | `416` | Faster than 640 on CPU |
| `batch_size` | `8` | Lower than 16 avoids CPU OOM |
| `device` | `cpu` | Change to `0` for CUDA GPU |
| `workers` | `0` | Avoids Windows fork errors |
| `cache` | `false` | Saves RAM on CPU |
| `optimizer` | `SGD` | Same as original notebook |
| `lr0` | `0.01` | Initial learning rate |
| `pretrained` | `true` | Fine-tune from `yolov8n.pt` |

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `mAP50: 0.0` in metrics.json | Labels empty — run `python verify_labels.py` then force retrain |
| `Plate detected but text unreadable` | Tesseract not installed — see [Tesseract Setup](#tesseract-ocr-setup-windows) |
| `NumPy 2.x` crash on startup | `uv pip install "numpy<2.0" --force-reinstall` |
| `No remote provided` on `dvc pull` | Run `python main.py` directly — no DVC remote configured |
| `labels.cache` blocking `dvc pull` | `dvc pull --force` |
| `best.pt not found` | Training not run yet — run `python main.py` first |
| `TesseractNotFoundError` | Install Tesseract — see [Tesseract Setup](#tesseract-ocr-setup-windows) |