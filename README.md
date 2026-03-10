# PlateVision — YOLOv8 Licence Plate Detection

End-to-end production ML pipeline: data ingestion → XML label conversion →
YOLOv8 training → evaluation → Flask API with OCR (Tesseract).

## Project structure

```
car-plate-detection/
├── src/carPlateDetection/
│   ├── components/
│   │   ├── data_ingestion.py       Download + extract + XML→YOLO split
│   │   ├── data_validation.py      Verify split folders exist + non-empty
│   │   ├── data_transformation.py  Copy + XML→YOLO convert to output dir
│   │   ├── model_trainer.py        YOLOv8 fine-tune
│   │   └── model_evaluation.py     YOLO val() → metrics.json
│   ├── pipeline/
│   │   ├── stage_0[1-5]_*.py       DVC stage entry points
│   │   └── prediction_pipeline.py  Inference + OCR (used by app.py)
│   ├── config/configuration.py     ConfigurationManager
│   ├── entity/config_entity.py     Frozen dataclasses
│   └── utils/common.py             read_yaml, save_json, etc.
├── tests/
│   ├── test_data_ingestion.py      Unit tests — XML conversion, file ID parsing
│   ├── test_data_transformation.py Unit tests — label conversion, data.yaml paths
│   └── test_app.py                 Integration tests — Flask routes
├── .github/workflows/ci-cd.yml    Lint → Test → Docker build → Deploy
├── Dockerfile                      Multi-stage build (builder + runtime)
├── app.py                          Flask API (image + video inference + OCR)
├── main.py                         Run all 5 pipeline stages sequentially
├── dvc.yaml                        DVC pipeline definition
├── config/config.yaml              Artifact paths + Google Drive URL
├── params.yaml                     All training hyperparameters
├── requirements.txt
├── setup.py
└── verify_labels.py                Diagnose label conversion before training
```

## Quick start

```bash
# 1. Install
pip install -e .
pip install -r requirements.txt

# 2. Run full pipeline (download → train → evaluate)
python main.py
# OR with DVC (recommended — skips unchanged stages)
dvc repro

# 3. Start API server
python app.py          # → http://localhost:8080
# OR
uv run python app.py
```

## CI/CD pipeline

| Job | Trigger | What it does |
|-----|---------|-------------|
| lint | Every push | flake8 + isort |
| test | After lint | pytest (unit + integration) |
| build-docker | After tests, push only | Docker Hub push |
| deploy-staging | `develop` branch | SSH pull & run on staging |
| deploy-prod | `main` branch + approval | SSH pull & run on production |

## API endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Readiness probe |
| POST | `/predict` | Image inference (multipart or base64 JSON) |
| POST | `/predict/video` | Video inference → annotated MP4 |
| GET | `/train` | Re-run pipeline, reload model |
| GET | `/debug` | Label counts + metrics diagnostics |

## Response format (`/predict`)

```json
{
  "image_b64":  "<annotated JPEG base64>",
  "detections": [
    {
      "class_name":  "license_plate",
      "confidence":  0.91,
      "bbox":        [120, 200, 280, 250],
      "plate_text":  "LTM 378"
    }
  ],
  "total":      1,
  "latency_ms": 168
}
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_CONF` | `0.25` | Confidence threshold |
| `YOLO_IOU` | `0.45` | NMS IoU threshold |
| `YOLO_OCR` | `1` | Enable Tesseract OCR (0 to disable) |
| `PORT` | `8080` | Flask listen port |

## Docker

```bash
# Build
docker build -t platevision .

# Run (mount trained weights)
docker run -p 8080:8080 \
  -v /path/to/artifacts/model_trainer:/app/artifacts/model_trainer \
  platevision
```

## Diagnosing zero detections

```bash
# 1. Check label conversion
python verify_labels.py

# 2. Check via API
curl http://localhost:8080/debug

# 3. Force retransform + retrain
dvc repro --force data_transformation model_trainer model_evaluation
```
