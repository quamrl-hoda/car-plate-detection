"""
data_transformation.py
======================
Copies the ingested train/valid/test splits to artifacts/data_transformation/
and converts any Pascal VOC XML labels → YOLO .txt format.

Why XML conversion here AND in data_ingestion?
  - data_ingestion handles the flat Kaggle layout (images/ + annotations/)
    and converts XML while splitting.
  - data_transformation handles the case where the dataset arrived already
    split (Layout A / B) but still has XML labels in the labels/ folders.
  Both stages are idempotent — running twice is safe.

Output data.yaml always uses absolute resolved paths so YOLO never
prepends its own working directory and doubles the path.
"""
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from carPlateDetection import logger
from carPlateDetection.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    # ── Public API ────────────────────────────────────────────────────

    def process_images(self) -> None:
        total_converted = 0
        splits = ["train", "valid", "test"]

        for split in splits:
            src_img = Path(self.config.data_dir) / split / "images"
            src_lbl = Path(self.config.data_dir) / split / "labels"
            dst_img = Path(self.config.root_dir)  / split / "images"
            dst_lbl = Path(self.config.root_dir)  / split / "labels"

            if not src_img.exists():
                logger.warning(f"Split '{split}' images not found at {src_img} — skipping.")
                continue

            # Clean and recreate destination folders
            for d in [dst_img, dst_lbl]:
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)

            # Copy images
            imgs = list(src_img.glob("*"))
            for f in imgs:
                shutil.copy2(f, dst_img / f.name)
            logger.info(f"[{split}] Copied {len(imgs)} images")

            # Convert / copy labels
            if src_lbl.exists():
                converted, skipped = self._convert_labels(src_lbl, dst_lbl)
                total_converted += converted
                logger.info(f"[{split}] Labels: {converted} written, {skipped} skipped")
            else:
                logger.warning(f"[{split}] No labels folder at {src_lbl}")

        if total_converted == 0:
            raise RuntimeError(
                "ZERO labels were written to the transformation directory.\n"
                "The model will train on images with no ground-truth — abort.\n"
                "Run verify_labels.py to diagnose the annotation files."
            )

        self._write_data_yaml(Path(self.config.root_dir))
        logger.info(f"Data transformation complete. Total labels: {total_converted}")

    # ── Label conversion ──────────────────────────────────────────────

    def _convert_labels(self, src: Path, dst: Path) -> tuple[int, int]:
        converted = skipped = 0
        for f in src.iterdir():
            ext = f.suffix.lower()
            if ext == ".xml":
                try:
                    lines    = self._xml_to_yolo_lines(f)
                    txt_path = dst / (f.stem + ".txt")
                    txt_path.write_text("\n".join(lines))
                    converted += 1
                except Exception as e:
                    logger.warning(f"  XML→TXT failed {f.name}: {e}")
                    skipped += 1
            elif ext == ".txt":
                shutil.copy2(f, dst / f.name)
                converted += 1
            else:
                skipped += 1
        return converted, skipped

    @staticmethod
    def _xml_to_yolo_lines(xml_path: Path) -> list[str]:
        """
        Pascal VOC XML → YOLO normalised [class cx cy w h] per bounding box.
        class_id is always 0 (license_plate is the only class).
        All coordinates normalised to [0, 1] and clamped to image bounds.
        """
        root  = ET.parse(xml_path).getroot()
        sz    = root.find("size")
        img_w = float(sz.findtext("width",  "0") if sz else "0")
        img_h = float(sz.findtext("height", "0") if sz else "0")
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"Invalid <size> {img_w}x{img_h} in {xml_path.name}")

        lines = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = max(0.0, min(float(bb.findtext("xmin", "0")), img_w))
            ymin = max(0.0, min(float(bb.findtext("ymin", "0")), img_h))
            xmax = max(0.0, min(float(bb.findtext("xmax", "0")), img_w))
            ymax = max(0.0, min(float(bb.findtext("ymax", "0")), img_h))
            if xmax <= xmin or ymax <= ymin:
                logger.warning(f"Degenerate bbox in {xml_path.name} — skipping")
                continue
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return lines

    # ── data.yaml ────────────────────────────────────────────────────

    def _write_data_yaml(self, root: Path) -> None:
        """
        Write data.yaml with ABSOLUTE resolved paths.
        Using relative paths causes YOLO to prepend its cwd and double the path,
        leading to 'directory not found' errors on training.
        """
        import yaml

        src_yaml = Path(self.config.data_dir) / "data.yaml"
        data: dict = {}
        if src_yaml.exists():
            with open(src_yaml) as f:
                data = yaml.safe_load(f) or {}

        data["train"] = (root / "train" / "images").resolve().as_posix()
        data["val"]   = (root / "valid" / "images").resolve().as_posix()
        data["test"]  = (root / "test"  / "images").resolve().as_posix()
        data["nc"]    = 1
        data["names"] = ["license_plate"]

        out = root / "data.yaml"
        with open(out, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.info(f"data.yaml written → {out}")
        logger.info(f"  train : {data['train']}")
        logger.info(f"  val   : {data['val']}")
        logger.info(f"  test  : {data['test']}")
