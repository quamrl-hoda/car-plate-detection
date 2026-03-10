"""
data_ingestion.py
=================
Downloads the dataset zip from Google Drive, extracts it, and normalises
the folder structure to a strict YOLO layout:

    artifacts/data_ingestion/
        train/images/   train/labels/
        valid/images/   valid/labels/
        test/images/    test/labels/
        data.yaml

Handles four real-world dataset layouts automatically:
  A) Already split  — train/ valid/ test/ exist at root
  B) Nested         — SomeName/train/ valid/ test/ inside zip
  C) Flat Kaggle    — images/ + annotations/ (Pascal VOC XML or YOLO TXT)
  D) Inner zip      — zip-inside-zip
"""
import os
import re
import shutil
import zipfile
import random
from pathlib import Path

import gdown

from carPlateDetection import logger
from carPlateDetection.utils.common import get_size
from carPlateDetection.entity.config_entity import DataIngestionConfig

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTS = {".xml", ".txt"}


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    # ── Public API ────────────────────────────────────────────────────

    def download_file(self) -> None:
        """Download dataset zip from Google Drive (skips if already present)."""
        dst = Path(self.config.local_data_file)
        if dst.exists() and dst.stat().st_size > 0:
            logger.info(f"Zip already present ({get_size(dst)}). Skipping download.")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        file_id = self._extract_file_id(self.config.source_URL)
        logger.info(f"Downloading Google Drive file id={file_id} → {dst}")
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            str(dst), quiet=False, fuzzy=True,
        )
        if not dst.exists() or dst.stat().st_size == 0:
            raise RuntimeError(
                f"Download failed — file missing or empty at {dst}.\n"
                "Check that the Drive file is shared as 'Anyone with the link'."
            )
        logger.info(f"Download complete ({get_size(dst)})")

    def extract_zip_file(self) -> None:
        """Extract and normalise dataset into YOLO train/valid/test layout."""
        src  = Path(self.config.local_data_file)
        dest = Path(self.config.unzip_dir)
        dest.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting {src} → {dest}")
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(dest)

        self._extract_inner_zips(dest, src.resolve())
        self._normalise(dest)

    # ── Private helpers ───────────────────────────────────────────────

    @staticmethod
    def _extract_file_id(url: str) -> str:
        for pat in [r"/file/d/([^/?]+)", r"[?&]id=([^&]+)"]:
            m = re.search(pat, url)
            if m:
                return m.group(1)
        raise ValueError(
            f"Cannot extract Google Drive file ID from URL: {url}\n"
            "Expected format: https://drive.google.com/file/d/FILE_ID/..."
        )

    @staticmethod
    def _extract_inner_zips(base: Path, outer: Path) -> None:
        for item in list(base.rglob("*.zip")):
            if item.resolve() == outer:
                continue
            logger.info(f"Inner zip found: {item.name}. Extracting in place.")
            with zipfile.ZipFile(item, "r") as zf:
                zf.extractall(item.parent)
            item.unlink()

    def _normalise(self, base: Path) -> None:
        """Route dataset to the correct normalisation strategy."""
        visible = [p for p in base.iterdir()
                   if not p.name.startswith(".") and p.name != "__MACOSX"]
        names   = {p.name for p in visible}
        logger.info(f"Dataset root contains: {sorted(names)}")

        # ── A: already correctly split ─────────────────────────────
        if {"train", "valid", "test"}.issubset(names):
            logger.info("Layout A detected: already train/valid/test split.")
            self._write_data_yaml(base)
            return

        # ── B: nested single subfolder ────────────────────────────
        dirs = [p for p in visible if p.is_dir()]
        if len(dirs) == 1:
            inner_names = {p.name for p in dirs[0].iterdir()}
            if {"train", "valid", "test"}.issubset(inner_names):
                logger.info(f"Layout B detected: nested in '{dirs[0].name}'. Flattening.")
                for item in dirs[0].iterdir():
                    dst = base / item.name
                    if dst.exists():
                        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
                    shutil.move(str(item), str(base))
                dirs[0].rmdir()
                self._write_data_yaml(base)
                return

        # ── C: flat images + annotations (Kaggle car-plate-detection) ─
        img_dir = base / "images"
        ann_dir = (base / "annotations") if (base / "annotations").exists() \
                  else (base / "Annotations") if (base / "Annotations").exists() \
                  else None
        if img_dir.exists() and img_dir.is_dir():
            logger.info("Layout C detected: flat images/ + annotations/. Splitting 70/20/10.")
            self._split_flat(base, img_dir, ann_dir)
            self._write_data_yaml(base)
            return

        raise RuntimeError(
            f"Unrecognised dataset layout. Root contains: {sorted(names)}\n"
            "Expected: train/valid/test directories OR images/annotations OR "
            "a single subfolder containing train/valid/test."
        )

    def _split_flat(self, base: Path, img_dir: Path, ann_dir: "Path | None") -> None:
        """
        Split a flat images/ + annotations/ dataset into YOLO train/valid/test.

        This is the exact layout of the Kaggle 'andrewmvd/car-plate-detection'
        dataset used in the notebook:
            images/Cars1.png  ...  Cars400.png
            annotations/Cars1.xml ... Cars400.xml

        Conversion happens here for XML files. TXT files are copied as-is.
        Split ratio: 70% train / 20% valid / 10% test (random_state=42).
        """
        images = sorted([p for p in img_dir.iterdir()
                         if p.suffix.lower() in IMAGE_EXTS])
        if not images:
            raise RuntimeError(f"No images found in {img_dir}")

        logger.info(f"Found {len(images)} images. Samples: {[i.name for i in images[:5]]}")

        # Build annotation stem → path lookup
        ann_map: dict[str, Path] = {}
        if ann_dir and ann_dir.exists():
            for f in ann_dir.glob("**/*"):
                if f.suffix.lower() in LABEL_EXTS:
                    ann_map[f.stem] = f
            if ann_map:
                logger.info(f"Found {len(ann_map)} annotation files. "
                            f"Samples: {list(ann_map.keys())[:5]}")
            else:
                logger.warning(f"No annotation files found in {ann_dir}")

        # 70 / 20 / 10 split — same seed as notebook
        random.seed(42)
        shuffled = images.copy()
        random.shuffle(shuffled)
        n     = len(shuffled)
        n_tr  = int(n * 0.70)
        n_val = int(n * 0.20)
        splits = {
            "train": shuffled[:n_tr],
            "valid": shuffled[n_tr:n_tr + n_val],
            "test":  shuffled[n_tr + n_val:],
        }

        total_matched = 0
        for split_name, split_imgs in splits.items():
            img_out = base / split_name / "images"
            lbl_out = base / split_name / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            matched = 0

            for img in split_imgs:
                shutil.copy2(img, img_out / img.name)
                ann = ann_map.get(img.stem) or self._icase_lookup(ann_map, img.stem)
                if ann is None:
                    continue
                if ann.suffix.lower() == ".xml":
                    # Convert Pascal VOC XML → YOLO TXT inline
                    try:
                        lines = self._xml_to_yolo_lines(ann)
                        (lbl_out / (ann.stem + ".txt")).write_text("\n".join(lines))
                        matched += 1
                    except Exception as e:
                        logger.warning(f"XML convert failed {ann.name}: {e}")
                else:  # .txt already in YOLO format
                    shutil.copy2(ann, lbl_out / ann.name)
                    matched += 1

            logger.info(f"  {split_name}: {len(split_imgs)} images, {matched} labels written")
            total_matched += matched

        if ann_map and total_matched == 0:
            raise RuntimeError(
                "ZERO labels matched any image.\n"
                f"  Image stem example   : '{images[0].stem}'\n"
                f"  Annotation stem example: '{next(iter(ann_map))}'\n"
                "Stems must match (case-insensitive)."
            )

        shutil.rmtree(img_dir)
        if ann_dir and ann_dir.exists():
            shutil.rmtree(ann_dir)

    @staticmethod
    def _icase_lookup(ann_map: dict, stem: str) -> "Path | None":
        lo = stem.lower()
        for k, v in ann_map.items():
            if k.lower() == lo:
                return v
        return None

    @staticmethod
    def _xml_to_yolo_lines(xml_path: Path) -> list[str]:
        """
        Parse Pascal VOC XML → list of YOLO format strings.
        Each line: "0 cx cy bw bh"  (class 0 = license_plate, all values 0-1)
        Mirrors the notebook's conversion logic exactly.
        """
        import xml.etree.ElementTree as ET
        root = ET.parse(xml_path).getroot()

        sz    = root.find("size")
        img_w = float(sz.findtext("width",  "0") if sz else "0")
        img_h = float(sz.findtext("height", "0") if sz else "0")
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"Invalid image size {img_w}x{img_h} in {xml_path.name}")

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
                logger.warning(f"Degenerate bbox in {xml_path.name} — skipping object")
                continue
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return lines

    @staticmethod
    def _write_data_yaml(base: Path) -> None:
        """Write data.yaml with ABSOLUTE resolved paths (prevents YOLO path-doubling bug)."""
        train = (base / "train" / "images").resolve().as_posix()
        valid = (base / "valid" / "images").resolve().as_posix()
        test  = (base / "test"  / "images").resolve().as_posix()
        content = (
            f"train: {train}\n"
            f"val:   {valid}\n"
            f"test:  {test}\n\n"
            "nc: 1\n"
            "names: ['license_plate']\n"
        )
        (base / "data.yaml").write_text(content)
        logger.info(f"data.yaml written → {base / 'data.yaml'}")
        logger.info(f"  train: {train}")
        logger.info(f"  val:   {valid}")
        logger.info(f"  test:  {test}")
