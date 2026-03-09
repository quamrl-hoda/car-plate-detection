import os
import shutil
import zipfile
import random
import gdown
from pathlib import Path
from carPlateDetection import logger
from carPlateDetection.utils.common import get_size
from carPlateDetection.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    @staticmethod
    def _extract_gdrive_file_id(url: str) -> str:
        if "/file/d/" in url:
            return url.split("/file/d/")[1].split("/")[0].split("?")[0]
        elif "id=" in url:
            return url.split("id=")[1].split("&")[0]
        else:
            raise ValueError(f"Cannot extract file ID from URL: {url}")

    def download_file(self):
        """Download the dataset zip from Google Drive using gdown."""
        local_path = Path(self.config.local_data_file)

        if local_path.exists() and local_path.stat().st_size > 0:
            logger.info(f"Zip already exists ({get_size(local_path)}), skipping download.")
            return

        os.makedirs(local_path.parent, exist_ok=True)
        file_id = self._extract_gdrive_file_id(self.config.source_URL)
        gdrive_url = f"https://drive.google.com/uc?id={file_id}"

        logger.info(f"Downloading from Google Drive (id={file_id}) -> {local_path}")
        gdown.download(gdrive_url, str(local_path), quiet=False, fuzzy=True)

        if not local_path.exists() or local_path.stat().st_size == 0:
            raise RuntimeError("Download failed — file is missing or empty.")

        logger.info(f"Download complete. File size: {get_size(local_path)}")

    def extract_zip_file(self):
        """Extract zip and normalise into YOLO train/valid/test layout."""
        local_path = Path(self.config.local_data_file)
        unzip_path = Path(self.config.unzip_dir)
        os.makedirs(unzip_path, exist_ok=True)

        logger.info(f"Extracting {local_path} -> {unzip_path}")
        with zipfile.ZipFile(local_path, "r") as zf:
            zf.extractall(unzip_path)
        logger.info("Initial extraction complete. Analysing structure…")

        self._extract_inner_zips(unzip_path)
        self._normalise_structure(unzip_path)

    # ------------------------------------------------------------------ #
    def _extract_inner_zips(self, base: Path):
        for item in list(base.iterdir()):
            if item.suffix == ".zip" and item.resolve() != Path(self.config.local_data_file).resolve():
                logger.info(f"Found inner zip: {item.name} — extracting…")
                with zipfile.ZipFile(item, "r") as zf:
                    zf.extractall(base)
                logger.info(f"Inner zip '{item.name}' extracted.")

    def _normalise_structure(self, base: Path):
        contents = {p.name: p for p in base.iterdir() if not p.name.startswith(".")}
        names = set(contents.keys())

        # Case A: already YOLO split
        if {"train", "valid", "test"}.issubset(names):
            logger.info("Structure A: already YOLO split.")
            self._write_data_yaml(base)
            return

        # Case B: single nested subfolder
        subdirs = [p for p in base.iterdir()
                   if p.is_dir() and p.name not in {"__MACOSX"}
                   and not p.name.startswith(".")]
        if len(subdirs) == 1:
            inner = subdirs[0]
            if {"train", "valid", "test"}.issubset({p.name for p in inner.iterdir()}):
                logger.info(f"Structure B: nested in '{inner.name}'. Flattening…")
                for item in inner.iterdir():
                    dest = base / item.name
                    if dest.exists():
                        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                    shutil.move(str(item), str(base))
                inner.rmdir()
                self._write_data_yaml(base)
                return

        # Case C: flat images/ + annotations/
        if "images" in names and contents["images"].is_dir():
            logger.info("Structure C: flat images/annotations — auto-splitting 70/20/10…")
            self._split_flat_dataset(base, contents)
            self._write_data_yaml(base)
            return

        raise RuntimeError(
            f"Unrecognised dataset structure. Found: {sorted(names)}\n"
            "Expected: train/valid/test  OR  images/annotations  OR  nested subfolder."
        )

    def _split_flat_dataset(self, base: Path, contents: dict):
        images_dir = contents["images"]
        ann_dir    = contents.get("annotations")

        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        all_images = sorted([p for p in images_dir.iterdir()
                             if p.suffix.lower() in IMAGE_EXTS])

        if not all_images:
            raise RuntimeError(f"No images found in {images_dir}")

        random.seed(42)
        random.shuffle(all_images)
        n       = len(all_images)
        n_train = int(n * 0.70)
        n_valid = int(n * 0.20)

        splits = {
            "train": all_images[:n_train],
            "valid": all_images[n_train:n_train + n_valid],
            "test":  all_images[n_train + n_valid:],
        }

        for split, imgs in splits.items():
            (base / split / "images").mkdir(parents=True, exist_ok=True)
            (base / split / "labels").mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, base / split / "images" / img.name)
                if ann_dir:
                    for ext in [".txt", ".xml"]:
                        lbl = ann_dir / (img.stem + ext)
                        if lbl.exists():
                            shutil.copy2(lbl, base / split / "labels" / lbl.name)
                            break
            logger.info(f"  {split}: {len(imgs)} images")

        shutil.rmtree(images_dir)
        if ann_dir and ann_dir.exists():
            shutil.rmtree(ann_dir)

    @staticmethod
    def _write_data_yaml(base: Path):
        """Write data.yaml with ABSOLUTE paths so YOLO resolves them correctly."""
        yaml_path = base / "data.yaml"

        # Use resolved absolute paths — prevents YOLO from doubling the path
        train_path = (base / "train" / "images").resolve().as_posix()
        valid_path = (base / "valid" / "images").resolve().as_posix()
        test_path  = (base / "test"  / "images").resolve().as_posix()

        content = (
            f"train: {train_path}\n"
            f"val:   {valid_path}\n"
            f"test:  {test_path}\n\n"
            "nc: 1\n"
            "names: ['license-plate']\n"
        )

        yaml_path.write_text(content)
        logger.info(f"Written data.yaml with absolute paths at {yaml_path}")
        logger.info(f"  train -> {train_path}")
        logger.info(f"  val   -> {valid_path}")
        logger.info(f"  test  -> {test_path}")