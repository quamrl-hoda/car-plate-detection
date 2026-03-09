import os
import shutil
from pathlib import Path
from carPlateDetection import logger
from carPlateDetection.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def process_images(self):
        """Copy train/valid/test splits to the transformation output directory.

        YOLOv8 handles all resizing and augmentation internally, so this stage
        simply copies files and rewrites data.yaml with correct absolute paths.
        """
        splits = ["train", "valid", "test"]

        for split in splits:
            src_img = Path(self.config.data_dir) / split / "images"
            src_lbl = Path(self.config.data_dir) / split / "labels"
            dst_img = Path(self.config.root_dir) / split / "images"
            dst_lbl = Path(self.config.root_dir) / split / "labels"

            if not src_img.exists():
                logger.warning(f"Skipping missing split: {src_img}")
                continue

            # Wipe stale destination
            for d in [dst_img, dst_lbl]:
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)

            # Copy images
            img_files = list(src_img.glob("*"))
            for f in img_files:
                shutil.copy2(f, dst_img / f.name)

            # Copy labels
            if src_lbl.exists():
                for f in src_lbl.glob("*"):
                    shutil.copy2(f, dst_lbl / f.name)

            logger.info(f"Copied '{split}': {len(img_files)} images -> {dst_img}")

        # Write data.yaml with absolute paths into the transformation dir
        self._write_data_yaml(Path(self.config.root_dir))
        logger.info("Data transformation complete.")

    def _write_data_yaml(self, root: Path):
        """Write data.yaml with ABSOLUTE paths so YOLO resolves them correctly."""
        import yaml

        src_yaml = Path(self.config.data_dir) / "data.yaml"
        dst_yaml = root / "data.yaml"

        # Build absolute paths for each split
        train_path = (root / "train" / "images").resolve().as_posix()
        valid_path = (root / "valid" / "images").resolve().as_posix()
        test_path  = (root / "test"  / "images").resolve().as_posix()

        # Start from existing data.yaml to preserve nc / names / extra keys
        if src_yaml.exists():
            with open(src_yaml) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {"nc": 1, "names": ["license-plate"]}

        data["train"] = train_path
        data["val"]   = valid_path
        data["test"]  = test_path

        with open(dst_yaml, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Written data.yaml with absolute paths at {dst_yaml}")
        logger.info(f"  train -> {train_path}")
        logger.info(f"  val   -> {valid_path}")
        logger.info(f"  test  -> {test_path}")