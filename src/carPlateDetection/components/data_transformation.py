"""
data_transformation.py
======================
Converts Pascal VOC XML → YOLO .txt labels and copies images to
artifacts/data_transformation/. Writes data.yaml with absolute paths
so YOLO never doubles the path.
"""
import shutil, xml.etree.ElementTree as ET
from pathlib import Path
from carPlateDetection import logger
from carPlateDetection.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def process_images(self):
        total = 0
        for split in ["train", "valid", "test"]:
            src_img = Path(self.config.data_dir) / split / "images"
            src_lbl = Path(self.config.data_dir) / split / "labels"
            dst_img = Path(self.config.root_dir)  / split / "images"
            dst_lbl = Path(self.config.root_dir)  / split / "labels"

            if not src_img.exists():
                logger.warning(f"Missing split images: {src_img}")
                continue

            for d in [dst_img, dst_lbl]:
                if d.exists(): shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)

            imgs = list(src_img.glob("*"))
            for f in imgs: shutil.copy2(f, dst_img/f.name)
            logger.info(f"[{split}] {len(imgs)} images copied")

            if src_lbl.exists():
                conv, skip = self._convert_labels(src_lbl, dst_lbl)
                total += conv
                logger.info(f"[{split}] {conv} labels written, {skip} skipped")
            else:
                logger.warning(f"[{split}] No labels folder at {src_lbl}")

        if total == 0:
            raise RuntimeError(
                "ZERO labels written to transformation output.\n"
                "Run verify_labels.py to diagnose annotation files."
            )

        self._write_yaml(Path(self.config.root_dir))
        logger.info(f"Transformation complete. Total labels: {total}")

    def _convert_labels(self, src, dst):
        converted = skipped = 0
        for f in src.iterdir():
            ext = f.suffix.lower()
            if ext == ".xml":
                try:
                    lines = self._xml_to_yolo(f)
                    (dst / (f.stem + ".txt")).write_text("\n".join(lines))
                    converted += 1
                except Exception as e:
                    logger.warning(f"XML failed {f.name}: {e}")
                    skipped += 1
            elif ext == ".txt":
                shutil.copy2(f, dst/f.name)
                converted += 1
            else:
                skipped += 1
        return converted, skipped

    @staticmethod
    def _xml_to_yolo(xml_path):
        root  = ET.parse(xml_path).getroot()
        sz    = root.find("size")
        img_w = float(sz.findtext("width",  "0") if sz else "0")
        img_h = float(sz.findtext("height", "0") if sz else "0")
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"Invalid size {img_w}x{img_h} in {xml_path.name}")
        lines = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None: continue
            xmin = max(0., min(float(bb.findtext("xmin","0")), img_w))
            ymin = max(0., min(float(bb.findtext("ymin","0")), img_h))
            xmax = max(0., min(float(bb.findtext("xmax","0")), img_w))
            ymax = max(0., min(float(bb.findtext("ymax","0")), img_h))
            if xmax <= xmin or ymax <= ymin: continue
            cx = ((xmin+xmax)/2)/img_w; cy = ((ymin+ymax)/2)/img_h
            bw = (xmax-xmin)/img_w;     bh = (ymax-ymin)/img_h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return lines

    def _write_yaml(self, root):
        import yaml
        src = Path(self.config.data_dir) / "data.yaml"
        data = {}
        if src.exists():
            with open(src) as f: data = yaml.safe_load(f) or {}
        data["train"] = (root/"train"/"images").resolve().as_posix()
        data["val"]   = (root/"valid"/"images").resolve().as_posix()
        data["test"]  = (root/"test" /"images").resolve().as_posix()
        data["nc"]    = 1
        data["names"] = ["license_plate"]
        with open(root/"data.yaml","w") as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.info(f"data.yaml written → {root/'data.yaml'}")
