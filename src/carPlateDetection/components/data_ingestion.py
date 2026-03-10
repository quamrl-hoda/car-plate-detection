"""
data_ingestion.py
=================
Downloads dataset zip from Google Drive, extracts it, and normalises to:
  artifacts/data_ingestion/
    train/images/  train/labels/
    valid/images/  valid/labels/
    test/images/   test/labels/
    data.yaml

Handles 4 layouts:
  A) Already split — train/ valid/ test/ at root
  B) Nested folder — SomeName/train/ valid/ test/ inside zip
  C) Flat Kaggle   — images/ + annotations/ (Pascal VOC XML or YOLO TXT)
  D) Inner zip     — zip inside zip
"""
import os, re, shutil, zipfile, random, xml.etree.ElementTree as ET
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

    def download_file(self):
        dst = Path(self.config.local_data_file)
        if dst.exists() and dst.stat().st_size > 0:
            logger.info(f"Zip already present ({get_size(dst)}). Skipping.")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        fid = self._file_id(self.config.source_URL)
        logger.info(f"Downloading id={fid} → {dst}")
        gdown.download(f"https://drive.google.com/uc?id={fid}",
                       str(dst), quiet=False, fuzzy=True)
        if not dst.exists() or dst.stat().st_size == 0:
            raise RuntimeError(
                f"Download failed. Check the Drive link is shared publicly.\n"
                f"URL: {self.config.source_URL}"
            )
        logger.info(f"Download complete ({get_size(dst)})")

    def extract_zip_file(self):
        src  = Path(self.config.local_data_file)
        dest = Path(self.config.unzip_dir)
        dest.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting {src} → {dest}")
        with zipfile.ZipFile(src) as zf:
            zf.extractall(dest)
        self._extract_inner_zips(dest, src.resolve())
        self._normalise(dest)

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _file_id(url):
        for pat in [r"/file/d/([^/?]+)", r"[?&]id=([^&]+)"]:
            m = re.search(pat, url)
            if m: return m.group(1)
        raise ValueError(f"Cannot parse Google Drive URL: {url}")

    @staticmethod
    def _extract_inner_zips(base, outer):
        for p in list(base.rglob("*.zip")):
            if p.resolve() == outer: continue
            logger.info(f"Inner zip: {p.name}")
            with zipfile.ZipFile(p) as zf:
                zf.extractall(p.parent)
            p.unlink()

    def _normalise(self, base):
        visible = [p for p in base.iterdir()
                   if not p.name.startswith(".") and p.name != "__MACOSX"]
        names   = {p.name for p in visible}
        logger.info(f"Dataset root contents: {sorted(names)}")

        # A: already split
        if {"train", "valid", "test"}.issubset(names):
            logger.info("Layout A: already split.")
            self._write_yaml(base); return

        # B: nested subfolder
        dirs = [p for p in visible if p.is_dir()]
        if len(dirs) == 1:
            inner = {p.name for p in dirs[0].iterdir()}
            if {"train", "valid", "test"}.issubset(inner):
                logger.info(f"Layout B: nested in '{dirs[0].name}'. Flattening.")
                for item in dirs[0].iterdir():
                    dst = base / item.name
                    if dst.exists():
                        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
                    shutil.move(str(item), str(base))
                dirs[0].rmdir()
                self._write_yaml(base); return

        # C: flat images + annotations (Kaggle andrewmvd/car-plate-detection)
        img_dir = base / "images"
        ann_dir = next((base / d for d in ["annotations", "Annotations"]
                        if (base / d).exists()), None)
        if img_dir.exists() and img_dir.is_dir():
            logger.info("Layout C: flat images/annotations. Splitting 70/20/10.")
            self._split_flat(base, img_dir, ann_dir)
            self._write_yaml(base); return

        raise RuntimeError(
            f"Unrecognised dataset layout. Root: {sorted(names)}\n"
            "Expected: train/valid/test  OR  images/annotations  OR  single subfolder."
        )

    def _split_flat(self, base, img_dir, ann_dir):
        imgs = sorted([p for p in img_dir.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS])
        if not imgs:
            raise RuntimeError(f"No images found in {img_dir}")
        logger.info(f"{len(imgs)} images. Samples: {[i.name for i in imgs[:4]]}")

        ann_map = {}
        if ann_dir and ann_dir.exists():
            for f in ann_dir.glob("**/*"):
                if f.suffix.lower() in LABEL_EXTS:
                    ann_map[f.stem] = f
            logger.info(f"{len(ann_map)} annotations. Samples: {list(ann_map.keys())[:4]}")

        random.seed(42); random.shuffle(imgs)
        n = len(imgs); nt = int(n*.7); nv = int(n*.2)
        splits = {"train": imgs[:nt], "valid": imgs[nt:nt+nv], "test": imgs[nt+nv:]}

        total = 0
        for split, split_imgs in splits.items():
            (base/split/"images").mkdir(parents=True, exist_ok=True)
            (base/split/"labels").mkdir(parents=True, exist_ok=True)
            matched = 0
            for img in split_imgs:
                shutil.copy2(img, base/split/"images"/img.name)
                ann = ann_map.get(img.stem)
                if ann is None:
                    for k, v in ann_map.items():
                        if k.lower() == img.stem.lower(): ann = v; break
                if ann is None: continue
                if ann.suffix.lower() == ".xml":
                    try:
                        lines = self._xml_to_yolo(ann)
                        (base/split/"labels"/(ann.stem+".txt")).write_text("\n".join(lines))
                        matched += 1
                    except Exception as e:
                        logger.warning(f"XML failed {ann.name}: {e}")
                else:
                    shutil.copy2(ann, base/split/"labels"/ann.name)
                    matched += 1
            logger.info(f"  {split}: {len(split_imgs)} imgs, {matched} labels")
            total += matched

        if ann_map and total == 0:
            raise RuntimeError(
                f"ZERO labels matched images.\n"
                f"Image stem: '{imgs[0].stem}'  Ann stem: '{next(iter(ann_map))}'\n"
                "Stems must match (case-insensitive)."
            )
        shutil.rmtree(img_dir)
        if ann_dir and ann_dir.exists(): shutil.rmtree(ann_dir)

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

    @staticmethod
    def _write_yaml(base):
        base = Path(base)
        t = (base/"train"/"images").resolve().as_posix()
        v = (base/"valid"/"images").resolve().as_posix()
        s = (base/"test" /"images").resolve().as_posix()
        (base/"data.yaml").write_text(
            f"train: {t}\nval:   {v}\ntest:  {s}\nnc: 1\nnames: ['license_plate']\n"
        )
        logger.info(f"data.yaml written → {base/'data.yaml'}")
