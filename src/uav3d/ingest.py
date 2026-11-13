"""Find input images and summarise their EXIF (camera, GPS) without modifying them."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
_GPS_IFD = 0x8825


def find_images(folder: Path) -> list[Path]:
    """Images in ``folder/images`` (the ODM dataset convention) when present, else in ``folder``.

    The subfolder wins because dataset roots often hold previews or products (e.g. a dsm.tif)."""
    for candidate in (folder / "images", folder):
        if candidate.is_dir():
            files = sorted(p for p in candidate.iterdir()
                           if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if files:
                return files
    raise FileNotFoundError(f"No images ({', '.join(sorted(IMAGE_SUFFIXES))}) in {folder} or {folder / 'images'}")


_GCP_HEADER = re.compile(r"^\s*(\+proj=|EPSG:\d+|WGS84 UTM)", re.IGNORECASE)


def find_gcp(images_dir: Path, images: list[Path]) -> Path | None:
    """An ODM-format GCP file next to the images or in the dataset folder.

    Accepted names: gcp_list.txt (ODM convention) or *gcp*.txt; the first line must be an ODM
    GCP projection header, so unrelated text files are never picked up."""
    folders = dict.fromkeys([images_dir, images[0].parent, images_dir.parent])
    for folder in folders:
        candidates = [folder / "gcp_list.txt", *sorted(folder.glob("*gcp*.txt"))]
        for path in candidates:
            if path.is_file():
                with open(path, encoding="utf-8", errors="replace") as fh:
                    if _GCP_HEADER.match(fh.readline()):
                        return path
    return None


def summarise(images: list[Path]) -> dict:
    cameras: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    gps = 0
    unreadable: list[str] = []
    megapixels = 0.0
    for path in images:
        try:
            with Image.open(path) as im:
                exif = im.getexif()
                w, h = im.size
        except OSError:
            unreadable.append(path.name)
            continue
        make = str(exif.get(0x010F, "")).strip("\x00 ")
        model = str(exif.get(0x0110, "")).strip("\x00 ")
        cameras[f"{make} {model}".strip() or "unknown"] += 1
        sizes[f"{w}x{h}"] += 1
        megapixels += w * h / 1e6
        if exif.get_ifd(_GPS_IFD):
            gps += 1
    readable = len(images) - len(unreadable)
    if readable == 0:
        raise ValueError("None of the input images could be read.")
    return {
        "count": readable,
        "unreadable": unreadable,
        "cameras": dict(cameras),
        "sizes": dict(sizes),
        "mean_megapixels": round(megapixels / readable, 2),
        "gps_fraction": round(gps / readable, 3),
    }
