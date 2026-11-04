"""Prepare pinned local benchmark subsets; reference poses are stored separately."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np
from PIL import Image
from sceaux_experiment import SAMPLES, BASE
from vision3d.geometry.camera import Camera
from vision3d.reconstruction.pair import write_json

TEMPLE_URL = "https://vision.middlebury.edu/mview/data/data/templeRing.zip"
TEMPLE_SHA256 = "5f871fe96d25f510eac026c66c3a4c38229326260986e9926cba8a64e88c8359"


def link_or_copy(source, target):
    if target.exists():
        if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
            raise ValueError(f"Existing subset image differs: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    castle = root / "datasets/sceaux/images"
    castle.mkdir(parents=True, exist_ok=True)
    for name, sha in SAMPLES.items():
        path = castle / name
        if not path.exists() and args.download:
            with urllib.request.urlopen(BASE + "images/" + name, timeout=60) as response:
                data = response.read(2_000_001)
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError("Castle checksum failed")
            path.write_bytes(data)
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            parser.error(f"Missing/modified sample {path}; use --download for missing samples")
    directory = root / "datasets/temple"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "templeRing.zip"
    if not archive.exists() and args.download:
        with urllib.request.urlopen(TEMPLE_URL, timeout=60) as response:
            data = response.read(20_000_001)
        if hashlib.sha256(data).hexdigest() != TEMPLE_SHA256:
            raise ValueError("Temple archive checksum failed")
        archive.write_bytes(data)
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != TEMPLE_SHA256:
        parser.error("Missing/modified Temple archive; use --download for a missing archive")
    originals = directory / "selected"
    originals.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        rows = z.read("templeRing/templeR_par.txt").decode().splitlines()[1:31]
        names, cameras, references, checksums = [], {}, {}, {}
        for row in rows:
            fields = row.split()
            name = fields[0]
            if Path(name).name != name or not name.endswith(".png"):
                raise ValueError("Unexpected archive image name")
            values = np.array(fields[1:], dtype=float)
            k, rotation, translation = values[:9].reshape(3, 3), values[9:18].reshape(3, 3), values[18:21]
            data = z.read("templeRing/" + name)
            (originals / name).write_bytes(data)
            with Image.open(originals / name) as image:
                width, height = image.size
            model = Camera(width, height, k[0, 0], k[1, 1], k[0, 2], k[1, 2],
                source="Middlebury TempleRing published K", assumptions="Provider removed radial distortion; EXIF orientation absent")
            cameras[name] = model.to_dict()
            references[name] = np.r_[np.c_[rotation, translation], [[0, 0, 0, 1]]].tolist()
            checksums[name] = hashlib.sha256(data).hexdigest()
            names.append(name)
        (directory / "UPSTREAM_README.txt").write_bytes(z.read("templeRing/README.txt"))
    write_json(directory / "cameras.json", {"images": cameras})
    write_json(directory / "reference_poses.json", references)
    for count in (6, 15, 30):
        subset = directory / f"images{count}"
        subset.mkdir(exist_ok=True)
        for name in names[:count]:
            link_or_copy(originals / name, subset / name)
        if {p.name for p in subset.glob("*.png")} != set(names[:count]):
            raise ValueError(f"Unexpected files in subset {subset}")
    write_json(directory / "provenance.json", {"archive_url": TEMPLE_URL, "archive_sha256": TEMPLE_SHA256,
        "selected_images": names, "image_sha256": checksums, "selection": "first 30 original filename/parameter-file rows; no pose-based reordering",
        "reference_poses": "evaluation only; separate from cameras.json and image directories",
        "source": "https://vision.middlebury.edu/mview/data/", "license_note": "Provider benchmark download and attribution README; no explicit redistribution license found. Local testing only; do not redistribute images.",
        "citation": "Seitz, Curless, Diebel, Scharstein and Szeliski, A Comparison and Evaluation of Multi-View Stereo Reconstruction Algorithms, CVPR 2006"})
    print("Prepared Castle 6 and Temple 6/15/30; camera-only inputs and references are separate")


if __name__ == "__main__":
    main()
