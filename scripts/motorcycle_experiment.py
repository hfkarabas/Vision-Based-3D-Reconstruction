"""Small reproducible real-image experiment. Reference pose is evaluation-only."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

from vision3d.reconstruction.pair import reconstruct_pair, write_json


SAMPLES = {
    "motorcycle_left.png": "db18e9c4157617403c3537a6ba355dfeafe9a7eabb6b9b94cb33f6525dd49179",
    "motorcycle_right.png": "5fc913ae870e42a4b662314bc904d1786bcad8e2f0b9b67dba5a229406357797",
}
BASE_URL = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/"


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "datasets/middlebury-motorcycle")
    parser.add_argument("--download", action="store_true", help="Download only two images (~1.3 MB total)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", choices=["sift", "orb"], default="sift")
    parser.add_argument("--max-dimension", type=int, default=1600)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    args.dataset.mkdir(parents=True, exist_ok=True)
    for name, expected in SAMPLES.items():
        path = args.dataset / name
        if not path.exists() and args.download:
            with urllib.request.urlopen(BASE_URL + name, timeout=60) as response:
                payload = response.read(2_000_001)
            if len(payload) > 2_000_000 or hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Unexpected sample content: {name}")
            path.write_bytes(payload)
        if not path.exists():
            parser.error(f"Missing {path}; supply --download to obtain the small sample")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Dataset checksum mismatch: {path}")
    result = reconstruct_pair(args.dataset / "motorcycle_left.png", args.dataset / "motorcycle_right.png",
                              args.output, camera=root / "configs/motorcycle.json", detector=args.detector,
                              max_dimension=args.max_dimension, preview=args.preview)
    pose = np.array(json.loads((args.output / "poses.json").read_text())["world_to_camera2"])
    # Rectified stereo reference: R=I, t points along negative X. No reference
    # rotation, translation, baseline magnitude or disparity enters reconstruction.
    reference_t = np.array([-1.0, 0, 0])
    evaluation = {
        "reference": "Middlebury rectified stereo convention; R=identity, t direction=(-1,0,0)",
        "rotation_error_degrees": float(np.rad2deg(np.arccos(np.clip((np.trace(pose[:3, :3]) - 1) / 2, -1, 1)))),
        "translation_direction_error_degrees": float(np.rad2deg(np.arccos(np.clip(pose[:3, 3] @ reference_t, -1, 1)))),
        "absolute_scale_evaluated": False,
        "dataset_urls": [BASE_URL + name for name in SAMPLES], "sha256": SAMPLES,
        "permission": "https://vision.middlebury.edu/stereo/data/",
        "citation": "Scharstein et al., High-resolution stereo datasets with subpixel-accurate ground truth, GCPR 2014",
    }
    write_json(args.output / "evaluation.json", evaluation)
    print(json.dumps({"metrics": result, "evaluation": evaluation}, indent=2))


if __name__ == "__main__":
    main()
