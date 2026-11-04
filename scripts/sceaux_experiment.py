"""Reproduce unrectified pair validation and six-view SfM on an upstream sample."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

from vision3d.reconstruction.pair import reconstruct_pair, write_json
from vision3d.reconstruction.sequence import reconstruct_sequence

REVISION = "fde7f5faba555e3c54700477c304488613346a19"
BASE = f"https://raw.githubusercontent.com/openMVG/ImageDataset_SceauxCastle/{REVISION}/"
SAMPLES = {
    "100_7100.JPG": "b475a1f74cd54536b6fe186fb165431ef0ee502f56b0d9895d6da576e364ee1e",
    "100_7101.JPG": "05de696ef02054a3d22af94f48fa3cb88df801cfe48561ad82010a130f408723",
    "100_7102.JPG": "9c25103cf8411909773a189d524eb959a0f8a9ac74a1632b3ca343b605b7e034",
    "100_7103.JPG": "eebf00b09b773aff229e53c4d8c744c5f64f3c004fa8edece0f3d8c917a19411",
    "100_7104.JPG": "3b18d6112bb68aa15e1e20d3be234ea4fa88ecf349f42d89bd7830d75ebe4fc6",
    "100_7105.JPG": "e67d29ccf210fab3227fec99e5a32862d3266f81bf4694eeb30cbe984a0b309f",
}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "datasets/sceaux/images")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--download", action="store_true", help="Obtain six copyrighted upstream test images (~7 MB), local testing only")
    parser.add_argument("--local-ba", action="store_true")
    args = parser.parse_args()
    args.dataset.mkdir(parents=True, exist_ok=True)
    for name, expected in SAMPLES.items():
        path = args.dataset / name
        if not path.exists() and args.download:
            with urllib.request.urlopen(BASE + "images/" + name, timeout=60) as response:
                payload = response.read(2_000_001)
            if len(payload) > 2_000_000 or hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Unexpected dataset content: {name}")
            path.write_bytes(payload)
        if not path.exists():
            parser.error(f"Missing {path}; provide --download to obtain the upstream sample")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Dataset checksum mismatch: {name}")
    if set(p.name for p in args.dataset.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) != set(SAMPLES):
        parser.error("This experiment requires exactly the six selected photographs")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "provenance.json", {
        "repository": "https://github.com/openMVG/ImageDataset_SceauxCastle", "revision": REVISION,
        "source_urls": [BASE + "images/" + name for name in SAMPLES], "sha256": SAMPLES,
        "copyright": "2012 Pierre MOULON", "notice_url": BASE + "Readme.txt",
        "license_note": "Upstream supplies these as an SfM test sample, but supplies no explicit redistribution license. Images remain local and are not redistributed.",
        "ground_truth_poses_available": False})
    pair_args = dict(image1=args.dataset / "100_7100.JPG", image2=args.dataset / "100_7101.JPG")
    calibrated = reconstruct_pair(**pair_args, camera=root / "configs/sceaux.json", output=args.output / "calibrated-pair")
    approximate = reconstruct_pair(**pair_args, approximate=True, fov_degrees=60, output=args.output / "approximate-pair")
    sequence = reconstruct_sequence(args.dataset, args.output / "sequence", camera=root / "configs/sceaux.json",
        local_ba=args.local_ba, quality_config=root / "configs/sequence_quality.json")
    write_json(args.output / "summary.json", {"calibrated_pair": calibrated, "approximate_pair": approximate, "sequence": sequence})
    print(f"Registered {sequence['registered_images']}/{sequence['input_images']} images; {sequence['valid_landmarks']} landmarks")
    if sequence["status"] != "success":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
