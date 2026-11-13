"""Run the same local raw-photograph workflow on any projected UAV block."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from omega_colmap_pilot import save


SCRIPTS = Path(__file__).resolve().parent
PHASES = ("reconstruction", "orthophoto", "ground", "buildings")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def measured(output, name, command):
    command = list(map(str, command))
    started = time.monotonic()
    available = psutil.virtual_memory().available
    with (output / f"{name}.log").open("wb") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    save(output / f"{name}.json", dict(command=command, seconds=time.monotonic() - started,
                                      available_before_bytes=available, returncode=process.returncode))
    if process.returncode:
        raise RuntimeError(f"{name} failed. Inspect {output / (name + '.log')}.")


def select(args):
    root = args.images.resolve()
    if args.selection:
        names = json.loads(args.selection.read_text(encoding="utf-8-sig"))["images"]
    else:
        names = sorted(path.name for path in root.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(names) != len(set(names)) or not 3 <= len(names) <= 180:
        raise ValueError("Specify 3–180 distinct original photographs.")
    paths = []
    for name in names:
        path = root / name
        if Path(name).name != name or path.suffix.lower() not in (".jpg", ".jpeg", ".png") or not path.is_file():
            raise ValueError(f"Not a supported image in the acquisition directory: {name}")
        paths.append(path)
    return paths


def fingerprint(args, paths):
    return dict(
        images=[dict(path=str(path), sha256=sha256(path)) for path in paths],
        settings=json.loads(args.settings.read_text(encoding="utf-8")), crs=args.crs,
        extent=args.extent, colmap_sha256=sha256(args.colmap),
        reconstruction_adapter_sha256=sha256(SCRIPTS / "omega_colmap_pilot.py"),
    )


def reconstruct(args, root, paths, inputs):
    root.mkdir(parents=True, exist_ok=False)
    staged = root / "images"
    staged.mkdir()
    for path, record in zip(paths, inputs["images"], strict=True):
        copy = staged / path.name
        shutil.copy2(path, copy)
        if sha256(copy) != record["sha256"]:
            raise IOError(f"Staged image differs from the original: {path}")
    save(root / "inputs.json", inputs)
    measured(root, "reconstruction", [
        sys.executable, SCRIPTS / "omega_colmap_pilot.py", "--images", staged,
        "--gps-crs", args.crs, "--colmap", args.colmap, "--pycolmap", args.pycolmap,
        "--output", root / "colmap", "--settings", args.settings,
    ])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("images", "settings", "output", "colmap", "pycolmap", "pdal", "blender", "hub-cache"):
        parser.add_argument(f"--{key}", type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--crs", required=True)
    parser.add_argument("--extent", type=float, nargs=4, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--through", choices=PHASES, default="buildings")
    return parser.parse_args()


def main():
    args = parse_args()
    west, south, east, north = args.extent
    if not west < east or not south < north:
        raise ValueError("Expected a non-empty projected extent in metres.")
    paths = select(args)
    inputs = fingerprint(args, paths)
    if not args.output.exists():
        reconstruct(args, args.output, paths, inputs)
    elif json.loads((args.output / "inputs.json").read_text(encoding="utf-8")) != inputs:
        raise ValueError("Inputs, settings or reconstruction tool changed. Choose a new output directory.")
    previous = json.loads((args.output / "reconstruction.json").read_text(encoding="utf-8"))
    if previous["returncode"] or not (args.output / "colmap" / "dense" / "fused.ply").is_file():
        raise ValueError("The cached reconstruction is incomplete.")
    if args.through != "reconstruction":
        from omega_uav_products import products
        products(args, inputs["settings"], PHASES.index(args.through), measured)


if __name__ == "__main__":
    main()
