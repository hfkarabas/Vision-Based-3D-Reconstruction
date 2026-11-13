"""Build orthophoto and terrain products from an existing, verified COLMAP workspace."""

import argparse
import json
from pathlib import Path

from omega_uav_pipeline import measured, sha256
from omega_uav_products import ground, orthophoto, read


def run(args):
    source = args.workspace
    inputs = read(source / "inputs.json")
    actual = {item.name: item for item in args.images.iterdir()
              if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    registered = source / "dense/images"
    if {item.relative_to(registered).as_posix() for item in registered.rglob("*") if item.is_file()} != set(actual):
        raise ValueError("The undistorted camera workspace does not contain the supplied image set")
    expected = {record.get("name", Path(record.get("path", "")).name): record["sha256"]
                for record in inputs["images"]}
    if actual.keys() != expected.keys() or any(sha256(path) != expected[name] for name, path in actual.items()):
        raise ValueError("COLMAP workspace provenance does not match the original photographs")
    frame = read(source / "frame.json")
    if frame["crs"] != args.crs:
        raise ValueError("The cached reconstruction and requested output CRS differ")
    if not (source / "dense/fused.ply").is_file():
        raise FileNotFoundError("The cached dense workspace has no fused point cloud")
    config = read(args.settings)
    if args.classifier:
        config["surface"]["ground_classifier"] = args.classifier
    record = dict(extent=args.extent, crs=args.crs, source_colmap=str(source), config=config,
                  image_count=len(actual), source_inventory_sha256=sha256(source / "inputs.json"))
    if args.verify_only:
        print(json.dumps(record, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    orthophoto(args, config, source, measured)
    if args.classifier:
        ground(args, config, source, measured)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "workspace", "output", "settings", "pdal", "colmap", "blender"):
        parser.add_argument(f"--{name}", required=True, type=lambda path: Path(path).resolve())
    parser.add_argument("--extent", nargs=4, type=float, required=True)
    parser.add_argument("--crs", required=True)
    parser.add_argument("--classifier", choices=("csf", "smrf"))
    parser.add_argument("--verify-only", action="store_true")
    run(parser.parse_args())
