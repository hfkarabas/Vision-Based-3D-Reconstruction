"""Build an experimental UAV asset from fresh PatchMatch depth at saved project poses."""
import argparse
from pathlib import Path

import cv2

from analyze_airzoo_geometry import read
from benchmark_airzoo_colmap import export_project_model
from replay_patchmatch_depth import replay
from run_airzoo_colmap_dense import run
from vision3d.geometry.camera import parse_camera
from vision3d.io import load_image
from vision3d.reconstruction.asset_workflow import reconstruct_asset
from vision3d.reconstruction.pair import source_hash


def candidate(sparse, grid, colmap, output, footprint_edges, target_faces):
    metadata = read(grid / "metrics.json")
    if Path(metadata["source_sparse"]).resolve() != sparse.resolve():
        raise ValueError("Voxel grid and sparse cameras must have the same coordinate gauge")
    output.mkdir(parents=True, exist_ok=False)
    model, images = output / "model", output / "images"
    export_project_model(sparse, model)
    images.mkdir()
    camera = {row["image_id"]: parse_camera(row["processed"]) for row in read(sparse / "intrinsics.json")}
    registered = {row["image_id"] for row in read(sparse / "poses.json") if row["status"] == "registered"}
    for row in read(sparse / "manifest.json")["inputs"]:
        key = row["image_id"]
        if key not in registered:
            continue
        path = Path(row["source"])
        if source_hash(path) != row["sha256"]:
            raise ValueError(f"Source photograph changed: {path}")
        source = load_image(path).image
        size = camera[key].width, camera[key].height
        resized = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(images / f"{key:04d}.png"), resized):
            raise OSError(f"Could not save processed photograph {key}")
    external = output / "patchmatch"
    run(colmap, model, images, external)
    dense = output / "verified-dense"
    replay(sparse, external / "workspace", grid, dense, footprint_edges)
    reconstruct_asset(output / "asset", dense_dir=dense, target_faces=target_faces, atlas_size=2048, obj=True,
                      stage=lambda stage: print(stage, flush=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sparse", "grid", "colmap", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--footprint-edges", action="store_true")
    parser.add_argument("--target-faces", type=int, default=450000)
    args = parser.parse_args()
    candidate(args.sparse, args.grid, args.colmap, args.output, args.footprint_edges, args.target_faces)
