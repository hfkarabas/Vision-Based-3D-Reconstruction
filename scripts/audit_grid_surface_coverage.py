"""Localize surface coverage loss using saved pre-cleanup grid connectivity."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from audit_reconstruction_visuals import View, VIEWS, montage, render_mesh
from analyze_airzoo_geometry import read


def audit(dense, baseline, output):
    output.mkdir(parents=True, exist_ok=False)
    metadata = read(dense / "metrics.json")
    frame = read(baseline / "frame.json")
    views = {key: View(frame, np.asarray(VIEWS[key]), read(baseline / "metrics.json")["resolution"])
             for key in ("top", "oblique")}
    cloud = o3d.io.read_point_cloud(str(dense / "dense.ply"))
    xyz = np.asarray(cloud.points)
    with np.load(dense / "depth-observations/grid-faces.npz") as store:
        triangles = store["triangles"]
    keys = np.sort(triangles, axis=1)
    rows = np.flatnonzero((np.diff(keys, axis=1) > 0).all(axis=1))
    _, unique = np.unique(keys[rows], axis=0, return_index=True)
    faces = triangles[rows[unique]]
    del triangles, keys
    a, b, c = xyz[faces].transpose(1, 0, 2)
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    edge = np.maximum.reduce([np.linalg.norm(u - v, axis=1) for u, v in ((a, b), (b, c), (c, a))])
    del a, b, c
    cap = metadata["mesh"]["max_edge_world_units"]
    summaries = {}
    with np.load(baseline / "occupancy.npz") as baseline_masks:
        for label, keep in (("pre-clean", area > 1e-14), ("edge-clean", (area > 1e-14) & (edge <= cap))):
            mesh = o3d.geometry.TriangleMesh(cloud.points, o3d.utility.Vector3iVector(faces[keep]))
            result, masks = render_mesh([mesh], views, output, label)
            result["new_pixels_over_raw"] = {key: int((mask & ~baseline_masks[f"raw_{key}"]).sum())
                                             for key, mask in masks.items()}
            result["dense_pixels_without_triangle"] = {
                key: int((baseline_masks[f"dense_{key}"] & ~mask).sum()) for key, mask in masks.items()}
            summaries[label] = result
            del mesh
    summaries["max_edge_sfm_units"] = cap
    summaries["scope"] = "Uncleaned diagnostic faces can bridge unsupported geometry and are not candidate assets."
    montage(output, views, ("pre-clean-neutral", "edge-clean-neutral"), "grid-cleanup.png")
    (output / "metrics.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dense", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    audit(args.dense, args.baseline, args.output)
