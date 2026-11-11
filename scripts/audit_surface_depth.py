"""Compare original and optimized surfaces with reserved AirZoo stereo depths."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from vision3d.io import load_image
from vision3d.reconstruction.dense_quality import landmark_visibility
from vision3d.reconstruction.multiview import make_depth, overlapping_pairs, plan_depth
from vision3d.reconstruction.pair import source_hash
from vision3d.reconstruction.stereo import DisparitySearchLimit
from vision3d.reconstruction.surface_validation import ray_scene, surface_views


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def observations(view, dimension=750):
    if source_hash(view.source) != view.sha256:
        raise ValueError(f"Source photograph changed: {view.source}")
    image = load_image(view.source).image
    camera = view.camera
    if image.shape[:2] != (camera.height, camera.width):
        raise ValueError(f"Source dimensions disagree with saved intrinsics: {view.source}")
    factor = min(1., dimension / max(camera.width, camera.height))
    size = round(camera.width * factor), round(camera.height * factor)
    return camera.resized(*size), cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def evaluate(surface_dir, output, baseline_surface=None):
    surface_metrics = read(surface_dir / "metrics.json")
    dense = Path(surface_metrics["dense_dir"])
    sparse = Path(surface_metrics["sparse_dir"])
    dense_metrics = read(dense / "metrics.json")
    def used_views(metrics):
        primary = {entry[name] for entry in metrics["pair_attempts"] if "depth_pixels" in entry
                   for name in ("reference", "source")}
        recovery = metrics.get("depth_recovery") or {}
        return primary | {entry[name] for entry in recovery.get("attempts", []) if entry["rejection"] is None
                          for name in ("reference", "source")}

    used = used_views(dense_metrics)
    input_mesh = dense
    if baseline_surface is not None:
        baseline_metrics = read(baseline_surface / "metrics.json")
        if Path(baseline_metrics["sparse_dir"]).resolve() != sparse.resolve():
            raise ValueError("Before/after stereo comparison requires the same sparse poses")
        used |= used_views(read(Path(baseline_metrics["dense_dir"]) / "metrics.json"))
        input_mesh = baseline_surface
    views = {view.id: view for view in surface_views(sparse) if view.id not in used}
    poses = {i: view.pose for i, view in views.items()}
    landmarks = read(sparse / "landmarks.json")
    neighbors = overlapping_pairs(landmarks, poses, landmark_visibility(landmarks, poses, 2.5))
    planned = [(score, i, j, xyz) for i, candidates in neighbors.items() for score, j, xyz in candidates
               if i < j]
    planned.sort(key=lambda row: (-row[0], row[1], row[2]))
    report = {"dense_camera_ids": sorted(used), "reserved_camera_ids": sorted(views), "pairs": [], "rejected": [],
              "method": "first mesh hit versus reciprocal stereo depth; reserved cameras still contributed to SfM"}
    if planned:
        scenes = [ray_scene(o3d.io.read_triangle_mesh(str(path / "mesh.ply"))) for path in (input_mesh, surface_dir)]
        cached = {}
        cv2.setNumThreads(1)
        for _, i, j, sparse_xyz in planned[:16]:
            for k in (i, j):
                if k not in cached:
                    cached[k] = observations(views[k])
            ca, image_a = cached[i]
            cb, image_b = cached[j]
            try:
                try:
                    plan = plan_depth(poses[i], poses[j], ca, cb, sparse_xyz)
                except DisparitySearchLimit:
                    plan = plan_depth(poses[i], poses[j], ca, cb, sparse_xyz, recenter=True)
                depth = make_depth(i, j, poses[i], poses[j], ca, cb, image_a, image_b, sparse_xyz, plan=plan)
            except (ValueError, cv2.error) as exc:
                report["rejected"].append({"pair": [i, j], "reason": str(exc)})
                continue
            xyz = depth.xyz[::max(1, len(depth.xyz) // 10000)]
            rays = np.empty((len(xyz), 6), np.float32)
            rays[:, :3] = poses[i].center
            rays[:, 3:] = xyz - poses[i].center
            tensor = o3d.core.Tensor(rays)
            hits = [scene.cast_rays(tensor, nthreads=2)["t_hit"].numpy() for scene in scenes]
            common = np.isfinite(hits[0]) & np.isfinite(hits[1])
            result = {"pair": [i, j], "stereo_samples": len(xyz), "common_hits": int(common.sum()),
                      "recentered": plan.recentered}
            for name, hit in zip(("input_mesh", "output_mesh"), hits):
                finite = np.isfinite(hit)
                relative_error = np.abs(hit - 1)
                result[name] = {"hit_fraction": float(finite.mean()),
                                "fraction_of_stereo_samples_within_2pct": float((relative_error <= .02).mean()),
                                "common_relative_depth_p50_p95": np.percentile(relative_error[common], [50, 95]).tolist()
                                if common.any() else None}
            report["pairs"].append(result)
            if len(report["pairs"]) == 4:
                break
    report["status"] = "measured" if report["pairs"] else "no_viable_reserved_pair"
    report["input_mesh"] = str(input_mesh)
    report["output_mesh"] = str(surface_dir)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-dir", required=True, type=Path)
    parser.add_argument("--baseline-surface", type=Path, help="Compare both meshes using cameras unused by either dense run")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.surface_dir, args.output, args.baseline_surface)
