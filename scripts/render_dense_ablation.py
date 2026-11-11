"""Compare saved project and COLMAP clouds in a common project coordinate frame."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from analyze_airzoo_geometry import read
from audit_reconstruction_visuals import VIEWS, View, cloud_image, montage


def poses(directory):
    manifest = read(directory / "manifest.json")
    names = {row["image_id"]: row["name"] for row in manifest["inputs"]}
    return {names[row["image_id"]]: np.asarray(row["world_to_camera"]) for row in read(directory / "poses.json")
            if row["status"] == "registered"}


def common_pose_similarity(source, target):
    """Fit the rotation from paired orientations, then fit center scale and shift."""
    common = sorted(source.keys() & target.keys())
    if len(common) < 3:
        raise ValueError("Three common registered images are required")
    a, b = (np.stack([group[key] for key in common]) for group in (source, target))
    ra, rb = a[:, :3, :3], b[:, :3, :3]
    u, _, vt = np.linalg.svd((rb.transpose(0, 2, 1) @ ra).sum(axis=0))
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    ca = -np.einsum("nji,nj->ni", ra, a[:, :3, 3])
    cb = -np.einsum("nji,nj->ni", rb, b[:, :3, 3])
    ca = ca @ rotation.T
    xa, xb = ca - ca.mean(axis=0), cb - cb.mean(axis=0)
    denominator = np.square(xa).sum()
    if denominator < 1e-12:
        raise ValueError("Common camera centers do not span a scale")
    scale = (xa * xb).sum() / denominator
    if scale <= 0:
        raise ValueError("Common camera alignment does not support positive scale")
    shift = cb.mean(axis=0) - scale * ca.mean(axis=0)
    angular = np.trace(rb @ rotation @ ra.transpose(0, 2, 1), axis1=1, axis2=2)
    residual = np.linalg.norm(scale * ca + shift - cb, axis=1)
    return scale, rotation, shift, {"common_cameras": len(common), "scale": float(scale), "rotation": rotation.tolist(),
                                    "shift": shift.tolist(),
                                    "center_residual_sfm_p50_p95": np.percentile(residual, [50, 95]).tolist(),
                                    "orientation_residual_deg_p50_p95":
                                        np.percentile(np.degrees(np.arccos(((angular - 1) / 2).clip(-1, 1))), [50, 95]).tolist()}


def render(baseline, frame_path, variants, output, size):
    output.mkdir(parents=True, exist_ok=False)
    target = poses(baseline)
    frame = read(frame_path)
    views = {name: View(frame, np.asarray(direction), size) for name, direction in VIEWS.items()}
    metrics = {}
    for label, path, source in variants:
        cloud = o3d.io.read_point_cloud(path)
        xyz = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors)
        if not len(xyz) or not len(colors):
            raise ValueError(f"Expected a colored cloud: {path}")
        report = {"path": path, "points": len(xyz), "common_pose_fit": None}
        if source != "same":
            scale, rotation, shift, alignment = common_pose_similarity(poses(Path(source)), target)
            xyz = scale * xyz @ rotation.T + shift
            report["common_pose_fit"] = alignment
        report["views"] = {}
        for name, view in views.items():
            xy, distance = view.pixels(xyz)
            inside = ((xy >= 0) & (xy < size)).all(axis=1) & (distance < 2 * view.span)
            rgb, mask = cloud_image(xyz, colors, view)
            Image.fromarray(rgb).save(output / f"{name}-{label}.png")
            report["views"][name] = {"occupied_pixels": int(mask.sum()), "points_inside_frame": int(inside.sum())}
        metrics[label] = report
    montage(output, views, [label for label, *_ in variants], "crossed-backends.png")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-sparse", type=Path, required=True)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--variant", nargs=3, metavar=("LABEL", "PLY", "SPARSE_DIR_OR_SAME"), action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=480)
    args = parser.parse_args()
    render(args.baseline_sparse, args.frame, args.variant, args.output, args.size)
