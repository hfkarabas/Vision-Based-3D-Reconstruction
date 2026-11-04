"""Reconstruct a photographic pair and export auditable geometry."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
import warnings

import cv2
import numpy as np

from vision3d.features import extract
from vision3d.features.matching import match
from vision3d.geometry.camera import approximate_camera, load_cameras
from vision3d.geometry.two_view import Quality, ReconstructionRejected, reconstruct
from vision3d.io import load_image
from vision3d.preprocessing import preprocess


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def source_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sample_colors(original_bgr, processed_pixels, resize):
    """Nearest original oriented pixel, reversing the recorded pixel-center transform."""
    pixels = np.asarray(processed_pixels, np.float64)
    xy = (pixels + 0.5) / [resize["scale_x"], resize["scale_y"]] - 0.5
    xy = np.rint(xy).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, original_bgr.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, original_bgr.shape[0] - 1)
    return original_bgr[xy[:, 1], xy[:, 0], ::-1].astype(np.float64) / 255.0


def export_cloud(path, xyz, colors):
    import open3d as o3d
    xyz, colors = np.asarray(xyz), np.asarray(colors)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or colors.shape != xyz.shape
            or not np.isfinite(xyz).all() or not np.isfinite(colors).all()
            or (colors < 0).any() or (colors > 1).any()):
        raise ValueError("PLY requires nonempty finite Nx3 points and RGB colors in [0,1]")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise OSError(f"Cannot export point cloud: {path}")
    reloaded = o3d.io.read_point_cloud(str(path))
    if (len(reloaded.points) != len(xyz) or not reloaded.has_colors()
            or not np.allclose(np.asarray(reloaded.points), xyz, rtol=1e-6, atol=1e-8)
            or not np.allclose(np.asarray(reloaded.colors), colors, atol=1 / 255 + 1e-9)):
        raise OSError("PLY reload validation failed")


def preview_cloud(path, xyz, colors, rotation, translation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*xyz.T, c=colors, s=3, depthshade=False)
    centers = np.array([np.zeros(3), -rotation.T @ translation])
    ax.scatter(*centers.T, c=["red", "blue"], marker="^", s=60)
    ax.set(xlabel="X (baseline units)", ylabel="Y (down)", zlabel="Z (forward)",
           title=f"Two-view sparse cloud: {len(xyz)} points; cameras red/blue")
    ax.set_box_aspect(np.maximum(np.ptp(np.r_[xyz, centers], axis=0), 0.01))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def reconstruct_pair(image1, image2, output, camera=None, approximate=False, focal_px=None,
                     fov_degrees=None, detector="sift", max_dimension=1600, max_features=6000,
                     ratio=0.75, seed=0, quality=Quality(), preview=False):
    start = perf_counter()
    if (camera is None) == (not approximate):
        raise ValueError("Choose --camera or explicit --approximate-intrinsics")
    if not approximate and (focal_px is not None or fov_degrees is not None):
        raise ValueError("Focal/FOV assumptions require --approximate-intrinsics")
    cv2.setRNGSeed(seed)
    cv2.setNumThreads(1)
    originals = [load_image(image1), load_image(image2)]
    if camera is not None:
        cameras = load_cameras(camera)
        if any(c.approximate for c in cameras):
            warnings.warn("Camera configuration declares approximate intrinsics; reliability is reduced", stacklevel=2)
    else:
        cameras = [approximate_camera(f.image.shape[1], f.image.shape[0], focal_px, fov_degrees) for f in originals]
    processed, resizes, scaled = [], [], []
    for frame, model in zip(originals, cameras):
        h, w = frame.image.shape[:2]
        if (w, h) != (model.width, model.height):
            raise ValueError("Calibration dimensions must match the original EXIF-oriented image")
        image, resize = preprocess(frame.image, max_dimension)
        processed.append(image)
        resizes.append(resize)
        scaled.append(model.resized(image.shape[1], image.shape[0]))
    features = [extract(image, detector, max_features) for image in processed]
    # Existing fundamental verification is diagnostic only: distortion-aware essential
    # estimation starts from reciprocal descriptor matches, avoiding biased F prefiltering.
    matches = match(*features, ratio=ratio, mutual=True)
    candidates = matches.consistent
    p1 = features[0].points[[m.queryIdx for m in candidates]]
    p2 = features[1].points[[m.trainIdx for m in candidates]]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": [{**f.metadata, "sha256": source_hash(f.metadata["source"])} for f in originals],
                "resizes": resizes, "detector_configs": [f.config for f in features],
                "config": {"detector": detector, "max_dimension": max_dimension, "max_features": max_features,
                           "ratio": ratio, "mutual": True, "seed": seed, "opencv_threads": 1,
                           "quality": asdict(quality), "preview": preview},
                "versions": {n: importlib.metadata.version(n) for n in ("vision3d", "numpy", "opencv-contrib-python", "open3d")},
                "coordinate_convention": "X2=R@X1+t; world=camera1; x right, y down, z forward; ||t||=1",
                "scale": "arbitrary baseline units; absolute scale is unobservable",
                "warnings": ["Approximate intrinsics; geometry may be biased"] if any(c.approximate for c in cameras) else []}
    write_json(output / "manifest.json", manifest)
    write_json(output / "intrinsics.json", {"original": [c.to_dict() for c in cameras], "processed": [c.to_dict() for c in scaled]})
    metrics = {"features": [{"count": f.count, "seconds": f.seconds} for f in features],
               "matching": matches.metrics()}
    try:
        result = reconstruct(p1, p2, *scaled, quality)
    except ReconstructionRejected as exc:
        metrics.update(exc.metrics, status="rejected", reason=str(exc), runtime_seconds=perf_counter() - start)
        write_json(output / "metrics.json", metrics)
        np.savez_compressed(output / "correspondences.npz", pixels1=p1, pixels2=p2,
                            feature_indices=np.array([(m.queryIdx, m.trainIdx) for m in candidates], dtype=int).reshape(-1, 2))
        raise
    valid = result["valid"]
    xyz = result["xyz"][valid]
    colors = sample_colors(originals[0].image, p1[result["point_indices"]], resizes[0])
    export_cloud(output / "sparse.ply", xyz, colors)
    rotation, translation = result["rotation"], result["translation"]
    write_json(output / "poses.json", {"world_to_camera1": np.eye(4).tolist(),
               "world_to_camera2": np.r_[np.c_[rotation, translation], [[0, 0, 0, 1]]].tolist(),
               "camera2_center_world": (-rotation.T @ translation).tolist(),
               "essential_matrix": result["essential"].tolist(), "translation_scale": "unit baseline, not meters"})
    np.savez_compressed(output / "correspondences.npz", pixels1=p1, pixels2=p2,
                        feature_indices=np.array([(m.queryIdx, m.trainIdx) for m in candidates], dtype=int),
                        essential_mask=result["essential_mask"], pose_mask=result["pose_mask"],
                        pose_indices=result["pose_indices"], valid_pose_points=valid,
                        ply_observation_indices=result["point_indices"], xyz_pose_points=result["xyz"],
                        reprojection_errors_px=result["errors_px"], angles_degrees=result["angles_degrees"],
                        positive_depth=result["positive_depth"], bounded=result["bounded"])
    # Original observations are recoverable exactly via the inverse resize transform.
    if preview:
        preview_cloud(output / "sparse_preview.png", xyz, colors, rotation, translation)
    metrics.update(result["metrics"], status="success", runtime_seconds=perf_counter() - start,
                   peak_memory_bytes=None, memory_basis="process lifetime peak working set when available")
    try:
        import psutil
        metrics["peak_memory_bytes"] = getattr(psutil.Process().memory_info(), "peak_wset", None)
    except (ImportError, OSError):
        pass
    write_json(output / "metrics.json", metrics)
    return metrics
