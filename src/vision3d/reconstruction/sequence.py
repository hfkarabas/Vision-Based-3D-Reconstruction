"""Ordered-input, track-based incremental SfM with bounded optional local BA."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import json
from time import perf_counter
import warnings

import cv2
import numpy as np
from PIL import Image

from vision3d.features import extract
from vision3d.features.matching import match
from vision3d.geometry import verify
from vision3d.geometry.camera import Camera, approximate_camera, load_cameras
from vision3d.geometry.two_view import Quality, ReconstructionRejected
from vision3d.io import iter_frames
from vision3d.preprocessing import preprocess
from .bundle import BundleOptions, local_bundle_adjust
from .incremental import RegistrationQuality, extend_landmarks, initialize, landmark_diagnostics, pnp_correspondences, register, select_initial, validate_landmarks
from .pair import export_cloud, sample_colors, source_hash, write_json
from .state import State, View, build_tracks


def load_quality(path):
    data = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(data, dict) or set(data) - {"geometry", "registration", "bundle"}:
        raise ValueError("Quality configuration accepts geometry, registration and bundle sections")
    try:
        return Quality(**data.get("geometry", {})), RegistrationQuality(**data.get("registration", {})), BundleOptions(**data.get("bundle", {}))
    except TypeError as exc:
        raise ValueError(f"Invalid quality configuration: {exc}") from exc


def match_sequence(views, window, long_gap, quality, ratio):
    pairs, pair_metrics, edges = {}, [], []
    ids = sorted(views)
    for i in ids:
        for j in ids:
            if j <= i or not (j - i <= window or (long_gap > 0 and j - i == long_gap)):
                continue
            a, b = views[i], views[j]
            result = match(a.features, b.features, ratio=ratio, mutual=True)
            matches = result.consistent
            fi = np.array([m.queryIdx for m in matches], dtype=int)
            fj = np.array([m.trainIdx for m in matches], dtype=int)
            focal = min(a.camera.fx, a.camera.fy, b.camera.fx, b.camera.fy)
            # Existing fundamental verifier, now on undistorted shared virtual pixels.
            _, mask, status = verify(a.camera.normalized(a.features.points[fi]) * focal,
                                     b.camera.normalized(b.features.points[fj]) * focal,
                                     threshold=quality.essential_threshold_px, confidence=quality.confidence)
            correspondences = np.c_[fi[mask], fj[mask]]
            if len(correspondences) >= quality.min_inliers:
                pairs[i, j] = correspondences
                for m, keep in zip(matches, mask):
                    if keep:
                        edges.append((float(m.distance), (i, m.queryIdx), (j, m.trainIdx)))
            pair_metrics.append({"pair": [i, j], **result.metrics(), "undistorted_verified": int(mask.sum()), "verification_status": status})
    tracks, track_metrics = build_tracks([(a, b) for _, a, b in sorted(edges)])
    return pairs, tracks, pair_metrics, track_metrics


def reconstruct_sequence(input_path, output, camera=None, approximate=False, focal_px=None, fov_degrees=None,
                         detector="sift", max_dimension=1600, max_features=6000, ratio=0.75,
                         window=2, long_gap=4, max_images=30, min_registered=5, local_ba=False,
                         quality_config=None, seed=0):
    start = perf_counter()
    if not Path(input_path).is_dir():
        raise ValueError("Sequence input must be an ordered JPEG/PNG directory")
    if (camera is None) == (not approximate):
        raise ValueError("Choose a camera configuration or explicit approximate intrinsics")
    if not approximate and (focal_px is not None or fov_degrees is not None):
        raise ValueError("Focal/FOV assumptions require explicit approximate intrinsics")
    if not 1 <= window <= 10 or not 0 <= long_gap <= 30 or not 2 <= max_images <= 50 or min_registered < 2:
        raise ValueError("Invalid sequence bounds: window 1..10, long gap 0..30, images 2..50, registration target >=2")
    if not 0 < ratio < 1:
        raise ValueError("Ratio must be between 0 and 1")
    if not 1 <= max_features <= 20000 or max_dimension < 2:
        raise ValueError("Feature limit must be 1..20000 and maximum dimension >=2")
    geometry, pnp_quality, bundle = load_quality(quality_config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cv2.setRNGSeed(seed)
    cv2.setNumThreads(1)
    camera_data = json.loads(Path(camera).read_text()) if camera else None
    per_image = camera_data.get("images") if isinstance(camera_data, dict) else None
    if camera_data is not None and per_image is None:
        shared = load_cameras(camera)
        if shared[0].to_dict() != shared[1].to_dict():
            raise ValueError("Sequence calibration uses one shared camera or an images mapping, not a two-camera pair")
        shared = shared[0]
    views, skipped, intrinsics = {}, [], []
    candidates = sorted((p for p in Path(input_path).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                        key=lambda p: (p.name.casefold(), p.name))
    source_order = {p.name: i for i, p in enumerate(candidates)}
    for i, frame in enumerate(iter_frames(input_path, errors=skipped)):
        if i >= max_images:
            raise ValueError("Image limit exceeded; choose a smaller subset or increase --max-images (maximum 50)")
        path = Path(frame.metadata["source"])
        h, w = frame.image.shape[:2]
        if camera is None:
            model = approximate_camera(w, h, focal_px, fov_degrees)
        elif per_image is not None:
            if path.name not in per_image:
                raise ValueError(f"Missing calibration for {path.name}")
            value = dict(per_image[path.name])
            matrix = value.pop("K", None)
            try:
                model = Camera(**value)
            except TypeError as exc:
                raise ValueError(f"Invalid calibration for {path.name}: {exc}") from exc
            if matrix is not None and not np.allclose(matrix, model.K):
                raise ValueError("K conflicts with scalar intrinsics")
        else:
            model = shared
        if (model.width, model.height) != (w, h):
            raise ValueError(f"Camera dimensions differ from EXIF-oriented input: {path.name}")
        if model.approximate:
            warnings.warn(f"Approximate intrinsics for {path.name}; reduced reliability", stacklevel=2)
        image, resize = preprocess(frame.image, max_dimension)
        features = extract(image, detector, max_features)
        scaled = model.resized(image.shape[1], image.shape[0])
        with Image.open(path) as source:
            exif = source.getexif()
            timestamp = exif.get_ifd(34665).get(36867, exif.get(306))
        metadata = {**frame.metadata, "sequence_index": i, "source_order_index": source_order[path.name], "timestamp": timestamp,
                    "sha256": source_hash(path), "resize": resize}
        views[i] = View(i, path.name, scaled, features,
                        sample_colors(frame.image, features.points, resize), metadata)
        intrinsics.append({"image_id": i, "original": model.to_dict(), "processed": scaled.to_dict()})
    if len(views) < 2:
        raise ValueError("At least two readable images are required")
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": [{"image_id": i, "name": v.name, **v.metadata} for i, v in views.items()],
                "skipped_unreadable_inputs": skipped, "input_order": "case-insensitive filename, deterministic tie-break",
                "convention": "Xcamera=R@Xworld+t; world=first selected anchor; scale=unit initial baseline",
                "config": {"detector": detector, "max_dimension": max_dimension, "max_features": max_features,
                           "ratio": ratio, "window": window, "long_gap": long_gap, "max_images": max_images,
                           "min_registered": min_registered, "local_ba": local_ba, "seed": seed,
                           "opencv_threads": 1, "geometry": asdict(geometry), "registration": asdict(pnp_quality), "bundle": asdict(bundle)},
                "versions": {n: importlib.metadata.version(n) for n in ("vision3d", "numpy", "scipy", "opencv-contrib-python", "open3d")}}
    write_json(output / "manifest.json", manifest)
    write_json(output / "intrinsics.json", intrinsics)
    pairs, tracks, pair_metrics, track_metrics = match_sequence(views, window, long_gap, geometry, ratio)
    write_json(output / "pairs.json", pair_metrics)
    state = State(views, tracks)
    try:
        initial, candidates = select_initial(views, pairs, geometry)
        initialize(state, initial, geometry)
    except ReconstructionRejected as exc:
        write_json(output / "metrics.json", {"status": "rejected", "reason": str(exc), **exc.metrics, "tracks": track_metrics})
        raise
    write_json(output / "initial_candidates.json", candidates)
    attempts, bundle_history = {i: [] for i in views}, []
    pending = set(views) - state.poses.keys()
    while pending:
        progress = False
        ranked = sorted(pending, key=lambda i: (-len(pnp_correspondences(state, i)[0]), i))
        for i in ranked:
            accepted, metrics = register(state, i, pnp_quality, geometry)
            attempts[i].append(metrics)
            if not accepted:
                continue
            pending.remove(i)
            progress = True
            state.history.append({"image_id": i, "pnp": metrics, **extend_landmarks(state, geometry)})
            if local_ba:
                bundle_history.append(local_bundle_adjust(state, bundle, geometry.max_reprojection_px,
                    geometry.min_angle_degrees, geometry.max_distance))
        if not progress:
            break
    validation_args = {"min_angle_degrees": geometry.min_angle_degrees, "max_distance": geometry.max_distance}
    current_quality = validate_landmarks(state, geometry.max_reprojection_px, **validation_args)
    before_final_ba = bundle_history[-1].get("before", current_quality) if bundle_history else current_quality
    final_quality = validate_landmarks(state, geometry.max_reprojection_px, prune=True, **validation_args)
    after_final_ba = validate_landmarks(state, geometry.max_reprojection_px, **validation_args)
    visibility = {i: sum(i in l.observations for l in state.landmarks.values()) for i in state.poses}
    weak = sorted(i for i, count in visibility.items() if count < pnp_quality.min_inliers)
    metrics = {"status": "success" if len(state.poses) >= min_registered and len(state.landmarks) >= geometry.min_points and not weak else "partial",
               "input_images": len(views), "registered_images": len(state.poses), "rejected_images": sorted(pending),
               "initial_pair": list(state.anchors), "registration_order": state.registration_order,
               "tracks": track_metrics, "valid_landmarks": len(state.landmarks), "weakly_supported_images": weak,
               "before_final_ba": before_final_ba, "after_final_ba": after_final_ba,
               "final_pruning": final_quality, "bundle_adjustment": bundle_history, "incremental_steps": state.history}
    write_json(output / "poses.json", [{"image_id": i, "name": v.name, "status": "registered" if i in state.poses else "rejected",
                "world_to_camera": state.poses[i].matrix.tolist() if i in state.poses else None,
                "center_world": state.poses[i].center.tolist() if i in state.poses else None,
                "timestamp": v.metadata["timestamp"], "attempts": attempts[i]} for i, v in views.items()])
    write_json(output / "tracks.json", [{"id": t.id, "observations": t.observations,
                "landmark_id": t.id if t.id in state.landmarks else None} for t in tracks.values()])
    landmarks = [state.landmarks[i] for i in sorted(state.landmarks)]
    write_json(output / "landmarks.json", [{"id": l.id, "track_id": l.id, "xyz": l.xyz.tolist(), "color": l.color.tolist(),
                "observations": l.observations, **landmark_diagnostics(state, l)} for l in landmarks])
    np.savez_compressed(output / "features.npz", **{f"image_{i}_points": v.features.points for i, v in views.items()})
    if landmarks:
        export_cloud(output / "sparse.ply", [l.xyz for l in landmarks], [l.color for l in landmarks])
        save_diagnostic(output / "overview.png", state)
    metrics["per_image"] = [{"image_id": i, "name": v.name, "features": v.features.count,
                "feature_seconds": v.features.seconds, "status": "registered" if i in state.poses else "rejected",
                "visible_landmarks": sum(i in l.observations for l in landmarks)} for i, v in views.items()]
    metrics.update(runtime_seconds=perf_counter() - start, peak_memory_bytes=None,
                   memory_basis="Windows process lifetime peak working set including imports; null when unavailable")
    try:
        import psutil
        metrics["peak_memory_bytes"] = getattr(psutil.Process().memory_info(), "peak_wset", None)
    except (ImportError, OSError):
        pass
    write_json(output / "metrics.json", metrics)
    return metrics


def save_diagnostic(path, state):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(121, projection="3d")
    xyz = np.array([l.xyz for l in state.landmarks.values()])
    ax.scatter(*xyz.T, c=[l.color for l in state.landmarks.values()], s=2)
    centers = np.array([p.center for p in state.poses.values()])
    ax.scatter(*centers.T, c="red", marker="^", s=30)
    for i, pose in state.poses.items():
        ax.text(*pose.center, str(i))
    ax.set(xlabel="X", ylabel="Y (down)", zlabel="Z", title="Sparse scene / registered camera IDs")
    ax.set_box_aspect(np.maximum(np.ptp(np.r_[xyz, centers], axis=0), 0.01))
    chart = fig.add_subplot(122)
    lengths = [len(t.observations) for t in state.tracks.values()]
    chart.hist(lengths, bins=np.arange(1.5, max(lengths) + 1.5), rwidth=0.8)
    chart.set(xlabel="Track length", ylabel="Tracks", title=f"{len(state.poses)} cameras; {len(xyz)} landmarks")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
