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

from vision3d.features.cache import cached_extract
from vision3d.features.pair_cache import descriptor_key, reciprocal_matches
from vision3d.geometry import verify
from vision3d.geometry.camera import approximate_camera, infer_camera, load_calibration, named_camera
from vision3d.geometry.two_view import Quality, ReconstructionRejected
from vision3d.io import image_paths, load_image
from vision3d.preprocessing import preprocess
from .bundle import BundleOptions, local_bundle_adjust
from .incremental import RegistrationQuality, extend_landmarks, initialize, landmark_diagnostics, measure_landmarks, register, registration_priority, select_initial, validate_landmarks
from .pair import export_cloud, sample_colors, source_hash, write_json
from .state import State, View, build_tracks, track_counts
from .graph import appearance_signatures, bridge_schedule, components, discovery_candidate
from .map_recovery import MapRecovery
from .seed_search import load_position_hints, search_seeds, station_labels


def load_quality(path):
    data = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(data, dict) or set(data) - {"geometry", "registration", "bundle"}:
        raise ValueError("Quality configuration accepts geometry, registration and bundle sections")
    try:
        return Quality(**data.get("geometry", {})), RegistrationQuality(**data.get("registration", {})), BundleOptions(**data.get("bundle", {}))
    except TypeError as exc:
        raise ValueError(f"Invalid quality configuration: {exc}") from exc


def pair_evaluator(views, quality, ratio, seed, verifier, pair_cache):
    keys = {i: descriptor_key(v.features) for i, v in views.items()} if pair_cache is not None else {}
    def evaluate(i, j, schedule, discovery_distance=None):
        a, b = views[i], views[j]
        cv2.setRNGSeed((seed + i * 1009 + j * 9176) % 2147483647)
        keys_for_pair = (keys[i], keys[j]) if keys else None
        metrics, matches, hit = reciprocal_matches(a.features, b.features, ratio, pair_cache, keys_for_pair)
        fi = np.array([m[0] for m in matches], dtype=int)
        fj = np.array([m[1] for m in matches], dtype=int)
        focal = min(a.camera.fx, a.camera.fy, b.camera.fx, b.camera.fy)
        # Existing fundamental verifier, now on undistorted shared virtual pixels.
        matrix, mask, status = verify(a.camera.normalized(a.features.points[fi]) * focal,
                                 b.camera.normalized(b.features.points[fj]) * focal,
                                 threshold=quality.essential_threshold_px, confidence=quality.confidence,
                                 method=cv2.USAC_MAGSAC if verifier == "magsac" else cv2.FM_RANSAC)
        correspondences = np.c_[fi[mask], fj[mask]]
        edges = [(distance, (i, query), (j, train))
                 for (query, train, distance), keep in zip(matches, mask) if keep]
        coverage = []
        for view, indices in ((a, fi), (b, fj)):
            pixels = view.features.points[indices[mask]]
            area = cv2.contourArea(cv2.convexHull(pixels)) if len(pixels) >= 3 else 0
            coverage.append(area / (view.camera.width * view.camera.height))
        n = int(mask.sum())
        metrics.update(ransac_inliers=n, inlier_ratio=n / len(matches) if matches else 0,
                       geometry_status=status, fundamental_matrix=None if matrix is None else matrix.tolist())
        row = {"pair": [i, j], **metrics, "undistorted_verified": n,
               "verification_status": status, "inlier_coverage": coverage,
               "edge_status": "verified" if n >= quality.min_inliers else
                   ("below_minimum_inliers" if status == "estimated" else status),
               "schedule": schedule, "pair_cache_hit": hit, "discovery_distance": discovery_distance}
        return row, correspondences, edges

    return evaluate

def match_sequence(views, window, long_gap, quality, ratio, seed=0, recovery_pairs=0, *,
                   exhaustive=False, verifier="ransac", pair_cache=None, bridge_labels=None, bridge_pairs=0):
    pairs, pair_metrics, edges = {}, [], []
    start = perf_counter()
    ids = sorted(views)
    evaluate = pair_evaluator(views, quality, ratio, seed, verifier, pair_cache)

    def process(i, j, discovery_distance=None, schedule="local"):
        row, correspondences, verified_edges = evaluate(i, j, schedule, discovery_distance)
        pair_metrics.append(row)
        if row["edge_status"] == "verified":
            pairs[i, j] = correspondences
            edges.extend(verified_edges)

    for i in ids:
        for j in ids:
            if j > i and (j - i <= window or (long_gap > 0 and j - i == long_gap)):
                process(i, j)
    original_components = components(views, pairs)
    attempted = {tuple(p['pair']) for p in pair_metrics}
    if exhaustive:
        for i in ids:
            for j in ids:
                if j > i and (i, j) not in attempted:
                    process(i, j, schedule="exhaustive")
    counts = {}
    signatures = appearance_signatures(views) if recovery_pairs else {}
    for _ in range(recovery_pairs):
        candidate = discovery_candidate(views, pairs, attempted, counts, signatures)
        if candidate is None:
            break
        i, j, distance = candidate
        attempted.add((i, j))
        counts[i], counts[j] = counts.get(i, 0) + 1, counts.get(j, 0) + 1
        process(i, j, distance, "discovery")
    if bridge_pairs:
        for i, j, support in bridge_schedule(views, bridge_labels, attempted, bridge_pairs, ratio,
                                              quality.essential_threshold_px, quality.confidence, seed):
            process(i, j, schedule="station_bridge")
            pair_metrics[-1]["coarse_inliers"] = support
    matching_seconds = perf_counter() - start
    start = perf_counter()
    tracks, track_metrics = build_tracks([(a, b) for _, a, b in sorted(edges)])
    track_metrics.update(matching_seconds=matching_seconds, track_construction_seconds=perf_counter() - start)
    track_metrics['graph'] = {'components_before': original_components, 'components_after': components(views, pairs),
                              'scheduled_pairs': sum(p['schedule'] == 'local' for p in pair_metrics),
                              'exhaustive_attempts': sum(p['schedule'] == 'exhaustive' for p in pair_metrics),
                              'pair_cache_hits': sum(p['pair_cache_hit'] for p in pair_metrics),
                              'discovery_attempts': sum(p['discovery_distance'] is not None for p in pair_metrics),
                              'verified_edges': len(pairs), 'discovery_budget': recovery_pairs}
    if bridge_pairs:
        bridge_rows = [row for row in pair_metrics if row["schedule"] == "station_bridge"]
        track_metrics["graph"].update(station_bridge_attempts=len(bridge_rows),
                                      station_bridge_verified=sum(row["edge_status"] == "verified" for row in bridge_rows))
    return pairs, tracks, pair_metrics, track_metrics


def reconstruct_sequence(input_path, output, camera=None, approximate=False, focal_px=None, fov_degrees=None,
                         detector="sift", max_dimension=1600, max_features=6000, ratio=0.75,
                         window=2, long_gap=4, max_images=30, min_registered=5, local_ba=False,
                         quality_config=None, seed=0, feature_cache=None, recovery_pairs=0, joint_ba=False,
                         refine_points=False, auto_intrinsics=False, exhaustive=False, verifier="ransac", pair_cache=None,
                         continuation_seed=False, probe_seeds=False, map_recovery_pairs=0, map_recovery_rounds=3,
                         seed_search=0, seed_search_views=8, position_hints=None, station_radius=15.0,
                         quality_pnp=False, final_retry=0, station_recovery=False, initial_pair=None, bridge_pairs=0):
    start = perf_counter()
    stages = {k: 0.0 for k in ("loading_preprocessing", "features", "matching", "tracks", "initial_pair",
                               "registration", "triangulation", "bundle_adjustment", "validation", "export")}
    feature_extraction_seconds = cache_hits = 0
    if isinstance(input_path, (str, Path)) and not Path(input_path).is_dir():
        raise ValueError("Sequence input must be an ordered JPEG/PNG directory")
    if sum((camera is not None, approximate, auto_intrinsics)) != 1:
        raise ValueError("Choose a camera configuration, automatic intrinsics or explicit approximate intrinsics")
    if not approximate and (focal_px is not None or fov_degrees is not None):
        raise ValueError("Focal/FOV assumptions require explicit approximate intrinsics")
    if not 1 <= window <= 10 or not 0 <= long_gap <= 30 or not 2 <= max_images <= 50 or min_registered < 2:
        raise ValueError("Invalid sequence bounds: window 1..10, long gap 0..30, images 2..50, registration target >=2")
    if not 0 < ratio < 1:
        raise ValueError("Ratio must be between 0 and 1")
    if not 1 <= max_features <= 20000 or max_dimension < 2:
        raise ValueError("Feature limit must be 1..20000 and maximum dimension >=2")
    if not isinstance(recovery_pairs, int) or not 0 <= recovery_pairs <= 30:
        raise ValueError('Recovery pair budget must be 0..30')
    if not (type(map_recovery_pairs) is int and 0 <= map_recovery_pairs <= 100
            and type(map_recovery_rounds) is int and 1 <= map_recovery_rounds <= 10):
        raise ValueError("Map recovery requires a pair budget of 0..100 and a round limit of 1..10")
    geometry, pnp_quality, bundle = load_quality(quality_config)
    if verifier not in ("ransac", "magsac"):
        raise ValueError("Unknown fundamental matrix verifier")
    if exhaustive and recovery_pairs:
        raise ValueError("Exhaustive scheduling already evaluates every pair")
    if (type(seed_search) is not int or not 0 <= seed_search <= 16
            or type(seed_search_views) is not int or not 1 <= seed_search_views <= 16
            or not np.isfinite(station_radius) or station_radius <= 0):
        raise ValueError("Invalid seed search budget or station radius")
    if type(bridge_pairs) is not int or not 0 <= bridge_pairs <= 40:
        raise ValueError("Station bridge budget must be between 0 and 40")
    if bridge_pairs and (position_hints is None or exhaustive):
        raise ValueError("Station bridges require position hints and non-exhaustive matching")
    if position_hints is not None and not (seed_search or station_recovery or bridge_pairs):
        raise ValueError("Position hints require seed search, station bridges or station-aware recovery")
    if station_recovery and (position_hints is None or not map_recovery_pairs):
        raise ValueError("Station-aware recovery requires position hints and a positive map recovery budget")
    if type(final_retry) is not int or not 0 <= final_retry <= 10:
        raise ValueError("Final retry camera budget must be between 0 and 10")
    if initial_pair is not None and (seed_search or len(initial_pair) != 2 or initial_pair[0] == initial_pair[1]):
        raise ValueError("An explicit seed requires two distinct filenames and no seed search")
    hints = load_position_hints(position_hints) if position_hints is not None else {}
    candidates = image_paths(input_path)
    if hints and len({p.name for p in candidates}) != len(candidates):
        raise ValueError("Position hints require unique input filenames")
    if hints and not any(p.name in hints for p in candidates):
        raise ValueError("No position hints match the input filenames")
    if exhaustive and len(candidates) > 30:
        raise ValueError("Exhaustive matching is limited to 30 images")
    if len(candidates) > max_images:
        raise ValueError("Image limit exceeded; select a smaller directory or increase --max-images")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cv2.setRNGSeed(seed)
    cv2.setNumThreads(1)
    mapping, explicit = load_calibration(camera) if camera is not None else (None, None)
    if mapping is not None and len({p.name for p in candidates}) != len(candidates):
        raise ValueError("Per-image calibration requires unique input filenames")
    views, skipped, intrinsics = {}, [], []
    loading_start = perf_counter()
    for source_index, path in enumerate(candidates):
        try:
            frame = load_image(path)
        except ValueError as exc:
            skipped.append(str(exc))
            warnings.warn(str(exc), stacklevel=2)
            continue
        i = len(views)
        h, w = frame.image.shape[:2]
        if mapping is not None:
            model = named_camera(mapping, path.name)
        elif explicit:
            model = explicit[0]
        elif auto_intrinsics:
            model = infer_camera(w, h, frame.metadata)
        else:
            model = approximate_camera(w, h, focal_px, fov_degrees)
        if (model.width, model.height) != (w, h):
            raise ValueError(f"Camera dimensions differ from EXIF-oriented input: {path.name}")
        if model.approximate:
            warnings.warn(f"Approximate intrinsics for {path.name}; reduced reliability", stacklevel=2)
        image, resize = preprocess(frame.image, max_dimension)
        content_hash = source_hash(path)
        feature_start = perf_counter()
        features, cache_hit = cached_extract(image, content_hash, max_dimension, detector, max_features, feature_cache)
        stages["features"] += perf_counter() - feature_start
        feature_extraction_seconds += features.seconds
        cache_hits += int(cache_hit)
        scaled = model.resized(image.shape[1], image.shape[0])
        metadata = {**frame.metadata, "sequence_index": i, "source_order_index": source_index,
                    "sha256": content_hash, "resize": resize, "feature_cache_hit": cache_hit}
        views[i] = View(i, path.name, scaled, features,
                        sample_colors(frame.image, features.points, resize), metadata)
        intrinsics.append({"image_id": i, "original": model.to_dict(), "processed": scaled.to_dict()})
    if len(views) < 2:
        raise ValueError("At least two readable images are required")
    requested_pair = None
    if initial_pair is not None:
        requested_pair = tuple(i for i, view in views.items() if view.name in initial_pair)
        if len(requested_pair) != 2 or {views[i].name for i in requested_pair} != set(initial_pair):
            raise ValueError("Each explicit seed filename must identify one readable image")
    stages["loading_preprocessing"] = perf_counter() - loading_start - stages["features"]
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": [{"image_id": i, "name": v.name, **v.metadata} for i, v in views.items()],
                "skipped_unreadable_inputs": skipped, "input_order": "case-insensitive filename, deterministic tie-break",
                "convention": "Xcamera=R@Xworld+t; world=first selected anchor; scale=unit initial baseline",
                "config": {"detector": detector, "max_dimension": max_dimension, "max_features": max_features,
                           "ratio": ratio, "window": window, "long_gap": long_gap, "max_images": max_images,
                           "min_registered": min_registered, "local_ba": local_ba, "seed": seed,
                           "opencv_threads": 1, "feature_cache": str(Path(feature_cache).resolve()) if feature_cache else None,
                           "recovery_pairs": recovery_pairs, "joint_ba": joint_ba, "refine_points": refine_points,
                            "auto_intrinsics": auto_intrinsics, "exhaustive": exhaustive, "verifier": verifier,
                            "continuation_seed": continuation_seed, "probe_seeds": probe_seeds,
                           "geometry": asdict(geometry), "registration": asdict(pnp_quality), "bundle": asdict(bundle)},
                "versions": {n: importlib.metadata.version(n) for n in ("vision3d", "numpy", "scipy", "opencv-contrib-python", "open3d")}}
    if map_recovery_pairs:
        manifest["config"].update(map_recovery_pairs=map_recovery_pairs, map_recovery_rounds=map_recovery_rounds)
    if seed_search or station_recovery or bridge_pairs:
        manifest["config"].update(seed_search=seed_search, seed_search_views=seed_search_views, station_radius=station_radius,
                                  position_hints={name: position.tolist() for name, position in hints.items()},
                                  station_recovery=station_recovery, bridge_pairs=bridge_pairs)
    if quality_pnp or final_retry:
        manifest["config"].update(quality_pnp=quality_pnp, final_retry=final_retry)
    if requested_pair is not None:
        manifest["config"]["requested_initial_pair"] = list(initial_pair)
    write_json(output / "manifest.json", manifest)
    write_json(output / "intrinsics.json", intrinsics)
    positions = {i: hints[v.name] for i, v in views.items() if v.name in hints}
    labels = station_labels(positions, station_radius) if bridge_pairs or station_recovery else None
    pairs, tracks, pair_metrics, track_metrics = match_sequence(views, window, long_gap, geometry, ratio, seed, recovery_pairs,
                                                               exhaustive=exhaustive, verifier=verifier, pair_cache=pair_cache,
                                                               bridge_labels=labels, bridge_pairs=bridge_pairs)
    stages["matching"] = track_metrics["matching_seconds"]
    stages["tracks"] = track_metrics["track_construction_seconds"]
    write_json(output / "pairs.json", pair_metrics)
    if exhaustive or recovery_pairs or bridge_pairs:
        np.savez_compressed(output / "verified_pairs.npz", **{f"{a}_{b}": matches for (a, b), matches in pairs.items()})
    state = State(views, tracks)
    phase_start = perf_counter()
    search = None
    if seed_search:
        def search(baseline, ranked, diagnostics):
            return search_seeds(baseline, ranked, diagnostics, views, tracks, geometry, pnp_quality, seed,
                                positions, station_radius, seed_search, seed_search_views)

    try:
        initial, candidates = select_initial(views, pairs, geometry, tracks, seed,
                                             continuation_min=pnp_quality.min_inliers if continuation_seed or probe_seeds or seed_search else None,
                                             probe_quality=pnp_quality if probe_seeds else None, search=search,
                                             required_pair=requested_pair)
        initialize(state, initial, geometry)
    except ReconstructionRejected as exc:
        stages["initial_pair"] = perf_counter() - phase_start
        write_json(output / "metrics.json", {"status": "rejected", "reason": str(exc), **exc.metrics, "tracks": track_metrics,
                   "input_images": len(views), "registered_images": 0, "valid_landmarks": 0,
                   "stage_seconds": stages, "runtime_seconds": perf_counter() - start,
                   "feature_extraction_seconds": feature_extraction_seconds, "feature_cache_hits": cache_hits})
        raise
    stages["initial_pair"] = perf_counter() - phase_start
    write_json(output / "initial_candidates.json", candidates)
    attempts, bundle_history = {i: [] for i in views}, []
    pending = set(views) - state.poses.keys()
    last_attempt_revision = {}
    recovery = MapRecovery(state, pair_metrics, map_recovery_pairs, map_recovery_rounds,
                           labels if station_recovery else None) if map_recovery_pairs else None
    if recovery:
        evaluate = pair_evaluator(views, geometry, ratio, seed, verifier, pair_cache)
        stages["map_recovery"] = 0.0

    def attempt_camera(i, revision, *, late=False):
        cv2.setRNGSeed((seed + i * 1009 + revision * 9176) % 2147483647)
        started = perf_counter()
        accepted, metrics = register(state, i, pnp_quality, geometry, quality_pnp=quality_pnp)
        metrics["map_revision"] = revision
        if recovery and recovery.rounds:
            metrics["map_recovery_round"] = len(recovery.rounds)
        if late:
            metrics["after_point_refinement"] = True
        attempts[i].append(metrics)
        last_attempt_revision[i] = revision
        stages["registration"] += perf_counter() - started
        if not accepted:
            return False
        pending.remove(i)
        started = perf_counter()
        state.history.append({"image_id": i, "pnp": metrics, **extend_landmarks(state, geometry)})
        stages["triangulation"] += perf_counter() - started
        if local_ba:
            started = perf_counter()
            bundle_history.append(local_bundle_adjust(state, bundle, geometry.max_reprojection_px,
                geometry.min_angle_degrees, geometry.max_distance))
            stages["bundle_adjustment"] += perf_counter() - started
        return True

    while pending:
        phase_start = perf_counter()
        revision = len(state.poses)
        available = [i for i in pending if last_attempt_revision.get(i) != revision]
        if not available:
            if recovery is None:
                break
            changed = recovery.run_round(state, pending, pairs, pair_metrics, attempts, evaluate, geometry,
                                         pnp_quality.min_inliers)
            stages["map_recovery"] += perf_counter() - phase_start
            if changed is None:
                break
            for image in changed:
                last_attempt_revision.pop(image, None)
            continue
        i = max(available, key=lambda image: registration_priority(state, image))
        stages["registration"] += perf_counter() - phase_start
        attempt_camera(i, revision)
    if recovery:
        graph = track_metrics["graph"]
        graph.update(components_before_map_recovery=graph["components_after"], components_after=components(views, pairs),
                     verified_edges=len(pairs), map_recovery_attempts=recovery.pair_attempts,
                     map_recovery_verified=recovery.verified_pairs,
                     pair_cache_hits=sum(row["pair_cache_hit"] for row in pair_metrics))
        track_metrics.update(track_counts(state.tracks))
        write_json(output / "pairs.json", pair_metrics)
        np.savez_compressed(output / "verified_pairs.npz", **{f"{a}_{b}": matches for (a, b), matches in pairs.items()})
    if joint_ba:
        phase_start = perf_counter()
        bundle_history.append(local_bundle_adjust(state, bundle, geometry.max_reprojection_px,
            geometry.min_angle_degrees, geometry.max_distance, joint=True))
        stages['bundle_adjustment'] += perf_counter() - phase_start
    point_refinement = {"status": "disabled"}
    if refine_points:
        from .refinement import refine_landmarks
        before_points = validate_landmarks(state, geometry.max_reprojection_px,
            min_angle_degrees=geometry.min_angle_degrees, max_distance=geometry.max_distance)
        point_refinement = refine_landmarks(state, geometry.max_reprojection_px,
            geometry.min_angle_degrees, geometry.max_distance)
        point_refinement["full_map_before"] = before_points
        point_refinement["full_map_after"] = validate_landmarks(state, geometry.max_reprojection_px,
            min_angle_degrees=geometry.min_angle_degrees, max_distance=geometry.max_distance)
    late_registered = []
    if final_retry and pending:
        # The map may have moved even though the camera count stayed unchanged.
        validate_landmarks(state, geometry.max_reprojection_px, prune=True,
                           min_angle_degrees=geometry.min_angle_degrees, max_distance=geometry.max_distance)
        for _ in range(final_retry):
            revision = len(state.poses)
            priorities = sorted(((registration_priority(state, i), i) for i in pending), reverse=True)
            for priority, i in priorities:
                if priority[1] >= pnp_quality.min_inliers and attempt_camera(i, revision, late=True):
                    late_registered.append(i)
                    break
            else:
                break
        stages["point_refinement"] = point_refinement["seconds"]
    phase_start = perf_counter()
    validation_args = {"min_angle_degrees": geometry.min_angle_degrees, "max_distance": geometry.max_distance}
    current_quality = validate_landmarks(state, geometry.max_reprojection_px, **validation_args)
    before_final_ba = bundle_history[-1].get("before", current_quality) if bundle_history else current_quality
    final_quality = validate_landmarks(state, geometry.max_reprojection_px, prune=True, **validation_args)
    after_final_ba = validate_landmarks(state, geometry.max_reprojection_px, **validation_args)
    stages["validation"] = perf_counter() - phase_start
    visibility = {i: sum(i in l.observations for l in state.landmarks.values()) for i in state.poses}
    weak = sorted(i for i, count in visibility.items() if count < pnp_quality.min_inliers)
    metrics = {"status": "success" if len(state.poses) >= min_registered and len(state.landmarks) >= geometry.min_points and not weak else "partial",
               "input_images": len(views), "registered_images": len(state.poses), "rejected_images": sorted(pending),
               "initial_pair": list(state.anchors), "registration_order": state.registration_order,
               "tracks": track_metrics, "valid_landmarks": len(state.landmarks), "weakly_supported_images": weak,
               "before_final_ba": before_final_ba, "after_final_ba": after_final_ba,
               "final_pruning": final_quality, "bundle_adjustment": bundle_history, "incremental_steps": state.history,
               "point_refinement": point_refinement}
    if final_retry:
        metrics["final_retry_registered"] = late_registered
    if recovery:
        metrics["map_recovery"] = {"pair_attempts": recovery.pair_attempts, "verified_pairs": recovery.verified_pairs,
                                   "rounds": recovery.rounds, "stop_reason": recovery.stop_reason or "all_registered",
                                   "runtime_seconds": stages["map_recovery"]}
    phase_start = perf_counter()
    write_json(output / "poses.json", [{"image_id": i, "name": v.name, "status": "registered" if i in state.poses else "rejected",
                "world_to_camera": state.poses[i].matrix.tolist() if i in state.poses else None,
                "center_world": state.poses[i].center.tolist() if i in state.poses else None,
                "timestamp": v.metadata["timestamp"], "attempts": attempts[i]} for i, v in views.items()])
    write_json(output / "tracks.json", [{"id": t.id, "observations": t.observations,
                "landmark_id": t.id if t.id in state.landmarks else None} for t in sorted(state.tracks.values(), key=lambda t: t.id)])
    landmarks = [state.landmarks[i] for i in sorted(state.landmarks)]
    measurements = measure_landmarks(state)
    write_json(output / "landmarks.json", [{"id": l.id, "track_id": l.id, "xyz": l.xyz.tolist(), "color": l.color.tolist(),
                "observations": l.observations, **landmark_diagnostics(state, l, measurements)} for l in landmarks])
    np.savez_compressed(output / "features.npz", **{f"image_{i}_points": v.features.points for i, v in views.items()})
    if landmarks:
        export_cloud(output / "sparse.ply", [l.xyz for l in landmarks], [l.color for l in landmarks])
        save_diagnostic(output / "overview.png", state)
    metrics["per_image"] = [{"image_id": i, "name": v.name, "features": v.features.count,
                "feature_seconds": v.features.seconds, "status": "registered" if i in state.poses else "rejected",
                "visible_landmarks": sum(i in l.observations for l in landmarks)} for i, v in views.items()]
    stages["export"] = perf_counter() - phase_start
    metrics.update(runtime_seconds=perf_counter() - start, peak_memory_bytes=None, stage_seconds=stages,
                   feature_extraction_seconds=feature_extraction_seconds, feature_cache_hits=cache_hits,
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
