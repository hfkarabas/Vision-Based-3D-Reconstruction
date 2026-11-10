"""Track-based PnP registration and shared-frame landmark maintenance."""
from dataclasses import dataclass
from itertools import combinations
import cv2
import numpy as np

from vision3d.geometry.two_view import Quality, ReconstructionRejected, project, reconstruct, triangulate
from .state import Landmark, Pose


@dataclass(frozen=True)
class RegistrationQuality:
    min_inliers: int = 20
    min_inlier_ratio: float = 0.4
    max_error_px: float = 2.0
    iterations: int = 2000
    min_image_coverage: float = 0.01

    def __post_init__(self):
        if (not isinstance(self.min_inliers, int) or self.min_inliers < 6
                or not isinstance(self.iterations, int) or self.iterations < 1
                or not np.isfinite([self.min_inlier_ratio, self.max_error_px, self.min_image_coverage]).all()
                or not 0 < self.min_inlier_ratio <= 1 or self.max_error_px <= 0
                or not 0 <= self.min_image_coverage <= 1):
            raise ValueError("Invalid PnP quality thresholds")


def select_initial(views, pairs, quality=Quality(), tracks=None, seed=0, continuation_min=None, probe_quality=None):
    """Rank only geometrically accepted pairs, using support, angle and cheirality."""
    from collections import Counter

    candidates, diagnostics = [], []
    lookup = {(i, f): tid for tid, t in (tracks or {}).items() for i, f in t.observations.items()}
    for (i, j), correspondences in sorted(pairs.items()):
        a, b = views[i], views[j]
        p1 = a.features.points[correspondences[:, 0]]
        p2 = b.features.points[correspondences[:, 1]]
        try:
            cv2.setRNGSeed((seed + i * 1009 + j * 9176) % 2147483647)
            result = reconstruct(p1, p2, a.camera, b.camera, quality)
        except ReconstructionRejected as exc:
            diagnostics.append({"pair": [i, j], "status": "rejected", "reason": str(exc), **exc.metrics})
            continue
        m = result["metrics"]
        compatible = m["valid_points"]
        if tracks is not None:
            compatible = sum(lookup.get((i, int(correspondences[k, 0]))) is not None and
                lookup.get((i, int(correspondences[k, 0]))) == lookup.get((j, int(correspondences[k, 1]))) for k in result["point_indices"])
        if compatible < quality.min_points:
            diagnostics.append({"pair": [i, j], "status": "rejected", "reason": "too_few_conflict_free_tracks", "compatible_tracks": compatible, **m})
            continue
        score = compatible * min(m["median_triangulation_angle_degrees"] / 5, 2) * m["cheirality_ratio"] * m["valid_point_ratio"]
        diagnostics.append({"pair": [i, j], "status": "accepted", "score": score, "compatible_tracks": compatible, **m})
        candidates.append((score, i, j, result, correspondences))
        if continuation_min is not None and tracks is not None:
            support = Counter()
            for k in result["point_indices"]:
                fi, fj = map(int, correspondences[k])
                tid = lookup.get((i, fi))
                if tid is not None and tid == lookup.get((j, fj)):
                    support.update(set(tracks[tid].observations) - {i, j})
            diagnostics[-1]["best_third_view_support"] = max(support.values(), default=0)
    if not candidates:
        raise ReconstructionRejected("no_geometrically_valid_initial_pair", {"initial_candidates": diagnostics})
    viable = {(d["pair"][0], d["pair"][1]) for d in diagnostics
              if continuation_min is not None and d.get("best_third_view_support", 0) >= continuation_min}
    ranked = sorted(candidates, key=lambda c: ((c[1], c[2]) in viable, c[0], -c[1], -c[2]), reverse=True)
    if probe_quality is not None and tracks is not None:
        from .state import State

        records = {tuple(d["pair"]): d for d in diagnostics if d["status"] == "accepted"}
        for candidate in ranked[:12]:
            _, i, j, _, _ = candidate
            if (i, j) not in viable:
                continue
            trial = State(views, tracks)
            initialize(trial, candidate, quality)
            others = sorted((k for k in views if k not in trial.poses),
                            key=lambda k: registration_priority(trial, k), reverse=True)
            attempts = records[i, j]["continuation_probes"] = []
            for k in others[:5]:
                _, xyz, _ = pnp_correspondences(trial, k)
                if len(xyz) < probe_quality.min_inliers:
                    continue
                cv2.setRNGSeed((seed + k * 1009 + 2 * 9176) % 2147483647)
                accepted, metrics = register(trial, k, probe_quality, quality)
                attempts.append({"image_id": k, **metrics})
                if accepted:
                    return candidate, diagnostics
    return ranked[0], diagnostics


def observation_quality(state, xyz, image_id, feature_id, max_error):
    pose, view = state.poses[image_id], state.views[image_id]
    depth = (pose.rotation @ xyz + pose.translation)[2]
    error = float(np.linalg.norm(project(np.asarray(xyz).reshape(1, 3), view.camera, pose.rotation, pose.translation)[0]
                                 - view.features.points[feature_id]))
    return bool(np.isfinite(error) and depth > 1e-9 and error <= max_error), error


def measure_landmarks(state):
    """Vectorize projection per camera rather than making one OpenCV call per observation."""
    result = {lid: {"reprojection_errors_px": {}, "depths": {}} for lid in state.landmarks}
    groups = {}
    for lid, landmark in state.landmarks.items():
        for image, feature in landmark.observations.items():
            groups.setdefault(image, []).append((lid, feature))
    for image, observations in groups.items():
        xyz = np.array([state.landmarks[lid].xyz for lid, _ in observations])
        pose, view = state.poses[image], state.views[image]
        pixels = view.features.points[[f for _, f in observations]]
        errors = np.linalg.norm(project(xyz, view.camera, pose.rotation, pose.translation) - pixels, axis=1)
        depths = (xyz @ pose.rotation.T + pose.translation)[:, 2]
        for (lid, _), error, depth in zip(observations, errors, depths):
            result[lid]["reprojection_errors_px"][image] = float(error)
            result[lid]["depths"][image] = float(depth)
    return result


def max_ray_angle(state, xyz, images):
    if len(images) < 2:
        return 0.0
    rays = np.array([xyz - state.poses[i].center for i in sorted(images)])
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    return float(np.rad2deg(np.arccos(np.clip(np.min(rays @ rays.T), -1, 1))))


def invalid_observations(state, max_error):
    measurements = measure_landmarks(state)
    return {(lid, i, state.landmarks[lid].observations[i]) for lid, m in measurements.items()
            for i, error in m["reprojection_errors_px"].items()
            if not np.isfinite(error) or m["depths"][i] <= 1e-9 or error > max_error}


def validate_landmarks(state, max_error=2.0, prune=False, min_angle_degrees=1.0, max_distance=1000.0):
    errors, invalid, lost, angles = [], 0, [], []
    low_angle = extreme = 0
    invalid_geometry_ids = []
    measurements = measure_landmarks(state)
    for lid, landmark in state.landmarks.items():
        bad = []
        for image_id, feature_id in landmark.observations.items():
            error = measurements[lid]["reprojection_errors_px"][image_id]
            valid = np.isfinite(error) and measurements[lid]["depths"][image_id] > 1e-9 and error <= max_error
            errors.append(error)
            if not valid:
                invalid += 1
                bad.append(image_id)
        angle = max_ray_angle(state, landmark.xyz, landmark.observations.keys() - set(bad))
        angles.append(angle)
        angle_bad = angle < min_angle_degrees
        bounds_bad = not np.isfinite(landmark.xyz).all() or np.linalg.norm(landmark.xyz) >= max_distance
        low_angle += int(angle_bad)
        extreme += int(bounds_bad)
        if angle_bad or bounds_bad:
            invalid_geometry_ids.append(lid)
        if prune:
            for image_id in bad:
                del landmark.observations[image_id]
            if len(landmark.observations) < 2 or angle_bad or bounds_bad:
                lost.append(lid)
    if prune:
        for lid in lost:
            del state.landmarks[lid]
    return {"observations": len(errors), "invalid_observations": invalid,
            "median_error_px": float(np.median(errors)) if errors else None,
            "rms_error_px": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            "median_triangulation_angle_degrees": float(np.median(angles)) if angles else None,
            "low_parallax_landmarks": low_angle, "extreme_landmarks": extreme, "retired_landmarks": len(lost),
            "invalid_landmark_ids": invalid_geometry_ids}


def landmark_diagnostics(state, landmark, measurements=None):
    if measurements is not None:
        return {**measurements[landmark.id], "max_triangulation_angle_degrees": max_ray_angle(state, landmark.xyz, landmark.observations)}
    errors, depths, rays = {}, {}, []
    for i, f in landmark.observations.items():
        pose = state.poses[i]
        _, error = observation_quality(state, landmark.xyz, i, f, np.inf)
        errors[i] = error
        depths[i] = float((pose.rotation @ landmark.xyz + pose.translation)[2])
        direction = landmark.xyz - pose.center
        rays.append(direction / max(np.linalg.norm(direction), 1e-12))
    angle = max((float(np.rad2deg(np.arccos(np.clip(a @ b, -1, 1)))) for a, b in combinations(rays, 2)), default=0.0)
    return {"reprojection_errors_px": errors, "depths": depths, "max_triangulation_angle_degrees": angle}


def initialize(state, initial, quality=Quality()):
    _, i, j, result, correspondences = initial
    state.anchors = (i, j)
    state.poses[i] = Pose(np.eye(3), np.zeros(3))
    state.poses[j] = Pose(result["rotation"], result["translation"])
    state.registration_order = [i, j]
    lookup = {(image, feature): tid for tid, track in state.tracks.items() for image, feature in track.observations.items()}
    for index, xyz in zip(result["point_indices"], result["xyz"][result["valid"]]):
        fi, fj = map(int, correspondences[index])
        tid = lookup.get((i, fi))
        if tid is not None and tid == lookup.get((j, fj)) and tid not in state.landmarks:
            state.landmarks[tid] = Landmark(tid, xyz.copy(), state.views[i].colors[fi], {i: fi, j: fj})
    if len(state.landmarks) < quality.min_points:
        raise ReconstructionRejected("initial_pair_has_too_few_conflict_free_tracks")


def pnp_correspondences(state, image_id):
    ids, xyz, pixels = [], [], []
    from collections import Counter
    feature_counts = Counter(state.tracks[tid].observations.get(image_id) for tid in state.landmarks)
    for tid, landmark in sorted(state.landmarks.items()):
        feature = state.tracks[tid].observations.get(image_id)
        if feature is not None and feature_counts[feature] == 1 and np.isfinite(landmark.xyz).all():
            ids.append(tid)
            xyz.append(landmark.xyz)
            pixels.append(state.views[image_id].features.points[feature])
    return ids, np.asarray(xyz, np.float64).reshape(-1, 3), np.asarray(pixels, np.float64).reshape(-1, 2)


def estimate_pnp(xyz, pixels, camera, quality=RegistrationQuality()):
    xyz, pixels = np.asarray(xyz, np.float64), np.asarray(pixels, np.float64)
    metrics = {"correspondences": len(xyz), "pnp_attempted": False}
    if xyz.shape != (len(xyz), 3) or pixels.shape != (len(xyz), 2) or not np.isfinite(xyz).all() or not np.isfinite(pixels).all():
        raise ValueError("PnP requires finite paired Nx3 and Nx2 arrays")
    if len(pixels) >= 3:
        metrics["candidate_image_coverage"] = float(cv2.contourArea(cv2.convexHull(pixels.astype(np.float32)))) / (camera.width * camera.height)
    if len(xyz) < quality.min_inliers or np.linalg.matrix_rank(xyz - xyz.mean(axis=0)) < 2:
        return None, np.zeros(len(xyz), bool), {**metrics, "reason": "insufficient_or_collinear_2d3d"}
    try:
        metrics["pnp_attempted"] = True
        ok, rvec, tvec, indices = cv2.solvePnPRansac(xyz, pixels, camera.K, camera.D,
            iterationsCount=quality.iterations, reprojectionError=quality.max_error_px,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        metrics["ransac_inliers"] = 0 if indices is None else len(indices)
        if not ok or indices is None or len(indices) < quality.min_inliers:
            return None, np.zeros(len(xyz), bool), {**metrics, "reason": "pnp_ransac_failed"}
        rvec, tvec = cv2.solvePnPRefineLM(xyz[indices.ravel()], pixels[indices.ravel()], camera.K, camera.D, rvec, tvec)
        rotation = cv2.Rodrigues(rvec)[0]
        translation = tvec.ravel()
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            return None, np.zeros(len(xyz), bool), {**metrics, "reason": "nonfinite_refined_pose"}
        errors = np.linalg.norm(project(xyz, camera, rotation, translation) - pixels, axis=1)
        depths = (xyz @ rotation.T + translation)[:, 2]
        mask = (errors <= quality.max_error_px) & (depths > 1e-9)
        metrics.update(inliers=int(mask.sum()), inlier_ratio=float(mask.mean()),
                       median_error_px=float(np.median(errors[mask])) if mask.any() else None)
        if mask.sum() < quality.min_inliers or mask.mean() < quality.min_inlier_ratio:
            return None, mask, {**metrics, "reason": "pnp_quality_rejected"}
        coverage = float(cv2.contourArea(cv2.convexHull(pixels[mask].astype(np.float32)))) / (camera.width * camera.height)
        metrics["image_coverage"] = coverage
        if coverage < quality.min_image_coverage:
            return None, mask, {**metrics, "reason": "pnp_spatial_support_too_small"}
        return Pose(rotation, translation), mask, {**metrics, "reason": "accepted"}
    except cv2.error:
        return None, np.zeros(len(xyz), bool), {**metrics, "reason": "pnp_solver_failed"}


def register(state, image_id, quality=RegistrationQuality(), geometry=Quality()):
    if image_id in state.poses:
        return False, {"reason": "already_registered"}
    ids, xyz, pixels = pnp_correspondences(state, image_id)
    pose, mask, metrics = estimate_pnp(xyz, pixels, state.views[image_id].camera, quality)
    metrics["invalid_or_ambiguous_associations_discarded"] = sum(image_id in state.tracks[tid].observations for tid in state.landmarks) - len(ids)
    if pose is None:
        return False, metrics
    if min(np.linalg.norm(pose.center - p.center) for p in state.poses.values()) >= geometry.max_distance:
        return False, {**metrics, "reason": "implausible_camera_distance"}
    state.poses[image_id] = pose
    state.registration_order.append(image_id)
    for lid, accepted in zip(ids, mask):
        if accepted:
            state.landmarks[lid].observations[image_id] = state.tracks[lid].observations[image_id]
    return True, metrics


def registration_priority(state, image_id):
    _, xyz, pixels = pnp_correspondences(state, image_id)
    if not len(xyz):
        return (0, 0, -image_id)
    view = state.views[image_id]
    cells = np.floor(pixels / [view.camera.width, view.camera.height] * 4).astype(int)
    coverage = len(np.unique(np.clip(cells, 0, 3), axis=0))
    return (len(xyz) * coverage, len(xyz), -image_id)


def extend_landmarks(state, quality=Quality()):
    """One landmark per track; triangulate the widest ray pair and validate all views."""
    added = attached = 0
    rejected, examples = {}, []
    fallback_points = 0
    for tid, track in sorted(state.tracks.items()):
        observations = {i: f for i, f in track.observations.items() if i in state.poses}
        if tid in state.landmarks:
            landmark = state.landmarks[tid]
            for i, f in observations.items():
                if i not in landmark.observations and observation_quality(state, landmark.xyz, i, f, quality.max_reprojection_px)[0]:
                    landmark.observations[i] = f
                    attached += 1
            continue
        if len(observations) < 2:
            continue
        rays = {}
        for i, f in observations.items():
            ray = np.r_[state.views[i].camera.normalized(state.views[i].features.points[[f]])[0], 1]
            ray = state.poses[i].rotation.T @ ray
            rays[i] = ray / np.linalg.norm(ray)
        ranked = sorted(combinations(observations, 2), key=lambda ij: (float(rays[ij[0]] @ rays[ij[1]]), ij))
        reason = "no_valid_triangulation"
        for attempt, (i, j) in enumerate(ranked[:3]):
            angle = np.rad2deg(np.arccos(np.clip(rays[i] @ rays[j], -1, 1)))
            if angle < quality.min_angle_degrees:
                reason = "insufficient_angle"
                break
            a, b = state.poses[i], state.poses[j]
            rotation = b.rotation @ a.rotation.T
            translation = b.translation - rotation @ a.translation
            if np.linalg.norm(translation) < 1e-9:
                reason = "zero_baseline"
                continue
            result = triangulate(state.views[i].features.points[[observations[i]]], state.views[j].features.points[[observations[j]]],
                                 state.views[i].camera, state.views[j].camera, rotation, translation, quality)
            if not result["valid"][0]:
                reason = "pair_depth_reprojection_or_angle"
                continue
            xyz = a.rotation.T @ (result["xyz"][0] - a.translation)
            if not np.isfinite(xyz).all() or np.linalg.norm(xyz) >= quality.max_distance:
                reason = "nonfinite_or_extreme"
                continue
            accepted = {k: f for k, f in observations.items() if observation_quality(state, xyz, k, f, quality.max_reprojection_px)[0]}
            if i not in accepted or j not in accepted or len(accepted) < max(2, len(observations) / 2):
                reason = "inconsistent_multiview_observations"
                continue
            state.landmarks[tid] = Landmark(tid, xyz, state.views[i].colors[observations[i]], accepted)
            added += 1
            fallback_points += int(attempt > 0)
            break
        if tid not in state.landmarks:
            rejected[reason] = rejected.get(reason, 0) + 1
            if len(examples) < 100:
                examples.append({"track_id": tid, "reason": reason})
    return {"new_landmarks": added, "new_observations": attached, "fallback_triangulations": fallback_points,
            "triangulation_rejections": rejected, "rejection_examples": examples}
