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

    def __post_init__(self):
        if (not isinstance(self.min_inliers, int) or self.min_inliers < 6
                or not isinstance(self.iterations, int) or self.iterations < 1
                or not np.isfinite([self.min_inlier_ratio, self.max_error_px]).all()
                or not 0 < self.min_inlier_ratio <= 1 or self.max_error_px <= 0):
            raise ValueError("Invalid PnP quality thresholds")


def select_initial(views, pairs, quality=Quality()):
    """Rank only geometrically accepted pairs, using support, angle and cheirality."""
    candidates, diagnostics = [], []
    for (i, j), correspondences in sorted(pairs.items()):
        a, b = views[i], views[j]
        p1 = a.features.points[correspondences[:, 0]]
        p2 = b.features.points[correspondences[:, 1]]
        try:
            result = reconstruct(p1, p2, a.camera, b.camera, quality)
        except ReconstructionRejected as exc:
            diagnostics.append({"pair": [i, j], "status": "rejected", "reason": str(exc), **exc.metrics})
            continue
        m = result["metrics"]
        score = m["valid_points"] * min(m["median_triangulation_angle_degrees"] / 5, 2) * m["cheirality_ratio"] * m["valid_point_ratio"]
        diagnostics.append({"pair": [i, j], "status": "accepted", "score": score, **m})
        candidates.append((score, i, j, result, correspondences))
    if not candidates:
        raise ReconstructionRejected("no_geometrically_valid_initial_pair", {"initial_candidates": diagnostics})
    return max(candidates, key=lambda c: (c[0], -c[1], -c[2])), diagnostics


def observation_quality(state, xyz, image_id, feature_id, max_error):
    pose, view = state.poses[image_id], state.views[image_id]
    depth = (pose.rotation @ xyz + pose.translation)[2]
    error = float(np.linalg.norm(project(np.asarray(xyz).reshape(1, 3), view.camera, pose.rotation, pose.translation)[0]
                                 - view.features.points[feature_id]))
    return bool(np.isfinite(error) and depth > 1e-9 and error <= max_error), error


def validate_landmarks(state, max_error=2.0, prune=False, min_angle_degrees=1.0, max_distance=1000.0):
    errors, invalid, lost, angles = [], 0, [], []
    low_angle = extreme = 0
    for lid, landmark in state.landmarks.items():
        bad = []
        for image_id, feature_id in landmark.observations.items():
            valid, error = observation_quality(state, landmark.xyz, image_id, feature_id, max_error)
            errors.append(error)
            if not valid:
                invalid += 1
                bad.append(image_id)
        directions = []
        for image_id in landmark.observations.keys() - set(bad):
            ray = landmark.xyz - state.poses[image_id].center
            directions.append(ray / max(np.linalg.norm(ray), 1e-12))
        angle = max((float(np.rad2deg(np.arccos(np.clip(a @ b, -1, 1)))) for a, b in combinations(directions, 2)), default=0.0)
        angles.append(angle)
        angle_bad = angle < min_angle_degrees
        bounds_bad = not np.isfinite(landmark.xyz).all() or np.linalg.norm(landmark.xyz) >= max_distance
        low_angle += int(angle_bad)
        extreme += int(bounds_bad)
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
            "low_parallax_landmarks": low_angle, "extreme_landmarks": extreme, "retired_landmarks": len(lost)}


def landmark_diagnostics(state, landmark):
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
    for tid, landmark in sorted(state.landmarks.items()):
        feature = state.tracks[tid].observations.get(image_id)
        if feature is not None:
            ids.append(tid)
            xyz.append(landmark.xyz)
            pixels.append(state.views[image_id].features.points[feature])
    return ids, np.asarray(xyz, np.float64).reshape(-1, 3), np.asarray(pixels, np.float64).reshape(-1, 2)


def estimate_pnp(xyz, pixels, camera, quality=RegistrationQuality()):
    xyz, pixels = np.asarray(xyz, np.float64), np.asarray(pixels, np.float64)
    metrics = {"correspondences": len(xyz)}
    if xyz.shape != (len(xyz), 3) or pixels.shape != (len(xyz), 2) or not np.isfinite(xyz).all() or not np.isfinite(pixels).all():
        raise ValueError("PnP requires finite paired Nx3 and Nx2 arrays")
    if len(xyz) < quality.min_inliers or np.linalg.matrix_rank(xyz - xyz.mean(axis=0)) < 2:
        return None, np.zeros(len(xyz), bool), {**metrics, "reason": "insufficient_or_collinear_2d3d"}
    try:
        ok, rvec, tvec, indices = cv2.solvePnPRansac(xyz, pixels, camera.K, camera.D,
            iterationsCount=quality.iterations, reprojectionError=quality.max_error_px,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
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
        return Pose(rotation, translation), mask, {**metrics, "reason": "accepted"}
    except cv2.error:
        return None, np.zeros(len(xyz), bool), {**metrics, "reason": "pnp_solver_failed"}


def register(state, image_id, quality=RegistrationQuality(), geometry=Quality()):
    if image_id in state.poses:
        return False, {"reason": "already_registered"}
    ids, xyz, pixels = pnp_correspondences(state, image_id)
    pose, mask, metrics = estimate_pnp(xyz, pixels, state.views[image_id].camera, quality)
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


def extend_landmarks(state, quality=Quality()):
    """One landmark per track; triangulate the widest ray pair and validate all views."""
    added = attached = 0
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
        i, j = ranked[0]
        angle = np.rad2deg(np.arccos(np.clip(rays[i] @ rays[j], -1, 1)))
        if angle < quality.min_angle_degrees:
            continue
        a, b = state.poses[i], state.poses[j]
        rotation = b.rotation @ a.rotation.T
        translation = b.translation - rotation @ a.translation
        if np.linalg.norm(translation) < 1e-9:
            continue
        result = triangulate(state.views[i].features.points[[observations[i]]], state.views[j].features.points[[observations[j]]],
                             state.views[i].camera, state.views[j].camera, rotation, translation, quality)
        if not result["valid"][0]:
            continue
        xyz = a.rotation.T @ (result["xyz"][0] - a.translation)
        if not np.isfinite(xyz).all() or np.linalg.norm(xyz) >= quality.max_distance:
            continue
        accepted = {k: f for k, f in observations.items() if observation_quality(state, xyz, k, f, quality.max_reprojection_px)[0]}
        if i not in accepted or j not in accepted or len(accepted) < max(2, len(observations) / 2):
            continue
        state.landmarks[tid] = Landmark(tid, xyz, state.views[i].colors[observations[i]], accepted)
        added += 1
    return {"new_landmarks": added, "new_observations": attached}
