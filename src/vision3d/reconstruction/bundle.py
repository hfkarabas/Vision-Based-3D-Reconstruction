"""Bounded transactional local BA with fixed initial-pair gauge anchors."""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from vision3d.geometry.two_view import project
from .incremental import validate_landmarks
from .state import Landmark, Pose, State


@dataclass(frozen=True)
class BundleOptions:
    max_cameras: int = 5  # includes both fixed initial cameras
    max_points: int = 400
    max_evaluations: int = 30
    loss_scale_px: float = 1.0
    max_camera_shift: float = 1.0

    def __post_init__(self):
        if (any(not isinstance(v, int) or isinstance(v, bool) for v in (self.max_cameras, self.max_points, self.max_evaluations))
                or not 3 <= self.max_cameras <= 10 or not 20 <= self.max_points <= 2000
                or not 1 <= self.max_evaluations <= 100 or self.loss_scale_px <= 0
                or self.max_camera_shift <= 0 or not np.isfinite([self.loss_scale_px, self.max_camera_shift]).all()):
            raise ValueError("BA limits: 3..10 cameras, 20..2000 points, 1..100 evaluations, positive finite scales")


def local_bundle_adjust(state, options=BundleOptions(), max_error_px=2.0, min_angle_degrees=1.0, max_distance=1000.0):
    fixed = list(state.anchors)
    recent = [i for i in state.registration_order if i not in fixed][-(options.max_cameras - len(fixed)):]
    if len(fixed) != 2 or not recent:
        return {"status": "skipped", "reason": "need_two_anchors_and_a_registered_camera"}
    selected = set(fixed + recent)
    landmarks = [l for l in state.landmarks.values() if any(i in l.observations for i in recent)
                 and all(i in l.observations for i in fixed)]
    landmarks = sorted(landmarks, key=lambda l: (-len(selected & l.observations.keys()), l.id))[:options.max_points]
    if len(landmarks) < 20:
        return {"status": "skipped", "reason": "too_few_local_landmarks"}
    camera_index = {image: k for k, image in enumerate(recent)}
    camera_size = 6 * len(recent)
    initial = np.r_[np.concatenate([np.r_[cv2.Rodrigues(state.poses[i].rotation)[0].ravel(), state.poses[i].translation] for i in recent]),
                    np.asarray([l.xyz for l in landmarks]).ravel()]
    groups, row_count = [], 0
    for image in sorted(selected):
        point_ids = np.array([k for k, l in enumerate(landmarks) if image in l.observations], dtype=int)
        pixels = np.array([state.views[image].features.points[landmarks[k].observations[image]] for k in point_ids]).reshape(-1, 2)
        groups.append((image, point_ids, pixels, row_count))
        row_count += 2 * len(point_ids)
    sparsity = lil_matrix((row_count, len(initial)), dtype=int)
    for image, indices, _, offset in groups:
        for k, point in enumerate(indices):
            rows = slice(offset + 2 * k, offset + 2 * k + 2)
            sparsity[rows, camera_size + 3 * point:camera_size + 3 * point + 3] = 1
            if image in camera_index:
                c = 6 * camera_index[image]
                sparsity[rows, c:c + 6] = 1
    def decode(parameters):
        poses = dict(state.poses)
        for image, index in camera_index.items():
            c = 6 * index
            poses[image] = Pose(cv2.Rodrigues(parameters[c:c + 3])[0], parameters[c + 3:c + 6])
        return poses, parameters[camera_size:].reshape(-1, 3)
    def residual(parameters):
        poses, points = decode(parameters)
        result = np.empty(row_count)
        for image, indices, pixels, offset in groups:
            if len(indices):
                pose = poses[image]
                result[offset:offset + 2 * len(indices)] = (project(points[indices], state.views[image].camera,
                    pose.rotation, pose.translation) - pixels).ravel()
        return result
    before = validate_landmarks(state, max_error_px, min_angle_degrees=min_angle_degrees, max_distance=max_distance)
    initial_residual = residual(initial)
    robust_before = float(np.sum(options.loss_scale_px ** 2 * (np.sqrt(1 + (initial_residual / options.loss_scale_px) ** 2) - 1)))
    info = {"cameras_optimized": recent, "fixed_cameras": fixed, "points_optimized": len(landmarks),
            "residuals": row_count, "before": before}
    try:
        fit = least_squares(residual, initial, jac_sparsity=sparsity.tocsr(), method="trf", loss="soft_l1",
                            f_scale=options.loss_scale_px, x_scale="jac", max_nfev=options.max_evaluations,
                            ftol=1e-6, xtol=1e-6, gtol=1e-6)
        if fit.status < 0 or not np.isfinite(fit.x).all() or not np.isfinite(fit.fun).all() or not np.isfinite(fit.cost):
            return {**info, "status": "rolled_back", "reason": "nonfinite_or_failed_solver"}
        poses, points = decode(fit.x)
        if np.linalg.norm(points, axis=1).max() >= max_distance:
            return {**info, "status": "rolled_back", "reason": "extreme_points"}
        candidate = State(state.views, state.tracks, poses=poses,
                          landmarks={i: Landmark(i, l.xyz.copy(), l.color, dict(l.observations)) for i, l in state.landmarks.items()})
        for l, xyz in zip(landmarks, points):
            candidate.landmarks[l.id].xyz = xyz.copy()
        after = validate_landmarks(candidate, max_error_px, min_angle_degrees=min_angle_degrees, max_distance=max_distance)
        shift = max(np.linalg.norm(poses[i].center - state.poses[i].center) for i in recent)
        acceptable = (after["rms_error_px"] <= before["rms_error_px"] and fit.cost <= robust_before + 1e-9 and
                      after["invalid_observations"] <= before["invalid_observations"] and
                      after["low_parallax_landmarks"] <= before["low_parallax_landmarks"] and
                      after["extreme_landmarks"] <= before["extreme_landmarks"] and shift <= options.max_camera_shift)
        info.update(after=after, evaluations=int(fit.nfev), converged=bool(fit.success),
                    termination=str(fit.message), max_camera_shift=float(shift),
                    robust_cost_before=robust_before, robust_cost_after=float(fit.cost))
        if not acceptable:
            return {**info, "status": "rolled_back", "reason": "global_error_depth_or_movement_gate"}
        # Nothing in live state has changed before every acceptance check passes.
        state.poses = poses
        state.landmarks = candidate.landmarks
        info["pruning"] = validate_landmarks(state, max_error_px, prune=True, min_angle_degrees=min_angle_degrees, max_distance=max_distance)
        return {**info, "status": "accepted"}
    except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as exc:
        return {**info, "status": "rolled_back", "reason": str(exc)}
