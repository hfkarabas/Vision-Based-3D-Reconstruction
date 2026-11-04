"""Bounded transactional local BA with fixed initial-pair gauge anchors."""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from vision3d.geometry.two_view import project
from .incremental import invalid_observations, validate_landmarks
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


def local_bundle_adjust(state, options=BundleOptions(), max_error_px=2.0, min_angle_degrees=1.0, max_distance=1000.0, joint=False):
    if joint and len(state.poses) > 30:
        return {'status': 'skipped', 'reason': 'joint_camera_limit_30'}
    recent = [i for i in state.registration_order if i not in state.anchors]
    if not joint:
        recent = recent[-(options.max_cameras - 2):]
    if len(state.anchors) != 2 or not recent:
        return {"status": "skipped", "reason": "need_two_anchors_and_a_registered_camera"}
    # Fixed boundary cameras retain the current world frame/scale, even after
    # the active window no longer overlaps the initial anchor pair.
    boundary = sorted(set(state.poses) - set(recent))
    choices = []
    from itertools import combinations
    for a, b in combinations(boundary, 2):
        support = sum(a in l.observations and b in l.observations and any(i in l.observations for i in recent)
                      for l in state.landmarks.values())
        if np.linalg.norm(state.poses[a].center - state.poses[b].center) > 1e-6:
            choices.append((support, -a, -b))
    if not choices:
        return {"status": "skipped", "reason": "no_noncoincident_boundary_pair"}
    _, a, b = max(choices)
    fixed = list(state.anchors) if joint else [-a, -b]
    selected = set(fixed + recent)
    landmarks = [l for l in state.landmarks.values() if any(i in l.observations for i in recent)
                 and (len(selected & l.observations.keys()) >= 2 if joint else all(i in l.observations for i in fixed))]
    landmarks = sorted(landmarks, key=lambda l: (-len(selected & l.observations.keys()), l.id))[:(2000 if joint else options.max_points)]
    if len(landmarks) < 20:
        return {"status": "skipped", "reason": "too_few_local_landmarks"}
    if any(sum(i in l.observations for l in landmarks) < 20 for i in recent):
        return {'status': 'skipped', 'reason': 'insufficient_selected_camera_support'}
    if joint:
        from .graph import components
        connections = set()
        for landmark in landmarks:
            images = sorted(selected & landmark.observations.keys())
            connections.update((images[0], i) for i in images[1:])
        if (any(sum(i in l.observations for l in landmarks) < 20 for i in fixed)
                or not any(selected <= set(group) for group in components(selected, connections))):
            return {'status': 'skipped', 'reason': 'joint_window_not_anchored'}
    camera_index = {image: k for k, image in enumerate(recent)}
    camera_size = 6 * len(recent)
    initial = np.r_[np.concatenate([np.r_[cv2.Rodrigues(state.poses[i].rotation)[0].ravel(), state.poses[i].translation] for i in recent]),
                    np.asarray([l.xyz for l in landmarks]).ravel()]
    if not np.isfinite(initial).all():
        return {"status": "rolled_back", "reason": "nonfinite_initial_state"}
    groups, row_count = [], 0
    for image in sorted(selected):
        point_ids = np.array([k for k, l in enumerate(landmarks) if image in l.observations], dtype=int)
        pixels = np.array([state.views[image].features.points[landmarks[k].observations[image]] for k in point_ids]).reshape(-1, 2)
        if not np.isfinite(pixels).all():
            return {"status": "rolled_back", "reason": "nonfinite_initial_observations"}
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
    if not np.isfinite(initial_residual).all():
        return {"status": "rolled_back", "reason": "nonfinite_initial_residual"}
    robust_before = float(np.sum(options.loss_scale_px ** 2 * (np.sqrt(1 + (initial_residual / options.loss_scale_px) ** 2) - 1)))
    info = {"cameras_optimized": recent, "fixed_cameras": fixed, "points_optimized": len(landmarks),
            "residuals": row_count, "before": before, 'scope': 'bounded_joint' if joint else 'local',
            'selected_landmark_ids': [l.id for l in landmarks]}
    invalid_before = invalid_observations(state, max_error_px)
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
        new_invalid = invalid_observations(candidate, max_error_px) - invalid_before
        shift = max(np.linalg.norm(poses[i].center - state.poses[i].center) for i in recent)
        acceptable = (after["rms_error_px"] <= before["rms_error_px"] and fit.cost <= robust_before + 1e-9 and
                      not new_invalid and
                      not (set(after["invalid_landmark_ids"]) - set(before["invalid_landmark_ids"])) and
                      shift <= options.max_camera_shift)
        info.update(after=after, evaluations=int(fit.nfev), converged=bool(fit.success),
                    termination=str(fit.message), max_camera_shift=float(shift),
                    robust_cost_before=robust_before, robust_cost_after=float(fit.cost))
        local_before, local_after = residual(initial).reshape(-1, 2), residual(fit.x).reshape(-1, 2)
        info["local_same_observation_rms_px"] = {"before": float(np.sqrt(np.mean(np.sum(local_before ** 2, axis=1)))),
                                               "after": float(np.sqrt(np.mean(np.sum(local_after ** 2, axis=1))))}
        info["new_invalid_observations"] = len(new_invalid)
        if not acceptable:
            return {**info, "status": "rolled_back", "reason": "global_error_depth_or_movement_gate"}
        # Nothing in live state has changed before every acceptance check passes.
        info["pruning"] = validate_landmarks(candidate, max_error_px, prune=True, min_angle_degrees=min_angle_degrees, max_distance=max_distance)
        state.poses = candidate.poses
        state.landmarks = candidate.landmarks
        return {**info, "status": "accepted"}
    except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as exc:
        return {**info, "status": "rolled_back", "reason": str(exc)}
