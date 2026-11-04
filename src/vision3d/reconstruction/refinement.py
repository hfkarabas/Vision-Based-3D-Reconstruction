"""Optional fixed-pose point refinement on unchanged, verified observations.

This does not adjust camera poses, intrinsics, associations, colors or scale.
"""
from time import perf_counter
import cv2
import numpy as np
from scipy.optimize import least_squares

from vision3d.geometry.two_view import project
from .incremental import max_ray_angle


def refine_landmarks(state, max_error_px=2.0, min_angle_degrees=1.0,
                     max_distance=1000.0, max_evaluations=15):
    """Refine >=3-view points, accepting only finite, valid, bounded improvements.

    Pose gauge is unchanged because no camera variable is optimized. Each point
    is independent; a failed solve leaves that point unchanged. Updates are
    staged until the full pass completes. RMS statistics use identical observations.
    """
    if (not np.isfinite([max_error_px, min_angle_degrees, max_distance]).all()
            or min(max_error_px, min_angle_degrees, max_distance) <= 0
            or not isinstance(max_evaluations, int) or not 1 <= max_evaluations <= 50):
        raise ValueError("Invalid fixed-pose refinement limits")
    start = perf_counter()
    updates, reasons = {}, {}
    before_errors, after_errors, shifts = [], [], []
    def reject(reason):
        reasons[reason] = reasons.get(reason, 0) + 1
    for lid, landmark in sorted(state.landmarks.items()):
        if len(landmark.observations) < 3:
            reject("two_view_unchanged")
            continue
        observations = [(state.views[i], state.poses[i], f) for i, f in sorted(landmark.observations.items())]
        def residual(xyz):
            return np.concatenate([project(xyz.reshape(1, 3), v.camera, p.rotation, p.translation)[0]
                                   - v.features.points[f] for v, p, f in observations])
        try:
            original = np.asarray(landmark.xyz, dtype=float)
            before = residual(original)
            if not np.isfinite(before).all() or not np.isfinite(original).all():
                reject("nonfinite_initial")
                continue
            # Statistics include all eligible finite points, including rolled-back proposals.
            after = before
            fit = least_squares(residual, original.copy(), method="trf", loss="soft_l1", f_scale=1.0,
                                max_nfev=max_evaluations, ftol=1e-7, xtol=1e-7, gtol=1e-7)
            candidate = fit.x
            if fit.status < 0 or not np.isfinite(candidate).all() or not np.isfinite(fit.fun).all():
                reject("nonfinite_or_failed_solver")
            else:
                errors = fit.fun.reshape(-1, 2)
                original_errors = before.reshape(-1, 2)
                depths = [(p.rotation @ candidate + p.translation)[2] for _, p, _ in observations]
                distance = np.median([np.linalg.norm(original - p.center) for _, p, _ in observations])
                shift = float(np.linalg.norm(candidate - original))
                old_cost = np.sum(np.sqrt(1 + before ** 2) - 1)
                new_cost = np.sum(np.sqrt(1 + fit.fun ** 2) - 1)
                if (np.sum(errors ** 2) <= np.sum(original_errors ** 2) - 1e-12
                        and new_cost <= old_cost + 1e-12
                        and np.max(np.linalg.norm(errors, axis=1)) <= max_error_px
                        and min(depths) > 1e-9 and np.linalg.norm(candidate) < max_distance
                        and shift <= 0.05 * distance
                        and max_ray_angle(state, candidate, landmark.observations) >= min_angle_degrees):
                    updates[lid] = candidate.copy()
                    after = fit.fun
                    shifts.append(shift)
                else:
                    reject("error_depth_angle_or_movement_gate")
            before_errors.extend(np.linalg.norm(before.reshape(-1, 2), axis=1))
            after_errors.extend(np.linalg.norm(after.reshape(-1, 2), axis=1))
        except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError):
            reject("solver_exception")
    for lid, xyz in updates.items():
        state.landmarks[lid].xyz = xyz
    return {"status": "completed", "scope": "fixed_camera_multiview_points", "accepted_points": len(updates),
            "unchanged_reasons": reasons, "max_evaluations_per_point": max_evaluations,
            "eligible_observations": len(before_errors),
            "same_observation_rms_px": {"before": float(np.sqrt(np.mean(np.square(before_errors)))) if before_errors else None,
                                        "after": float(np.sqrt(np.mean(np.square(after_errors)))) if after_errors else None},
            "max_point_shift_world_units": max(shifts, default=0.0), "seconds": perf_counter() - start}
