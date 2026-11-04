"""Calibrated two-view geometry with measurable rejection gates."""
from dataclasses import asdict, dataclass
import cv2
import numpy as np


class ReconstructionRejected(ValueError):
    def __init__(self, reason, metrics=None):
        super().__init__(reason)
        self.metrics = metrics or {}


@dataclass(frozen=True)
class Quality:
    min_correspondences: int = 30
    min_inliers: int = 20
    min_points: int = 20
    essential_threshold_px: float = 1.0
    max_reprojection_px: float = 2.0
    min_angle_degrees: float = 1.0
    min_cheirality_ratio: float = 0.6
    min_valid_ratio: float = 0.5
    homography_dominance: float = 0.9
    max_distance: float = 1000.0
    confidence: float = 0.999

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Quality thresholds must be finite")
        if any(not isinstance(v, int) or v < 8 for v in (self.min_correspondences, self.min_inliers, self.min_points)):
            raise ValueError("Minimum geometry counts must be integers >= 8")
        if min(self.essential_threshold_px, self.max_reprojection_px, self.max_distance) <= 0:
            raise ValueError("Distance thresholds must be positive")
        if not 0 < self.min_angle_degrees < 90 or not 0 < self.confidence < 1:
            raise ValueError("Invalid angle or confidence")
        if any(not 0 < v <= 1 for v in (self.min_cheirality_ratio, self.homography_dominance, self.min_valid_ratio)):
            raise ValueError("Support ratios must lie in (0,1]")


def project(points, camera, rotation=None, translation=None):
    rotation = np.eye(3) if rotation is None else rotation
    translation = np.zeros(3) if translation is None else translation
    return cv2.projectPoints(np.asarray(points, np.float64), cv2.Rodrigues(rotation)[0],
                             np.asarray(translation, np.float64), camera.K, camera.D)[0].reshape(-1, 2)


def triangulate(p1, p2, camera1, camera2, rotation, translation, quality=Quality()):
    """Return one point and independent filter diagnostics per input observation."""
    rotation = np.asarray(rotation, np.float64)
    translation = np.asarray(translation, np.float64).reshape(-1)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
            or np.linalg.norm(translation) < 1e-12):
        raise ValueError("Expected a valid rotation and nonzero translation")
    n1, n2 = camera1.normalized(p1), camera2.normalized(p2)
    if len(n1) != len(n2) or not len(n1):
        raise ValueError("Triangulation requires nonempty paired observations")
    homogeneous = cv2.triangulatePoints(np.c_[np.eye(3), np.zeros(3)],
                                        np.c_[rotation, translation], n1.T, n2.T).T
    usable_w = np.abs(homogeneous[:, 3]) > 1e-12
    xyz = np.full((len(n1), 3), np.nan)
    xyz[usable_w] = homogeneous[usable_w, :3] / homogeneous[usable_w, 3:]
    second = xyz @ rotation.T + translation
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(second).all(axis=1)
    positive = finite & (xyz[:, 2] > 1e-9) & (second[:, 2] > 1e-9)
    bounded = finite & (np.linalg.norm(xyz, axis=1) < quality.max_distance) & (np.linalg.norm(second, axis=1) < quality.max_distance)
    errors = np.full((len(n1), 2), np.inf)
    if finite.any():
        errors[finite, 0] = np.linalg.norm(project(xyz[finite], camera1) - np.asarray(p1)[finite], axis=1)
        errors[finite, 1] = np.linalg.norm(project(xyz[finite], camera2, rotation, translation) - np.asarray(p2)[finite], axis=1)
    # Rays are expressed in camera 1/world coordinates before taking their angle.
    rays1 = np.c_[n1, np.ones(len(n1))]
    rays2 = np.c_[n2, np.ones(len(n2))] @ rotation
    cosine = np.sum(rays1 * rays2, axis=1) / (np.linalg.norm(rays1, axis=1) * np.linalg.norm(rays2, axis=1))
    angles = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    valid = positive & bounded & (errors.max(axis=1) <= quality.max_reprojection_px) & (angles >= quality.min_angle_degrees)
    return {"xyz": xyz, "finite": finite, "positive_depth": positive, "bounded": bounded,
            "errors_px": errors, "angles_degrees": angles, "valid": valid}


def reconstruct(p1, p2, camera1, camera2, quality=Quality()):
    """Inputs are distorted processed pixels; output world frame is camera 1."""
    n1, n2 = camera1.normalized(p1), camera2.normalized(p2)
    if len(n1) != len(n2):
        raise ValueError("Observation counts differ")
    metrics = {"correspondences": len(n1), "quality_thresholds": asdict(quality)}
    def reject(reason):
        raise ReconstructionRejected(reason, metrics)
    if len(n1) < quality.min_correspondences:
        reject("insufficient_correspondences")
    if any(len(np.unique(n, axis=0)) < quality.min_correspondences or np.linalg.matrix_rank(n - n.mean(axis=0)) < 2 for n in (n1, n2)):
        reject("degenerate_observations")
    # Conservative shared virtual focal converts a pixel tolerance to normalized units.
    focal = min(camera1.fx, camera1.fy, camera2.fx, camera2.fy)
    threshold = quality.essential_threshold_px / focal
    metrics.update(normalized_ransac_threshold=threshold, virtual_focal_px=focal)
    essential, mask = cv2.findEssentialMat(n1, n2, np.eye(3), method=cv2.RANSAC,
                                         prob=quality.confidence, threshold=threshold, maxIters=5000)
    if essential is None or mask is None or not np.isfinite(essential).all():
        reject("essential_estimation_failed")
    essential_mask = mask.ravel().astype(bool)
    metrics["essential_inliers"] = int(essential_mask.sum())
    if essential_mask.sum() < quality.min_inliers:
        reject("too_few_essential_inliers")
    bearings = [np.c_[n[essential_mask], np.ones(int(essential_mask.sum()))] for n in (n1, n2)]
    unit1, unit2 = [b / np.linalg.norm(b, axis=1, keepdims=True) for b in bearings]
    u, _, vt = np.linalg.svd(unit2.T @ unit1)
    rotation_only = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    predicted = bearings[0] @ rotation_only.T
    residual = np.full(len(predicted), np.inf)
    forward = predicted[:, 2] > 1e-12
    residual[forward] = np.linalg.norm(predicted[forward, :2] / predicted[forward, 2:] - n2[essential_mask][forward], axis=1) * focal
    rotation_support = float((residual <= 2 * quality.essential_threshold_px).mean())
    metrics.update(rotation_only_support=rotation_support,
                   rotation_only_median_residual_virtual_px=float(np.median(residual)) if np.isfinite(np.median(residual)) else None)
    homography, hmask = cv2.findHomography(n1, n2, cv2.RANSAC, 2 * threshold, maxIters=5000, confidence=quality.confidence)
    hmask = hmask.ravel().astype(bool) if homography is not None and hmask is not None else np.zeros(len(n1), bool)
    support = int((hmask & essential_mask).sum()) / int(essential_mask.sum())
    metrics.update(homography_inliers=int(hmask.sum()), homography_essential_support=support)
    if rotation_support >= quality.homography_dominance:
        reject("rotation_dominant_or_negligible_translation")
    if support >= quality.homography_dominance:
        reject("homography_dominant_planar_or_rotational_scene")
    candidates = []
    if essential.ndim != 2 or essential.shape[1] != 3 or essential.shape[0] % 3:
        reject("invalid_essential_shape")
    for matrix in essential.reshape(-1, 3, 3):
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular[1] < 1e-10 or singular[2] > 1e-3 * singular[1] or singular[0] / singular[1] > 1.01:
            continue
        count, rotation, translation, pose_mask, _ = cv2.recoverPose(
            matrix, n1, n2, np.eye(3), distanceThresh=quality.max_distance,
            mask=essential_mask.astype(np.uint8).reshape(-1, 1))
        candidates.append((count, matrix, rotation, translation.ravel(), pose_mask.ravel().astype(bool)))
    if not candidates:
        reject("unstable_essential_matrix")
    count, matrix, rotation, translation, pose_mask = max(candidates, key=lambda c: c[0])
    norm = np.linalg.norm(translation)
    if not np.isfinite(rotation).all() or not np.isfinite(norm) or norm < 1e-12:
        reject("invalid_or_zero_translation")
    translation /= norm
    metrics.update(pose_inliers=int(count), cheirality_ratio=float(count / essential_mask.sum()))
    if count < quality.min_inliers or metrics["cheirality_ratio"] < quality.min_cheirality_ratio:
        reject("poor_cheirality_or_low_parallax")
    indices = np.flatnonzero(pose_mask)
    result = triangulate(np.asarray(p1)[indices], np.asarray(p2)[indices], camera1, camera2, rotation, translation, quality)
    valid = result["valid"]
    metrics.update(triangulated_points=len(indices), finite_points=int(result["finite"].sum()),
                   positive_depth_points=int(result["positive_depth"].sum()), valid_points=int(valid.sum()),
                   median_candidate_angle_degrees=float(np.median(result["angles_degrees"])),
                   reprojection_pass_points=int((result["errors_px"].max(axis=1) <= quality.max_reprojection_px).sum()),
                   valid_point_ratio=float(valid.mean()))
    if valid.sum() < quality.min_points or valid.mean() < quality.min_valid_ratio:
        reject("too_few_points_after_depth_reprojection_and_parallax_filters")
    metrics.update(median_reprojection_error_px=float(np.median(result["errors_px"][valid])),
                   median_triangulation_angle_degrees=float(np.median(result["angles_degrees"][valid])))
    return {**result, "rotation": rotation, "translation": translation, "essential": matrix,
            "essential_mask": essential_mask, "pose_mask": pose_mask, "pose_indices": indices,
            "point_indices": indices[valid], "metrics": metrics}
