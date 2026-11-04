"""Fundamental matrix verification; no camera pose inference."""
import cv2
import numpy as np


def verify(points1, points2, threshold=1.5, confidence=0.99):
    if threshold <= 0 or not 0 < confidence < 1:
        raise ValueError("Invalid RANSAC threshold or confidence")
    empty = np.zeros(len(points1), dtype=bool)
    if len(points1) < 8:
        return None, empty, "insufficient_correspondences"
    if (not np.isfinite(points1).all() or not np.isfinite(points2).all()
            or len(np.unique(points1, axis=0)) < 8 or len(np.unique(points2, axis=0)) < 8
            or np.linalg.matrix_rank(points1 - points1.mean(axis=0)) < 2
            or np.linalg.matrix_rank(points2 - points2.mean(axis=0)) < 2):
        return None, empty, "degenerate_correspondences"
    try:
        matrix, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, threshold, confidence, 5000)
    except cv2.error:
        return None, empty, "estimation_failed"
    if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all() or mask is None:
        return None, empty, "estimation_failed"
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 8:
        return None, empty, "insufficient_inliers"
    return matrix, inliers, "estimated"
