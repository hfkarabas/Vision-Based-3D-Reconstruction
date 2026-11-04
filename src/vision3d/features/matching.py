"""Descriptor matching stages and geometric verification."""
from dataclasses import dataclass
from time import perf_counter
import cv2
from vision3d.geometry import verify


def ratio_filter(neighbors, ratio):
    if not 0 < ratio < 1:
        raise ValueError("Ratio must be between zero and one")
    return [pair[0] for pair in neighbors if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]


@dataclass
class Matches:
    raw: list
    ratio: list
    consistent: list
    inliers: list
    matrix: object
    status: str
    seconds: float

    def metrics(self):
        return {"raw_matches": len(self.raw), "ratio_matches": len(self.ratio),
                "consistent_matches": len(self.consistent), "ransac_inliers": len(self.inliers),
                "inlier_ratio": len(self.inliers) / len(self.consistent) if self.consistent else 0.0,
                "geometry_status": self.status, "matching_seconds": self.seconds,
                "fundamental_matrix": self.matrix.tolist() if self.matrix is not None else None}


def match(first, second, ratio=0.75, mutual=False, threshold=1.5, confidence=0.99):
    ratio_filter([], ratio)
    if first.config["detector"] != second.config["detector"]:
        raise ValueError("Cannot match different descriptor families")
    start = perf_counter()
    raw, filtered, consistent = [], [], []
    a, b = first.descriptors, second.descriptors
    if a is not None and b is not None and len(a) > 0 and len(b) >= 2:
        norm = cv2.NORM_L2 if first.config["detector"] == "sift" else cv2.NORM_HAMMING
        if first.config["detector"] == "orb" and first.config.get("WTA_K", 2) != 2:
            norm = cv2.NORM_HAMMING2
        matcher = cv2.BFMatcher(norm)
        neighbors = matcher.knnMatch(a, b, k=2)
        raw = [pair[0] for pair in neighbors if pair]
        filtered = ratio_filter(neighbors, ratio)
        consistent = filtered
        if mutual:
            reverse = ratio_filter(matcher.knnMatch(b, a, k=2), ratio) if len(a) >= 2 else []
            pairs = {(m.trainIdx, m.queryIdx) for m in reverse}
            consistent = [m for m in filtered if (m.queryIdx, m.trainIdx) in pairs]
    p1 = first.points[[m.queryIdx for m in consistent]]
    p2 = second.points[[m.trainIdx for m in consistent]]
    matrix, mask, status = verify(p1, p2, threshold, confidence)
    return Matches(raw, filtered, consistent, [m for m, keep in zip(consistent, mask) if keep],
                   matrix, status, perf_counter() - start)
