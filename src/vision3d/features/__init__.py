"""OpenCV feature extraction retaining native descriptor types."""
from dataclasses import dataclass
from time import perf_counter
import cv2
import numpy as np
from vision3d.preprocessing import grayscale


@dataclass
class Features:
    keypoints: tuple
    points: np.ndarray
    descriptors: np.ndarray | None
    dimensions: tuple
    config: dict
    seconds: float

    @property
    def count(self):
        return len(self.keypoints)


def extract(image, detector="sift", max_features=4000, **parameters):
    if max_features < 1:
        raise ValueError("Feature limit must be positive")
    if detector not in {"sift", "orb"}:
        raise ValueError(f"Unknown detector: {detector}")
    config = {"detector": detector, "nfeatures": max_features, **parameters}
    algorithm = (cv2.SIFT_create if detector == "sift" else cv2.ORB_create)(
        nfeatures=max_features, **parameters)
    start = perf_counter()
    keypoints, descriptors = algorithm.detectAndCompute(grayscale(image), None)
    return Features(tuple(keypoints), np.array([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 2),
                    descriptors, (image.shape[1], image.shape[0]), config, perf_counter() - start)
