"""Preprocessing in the oriented image coordinate system."""
import cv2
import numpy as np


def grayscale(image):
    validate(image)
    return image.copy() if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def validate(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Expected a nonempty uint8 image")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("Expected grayscale or BGR image")


def blur_score(image):
    return float(cv2.Laplacian(grayscale(image), cv2.CV_64F).var())


def preprocess(image, max_dimension=1600, score_blur=True):
    validate(image)
    if max_dimension < 1:
        raise ValueError("Maximum dimension must be positive")
    h, w = image.shape[:2]
    scale = min(1.0, max_dimension / max(h, w))
    width, height = max(1, round(w * scale)), max(1, round(h * scale))
    result = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    sx, sy = width / w, height / h
    gray = grayscale(result)
    return result, {"original_size": [w, h], "processed_size": [width, height],
                    "scale_x": sx, "scale_y": sy,
                    "pixel_transform": [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]],
                    "mean_intensity": float(gray.mean()), "std_intensity": float(gray.std()),
                    "blur_variance": blur_score(gray) if score_blur else None}
