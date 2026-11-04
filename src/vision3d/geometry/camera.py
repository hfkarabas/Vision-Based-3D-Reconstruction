"""Validated pinhole intrinsics in EXIF-oriented image coordinates."""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import warnings

import cv2
import numpy as np


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple = ()
    source: str = "known calibration"
    approximate: bool = False
    assumptions: str = ""
    coordinates: str = "exif_oriented"

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in (self.width, self.height)):
            raise ValueError("Camera width/height must be integers >= 2")
        values = [self.fx, self.fy, self.cx, self.cy]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ValueError("Camera intrinsics must be numeric scalars")
        if not np.isfinite(values).all() or min(self.fx, self.fy) <= 0:
            raise ValueError("Camera focal lengths must be positive and all intrinsics finite")
        if not (0 <= self.cx < self.width and 0 <= self.cy < self.height):
            raise ValueError("Principal point must lie inside the image")
        if not isinstance(self.distortion, (list, tuple)) or len(self.distortion) not in (0, 4, 5, 8, 12, 14):
            raise ValueError("Expected 0/4/5/8/12/14 finite OpenCV distortion coefficients")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in self.distortion) or not np.isfinite(self.distortion).all():
            raise ValueError("Distortion coefficients must be finite numeric scalars")
        object.__setattr__(self, "distortion", tuple(self.distortion))
        if self.coordinates != "exif_oriented":
            raise ValueError("Calibration must describe EXIF-oriented pixels; raw sensor calibration is unsupported")
        if (not isinstance(self.approximate, bool) or not isinstance(self.source, str) or not self.source
                or not isinstance(self.assumptions, str) or (self.approximate and not self.assumptions)):
            raise ValueError("Calibration requires a source and explicit approximate assumptions")

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    @property
    def D(self):
        return np.asarray(self.distortion, dtype=np.float64) if self.distortion else None

    def resized(self, width, height):
        sx, sy = width / self.width, height / self.height
        return replace(self, width=width, height=height, fx=self.fx * sx, fy=self.fy * sy,
                       cx=(self.cx + 0.5) * sx - 0.5, cy=(self.cy + 0.5) * sy - 0.5)

    def normalized(self, pixels):
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
            raise ValueError("Observations must be a finite Nx2 matrix")
        if len(pixels) == 0:
            return pixels.copy()
        normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)
        if not np.isfinite(normalized).all():
            raise ValueError("Undistortion produced non-finite coordinates")
        return normalized

    def to_dict(self):
        return {**asdict(self), "K": self.K.tolist()}


def load_cameras(path):
    """A single camera object or an object with camera1/camera2 entries."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        def parse(value):
            value = dict(value)
            matrix = value.pop("K", None)
            camera = Camera(**value)
            if matrix is not None and not np.allclose(np.asarray(matrix), camera.K, rtol=1e-9, atol=1e-9):
                raise ValueError("K conflicts with scalar intrinsics")
            return camera
        if "camera1" in data:
            if set(data) != {"camera1", "camera2"}:
                raise ValueError("Pair calibration requires exactly camera1 and camera2")
            return parse(data["camera1"]), parse(data["camera2"])
        camera = parse(data)
        return camera, camera
    except (TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid camera configuration: {exc}") from exc


def approximate_camera(width, height, focal_px=None, fov_degrees=None):
    if (focal_px is None) == (fov_degrees is None):
        raise ValueError("Specify exactly one focal length in original pixels or horizontal FOV")
    if fov_degrees is not None:
        if not np.isfinite(fov_degrees) or not 1 < fov_degrees < 179:
            raise ValueError("Horizontal FOV must be between 1 and 179 degrees")
        focal_px = width / (2 * np.tan(np.deg2rad(fov_degrees) / 2))
    assumptions = (f"horizontal_fov_degrees={fov_degrees}" if fov_degrees is not None else f"focal_px={focal_px}")
    assumptions += "; square pixels, centered principal point, zero distortion"
    camera = Camera(width, height, focal_px, focal_px, (width - 1) / 2, (height - 1) / 2,
                    source="explicit user approximation", approximate=True, assumptions=assumptions)
    warnings.warn("Approximate intrinsics reduce geometric reliability: " + assumptions, stacklevel=2)
    return camera
