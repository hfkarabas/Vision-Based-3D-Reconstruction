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
    confidence: str | None = None

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
        if self.confidence not in (None, "exif_estimate", "fallback"):
            raise ValueError("Unknown intrinsics confidence")
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


def parse_camera(value):
    fields = dict(value)
    matrix = fields.pop("K", None)
    camera = Camera(**fields)
    if matrix is not None and (np.shape(matrix) != (3, 3) or
                               not np.allclose(np.asarray(matrix, dtype=float), camera.K, rtol=1e-9, atol=1e-9)):
        raise ValueError("K conflicts with scalar intrinsics")
    return camera


def _parse_pair(data):
    try:
        if "camera1" in data:
            if set(data) != {"camera1", "camera2"}:
                raise ValueError("Pair calibration requires exactly camera1 and camera2")
            return parse_camera(data["camera1"]), parse_camera(data["camera2"])
        camera = parse_camera(data)
        return camera, camera
    except (TypeError, KeyError) as exc:
        raise ValueError(f"Invalid camera configuration: {exc}") from exc


def load_cameras(path):
    """A single camera object or an object with camera1/camera2 entries."""
    return _parse_pair(json.loads(Path(path).read_text(encoding="utf-8")))


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


def infer_camera(width: int, height: int, metadata: dict) -> Camera:
    """Initialize oriented pinhole intrinsics from EXIF or a per-image diagonal prior."""
    tags = metadata.get("exif_focal", {})
    diagonal = np.hypot(width, height)
    equivalent = tags.get("41989", 0)
    focal = tags.get("37386", 0)
    unit_mm = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}.get(tags.get("41488"))
    plane_x, plane_y = tags.get("41486", 0), tags.get("41487", 0)
    plane = None
    # Focal-plane resolution can refer to the original uncropped pixel grid.
    if (unit_mm and focal > 0 and min(plane_x, plane_y) > 0 and
            [tags.get("40962"), tags.get("40963")] == metadata.get("source_size") and
            tags.get("41988", 1) in (0, 1)):
        plane = (focal * plane_x / unit_mm, focal * plane_y / unit_mm)
        if metadata.get("exif_orientation") in (5, 6, 7, 8):
            plane = plane[::-1]
        if min(plane) < 0.2 * diagonal or max(plane) > 12 * diagonal or max(plane) / min(plane) > 1.1:
            plane = None
    estimate = diagonal * equivalent / np.hypot(36, 24) if 9 <= equivalent <= 500 else None
    if plane and estimate and max(abs(v / estimate - 1) for v in plane) > 0.25:
        source, focal_xy = "fallback (conflicting EXIF focal estimates)", None
    elif estimate:
        source, focal_xy = "EXIF 35 mm equivalent focal length", (estimate, estimate)
    elif plane:
        source, focal_xy = "EXIF focal length and focal-plane resolution", plane
    else:
        source, focal_xy = "fallback (missing or unusable focal EXIF)", None
    confidence = "exif_estimate" if focal_xy else "fallback"
    fx, fy = focal_xy if focal_xy else (diagonal, diagonal)
    assumptions = "centered principal point, zero distortion; focal initialization only"
    if confidence == "fallback":
        assumptions += "; focal length equals image diagonal (~53 degree diagonal FOV)"
    return Camera(width, height, float(fx), float(fy), (width - 1) / 2, (height - 1) / 2,
                  source=source, approximate=True, assumptions=assumptions, confidence=confidence)


def named_camera(mapping, name):
    if name not in mapping:
        raise ValueError(f"Missing calibration for {name}.")
    return parse_camera(mapping[name])


def load_calibration(path, *, pair=False):
    """Return filename mapping or explicit model tuple, rejecting ambiguous schemas."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Camera calibration must be a JSON object")
    if "images" in data:
        if set(data) != {"images"} or not isinstance(data["images"], dict):
            raise ValueError("Expected an images calibration object keyed by filename")
        return data["images"], None
    models = _parse_pair(data)
    if not pair and models[0] != models[1]:
        raise ValueError("Sequence calibration uses one shared camera or an images mapping, not a two-camera pair")
    return None, models
