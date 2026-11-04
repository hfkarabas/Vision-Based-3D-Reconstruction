"""Streaming input with explicit source provenance."""
from dataclasses import dataclass
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass
class Frame:
    image: np.ndarray
    metadata: dict


def load_image(path: Path) -> Frame:
    path = Path(path)
    try:
        with Image.open(path) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError(f"Unsupported image format: {path}")
            original = source.size
            orientation = source.getexif().get(274, 1)
            rgb = np.array(ImageOps.exif_transpose(source).convert("RGB"))
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unreadable image: {path}") from exc
    return Frame(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), {
        "source": str(path.resolve()), "source_size": list(original),
        "exif_orientation": orientation, "orientation_applied": True,
    })


def iter_frames(path, interval=1, max_frames=None, errors=None):
    """Yield oriented images or sampled video frames; interval is in source frames."""
    path = Path(path)
    if interval < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("Interval and maximum frame count must be positive")
    if not path.exists():
        raise ValueError(f"Input does not exist: {path}")
    if path.is_dir():
        files = sorted((p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                       key=lambda p: (p.name.casefold(), p.name))
        count = 0
        for file in files[::interval]:
            try:
                frame = load_image(file)
            except ValueError as exc:
                if errors is not None:
                    errors.append(str(exc))
                warnings.warn(str(exc), stacklevel=2)
                continue
            yield frame
            count += 1
            if max_frames is not None and count >= max_frames:
                break
        if count == 0:
            raise ValueError(f"No readable JPEG/PNG images in {path}")
        return
    if path.suffix.lower() != ".mp4":
        raise ValueError("Input must be an image directory or MP4 video")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no usable frame rate for timestamps")
        index = count = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if index % interval == 0:
                yield Frame(image, {"source": str(path.resolve()), "frame_index": index,
                                    "timestamp_seconds": index / fps, "timestamp_basis": "frame_index/fps"})
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
            index += 1
        if count == 0:
            raise ValueError(f"Video contains no readable frames: {path}")
    finally:
        capture.release()
