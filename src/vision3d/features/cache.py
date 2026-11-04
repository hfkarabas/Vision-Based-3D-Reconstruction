"""Content/configuration keyed feature cache; numeric NPZ only, never pickle."""
import hashlib
import json
from pathlib import Path
import tempfile
import warnings
import zipfile

import cv2
import numpy as np
from . import Features, extract


def cached_extract(image, content_hash, max_dimension, detector, max_features, cache=None):
    config = {"detector": detector, "nfeatures": max_features}
    if cache is None:
        return extract(image, detector, max_features), False
    signature = {"schema": 1, "source_sha256": content_hash, "max_dimension": max_dimension,
                 "config": config, "shape": image.shape, "opencv": cv2.__version__, "numpy": np.__version__}
    sources = [Path(__file__).with_name("__init__.py"), Path(__file__).parents[1] / "preprocessing" / "__init__.py"]
    signature["implementation_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    directory = Path(cache)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                values, descriptors = data["keypoints"], data["descriptors"]
                width = 128 if detector == "sift" else 32
                dtype = np.float32 if detector == "sift" else np.uint8
                if (str(data["signature"].item()) != key or values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all()
                        or len(values) > max_features * 2 + 100 or descriptors.shape != (len(values), width)
                        or descriptors.dtype != dtype or not np.isfinite(descriptors).all()):
                    raise ValueError("Invalid cached feature arrays")
                if len(values) and ((values[:, 0] < 0).any() or (values[:, 0] >= image.shape[1]).any()
                        or (values[:, 1] < 0).any() or (values[:, 1] >= image.shape[0]).any() or (values[:, 2] <= 0).any()):
                    raise ValueError("Cached keypoints exceed image bounds")
                keypoints = tuple(cv2.KeyPoint(float(v[0]), float(v[1]), float(v[2]), float(v[3]),
                                               float(v[4]), int(v[5]), int(v[6])) for v in values)
                return Features(keypoints, values[:, :2].astype(np.float32), descriptors if len(values) else None,
                                (image.shape[1], image.shape[0]), config, 0.0), True
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, cv2.error):
            warnings.warn(f"Ignoring invalid feature cache entry: {path.name}", stacklevel=2)
    features = extract(image, detector, max_features)
    values = np.array([[*k.pt, k.size, k.angle, k.response, k.octave, k.class_id] for k in features.keypoints], np.float64).reshape(-1, 7)
    descriptors = features.descriptors
    if descriptors is None:
        descriptors = np.empty((0, 128 if detector == "sift" else 32), np.float32 if detector == "sift" else np.uint8)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, keypoints=values, descriptors=descriptors, signature=np.array(key))
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return features, False
