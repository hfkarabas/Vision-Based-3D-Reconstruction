"""Content-keyed reciprocal descriptor matching cache; geometry is always recomputed."""
import hashlib
import json
from pathlib import Path
import tempfile

import cv2

from .matching import match


def descriptor_key(features):
    descriptors = features.descriptors
    digest = hashlib.sha256(repr(features.config).encode())
    if descriptors is not None:
        digest.update(str((descriptors.shape, descriptors.dtype)).encode())
        digest.update(descriptors.tobytes())
    return digest.hexdigest()


def reciprocal_matches(a, b, ratio, cache=None, keys=None):
    """Return diagnostics and (query, train, distance) triples."""
    if cache is not None:
        keys = keys or (descriptor_key(a), descriptor_key(b))
        fingerprint = hashlib.sha256(Path(__file__).with_name("matching.py").read_bytes()).hexdigest()
        key = hashlib.sha256(repr((keys, ratio, cv2.__version__, fingerprint)).encode()).hexdigest()
        path = Path(cache) / f"{key}.json"
        try:
            metrics, matches = json.loads(path.read_text(encoding="utf-8"))
            if not (0 <= len(matches) == metrics["consistent_matches"] <= metrics["ratio_matches"] <= metrics["raw_matches"] <= len(a.points)):
                raise ValueError("Invalid cached counts")
            if any(not (type(i) is int and type(j) is int and 0 <= i < len(a.points)
                        and 0 <= j < len(b.points) and 0 <= distance < float("inf")) for i, j, distance in matches):
                raise ValueError("Invalid cached match")
            return metrics, matches, True
        except (OSError, ValueError, TypeError, KeyError):
            pass
    result = match(a, b, ratio=ratio, mutual=True, geometric=False)
    metrics = result.metrics()
    matches = [(m.queryIdx, m.trainIdx, float(m.distance)) for m in result.consistent]
    if cache is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, encoding="utf-8", mode="w", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump([metrics, matches], stream, allow_nan=False)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return metrics, matches, False
