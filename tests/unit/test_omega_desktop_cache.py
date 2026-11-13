"""Test ODM handoff against mismatched inputs, tools and products."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


GIS = all(importlib.util.find_spec(name) for name in ("rasterio", "shapely", "psutil", "yaml"))
pytestmark = pytest.mark.skipif(not GIS, reason="optional GIS runtime required")
if GIS:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from omega_desktop import PRODUCTS, cache_project, register_cache, sha256


def test_only_matching_odm_products_are_reused(tmp_path):
    project = tmp_path / "odm"
    images = project / "images"
    images.mkdir(parents=True)
    image = images / "camera.jpg"
    image.write_bytes(b"original")
    inputs = {image.name: sha256(image)}
    options = {"dsm": True, "dtm": True, "max_concurrency": 4}
    metadata = {"version": "3.6.2", "files": {"run.py": "checksum"}, "options": options}
    (project / "options.json").write_text(json.dumps(dict(options, name="site")), encoding="utf-8")
    for file in PRODUCTS.values():
        path = project / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file.encode())
    manifest = tmp_path / "cache.json"
    register_cache(project, manifest, inputs, metadata)
    assert cache_project(manifest, inputs, metadata) == project
    assert cache_project(manifest, {image.name: "changed"}, metadata) is None
    assert cache_project(manifest, inputs, dict(metadata, version="other")) is None
    (project / "options.json").write_text(json.dumps(dict(options, max_concurrency=2)), encoding="utf-8")
    assert cache_project(manifest, inputs, metadata) is None
    (project / "options.json").write_text(json.dumps(options), encoding="utf-8")
    (project / PRODUCTS["dsm"]).write_bytes(b"changed")
    assert cache_project(manifest, inputs, metadata) is None
