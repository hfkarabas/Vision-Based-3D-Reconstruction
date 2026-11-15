import json
import struct

import laspy
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from uav3d import adaptive, plan, signals


def _tif(path, arr, nodata=-9999.0):
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                       dtype="float32", transform=from_origin(0, arr.shape[0], 1, 1), nodata=nodata) as ds:
        ds.write(arr.astype("float32"), 1)


def _dmap(path, w, h):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"DR" + b"\x00\x00" + struct.pack("<IIII", w * 8, h * 8, w, h))


def test_levels_one_finer_and_full_resolution_skips():
    assert adaptive.levels(7952, 994) == (3, 2)
    assert adaptive.levels(4000, 1000) == (2, 1)
    with pytest.raises(adaptive.Skip):
        adaptive.levels(994, 994)


def test_coarse_side_from_odm_depth_map(tmp_path):
    assert adaptive.coarse_side(tmp_path) is None
    _dmap(tmp_path / "opensfm" / "undistorted" / "openmvs" / "depthmaps" / "a.dmap", 994, 663)
    assert adaptive.coarse_side(tmp_path) == 994


def test_run_skips_without_metric_gsd_or_georeference(tmp_path):
    cfg = plan.defaults()
    with pytest.raises(adaptive.Skip, match="GSD"):
        adaptive.run(None, tmp_path, "c.laz", None, 0.5, cfg, tmp_path / "w", tmp_path / "l", 2, {})
    with pytest.raises(adaptive.Skip, match="coords"):
        adaptive.run(None, tmp_path, "c.laz", 0.02, 0.5, cfg, tmp_path / "w", tmp_path / "l", 2, {})
    with pytest.raises(ValueError):
        adaptive.run(None, tmp_path, "c.laz", 0.02, 1.5, cfg, tmp_path / "w", tmp_path / "l", 2, {})


def test_run_skips_on_pruned_project(tmp_path):
    (tmp_path / "odm_georeferencing").mkdir()
    (tmp_path / "odm_georeferencing" / "coords.txt").write_text("WGS84 UTM 33N\n1 2\n", encoding="utf-8")
    with pytest.raises(adaptive.Skip, match="intermediates"):
        adaptive.run(None, tmp_path, "c.laz", 0.02, 0.5, plan.defaults(), tmp_path / "w", tmp_path / "l", 2, {})


def test_frozen_gain_model_loads_from_defaults():
    m = adaptive.load_model(plan.defaults())
    assert any(w > 0 for w in m.weights.values())
    assert set(m.weights) <= set(adaptive.SIGNALS)


def test_prune_keeps_merged_cloud_and_records(tmp_path):
    w = tmp_path / "adaptive"
    _dmap(w / "g000" / "depthmaps" / "a.dmap", 4, 4)
    for name in ("g000/scene.mvs", "g000/scene_dense.ply", "refined_points.laz", "point_cloud.laz",
                 "coarse_dsm.tif"):
        (w / name).parent.mkdir(parents=True, exist_ok=True)
        (w / name).write_bytes(b"x" * 10)
    adaptive.prune(w)
    left = sorted(p.relative_to(w).as_posix() for p in w.rglob("*") if p.is_file())
    assert left == ["coarse_dsm.tif", "point_cloud.laz"]


def test_tile_signals_without_overlap_raster(tmp_path):
    _tif(tmp_path / "dsm.tif", np.zeros((20, 20)))
    h = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(h)
    xs = np.random.default_rng(0).uniform(0, 20, 200)
    las.x, las.y, las.z = xs, np.full(len(xs), 5.0), np.zeros(len(xs))
    las.write(tmp_path / "c.laz")
    rows = signals.tile_signals(tmp_path / "dsm.tif", None, tmp_path / "c.laz", (0, 0), 10.0, (2, 2))
    assert all(r["overlap"] is None for r in rows)
    assert rows[0]["void"] == 0.0


def test_cli_accepts_refine_budget(monkeypatch, tmp_path):
    from uav3d import cli

    seen = {}
    monkeypatch.setattr(cli, "cmd_run", lambda a: seen.setdefault("b", a.refine_budget) and 0)
    cli.main(["run", str(tmp_path), "--out", str(tmp_path / "o"), "--refine-budget", "0.5"])
    assert seen["b"] == 0.5
