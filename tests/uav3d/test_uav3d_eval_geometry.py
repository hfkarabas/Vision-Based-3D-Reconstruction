import laspy
import numpy as np
import pytest

from uav3d.evaluate import geometry


def write(path, xyz, cls=None):
    h = laspy.LasHeader(point_format=6, version="1.4")
    h.scales = [0.001] * 3
    h.offsets = [500000, 4000000, 0]
    las = laspy.LasData(h)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if cls is not None:
        las.classification = cls
    las.write(path)


def test_precision_recall_on_offset_and_partial_reconstruction(tmp_path):
    rng = np.random.default_rng(0)
    n = 200_000
    xy = rng.uniform(0, 40, (n, 2)) + [500000, 4000000]
    ref = np.column_stack([xy, np.full(n, 10.0)])
    # reconstruction: the same ground surface, 0.15 m too high, plus a 30 m high noise cluster
    ours = np.column_stack([xy[: n // 2], np.full(n // 2, 10.15)])
    noise = np.column_stack([xy[:500], np.full(500, 40.0)])
    write(tmp_path / "ref.las", ref)
    write(tmp_path / "ours.laz", np.vstack([ours, noise]),
          cls=np.r_[np.full(n // 2, 2), np.full(500, 7)].astype(np.uint8))
    r = geometry.evaluate(tmp_path / "ours.laz", tmp_path / "ref.las", taus=(0.1, 0.2), voxel=0.05,
                          footprint_cell=2.0, sample=50_000)
    assert r["tau_10cm"]["precision"] < 0.05            # 15 cm offset fails a 10 cm tolerance
    assert r["tau_20cm"]["precision"] > 0.95            # noise class excluded
    assert r["tau_20cm"]["recall"] > 0.95               # common footprint only: both cover the same area
    assert r["median_vertical_offset_m"] == pytest.approx(0.15, abs=0.02)
