import numpy as np
import rasterio
from rasterio.transform import from_origin

from uav3d import terrain

TCFG = {"health_cell_m": 0.5, "local_window_m": 5, "pit_depth_m": 1.0}


def test_low_surface_removes_mirrored_patch_but_keeps_steep_slope():
    rng = np.random.default_rng(0)
    n = 60000
    x, y = rng.uniform(0, 200, n), rng.uniform(0, 200, n)
    z = np.where(x < 100, 10.0, 10.0 + (x - 100) * 0.8)        # flat, then a real 39 deg slope
    z = z + rng.normal(0, 0.05, n)
    # dense "reflection" patch (like a pond mirroring a hill): 30 m below the flat ground
    m = 4000
    px, py = rng.uniform(30, 45, m), rng.uniform(80, 95, m)
    pz = np.full(m, -20.0) + rng.normal(0, 0.3, m)
    X, Y, Z = np.concatenate([x, px]), np.concatenate([y, py]), np.concatenate([z, pz])
    low = terrain.low_surface_mask(X, Y, Z, cell_m=5.0, window_cells=5, base_tolerance_m=3.0)
    assert low[n:].mean() > 0.95                 # mirrored patch removed
    assert low[:n][x > 100].mean() < 0.005       # real slope kept
    assert low[:n].mean() < 0.01


def test_health_detects_pits(tmp_path):
    a = np.full((200, 200), 10.0, dtype="float32")
    a[100:104, 100:104] = -40.0                  # a 2 m x 2 m pit, 50 m deep
    path = tmp_path / "dtm.tif"
    with rasterio.open(path, "w", driver="GTiff", height=200, width=200, count=1, dtype="float32",
                       crs="EPSG:32617", transform=from_origin(0, 100, 0.5, 0.5), nodata=-9999) as ds:
        ds.write(a, 1)
    h = terrain.health(path, TCFG)
    assert 0 < h["pit_fraction"] < 0.01 and h["deepest_pit_m"] < -40 and h["spike_fraction"] == 0


def test_overlap_summary(tmp_path):
    a = np.array([[2, 3, 4, 5], [6, 7, 8, 9], [255, 255, 255, 255]], dtype="uint8")
    path = tmp_path / "overlap.tif"
    with rasterio.open(path, "w", driver="GTiff", height=3, width=4, count=1, dtype="uint8",
                       crs="EPSG:32617", transform=from_origin(0, 3, 1, 1), nodata=255) as ds:
        ds.write(a, 1)
    s = terrain.overlap_summary(path)
    assert s == {"share_lt3": 0.125, "share_3to4": 0.25, "share_5to7": 0.375, "share_ge8": 0.25, "median": 5.5}
