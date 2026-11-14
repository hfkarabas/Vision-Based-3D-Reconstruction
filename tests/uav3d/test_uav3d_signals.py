import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin

from uav3d import signals


def _tif(path, arr, nodata=-9999.0):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", transform=from_origin(0, 20, 1, 1), nodata=nodata, crs="EPSG:32633") as ds:
        ds.write(arr.astype("float32"), 1)


def test_plane_rms_zero_for_plane_and_positive_for_box():
    yy, xx = np.mgrid[0:10, 0:10]
    plane = np.ma.masked_array(0.5 * xx + 0.2 * yy)
    assert signals.plane_rms(plane) < 1e-9
    box = plane.copy()
    box[3:6, 3:6] += 5
    assert signals.plane_rms(box) > 1


def test_tile_signals_void_density_overlap(tmp_path):
    dsm = np.zeros((20, 20))
    dsm[:, 10:] = -9999.0  # right half (tile 1) has no data
    _tif(tmp_path / "dsm.tif", dsm)
    ov = np.full((20, 20), 3.0)
    ov[:, 10:] = 8
    _tif(tmp_path / "ov.tif", ov, nodata=0)
    h = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(h)
    xs = np.r_[np.random.default_rng(0).uniform(0, 10, 400), np.random.default_rng(1).uniform(10, 20, 100)]
    las.x, las.y, las.z = xs, np.full(len(xs), 5.0), np.zeros(len(xs))
    las.write(tmp_path / "c.laz")
    rows = signals.tile_signals(tmp_path / "dsm.tif", tmp_path / "ov.tif", tmp_path / "c.laz", (0, 0), 10.0, (2, 2))
    assert rows[0]["void"] == 0.0 and rows[1]["void"] == 1.0
    assert rows[0]["overlap"] == 3.0 and rows[1]["overlap"] == 8.0
    assert rows[0]["density"] > rows[1]["density"]
    assert rows[0]["roughness"] < 1e-6 and rows[1]["roughness"] is None
