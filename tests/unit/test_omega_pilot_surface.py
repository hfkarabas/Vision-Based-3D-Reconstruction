"""Regression checks for bounded ground support under detected roofs."""

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

if importlib.util.find_spec("geopandas") and importlib.util.find_spec("rasterio"):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from omega_lidar_check import matched_evaluation
    from omega_pilot_surface import support_roofs
else:
    pytestmark = pytest.mark.skip(reason="optional GIS runtime required")


def test_ground_under_roofs_never_fills_beyond_observed_surface(tmp_path):
    profile = dict(driver="GTiff", width=12, height=12, count=1, dtype="float32",
                   transform=from_origin(100, 212, 1, 1), crs="EPSG:32617", nodata=-9999)
    surface = np.full((12, 12), 20, dtype=np.float32)
    base = surface.copy()
    anchors = np.ones_like(surface)
    surface[3:9, 3:9] = 27
    surface[5, 5] = -9999
    base[3:9, 3:9] = -9999
    anchors[3:9, 3:9] = 0
    for name, raster in (("dsm", surface), ("dtm", base), ("anchors", anchors)):
        with rasterio.open(tmp_path / f"{name}.tif", "w", **profile) as dst:
            dst.write(raster, 1)
    polygons = tmp_path / "roofs.geojson"
    gpd.GeoDataFrame(geometry=[box(103, 203, 109, 209)], crs=profile["crs"]).to_file(polygons)
    output = tmp_path / "derived"
    support_roofs(argparse.Namespace(
        dsm=tmp_path / "dsm.tif", dtm=tmp_path / "dtm.tif", anchors=tmp_path / "anchors.tif",
        polygons=polygons, output=output, radius=4,
    ))
    with rasterio.open(output / "dtm.tif") as src:
        result = src.read(1, masked=True)
    assert result[5, 5] is np.ma.masked
    assert result[3, 3] == pytest.approx(20)
    assert result[0, 0] == 20
    with rasterio.open(output / "observed_ground.tif") as src:
        assert src.read(1)[3, 3] == 0
    with rasterio.open(output / "inferred_ground.tif") as src:
        assert src.read(1)[3, 3] == 1


def test_lidar_ab_uses_the_same_cells_and_reference_samples():
    lidar = np.array([[0.2, 0.2, 0], [1.2, 0.2, 10], [2.2, 0.2, 20]])
    first = np.array([[0.2, 0.2, 1], [1.2, 0.2, 10]])
    second = np.array([[0.2, 0.2, 2], [2.2, 0.2, 20]])
    report = matched_evaluation(lidar, first, second)
    assert report["common_1m_cells"] == report["lidar_points"] == 1
    assert report["variants"][0]["reconstruction_to_lidar"]["median_m"] == 1
    assert report["variants"][1]["lidar_to_reconstruction"]["median_m"] == 2
