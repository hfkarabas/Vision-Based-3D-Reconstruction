"""Exercise geospatial format transfer and coverage accounting on small fixtures."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

requires_gis = all(importlib.util.find_spec(name) is not None
                   for name in ("rasterio", "shapely", "matplotlib"))
pytestmark = pytest.mark.skipif(not requires_gis, reason="optional GIS runtime required")
if requires_gis:
    import rasterio
    from rasterio.transform import from_origin

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from omega_odm_products import transfer
    from omega_site_coverage import statistics
    from omega_uav_foundation_trial import prepare


def test_site_components_are_clipped_to_the_evaluation_region():
    terrain = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=bool)
    region = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=bool)
    assert statistics(terrain, region) == dict(
        pixels=4, fraction=4 / 6, components=2, largest_fraction=3 / 6,
    )


def test_odm_transfer_tracks_inferred_ground_and_excludes_areas_without_photos(tmp_path):
    affine = from_origin(100, 204, 1, 1)
    rgb = np.full((3, 4, 4), 127, dtype=np.uint8)
    mask = np.full((4, 4), 255, dtype=np.uint8)
    mask[0, 0] = 0
    elevation = np.full((4, 4), 12, dtype=np.float32)
    elevation[3, 3] = -9999
    observed = elevation.copy()
    observed[1, 1] = -9999
    geo = dict(driver="GTiff", crs="EPSG:32617", width=4, height=4, transform=affine)
    rgb_path = tmp_path / "rgb.tif"
    with rasterio.open(rgb_path, "w", dtype="uint8", count=3, **geo) as dst:
        dst.write(rgb)
        dst.write_mask(mask)
    for name, values in (("dsm", elevation), ("dtm", elevation), ("observed", observed)):
        with rasterio.open(tmp_path / f"{name}.tif", "w", dtype="float32", count=1,
                           nodata=-9999, **geo) as dst:
            dst.write(values, 1)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(dict(surface=dict(
        ortho_cell_m=1, orthophoto_max_pixels=100, terrain_cell_m=1, terrain_max_cells=100,
    ))), encoding="utf-8")
    result = tmp_path / "result"
    transfer(argparse.Namespace(rgb=rgb_path, dsm=tmp_path / "dsm.tif", dtm=tmp_path / "dtm.tif",
                                observed=tmp_path / "observed.tif", settings=settings, output=result,
                                extent=[100, 200, 104, 204], crs="EPSG:32617"))
    with rasterio.open(result / "ground/dtm.tif") as terrain:
        z = terrain.read(1, masked=True)
        assert int(z.mask.sum()) == 2
        assert z[1, 1] == 12
    with rasterio.open(result / "ground/inferred.tif") as inferred:
        values = inferred.read(1)
        assert values.sum() == 1
        assert values[1, 1] == 1


def test_foundation_can_keep_observed_ground_inside_a_detection(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    terrain = tmp_path / "source"
    terrain.mkdir()
    profile = dict(driver="GTiff", crs="EPSG:32617", width=10, height=10, count=1,
                   transform=from_origin(100, 210, 1, 1), dtype="float32", nodata=-9999)
    original = np.full((10, 10), 20, dtype=np.float32)
    original[4, 4] = 21
    for name, array in [("dtm", original), ("classified_ground", original),
                        ("observed", np.ones_like(original))]:
        with rasterio.open(terrain / f"{name}.tif", "w", **profile) as dst:
            dst.write(array, 1)
    dense = tmp_path / "density.tif"
    with rasterio.open(dense, "w", **profile) as dst:
        dst.write(np.ones_like(original), 1)
    np.savez(terrain / "terrain.npz", x=np.arange(10), y=np.arange(10), z=original)
    (terrain / "complete.json").write_text('{"origin": [105, 205, 0]}', encoding="utf-8")
    roofs = tmp_path / "roofs.gpkg"
    gpd.GeoDataFrame(geometry=[box(103, 203, 107, 207)], crs=profile["crs"]).to_file(roofs)
    output = tmp_path / "foundation"
    prepare(terrain, dense, roofs, output, radius=8, surrounded=True, preserve_observed=True)
    with rasterio.open(output / "dtm.tif") as src:
        np.testing.assert_array_equal(src.read(1), original)
    with rasterio.open(output / "inferred_foundation.tif") as src:
        assert src.read(1).sum() == 0
