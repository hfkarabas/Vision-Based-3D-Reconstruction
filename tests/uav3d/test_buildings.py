import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from uav3d import buildings, footprints, ingest


def dsm(tmp_path, valid_cols: int):
    """10x10 m DSM at 1 m, EPSG:32617; only the first ``valid_cols`` columns hold data."""
    data = np.full((10, 10), -9999.0, dtype="float32")
    data[:, :valid_cols] = 5.0
    path = tmp_path / "dsm.tif"
    with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=1, dtype="float32",
                       crs="EPSG:32617", transform=from_origin(500000, 3100010, 1, 1), nodata=-9999) as ds:
        ds.write(data, 1)
    return path


def test_coverage_filter_keeps_only_observed_roofprints(tmp_path):
    path = dsm(tmp_path, valid_cols=5)
    fp = gpd.GeoDataFrame(geometry=[box(500000, 3100000, 500004, 3100004),     # fully covered
                                    box(500003, 3100000, 500009, 3100004)],    # mostly outside
                          crs="EPSG:32617")
    assert footprints.covered(fp, path, 0.9).tolist() == [True, False]


def test_dsm_area_reports_wgs84_bbox(tmp_path):
    (west, south, east, north), wkt = footprints.dsm_area(dsm(tmp_path, 10))
    assert -82 < west < east < -80 and 27 < south < north < 29 and "UTM" in wkt


def test_roofer_summary_counts_buildings_and_modes(tmp_path):
    line = {"type": "CityJSONFeature", "CityObjects": {
        "b1": {"type": "Building", "attributes": {"rf_extrusion_mode": "standard"}},
        "b1-0": {"type": "BuildingPart", "geometry": [{"lod": "1.2"}, {"lod": "2.2"}]},
        "b2": {"type": "Building", "attributes": {"rf_extrusion_mode": "skip"}, "geometry": [{"lod": "0"}]}}}
    (tmp_path / "t.city.jsonl").write_text(json.dumps(line) + "\n")
    s = buildings.summarise(tmp_path)
    assert s["buildings"] == 2 and s["extrusion_modes"] == {"standard": 1, "skip": 1}
    assert s["geometries_by_lod"] == {"1.2": 1, "2.2": 1, "0": 1}
    with pytest.raises(buildings.BuildingError):
        buildings.summarise(tmp_path / "empty")


def test_gcp_autodetect_requires_odm_header(tmp_path):
    (tmp_path / "images").mkdir()
    img = tmp_path / "images" / "a.jpg"
    img.write_bytes(b"")
    (tmp_path / "notes_gcp.txt").write_text("not a gcp file\n")
    assert ingest.find_gcp(tmp_path, [img]) is None
    (tmp_path / "gcp_list.txt").write_text("+proj=utm +zone=17 +datum=WGS84\n1 2 3 4 5 a.jpg\n")
    assert ingest.find_gcp(tmp_path, [img]) == tmp_path / "gcp_list.txt"
