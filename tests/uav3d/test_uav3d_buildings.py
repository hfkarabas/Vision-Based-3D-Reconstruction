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
    std = lambda h70, rmse: {"rf_extrusion_mode": "standard", "rf_h_ground": 3.0,  # noqa: E731
                             "rf_h_roof_70p": 3.0 + h70, "rf_rmse_lod22": rmse, "rf_roof_planes": 2}
    line = {"type": "CityJSONFeature", "CityObjects": {
        "b1": {"type": "Building", "attributes": std(4.0, 0.3)},
        "b1-0": {"type": "BuildingPart", "geometry": [{"lod": "1.2"}, {"lod": "2.2"}]},
        "b2": {"type": "Building", "attributes": {"rf_extrusion_mode": "skip"}, "geometry": [{"lod": "0"}]},
        "b3": {"type": "Building", "attributes": std(1.2, 0.2)},
        "b4": {"type": "Building", "attributes": std(5.0, 2.4)}}}
    (tmp_path / "t.city.jsonl").write_text(json.dumps(line) + "\n")
    rcfg = {"min_roof_height_m": 2.5, "low_confidence_rmse_m": 1.0}
    s = buildings.summarise(tmp_path, rcfg)
    assert s["buildings"] == 4 and s["extrusion_modes"] == {"standard": 3, "skip": 1}
    assert s["by_status"] == {"ok": 1, "skipped": 1, "too_low": 1, "low_confidence": 1}
    assert s["per_building"]["b1"]["height_m"] == 4.0
    with pytest.raises(buildings.BuildingError):
        buildings.summarise(tmp_path / "empty", rcfg)


def test_pointcloud_roofprints_recover_buildings_and_drop_noise(tmp_path):
    import laspy

    rng = np.random.default_rng(0)
    def square(x0, y0, side, n, cls):
        return np.column_stack([rng.uniform(x0, x0 + side, n), rng.uniform(y0, y0 + side, n),
                                np.full(n, 5.0), np.full(n, cls)])
    pts = np.vstack([square(0, 0, 10, 20000, 6),      # 100 m2 building
                     square(30, 0, 6, 8000, 6),       # 36 m2 building
                     square(15, 15, 0.4, 20, 6),      # speck of misclassified points
                     square(-5, -5, 50, 20000, 2)])   # ground
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001] * 3
    header.offsets = [500000, 3100000, 0]
    las = laspy.LasData(header)
    las.x, las.y, las.z = pts[:, 0] + 500000, pts[:, 1] + 3100000, pts[:, 2]
    las.classification = pts[:, 3].astype(np.uint8)
    path = tmp_path / "pc.laz"
    las.write(path)
    cfg = {"cell_gsd_factor": 10, "cell_min_m": 0.1, "cell_max_m": 0.5, "close_m": 1.0, "open_m": 0.5,
           "simplify_cells": 1.5, "rectangularity": 0.8}
    fp = footprints.from_pointcloud(path, "EPSG:32617", 0.03, cfg, building_class=6, min_area=10)
    areas = sorted(fp.geometry.area)
    assert len(fp) == 2
    assert areas[0] == pytest.approx(36, rel=0.15) and areas[1] == pytest.approx(100, rel=0.15)


def test_gcp_autodetect_requires_odm_header(tmp_path):
    (tmp_path / "images").mkdir()
    img = tmp_path / "images" / "a.jpg"
    img.write_bytes(b"")
    (tmp_path / "notes_gcp.txt").write_text("not a gcp file\n")
    assert ingest.find_gcp(tmp_path, [img]) is None
    (tmp_path / "gcp_list.txt").write_text("+proj=utm +zone=17 +datum=WGS84\n1 2 3 4 5 a.jpg\n")
    assert ingest.find_gcp(tmp_path, [img]) == tmp_path / "gcp_list.txt"
