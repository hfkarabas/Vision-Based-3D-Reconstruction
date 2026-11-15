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


def test_planar_points_keep_roof_and_drop_tree_crown():
    rng = np.random.default_rng(1)
    n = 20000
    xy = rng.uniform(0, 10, (n, 2))
    roof = np.column_stack([xy, 5 + 0.3 * xy[:, 0] + rng.normal(0, 0.02, n)])      # sloped plane, 2 cm noise
    c = rng.normal(0, 1, (n, 3))
    crown = c / np.linalg.norm(c, axis=1, keepdims=True) * rng.uniform(1.5, 3, (n, 1)) + [20, 5, 8]  # leafy ball
    kept = footprints.planar_points(np.vstack([roof, crown]), voxel_m=0.3, k=30, max_surface_variation=0.03)
    on_roof = (kept[:, 0] <= 10).sum()
    assert on_roof > 0.8 * len(np.unique(np.floor(roof / 0.3), axis=0))
    assert (kept[:, 0] > 10).sum() < 0.1 * on_roof


def test_no_osm_buildings_gives_empty_frame_not_error(monkeypatch):
    monkeypatch.setattr(footprints, "_overpass", lambda *a, **k: ([], "https://example.invalid"))
    gdf = footprints.fetch_osm((0, 0, 1, 1), ["https://example.invalid"])
    assert len(gdf) == 0 and gdf.crs.to_epsg() == 4326


def test_geo_txt_autodetect_requires_projection_header(tmp_path):
    (tmp_path / "images").mkdir()
    img = tmp_path / "images" / "a.jpg"
    img.write_bytes(b"")
    (tmp_path / "images" / "geo.txt").write_text("a.jpg 1 2 3\n")
    assert ingest.find_geo(tmp_path, [img]) is None
    (tmp_path / "images" / "geo.txt").write_text("EPSG:32633\na.jpg 498017.8 4379490.5 190.2\n")
    assert ingest.find_geo(tmp_path, [img]) == tmp_path / "images" / "geo.txt"


def test_osm_outage_degrades_and_successes_are_cached(tmp_path, monkeypatch):
    fcfg = {"osm_overpass_urls": ["https://example.invalid"], "osm_retries_per_endpoint": 1, "osm_timeout_s": 1}

    def down(*a, **k):
        raise footprints.FootprintError("All Overpass endpoints failed")
    monkeypatch.setattr(footprints, "fetch_osm", down)
    osm, note = footprints.osm_cached((0, 0, 1, 1), fcfg, tmp_path)
    assert len(osm) == 0 and note.startswith("unavailable") and not list(tmp_path.glob("*.gpkg"))

    calls = []

    def up(*a, **k):
        calls.append(1)
        g = gpd.GeoDataFrame({"osm_id": ["w1"]}, geometry=[box(0.1, 0.1, 0.2, 0.2)], crs="EPSG:4326")
        g.attrs["endpoint"] = "https://example.invalid"
        return g
    monkeypatch.setattr(footprints, "fetch_osm", up)
    first, _ = footprints.osm_cached((0, 0, 1, 1), fcfg, tmp_path)
    second, note2 = footprints.osm_cached((0, 0, 1, 1), fcfg, tmp_path)
    assert len(first) == len(second) == 1 and len(calls) == 1 and note2.startswith("cache:")


def test_gcp_autodetect_requires_odm_header(tmp_path):
    (tmp_path / "images").mkdir()
    img = tmp_path / "images" / "a.jpg"
    img.write_bytes(b"")
    (tmp_path / "notes_gcp.txt").write_text("not a gcp file\n")
    assert ingest.find_gcp(tmp_path, [img]) is None
    (tmp_path / "gcp_list.txt").write_text("+proj=utm +zone=17 +datum=WGS84\n1 2 3 4 5 a.jpg\n")
    assert ingest.find_gcp(tmp_path, [img]) == tmp_path / "gcp_list.txt"


def test_orthogonalise_squares_a_ragged_rotated_l_shape():
    import numpy as np
    from shapely.affinity import rotate
    from shapely.geometry import Polygon

    rng = np.random.default_rng(0)
    l_shape = Polygon([(0, 0), (20, 0), (20, 8), (8, 8), (8, 16), (0, 16)])
    # ragged outline: densify and jitter the vertices by up to 0.4 m, then rotate 30 deg
    ring = np.asarray(l_shape.exterior.segmentize(1.0).coords)[:-1]
    ragged = rotate(Polygon(ring + rng.uniform(-0.4, 0.4, ring.shape)), 30, origin=(0, 0))
    out = footprints.orthogonalise(ragged, 1.5, 0.75)
    edges = np.diff(np.asarray(out.exterior.coords), axis=0)
    ang = np.degrees(np.arctan2(edges[:, 1], edges[:, 0])) % 90
    assert np.all(np.minimum(np.abs(ang - 30), 90 - np.abs(ang - 30)) < 1.0)  # every edge along 30 / 120 deg
    assert 4 <= len(edges) <= 8                                              # L-shape: 6 corners, not hundreds
    target = rotate(l_shape, 30, origin=(0, 0))
    assert out.intersection(target).area / out.union(target).area > 0.9


def test_regularise_all_keeps_osm_outlines():
    import geopandas as gpd
    from shapely.geometry import Polygon

    tri = Polygon([(0, 0), (10, 0), (0, 10)])
    fp = gpd.GeoDataFrame({"source": ["osm", "pointcloud"]}, geometry=[tri, tri], crs="EPSG:32633")
    out = footprints.regularise_all(fp, {"step_m": 1.5, "min_iou": 0.75})
    assert out.geometry[0].equals(tri)


def test_pick_lod_uses_lod22_only_for_well_fitted_roofs():
    from uav3d import export

    assert export.pick_lod({"rmse_lod22_m": 0.2}, "1.2", 0.3)[0] == "1.2"
    assert export.pick_lod({"rmse_lod22_m": 0.6}, "2.2", 0.3)[0] == "2.2"
    assert export.pick_lod({"rmse_lod22_m": 0.2}, "auto", 0.3)[0] == "2.2"
    assert export.pick_lod({"rmse_lod22_m": 0.6}, "auto", 0.3)[0] == "1.2"
    assert export.pick_lod({"rmse_lod22_m": None}, "auto", 0.3)[0] == "1.2"


def test_roofer_run_clears_outputs_of_an_earlier_run(tmp_path, monkeypatch):
    from uav3d import buildings, plan

    out = tmp_path / "roofer"
    out.mkdir()
    (out / "1_2.city.jsonl").write_text("old", encoding="utf-8")
    monkeypatch.setattr(buildings, "run_measured", lambda cmd, log, env=None: {"returncode": 0})
    monkeypatch.setattr(buildings, "srs_of", lambda p: None)
    monkeypatch.setattr(buildings, "summarise", lambda d, rcfg: sorted(p.name for p in d.glob("*.city.jsonl")))
    r = buildings.run(tmp_path / "roofer.exe", tmp_path / "c.laz", tmp_path / "r.gpkg", out, None, plan.defaults(),
                      tmp_path / "l.log")
    assert r["summary"] == []


def test_assess_flags_ragged_traced_outlines_only():
    from uav3d import buildings, plan

    rcfg = plan.defaults()["roofer"]
    base = {"rf_h_ground": 0.0, "rf_h_roof_70p": 6.0, "rf_rmse_lod22": 0.3}
    assert buildings._assess({**base, "source": "pointcloud", "rectangularity": 0.6}, rcfg)["footprint_irregular"]
    assert not buildings._assess({**base, "source": "pointcloud", "rectangularity": 1.0}, rcfg)["footprint_irregular"]
    assert not buildings._assess({**base, "source": "osm", "rectangularity": 0.6}, rcfg)["footprint_irregular"]


def test_assess_flags_a_tower_above_the_roof():
    from uav3d import buildings, plan

    rcfg = plan.defaults()["roofer"]
    base = {"rf_h_ground": 0.0, "rf_h_roof_70p": 6.0, "rf_rmse_lod22": 0.3}
    assert buildings._assess({**base, "rf_h_roof_max": 13.0}, rcfg)["roof_spike"]
    assert not buildings._assess({**base, "rf_h_roof_max": 7.0}, rcfg)["roof_spike"]
