import json

import geopandas as gpd
import pyogrio
from shapely.geometry import box

from uav3d.evaluate import buildings as ev

CRS = "EPSG:32617"


def make(tmp_path, labels, uncertain, preds, statuses):
    gpkg = tmp_path / "labels.gpkg"
    pyogrio.write_dataframe(gpd.GeoDataFrame({"area_id": [1]}, geometry=[box(0, 0, 100, 100)], crs=CRS), gpkg, layer="area")
    pyogrio.write_dataframe(gpd.GeoDataFrame({"uncertain": uncertain}, geometry=labels, crs=CRS), gpkg, layer="buildings")
    run = tmp_path / "run"
    (run / "buildings").mkdir(parents=True)
    ids = [f"fp{i:05d}" for i in range(len(preds))]
    gpd.GeoDataFrame({"fp_id": ids}, geometry=preds, crs=CRS).to_file(run / "buildings" / "roofprints.gpkg")
    per = {i: {"status": s, "height_m": 5.0} for i, s in zip(ids, statuses)}
    (run / "run.json").write_text(json.dumps({"buildings": {"per_building": per}}))
    return run, gpkg


def test_touching_rectangles_are_one_building_and_counts_are_right(tmp_path):
    labels = [box(10, 10, 20, 20), box(20, 12, 24, 18),   # house + touching porch = 1 building
              box(40, 40, 50, 50),                         # missed
              box(70, 70, 75, 75),                         # uncertain, missed -> not a miss
              box(200, 200, 210, 210)]                     # outside the sample area -> ignored
    preds = [box(10, 10, 23, 20),                          # matches house
             box(60, 10, 70, 20),                          # false positive
             box(80, 80, 90, 90),                          # too_low -> not a prediction
             box(150, 150, 160, 160)]                      # outside area -> ignored
    run, gpkg = make(tmp_path, labels, [0, 0, 0, 1, 0], preds, ["ok", "low_confidence", "too_low", "ok"])
    r = ev.evaluate(run, gpkg)
    assert (r["labels"], r["predictions"], r["tp"], r["fp"], r["fn"]) == (2, 2, 1, 1, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5


def test_merged_prediction_over_two_houses_does_not_count_twice(tmp_path):
    labels = [box(0, 0, 10, 10), box(12, 0, 22, 10)]
    preds = [box(0, 0, 22, 10)]
    run, gpkg = make(tmp_path, labels, [0, 0], preds, ["ok"])
    r = ev.evaluate(run, gpkg)
    assert r["tp"] == 1 and r["fn"] == 1 and r["fp"] == 0
