"""Verify that rejected foundation trials cannot modify the published surface."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


GIS = all(importlib.util.find_spec(name) for name in ("geopandas", "rasterio", "yaml"))
pytestmark = pytest.mark.skipif(not GIS, reason="optional GIS runtime required")
if GIS:
    import geopandas as gpd
    from rasterio.transform import from_origin
    from shapely.geometry import box, mapping

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from omega_desktop import audit_ground, raster
    import omega_uav_publish as publish


def test_raster_rejection_after_map_change_removes_previously_approved_foundation(tmp_path, monkeypatch):
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "grid.json").write_text('{"bounds": [0, 0, 30, 30], "crs": "EPSG:32617"}')
    (scene / "terrain_photo.png").touch()
    original = tmp_path / "original"
    proposals = gpd.GeoDataFrame(geometry=[box(x, 0, x + 2, 2) for x in (0, 10, 20)], crs=32617)
    rounds = []

    def prepare(base, dense, candidates, output, *args, **kwargs):
        assert base == original
        assert kwargs["preserve_observed"] and kwargs["preserve_unfilled"]
        rounds.append(list(candidates.building_id))
        output.mkdir()

    def heights(adapter, trial, ortho, measured):
        eligible = {"Building_0001"} if len(rounds) > 1 else set(rounds[-1])
        features = [{"properties": {"building_id": name, "status": (
            "eligible" if name in eligible else "insufficient_raster_support"
        )}} for name in rounds[-1]]
        (trial / "building_footprints_projected.json").write_text(json.dumps({"features": features}))

    def assemble(run, blender_args, trial, mesh, *, validate):
        assert validate
        return [{"building_id": feature["properties"]["building_id"], "status": (
            "geometry_rejected" if feature["properties"]["building_id"] == "Building_0003" else "built"
        )} for feature in publish.read(trial / "building_footprints_projected.json")["features"]
            if feature["properties"]["status"] == "eligible"]

    monkeypatch.setattr(publish, "prepare", prepare)
    monkeypatch.setattr(publish, "building_heights", heights)
    monkeypatch.setattr(publish, "assemble_scene", assemble)
    monkeypatch.setattr(publish, "export_dem", lambda *args: None)
    run = SimpleNamespace(output=tmp_path, measured=None)
    ground, mesh, assessments, decisions = publish.stage_foundations(
        run, SimpleNamespace(pdal=None), [], original, None, None, proposals, scene,
    )
    assert rounds == [
        ["Building_0001", "Building_0002", "Building_0003"],
        ["Building_0001", "Building_0002"],
        ["Building_0001"],
    ]
    assert ground.parent == mesh.parent
    assert publish.built_ids(decisions) == {"Building_0001"}
    assert [item["properties"]["status"] for item in assessments] == [
        "eligible", "insufficient_raster_support", "geometry_rejected",
    ]


def test_published_dtm_rejects_unapproved_and_observed_changes(tmp_path):
    base, final = tmp_path / "base", tmp_path / "final"
    base.mkdir()
    final.mkdir()
    grid = dict(crs="EPSG:32617", transform=from_origin(0, 8, 1, 1), height=8, width=12)
    terrain = np.zeros((8, 12), dtype="float32")
    observed = np.zeros_like(terrain, dtype="uint8")
    observed[4, 1] = 1
    raster(base / "dtm.tif", terrain, grid, nodata=np.nan)
    raster(base / "observed.tif", observed, grid)
    features = [
        dict(properties=dict(building_id=identifier), geometry=mapping(box(x, 2, x + 3, 6)))
        for identifier, x in (("accepted", 0), ("rejected", 8))
    ]
    solids = [dict(building_id="accepted", status="built")]
    terrain[4, 2] = 1
    raster(final / "dtm.tif", terrain, grid, nodata=np.nan)
    assert audit_ground(base, final, tmp_path, features, solids)["changed_cells"] == 1
    terrain[4, 1] = 1
    raster(final / "dtm.tif", terrain, grid, nodata=np.nan)
    with pytest.raises(ValueError, match="directly observed"):
        audit_ground(base, final, tmp_path, features, solids)
    terrain[4, 1] = 0
    terrain[4, 9] = 1
    raster(final / "dtm.tif", terrain, grid, nodata=np.nan)
    with pytest.raises(ValueError, match="outside approved"):
        audit_ground(base, final, tmp_path, features, solids)
