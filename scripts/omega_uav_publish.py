"""Validate temporary foundation surfaces before publishing the scene terrain."""

from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from omega_uav_foundation_trial import prepare
from omega_uav_products import building_heights, read


def built_ids(records):
    return {item["building_id"] for item in records if item["status"] == "built"}


def export_dem(run, local, ground, mesh):
    run.command([sys.executable, Path(__file__).with_name("omega_qgis_dem.py"), "--dtm",
                 ground / "dtm.tif", "--plugin", local.qgis, "--node", local.node, "--output", mesh])


def assemble_scene(run, blender_args, scene, mesh, *, validate=False):
    run.command([*blender_args, Path(__file__).with_name("omega_building_blender.py"), "--",
                 "--output", scene, "--dem-mesh", mesh / "terrain.ply", "--dem-manifest",
                 mesh / "grid.json", "--allow-terrain-only", "--terrain-collection",
                 *(["--validate-only"] if validate else [])])
    return read(scene / "reports/blender_buildings.json")


def stage_foundations(run, local, blender_args, base, ortho, dense, candidates, scene):
    """Keep a monotonically shrinking set of candidates until the surface and solids agree."""
    root = run.output / "foundation-trials"
    root.mkdir()
    settings = read(scene / "grid.json")
    candidates = candidates.copy()
    candidates["building_id"] = [f"Building_{i:04d}" for i in range(1, len(candidates) + 1)]
    assessments = {}
    decisions = {}
    for number in range(len(candidates) + 1):
        if candidates.empty:
            return base, None, list(assessments.values()), list(decisions.values())
        trial = root / f"round-{number:03d}"
        trial.mkdir()
        prepared = trial / "ground"
        prepare(base, dense, candidates, prepared, 8, surrounded=True, preserve_observed=True,
                preserve_unfilled=True)
        candidates.to_file(trial / "geoai_polygons.gpkg", driver="GPKG", index=False)
        adapter = SimpleNamespace(output=trial, extent=settings["bounds"], crs=settings["crs"],
                                  pdal=local.pdal)
        building_heights(adapter, trial, ortho, run.measured)
        features = read(trial / "building_footprints_projected.json")["features"]
        for filename in ("grid.json", "terrain_photo.png"):
            shutil.copy2(scene / filename, trial / filename)
        mesh = trial / "mesh"
        export_dem(run, local, prepared, mesh)
        results = assemble_scene(run, blender_args, trial, mesh, validate=True)
        approved = built_ids(results)
        blender_results = {result["building_id"]: result for result in results}
        for feature in features:
            properties = feature["properties"]
            identifier = properties["building_id"]
            decision = blender_results.get(identifier, {"building_id": identifier, "status": properties["status"]})
            if properties["status"] == "eligible" and identifier not in blender_results:
                raise RuntimeError(f"Blender omitted foundation validation for {identifier}.")
            if identifier not in approved:
                properties["status"] = decision["status"]
            assessments[identifier] = feature
            decisions[identifier] = decision
        survivors = candidates["building_id"].isin(approved)
        if survivors.all():
            return prepared, mesh, list(assessments.values()), list(decisions.values())
        # An excluded footprint no longer influences anchors, the interpolation
        # mask or roof/ground measurements in any later round.
        candidates = candidates.loc[survivors]
    raise RuntimeError("Foundation verification did not reach a stable candidate set.")
