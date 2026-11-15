"""LoD1.2/LoD2.2 buildings with Roofer (3DBAG) from the ODM point cloud and roofprints.

ODM 3.6 classifies its point cloud with ASPRS codes (2 ground, 3 vegetation, 6 building),
matching Roofer's defaults; all Roofer parameters stay at their published defaults.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import rasterio

from uav3d.record import run_measured


class BuildingError(RuntimeError):
    pass


def relabel(laz_in: Path, roofprints: Path, dtm_path, out_laz: Path, min_above_m: float, building_class: int) -> dict:
    """Mark as building every point inside an accepted roofprint that stands at least
    min_above_m above the DTM, so Roofer does not depend on the point classifier (which
    calls tower roofs ground or terraces buildings on unfamiliar scenes)."""
    import geopandas as gpd
    import laspy
    import numpy as np
    from rasterio.features import rasterize

    fp = gpd.read_file(roofprints)
    las = laspy.read(laz_in)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    cls = np.asarray(las.classification).copy()
    with rasterio.open(dtm_path) as ds:
        dtm = ds.read(1, masked=True)
        inside = rasterize([(g, 1) for g in fp.to_crs(ds.crs).geometry], out_shape=ds.shape,
                           transform=ds.transform, fill=0, dtype="uint8").astype(bool)
        t = ds.transform
    col = np.floor((x - t.c) / t.a).astype(np.int64).clip(0, dtm.shape[1] - 1)
    row = np.floor((y - t.f) / t.e).astype(np.int64).clip(0, dtm.shape[0] - 1)
    ground = dtm.filled(np.nan)[row, col]
    above = (z - ground) >= min_above_m
    building = inside[row, col] & above & (cls != 7)
    before = int((cls == building_class).sum())
    cls[(cls == building_class) & ~inside[row, col]] = 1   # outside every accepted roofprint: not a building
    cls[building] = building_class
    las.classification = cls
    las.write(out_laz)
    return {"building_points_before": before, "building_points_after": int((cls == building_class).sum())}


def srs_of(dsm_path) -> str | None:
    with rasterio.open(dsm_path) as ds:
        epsg = ds.crs.to_epsg() if ds.crs else None
    return f"EPSG:{epsg}" if epsg else None


def run(roofer: Path, point_cloud: Path, roofprints: Path, out_dir: Path, dsm_path, cfg: dict, log: Path,
        n_threads: int | None = None) -> dict:
    rcfg = cfg["roofer"]
    out_dir.mkdir(parents=True, exist_ok=True)
    # Roofer names its output after the tile it computes; a rerun on changed roofprints can produce a
    # different name, and summarise() reads every *.city.jsonl here, so outputs of an earlier run must go
    for stale in out_dir.glob("*.city.jsonl"):
        stale.unlink()
    cmd = [str(roofer), *rcfg["lods"], "--bld-class", str(rcfg["building_class"]),
           "--grnd-class", str(rcfg["ground_class"]), "--id-attribute", "fp_id",
           "-j", str(n_threads or os.cpu_count() or 1)]
    if srs := srs_of(dsm_path):
        cmd += ["--srs", srs]
    cmd += [str(point_cloud), str(roofprints), str(out_dir)]
    # Roofer ships its own GDAL/PROJ data next to the executable.
    env = os.environ.copy()
    share = roofer.parent.parent / "share"
    env.update(GDAL_DATA=str(share / "gdal"), PROJ_DATA=str(share / "proj"), PROJ_LIB=str(share / "proj"))
    result = run_measured(cmd, log, env=env)
    result["command"] = cmd
    if result["returncode"] != 0:
        raise BuildingError(f"Roofer exited with {result['returncode']}; see {log}")
    result["summary"] = summarise(out_dir, rcfg)
    return result


def _assess(attrs: dict, rcfg: dict) -> dict:
    """Reference-free per-building quality from Roofer's own attributes.

    height: 70th-percentile roof height above Roofer's ground estimate (building points only);
    rmse: LoD2.2 fit error. Below min_roof_height_m -> not a building (excluded from export);
    rmse above low_confidence_rmse_m -> kept but flagged (trees merged into a roofprint, etc.)."""
    ground = attrs.get("rf_h_ground")
    h70 = attrs.get("rf_h_roof_70p")
    height = round(h70 - ground, 2) if h70 is not None and ground is not None else None
    rmse = attrs.get("rf_rmse_lod22")
    if attrs.get("rf_extrusion_mode") == "skip":
        status = "skipped"
    elif height is None or height < rcfg["min_roof_height_m"]:
        status = "too_low"
    elif rmse is not None and rmse > rcfg["low_confidence_rmse_m"]:
        status = "low_confidence"
    else:
        status = "ok"
    rect = attrs.get("rectangularity")
    # traced (non-OSM) outline far from rectangular (recorded only: most traced outlines are ragged)
    irregular = (attrs.get("source") not in (None, "osm") and rect is not None
                 and rect < rcfg.get("irregular_footprint_rectangularity", 0.8))
    # top of the modelled roof far above the 70th-percentile roof height: a tower in the model, typically
    # a tree crown over the roof that LoD2.2 extrudes as a separate roof part
    h_max = attrs.get("lod22_top", attrs.get("rf_h_roof_max"))
    if h_max is None:
        h_max = attrs.get("rf_h_roof_max")
    spike = round(h_max - h70, 2) if h_max is not None and h70 is not None else None
    return {"height_m": height, "rmse_lod22_m": None if rmse is None else round(rmse, 3),
            "roof_planes": attrs.get("rf_roof_planes"), "rectangularity": rect,
            "footprint_irregular": bool(irregular), "roof_spike_m": spike,
            "roof_spike": bool(spike is not None and spike > rcfg.get("roof_spike_m", 3.0)), "status": status}


def _indices(boundaries):
    if isinstance(boundaries, int):
        yield boundaries
    else:
        for b in boundaries:
            yield from _indices(b)


def summarise(out_dir: Path, rcfg: dict) -> dict:
    """Count buildings, LoDs and extrusion modes; assess each building (see _assess)."""
    files = sorted(out_dir.rglob("*.city.jsonl"))
    if not files:
        raise BuildingError(f"Roofer produced no CityJSONSeq output in {out_dir}")
    lods, modes, per_building = Counter(), Counter(), {}
    for f in files:
        scale, translate = None, None
        for line in f.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            if "transform" in obj:
                scale, translate = obj["transform"]["scale"][2], obj["transform"]["translate"][2]
            zs = obj.get("vertices") or []
            for key, co in obj.get("CityObjects", {}).items():
                attrs = co.get("attributes", {})
                if co.get("type") == "Building":
                    modes[str(attrs.get("rf_extrusion_mode", "?"))] += 1
                    # highest point of the modelled LoD2.2 geometry (Roofer's rf_h_roof_max can miss
                    # parts extruded to tree-crown height)
                    top = None
                    if scale is not None and zs:
                        idx = {i for c in co.get("children", []) for g in obj["CityObjects"].get(c, {}).get("geometry", [])
                               if g.get("lod") == "2.2" for i in _indices(g["boundaries"])}
                        if idx:
                            top = max(zs[i][2] for i in idx) * scale + translate
                    per_building[key] = {"osm_id": attrs.get("osm_id"), "source": attrs.get("source"),
                                         **_assess({**attrs, "lod22_top": top}, rcfg)}
                for g in co.get("geometry", []):
                    lods[str(g.get("lod"))] += 1
    status = Counter(b["status"] for b in per_building.values())
    return {"files": [str(f) for f in files], "buildings": len(per_building),
            "by_status": dict(status),
            "footprint_irregular": sum(b["footprint_irregular"] for b in per_building.values()),
            "roof_spike": sum(b["roof_spike"] for b in per_building.values()),
            "geometries_by_lod": dict(lods),
            "extrusion_modes": dict(modes), "per_building": per_building}
