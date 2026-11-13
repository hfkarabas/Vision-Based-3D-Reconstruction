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


def srs_of(dsm_path) -> str | None:
    with rasterio.open(dsm_path) as ds:
        epsg = ds.crs.to_epsg() if ds.crs else None
    return f"EPSG:{epsg}" if epsg else None


def run(roofer: Path, point_cloud: Path, roofprints: Path, out_dir: Path, dsm_path, cfg: dict, log: Path) -> dict:
    rcfg = cfg["roofer"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(roofer), *rcfg["lods"], "--bld-class", str(rcfg["building_class"]),
           "--grnd-class", str(rcfg["ground_class"]), "--id-attribute", "fp_id",
           "-j", str(os.cpu_count() or 1)]
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
    return {"height_m": height, "rmse_lod22_m": None if rmse is None else round(rmse, 3),
            "roof_planes": attrs.get("rf_roof_planes"), "status": status}


def summarise(out_dir: Path, rcfg: dict) -> dict:
    """Count buildings, LoDs and extrusion modes; assess each building (see _assess)."""
    files = sorted(out_dir.rglob("*.city.jsonl"))
    if not files:
        raise BuildingError(f"Roofer produced no CityJSONSeq output in {out_dir}")
    lods, modes, per_building = Counter(), Counter(), {}
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            for key, co in obj.get("CityObjects", {}).items():
                attrs = co.get("attributes", {})
                if co.get("type") == "Building":
                    modes[str(attrs.get("rf_extrusion_mode", "?"))] += 1
                    per_building[key] = {"osm_id": attrs.get("osm_id"), "source": attrs.get("source"),
                                         **_assess(attrs, rcfg)}
                for g in co.get("geometry", []):
                    lods[str(g.get("lod"))] += 1
    status = Counter(b["status"] for b in per_building.values())
    return {"files": [str(f) for f in files], "buildings": len(per_building),
            "by_status": dict(status), "geometries_by_lod": dict(lods),
            "extrusion_modes": dict(modes), "per_building": per_building}
