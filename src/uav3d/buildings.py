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
           "--grnd-class", str(rcfg["ground_class"]), "-j", str(os.cpu_count() or 1)]
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
    result["summary"] = summarise(out_dir)
    return result


def summarise(out_dir: Path) -> dict:
    """Count buildings and the LoDs/extrusion modes Roofer actually produced."""
    files = sorted(out_dir.rglob("*.city.jsonl"))
    buildings, lods, modes = 0, Counter(), Counter()
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            for co in obj.get("CityObjects", {}).values():
                if co.get("type") == "Building":
                    buildings += 1
                    modes[str(co.get("attributes", {}).get("rf_extrusion_mode", "?"))] += 1
                for g in co.get("geometry", []):
                    lods[str(g.get("lod"))] += 1
    if not files:
        raise BuildingError(f"Roofer produced no CityJSONSeq output in {out_dir}")
    return {"files": [str(f) for f in files], "buildings": buildings,
            "geometries_by_lod": dict(lods), "extrusion_modes": dict(modes)}
