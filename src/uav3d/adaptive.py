"""Quality-feedback regional refinement (the research method) as a stage of ``uav3d run``.

After ODM's uniform dense pass (the coarse pass, at the level OpenMVS actually used), the block is cut into
square tiles; reference-free signals of each tile (uav3d.signals) give a predicted gain with the frozen gain
model (configs/gain_model.json, fitted on the tuning block only); tiles are chosen greedily by predicted gain
per new image under an image budget (uav3d.schedule.select); each 4-connected group of chosen tiles is
densified one level finer from its own image subset (same SfM, same camera model; uav3d.refine), classified
with ODM's OpenPointClass model, and replaces the coarse points inside its tiles. Later stages (terrain,
buildings, export) run unchanged on the merged cloud.

The stage needs ODM's intermediates (undistorted images, OpenMVS depth maps) of a run without split-merge;
otherwise it records why it was skipped and the run continues on the coarse cloud.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import laspy
import numpy as np

from uav3d import refine, schedule, signals
from uav3d.record import run_measured

REPO = Path(__file__).resolve().parents[2]
SIGNALS = ("overlap", "void", "density", "roughness", "relief")


class Skip(Exception):
    """Refinement is not possible for this run; the reason is recorded, the run goes on."""


def coarse_side(project: Path) -> int | None:
    """Image side (px) of ODM's own dense pass, read from one of its OpenMVS depth maps."""
    d = project / "opensfm" / "undistorted" / "openmvs" / "depthmaps"
    first = next(iter(sorted(d.glob("*.dmap"))), None) if d.is_dir() else None
    return max(refine.dmap_size(first)) if first else None


def levels(max_size: int, side: int) -> tuple[int, int]:
    """(coarse level, fine level): the coarse level is the one that gave `side`; fine is one level finer."""
    lc = max(0, round(math.log2(max_size / side)))
    if lc == 0:
        raise Skip(f"coarse pass already at full resolution ({side} px)")
    return lc, lc - 1


def opc_model() -> Path | None:
    """ODM's OpenPointClass model (ODM downloads it into its model store on its first classified run)."""
    for base in (Path(os.environ.get("PROGRAMDATA", "")) / "ODM", REPO / ".tools" / "odm"):
        p = base / "storage" / "models" / "openpointclass" / "v1_0_0" / "model.bin"
        if p.is_file():
            return p
    return None


def load_model(cfg: dict) -> schedule.GainModel:
    p = Path(cfg["adaptive"]["gain_model"])
    p = p if p.is_absolute() else REPO / "configs" / p
    return schedule.GainModel.from_dict(json.loads(p.read_text(encoding="utf-8")))


def _pdal(odm, pipeline: list, path: Path, log: Path) -> dict:
    path.write_text(json.dumps(pipeline, indent=1), encoding="utf-8")
    info = run_measured([str(odm.exe("pdal")), "pipeline", str(path)], log, env=odm.env())
    if info["returncode"] != 0:
        raise refine.RefineError(f"PDAL {path.name} failed (exit {info['returncode']}); see {log}")
    return {"wall_s": info["wall_s"], "peak_rss_gb": info["peak_rss_gb"]}


def classify(odm, laz: Path, out: Path, log: Path, n_threads: int) -> dict:
    """ODM's own classification call (opendm/opc.py): OpenPointClass on unclassified points only."""
    model = opc_model()
    if model is None:
        shutil.copy2(laz, out)
        return {"skipped": "OpenPointClass model not found; refined points stay unclassified"}
    env = {**odm.env(), "OMP_NUM_THREADS": str(n_threads)}
    info = run_measured([str(odm.exe("pcclassify")), str(laz), str(out), str(model), "-u", "-s", "2,64"],
                        log, env=env, cpus=n_threads)
    if info["returncode"] != 0 or not out.is_file():
        raise refine.RefineError(f"pcclassify failed (exit {info['returncode']}); see {log}")
    return {"wall_s": info["wall_s"], "peak_rss_gb": info["peak_rss_gb"]}


def replace(odm, coarse_laz: str, refined_laz: Path, tiles: list[refine.Tile], project: Path, out_laz: Path,
            log: Path) -> dict:
    """Coarse points outside the refined tiles + the refined points (both in UTM) -> one LAZ."""
    ox, oy = refine.utm_offset(project)
    srs = refine.utm_srs(project)
    kept = out_laz.with_name(out_laz.stem + ".coarse_kept.laz")
    dropped = refine.drop_tiles(coarse_laz, tiles, (ox, oy), kept)
    pipe = [{"type": "readers.las", "filename": str(kept), "override_srs": srs, "tag": "keep"},
            {"type": "readers.las", "filename": str(refined_laz), "override_srs": srs, "tag": "fine"},
            {"type": "filters.merge", "inputs": ["keep", "fine"], "tag": "all"},
            {"type": "writers.las", "inputs": ["all"], "filename": str(out_laz), "compression": "laszip",
             "minor_version": 4, "dataformat_id": 3, "offset_x": ox, "offset_y": oy, "offset_z": 0,
             "scale_x": 0.001, "scale_y": 0.001, "scale_z": 0.001, "extra_dims": "all"}]
    info = _pdal(odm, pipe, out_laz.with_suffix(".pipeline.json"), log)
    kept.unlink()
    return {**info, "coarse": dropped}


def run(odm, project: Path, coarse_laz: str, gsd_m: float | None, budget: float, cfg: dict, work: Path,
        log: Path, n_threads: int, rec: dict, save=lambda: None) -> str | None:
    """Refine the block in place of the coarse cloud; returns the merged LAZ, or None if nothing was refined.
    Raises Skip when refinement is not possible (recorded by the caller)."""
    acfg = cfg["adaptive"]
    if not 0 < budget <= 1:
        raise ValueError("--refine-budget is a fraction of the block's images in (0, 1]")
    if gsd_m is None:
        raise Skip("no metric GSD (unreferenced reconstruction): tiles cannot be sized")
    if not (project / "odm_georeferencing" / "coords.txt").is_file():
        raise Skip("no odm_georeferencing/coords.txt (unreferenced reconstruction)")
    side = coarse_side(project)
    try:
        max_size = refine.undistorted_max_size(project)
    except (refine.RefineError, OSError):
        max_size = None
    if side is None or max_size is None:
        raise Skip("ODM intermediates missing (undistorted images / OpenMVS depth maps); "
                   "rerun ODM without --reuse-odm on a pruned project")
    lc, lf = levels(max_size, side)
    fine_level, fine_side = refine.effective_size(max_size, lf)
    if fine_level != lf:
        raise Skip(f"fine level {lf} is clamped by OpenMVS --min-resolution")
    rec.update(coarse_level=lc, coarse_side_px=side, fine_level=lf, fine_side_px=fine_side, budget=budget)
    work.mkdir(parents=True, exist_ok=True)

    # tiles (UTM grid anchored at the coarse cloud) and their image subsets
    size = float(acfg["tile_m"])
    h = laspy.open(coarse_laz).header
    x0, y0, x1, y1 = h.mins[0], h.mins[1], h.maxs[0], h.maxs[1]
    shape = (math.ceil((y1 - y0) / size), math.ceil((x1 - x0) / size))
    ox, oy = refine.utm_offset(project)
    # ids iy * nx + ix, as in signals.tile_signals (built from `shape` to avoid float round-off in the count)
    tiles = [refine.Tile(iy * shape[1] + ix, ix, iy, x0 - ox + ix * size, y0 - oy + iy * size,
                         x0 - ox + (ix + 1) * size, y0 - oy + (iy + 1) * size)
             for iy in range(shape[0]) for ix in range(shape[1])]
    k = int(acfg["k"])
    sub = refine.subsets(odm, project, tiles, k, work / "shots", log)
    shots = [set(sub["tiles"].get(str(t.id), {}).get("shots", [])) for t in tiles]
    total = len(set().union(*shots)) if shots else 0
    rec.update(tiles=len(tiles), tile_m=size, k=k, images_total=total, shots_wall_s=sub["measure"]["wall_s"])

    # reference-free signals on the coarse pass and predicted gains
    dsm = work / "coarse_dsm.tif"
    if not dsm.is_file():
        signals.dsm(odm, coarse_laz, dsm, signals.dsm_cell(gsd_m, max_size, side), log)
    overlap = project / "opensfm" / "stats" / "overlap.tif"
    sig = signals.tile_signals(dsm, overlap if overlap.is_file() else None, coarse_laz, (x0, y0), size, shape)
    model = load_model(cfg)
    pred = model.predict([{s: r.get(s) for s in SIGNALS} for r in sig])
    pred = np.where(np.isfinite(pred), pred, 0.0)
    sel = schedule.select(pred, shots, int(round(budget * total)))
    rec.update(selected=[int(i) for i in sel.order], images_budget=int(round(budget * total)),
               images_selected=sel.images[-1] if sel.images else 0,
               predicted_gain=round(float(sel.gain[-1]), 5) if sel.gain else 0.0,
               tiles_positive=int((pred > 0).sum()))
    save()
    if not sel.order:
        rec["result"] = "no tile with positive predicted gain"
        return None

    # densify each group of adjacent chosen tiles from its own images, one level finer
    chosen = [tiles[i] for i in sel.order]
    grp = refine.groups(chosen)
    rec["groups"] = [[t.id for t in g] for g in grp]
    if "export" not in rec:
        ex = refine.subsets(odm, project, chosen, k, work / "export", log,
                            export=[(f"g{j:03d}", g) for j, g in enumerate(grp)])
        rec["export"] = [{"name": g["name"], "shots": len(g["shots"]), "scene": g["scene"]} for g in ex["groups"]]
        save()
    rec.setdefault("densify", {})
    refined = []
    for g, tiles_g in zip(rec["export"], grp):
        d = rec["densify"].setdefault(g["name"], {})
        refine.densify_lowmem(odm, Path(g["scene"]), work / g["name"], lf, max_size, n_threads, log,
                              int(acfg["sub_scene_area_px"]), d, save)
        save()
        refined += [(p, tiles_g) for p in d["plys"]]

    fine_only = work / "refined_points.laz"
    rec["fine_points"] = refine.merge(odm, None, refined, project, fine_only, log)
    fine_cls = work / "refined_classified.laz"
    rec["classify"] = classify(odm, fine_only, fine_cls, log, n_threads)
    out = work / "point_cloud.laz"
    rec["replace"] = replace(odm, coarse_laz, fine_cls, chosen, project, out, log)
    rec["peak_rss_gb"] = max([v.get("peak_rss_gb", 0) for d in rec["densify"].values()
                              for v in [d.get("depthmaps", {}), *[x["fuse"] for x in d.get("subscenes", {}).values()]]]
                             or [0])
    rec["wall_s"] = round(sum(d.get("depthmaps", {}).get("wall_s", 0) + d.get("split", {}).get("wall_s", 0)
                              + sum(v["fuse"]["wall_s"] + v["filter"]["wall_s"] for v in d.get("subscenes", {}).values())
                              for d in rec["densify"].values()), 1)
    rec["result"] = "refined"
    save()
    return str(out)


def prune(work: Path) -> dict:
    """Delete the stage's regenerable OpenMVS work (depth maps, scenes, PLYs); keep the merged cloud,
    the signals DSM and the records."""
    freed = 0
    for p in sorted(work.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if p.is_file() and (p.suffix.lower() in (".dmap", ".mvs", ".ply") or "depthmaps" in p.parts
                            or p.name in ("refined_points.laz", "refined_classified.laz")):
            freed += p.stat().st_size
            p.unlink()
        elif p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    return {"freed_gb": round(freed / 2**30, 2)}
