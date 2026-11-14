"""Shots observing each tile of a run's tile grid (cost input of the scheduler).

usage: python bench/tile_shots.py <run dir> [--size 25] [--k 50 200 400]
Grid: same as bench/tile_gains.py (anchored at the coarse cloud's min XY, UTM). Tiles are shifted into ODM's
reconstruction frame with the coords.txt offset. Writes <run>/tile_shots.json:
{"size", "origin", "shape", "k": {k: {tile id: [shots]}}, "stats": {k: {...}}}
"""
import argparse
import json
import math
from pathlib import Path

import laspy
import numpy as np

from uav3d import refine, tools


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--size", type=float, default=25.0)
    ap.add_argument("--k", type=int, nargs="+", default=[50, 200, 400])
    a = ap.parse_args()
    project = a.run / "odm" / "project"
    h = laspy.open(project / "odm_georeferencing" / "odm_georeferenced_model.laz").header
    x0, y0, x1, y1 = h.mins[0], h.mins[1], h.maxs[0], h.maxs[1]
    shape = (math.ceil((y1 - y0) / a.size), math.ceil((x1 - x0) / a.size))
    ox, oy = refine.utm_offset(project)
    tiles = refine.grid(x0 - ox, y0 - oy, x0 - ox + shape[1] * a.size, y0 - oy + shape[0] * a.size, a.size)
    odm = tools.odm()
    out = {"size": a.size, "origin": [x0, y0], "shape": shape, "k": {}, "stats": {}}
    for k in a.k:
        work = a.run / "tile_shots" / f"k{k}"
        r = refine.subsets(odm, project, tiles, k, work, work / "subsets.log")
        per = {tid: v["shots"] for tid, v in r["tiles"].items()}
        n = np.array([len(v) for v in per.values()])
        nz = n[n > 0]
        out["k"][str(k)] = per
        out["stats"][str(k)] = {"shots_total": r["shots_total"], "tiles_with_shots": int(len(nz)),
                                "median": float(np.median(nz)) if len(nz) else 0,
                                "p10": float(np.percentile(nz, 10)) if len(nz) else 0,
                                "p90": float(np.percentile(nz, 90)) if len(nz) else 0,
                                "wall_s": r["measure"]["wall_s"]}
        print(k, out["stats"][str(k)], flush=True)
    (a.run / "tile_shots.json").write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
