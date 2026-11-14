"""End-to-end check of uav3d.refine on a kept ODM project: one tile -> image subset -> densify -> merge,
then the vertical agreement of refined and coarse points inside the tile (same frame => ~0).

usage: python bench/feasibility/refine_smoke.py <odm project> <work dir> <cx> <cy> <size> <k> <level> [threads]
"""
import json
import sys
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

from uav3d import refine, tools

project, work = Path(sys.argv[1]), Path(sys.argv[2])
cx, cy, size = map(float, sys.argv[3:6])
k, level = int(sys.argv[6]), int(sys.argv[7])
threads = int(sys.argv[8]) if len(sys.argv) > 8 else 2
o = tools.odm()
tile = refine.Tile(0, 0, 0, cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)
log = work / "refine.log"
sub = refine.subsets(o, project, [tile], k, work, log, export=[("g0", [tile])])
g = sub["groups"][0]
print("subset:", len(g["shots"]), "of", sub["shots_total"], "shots;", sub["tiles"]["0"]["points"], "sparse points")
d = refine.densify(o, Path(g["scene"]), level, refine.undistorted_max_size(project), threads, log)
print("densify:", json.dumps(d))
coarse = project / "odm_georeferencing" / "odm_georeferenced_model.laz"
m = refine.merge(o, str(coarse), [(d["ply"], [tile])], project, work / "merged.laz", log)
print("merge:", json.dumps(m))

ox, oy = refine.utm_offset(project)
x0, y0, x1, y1 = tile.x0 + ox, tile.y0 + oy, tile.x1 + ox, tile.y1 + oy
c = laspy.read(coarse)
cin = (c.x >= x0) & (c.x < x1) & (c.y >= y0) & (c.y < y1)
r = laspy.read(work / "merged.laz")
rin = (r.x >= x0) & (r.x < x1) & (r.y >= y0) & (r.y < y1)
cp = np.c_[c.x[cin], c.y[cin], c.z[cin]]
rp = np.c_[r.x[rin], r.y[rin], r.z[rin]]
dist, idx = cKDTree(cp[:, :2]).query(rp[:, :2])
dz = rp[:, 2] - cp[idx, 2]
print(json.dumps({"coarse_in_tile": int(cin.sum()), "refined_in_tile": int(rin.sum()),
                  "merged_total": int(len(r.x)), "coarse_total": int(len(c.x)),
                  "xy_nn_median_m": float(np.median(dist)), "dz_median_m": float(np.median(dz)),
                  "dz_abs_p90_m": float(np.percentile(np.abs(dz), 90))}))
