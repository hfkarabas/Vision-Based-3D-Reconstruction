"""Fine dense pass of a kept ODM project, tile by tile (tuning data for the adaptive scheduler).

The whole area is re-densified one OpenMVS level finer than ODM's run, as a grid of large tiles so that
peak memory stays bounded (each tile is its own image-subset scene, uav3d.refine). Output: per-tile
cost (wall time, peak RSS, images) in fine_pass.json and the merged fine cloud fine.laz in ODM's frame.
Per-tile quality gains (fine vs coarse against LiDAR) are computed afterwards from fine.laz.
Resumable: tiles whose PLY exists are skipped.

usage: python bench/fine_pass.py <odm project> <out> [--grid 3] [--level 3] [--k 20] [--threads 10]
       [--test-tile SIZE_M]   # one small tile at the centre, for a quick check
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import laspy

from uav3d import refine, tools


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--grid", type=int, default=3, help="tiles along the longer side")
    ap.add_argument("--level", type=int, default=3, help="OpenMVS --resolution-level of the fine pass")
    ap.add_argument("--k", type=int, default=20, help="min sparse points a shot must observe in the tile")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--test-tile", type=float, default=None)
    a = ap.parse_args()

    project, out = a.project.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "fine_pass.log"
    o = tools.odm()
    coarse = project / "odm_georeferencing" / "odm_georeferenced_model.laz"
    ox, oy = refine.utm_offset(project)
    h = laspy.open(coarse).header
    xmin, ymin, xmax, ymax = h.mins[0] - ox, h.mins[1] - oy, h.maxs[0] - ox, h.maxs[1] - oy
    if a.test_tile:
        cx, cy, s = (xmin + xmax) / 2, (ymin + ymax) / 2, a.test_tile / 2
        tiles = [refine.Tile(0, 0, 0, cx - s, cy - s, cx + s, cy + s)]
    else:
        tiles = refine.grid(xmin, ymin, xmax, ymax, max(xmax - xmin, ymax - ymin) / a.grid)

    record_path = out / "fine_pass.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {"tiles": {}}
    record.update(project=str(project), level=a.level, k=a.k, threads=a.threads, bbox=[xmin, ymin, xmax, ymax],
                  tile_size=tiles[0].x1 - tiles[0].x0)
    todo = [t for t in tiles if not record["tiles"].get(str(t.id), {}).get("ply")]
    if todo:
        sub = refine.subsets(o, project, tiles, a.k, out / "scenes", log,
                             export=[(f"t{t.id}", [t]) for t in todo])
        record["shots_total"] = sub["shots_total"]
        scenes = {g["name"]: g for g in sub["groups"]}
        max_res = refine.undistorted_max_size(project)
        for t in todo:
            g = scenes[f"t{t.id}"]
            entry = {"bounds": [t.x0, t.y0, t.x1, t.y1], "shots": len(g["shots"]),
                     "sparse_points": sub["tiles"][str(t.id)]["points"]}
            if not g["shots"]:
                entry["skipped"] = "no shots"
            else:
                t0 = time.time()
                try:
                    entry.update(refine.densify(o, Path(g["scene"]), a.level, max_res, a.threads, log))
                except refine.RefineError as exc:
                    entry["error"] = str(exc)
                entry["total_s"] = round(time.time() - t0, 1)
                shutil.rmtree(Path(g["scene"]).parent / "depthmaps", ignore_errors=True)
            record["tiles"][str(t.id)] = entry
            record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
            print(f"tile {t.id}: {entry.get('shots')} shots, {entry.get('total_s')} s, "
                  f"{entry.get('peak_rss_gb')} GB {entry.get('error', '')}", flush=True)
    done = [(e["ply"], [t]) for t in tiles if (e := record["tiles"].get(str(t.id), {})).get("ply")]
    if done:
        record["merge"] = refine.merge(o, str(coarse), done, project, out / "fine.laz", log)
        record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "tiles"}))
    return 0 if len(done) == len(tiles) else 1


if __name__ == "__main__":
    raise SystemExit(main())
