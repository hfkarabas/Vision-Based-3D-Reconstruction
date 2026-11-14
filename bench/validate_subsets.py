"""Real subset run of one selection: checks the two replay approximations (tile-border effects; a subset scene
densifies a tile like the full scene) and the image-count cost model.

The selection (adaptive at budget beta, or explicit tile ids) is grouped into 4-connected components; each group
is exported as an OpenMVS scene with only its shots (OpenSfM exporter), densified at the fine level with ODM's
low-memory path, cropped to its tiles and merged into the coarse cloud. The merged cloud is scored per tile
against the reference and its block P/R/F is compared with the replay prediction for the same selection.

usage: python bench/validate_subsets.py <run> <reference.las> (--budget 0.5 --model configs/gain_model.json | --tiles 12 13)
       [--k 200] [--level 2] [--threads 10]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from uav3d import refine, schedule, tools
from uav3d.evaluate import tiles as tile_eval

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_replay  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("--budget", type=float)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--tiles", type=int, nargs="+")
    ap.add_argument("--k", default="200")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--sub-scene-area", type=int, default=400000)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    run = a.run.resolve()
    project = run / "odm" / "project"
    o = tools.odm()
    tg = json.loads((run / "tile_gains.json").read_text(encoding="utf-8"))
    rows = tg["rows"]
    ts = json.loads((run / "tile_shots.json").read_text(encoding="utf-8"))
    size, origin, shape = ts["size"], ts["origin"], ts["shape"]
    shots = [set(ts["k"][a.k].get(str(r["id"]), [])) for r in rows]
    if a.tiles:
        sel = list(a.tiles)
        name = a.name or "tiles_" + "_".join(map(str, sel[:4]))
    else:
        model = schedule.GainModel.from_dict(json.loads(a.model.read_text(encoding="utf-8")))
        pred = model.predict([{k: r.get(k) for k in adaptive_replay.SIGNALS} for r in rows])
        total = len(set().union(*shots))
        sel = schedule.select(pred - np.nanmin(pred) + 1e-6, shots, int(round(a.budget * total))).order
        name = a.name or f"adaptive_{a.budget}"
    work = run / "validate" / name
    work.mkdir(parents=True, exist_ok=True)
    log = work / "validate.log"
    rec_path = work / "validate.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.is_file() else {}

    def save():
        rec_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")

    ox, oy = refine.utm_offset(project)
    by_id = {r["id"]: r for r in rows}
    sel_tiles = [refine.Tile(i, by_id[i]["ix"], by_id[i]["iy"], origin[0] + by_id[i]["ix"] * size - ox,
                             origin[1] + by_id[i]["iy"] * size - oy, origin[0] + (by_id[i]["ix"] + 1) * size - ox,
                             origin[1] + (by_id[i]["iy"] + 1) * size - oy) for i in sel]
    grp = refine.groups(sel_tiles)
    rec.update(selection=sel, groups=[[t.id for t in g] for g in grp], level=a.level, k=a.k,
               predicted_images=len(set().union(*[shots[i] for i in sel])) if sel else 0)
    if "export" not in rec:
        sub = refine.subsets(o, project, sel_tiles, int(a.k), work / "export", log,
                             export=[(f"g{j:03d}", g) for j, g in enumerate(grp)])
        rec["export"] = {"wall_s": sub["measure"]["wall_s"],
                         "groups": [{"name": g["name"], "shots": len(g["shots"]), "scene": g["scene"]}
                                    for g in sub["groups"]]}
        rec["images_used"] = len(set().union(*[set(g["shots"]) for g in sub["groups"]]))
        save()
    max_res = refine.undistorted_max_size(project)
    rec.setdefault("densify", {})
    refined = []
    for g, tiles_g in zip(rec["export"]["groups"], grp):
        d = rec["densify"].setdefault(g["name"], {})
        refine.densify_lowmem(o, Path(g["scene"]), work / g["name"], a.level, max_res, a.threads, log,
                              a.sub_scene_area, d, save)
        save()
        refined += [(p, tiles_g) for p in d["plys"]]
    coarse = run / "coarse_pass" / "fine.laz"
    out_laz = work / "merged.laz"
    if not out_laz.is_file():
        rec["merge"] = refine.merge(o, str(coarse), refined, project, out_laz, log)
        save()

    def wall(d):
        s = d.get("depthmaps", {}).get("wall_s", 0) + d.get("split", {}).get("wall_s", 0)
        return s + sum(v["fuse"]["wall_s"] + v["filter"]["wall_s"] for v in d.get("subscenes", {}).values())
    rec["measured"] = {"wall_s": round(sum(wall(d) for d in rec["densify"].values()), 1),
                       "depthmap_wall_s": round(sum(d.get("depthmaps", {}).get("wall_s", 0)
                                                    for d in rec["densify"].values()), 1),
                       "peak_rss_gb": max([v.get("peak_rss_gb", 0) for d in rec["densify"].values()
                                           for v in [d.get("depthmaps", {}), *[x["fuse"] for x in d.get("subscenes", {}).values()]]]
                                          or [0])}
    fp = json.loads((run / "fine_pass" / "fine_pass.json").read_text(encoding="utf-8"))
    rec["predicted"] = {"depthmap_wall_s": round(rec["images_used"] * fp["depthmaps"]["wall_s"] / fp["depthmaps"]["count"], 1)}

    bias = tg["summary"]["bias"]["offset_m"] if isinstance(tg["summary"].get("bias"), dict) else tg["summary"]["bias_m"]
    b = (origin[0], origin[1], origin[0] + shape[1] * size, origin[1] + shape[0] * size)
    ev = tile_eval.evaluate_clouds({"merged": out_laz}, a.ref, size, bias, b)["clouds"]["merged"]
    real_rows = []
    for r, e in zip(rows, ev):
        x = dict(r)
        x["n_fine"] = e["n_ours"]
        for t in adaptive_replay.TAUS:
            for m in ("precision", "recall"):
                x[f"fine_{m}{t}"] = e.get(f"tau_{t}cm", {}).get(m)
        real_rows.append(x)
    mask_all = np.ones(len(rows), bool)
    mask_sel = np.zeros(len(rows), bool)
    mask_sel[[list(by_id).index(i) for i in sel]] = True
    rec["scores"] = {"merged_measured": adaptive_replay.block_scores(real_rows, mask_all),
                     "replay_predicted": adaptive_replay.block_scores(rows, mask_sel),
                     "coarse": adaptive_replay.block_scores(rows, np.zeros(len(rows), bool))}
    save()
    print(json.dumps({k: rec[k] for k in ("images_used", "predicted_images", "measured", "predicted", "scores")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
