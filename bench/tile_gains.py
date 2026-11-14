"""Tuning analysis (UseGeo-1 only): per-tile gain of the fine pass over the coarse pass against LiDAR,
and how well each reference-free signal ranks that gain.

usage: python bench/tile_gains.py <run dir> <reference.las> [--size 25]
Reads <run>/odm/project (coarse), <run>/fine_pass/fine.laz (fine), <run>/eval_geometry.json (vertical bias).
Writes <run>/tile_gains.json and tile_gains.csv; prints Spearman correlations signal vs gain.
"""
import argparse
import csv
import json
from pathlib import Path

import laspy
import numpy as np
from scipy.stats import spearmanr

from uav3d import signals
from uav3d.evaluate import tiles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("--size", type=float, default=25.0)
    ap.add_argument("--min-cover", type=float, default=0.8, help="ref and coarse 2 m-cell coverage of a usable tile")
    a = ap.parse_args()

    project = a.run / "odm" / "project"
    coarse = project / "odm_georeferencing" / "odm_georeferenced_model.laz"
    fine = a.run / "fine_pass" / "fine.laz"
    bias = json.loads((a.run / "eval_geometry.json").read_text(encoding="utf-8"))["vertical_bias"]["offset_m"]
    h = laspy.open(coarse).header
    bounds = (h.mins[0], h.mins[1], h.maxs[0], h.maxs[1])
    shape = (int(np.ceil((bounds[3] - bounds[1]) / a.size)), int(np.ceil((bounds[2] - bounds[0]) / a.size)))

    sig = signals.tile_signals(project / "odm_dem" / "dsm.tif", project / "opensfm" / "stats" / "overlap.tif",
                               coarse, bounds[:2], a.size, shape)
    ev = tiles.evaluate_clouds({"coarse": coarse, "fine": fine}, a.ref, a.size, bias, bounds)
    rows = []
    for s, c, f in zip(sig, ev["clouds"]["coarse"], ev["clouds"]["fine"]):
        r = {**{k: s[k] for k in ("id", "ix", "iy", "overlap", "void", "density", "roughness", "relief")},
             "cover_ref": c["cover_ref"], "cover_coarse": c["cover_ours"], "cover_fine": f["cover_ours"],
             "n_ref": c["n_ref"], "n_coarse": c["n_ours"], "n_fine": f["n_ours"]}
        for t in ("10", "20", "50"):
            for m in ("precision", "recall", "f"):
                cv, fv = c.get(f"tau_{t}cm", {}).get(m), f.get(f"tau_{t}cm", {}).get(m)
                r[f"coarse_{m}{t}"], r[f"fine_{m}{t}"] = cv, fv
                r[f"gain_{m}{t}"] = None if cv is None or fv is None else round(fv - cv, 4)
        r["usable"] = bool(c["cover_ref"] >= a.min_cover and c["cover_ours"] >= a.min_cover
                           and r["gain_f20"] is not None)
        rows.append(r)

    use = [r for r in rows if r["usable"]]
    corr = {}
    for target in ("gain_f20", "gain_recall20", "gain_f10"):
        corr[target] = {}
        for k in ("overlap", "void", "density", "roughness", "relief", "coarse_f20"):
            x = [r[k] for r in use if r[k] is not None and r[target] is not None]
            y = [r[target] for r in use if r[k] is not None and r[target] is not None]
            if len(x) > 10:
                rho, p = spearmanr(x, y)
                corr[target][k] = {"rho": round(float(rho), 3), "p": float(p), "n": len(x)}
    g = np.array([r["gain_f20"] for r in use])
    summary = {"tiles": len(rows), "usable": len(use), "bias_m": bias, "size_m": a.size,
               "gain_f20": {"mean": round(float(g.mean()), 4), "median": round(float(np.median(g)), 4),
                            "p10": round(float(np.percentile(g, 10)), 4), "p90": round(float(np.percentile(g, 90)), 4)}
               if len(g) else None,
               "spearman": corr}
    (a.run / "tile_gains.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    with open(a.run / "tile_gains.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
