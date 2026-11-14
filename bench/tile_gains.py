"""Per-tile gain of the fine pass over the coarse pass against a reference, plus the reference-free signals of
each tile (tuning data on UseGeo-1; replay input on every block).

usage: python bench/tile_gains.py <run dir> <reference.las> [--size 25] [--min-cover 0.8]
Inputs (run dir):
  odm/project                 ODM project (SfM; opensfm/stats/overlap.tif; undistorted images)
  coarse_pass/fine.laz        coarse cloud from bench/fine_pass.py --coarse (falls back to ODM's cloud)
  fine_pass/fine.laz          fine cloud from bench/fine_pass.py
Signals use a DSM rasterised here from the coarse cloud (no gap filling; cell = 2 x coarse ground sampling),
so that tuning and hold-out blocks get identical signal definitions. The vertical bias is measured on the coarse
cloud (flat reference ground) and applied to both clouds.
Writes <run>/tile_gains.json and .csv; prints Spearman correlations signal vs gain.
"""
import argparse
import csv
import json
from pathlib import Path

import laspy
import numpy as np
from scipy.stats import spearmanr

from uav3d import odm as odm_mod
from uav3d import refine, signals, tools
from uav3d.evaluate import geometry, tiles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("--size", type=float, default=25.0)
    ap.add_argument("--min-cover", type=float, default=0.8, help="ref and coarse 2 m-cell coverage of a usable tile")
    a = ap.parse_args()

    project = a.run / "odm" / "project"
    coarse = a.run / "coarse_pass" / "fine.laz"
    if not coarse.is_file():
        coarse = project / "odm_georeferencing" / "odm_georeferenced_model.laz"
    fine = a.run / "fine_pass" / "fine.laz"
    o = tools.odm()

    h = laspy.open(coarse).header
    bounds = (h.mins[0], h.mins[1], h.maxs[0], h.maxs[1])
    shape = (int(np.ceil((bounds[3] - bounds[1]) / a.size)), int(np.ceil((bounds[2] - bounds[0]) / a.size)))

    rec_path = next((p for p in (project / "opensfm" / "reconstruction.json",
                                 *sorted(project.glob("submodels/submodel_*/opensfm/reconstruction.json")))
                     if p.is_file()), None)
    gsd = odm_mod.gsd_from_reconstruction(rec_path)["gsd_m"]
    cp = a.run / "coarse_pass" / "fine_pass.json"
    coarse_side = json.loads(cp.read_text(encoding="utf-8")).get("side_px") if cp.is_file() else None
    max_size = refine.undistorted_max_size(project)
    if coarse_side is None:
        coarse_side = refine.effective_size(max_size, 4)[1]
    cell = signals.dsm_cell(gsd, max_size, coarse_side)
    dsm_path = a.run / "signals" / "coarse_dsm.tif"
    dsm_path.parent.mkdir(exist_ok=True)
    if not dsm_path.is_file():
        signals.dsm(o, coarse, dsm_path, cell, a.run / "signals" / "dsm.log")

    bias_path = a.run / "signals" / "vertical_bias.json"
    if bias_path.is_file():
        bias_rec = json.loads(bias_path.read_text(encoding="utf-8"))
    else:
        bias_rec = geometry.flat_ground_offset(coarse, a.ref)
        bias_path.write_text(json.dumps(bias_rec), encoding="utf-8")
    bias = bias_rec["offset_m"]

    sig = signals.tile_signals(dsm_path, project / "opensfm" / "stats" / "overlap.tif", coarse, bounds[:2], a.size, shape)
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
    summary = {"tiles": len(rows), "usable": len(use), "bias": bias_rec, "size_m": a.size, "coarse": str(coarse),
               "dsm_cell_m": round(cell, 3), "gsd_m": gsd,
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
