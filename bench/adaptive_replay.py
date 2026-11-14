"""Replay evaluation of tile-selection methods under an image budget (Documentation/research/ADAPTIVE_SCHEDULER.md §4).

Instead of densifying every (method, budget, seed) separately, the coarse and the full fine cloud of a block are
scored once per tile against the reference (bench/tile_gains.py). A selection's cloud is "fine inside the
selected tiles, coarse elsewhere"; its block precision/recall are the point-weighted means of the per-tile values
(recall weights = reference points, identical for all methods). Cost = distinct images the selection needs
(bench/tile_shots.py) x the measured per-image cost of the fine level. Approximations (stated in the report):
tile-border effects within tau are ignored, and a subset scene is assumed to densify like the full one inside the
tile; both are checked with real subset runs.

Methods: adaptive (fitted gain model, cost aware), static (low overlap first, cost aware), random (seeds, no cost
awareness), oracle (true per-tile gain from the reference, cost aware; upper bound, analysis only).

usage:
  tune:     python bench/adaptive_replay.py <run> --fit --model-out configs/gain_model.json
  holdout:  python bench/adaptive_replay.py <run> --model configs/gain_model.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from uav3d import schedule

SIGNALS = ["overlap", "void", "density", "roughness", "relief"]
TAUS = ("10", "20", "50")


def block_scores(rows, fine_mask):
    out = {}
    for t in TAUS:
        num_p = den_p = num_r = den_r = 0.0
        for r, fine in zip(rows, fine_mask):
            src = "fine" if fine else "coarse"
            p, rc, n = r.get(f"{src}_precision{t}"), r.get(f"{src}_recall{t}"), r.get(f"n_{src}") or 0
            if p is None or rc is None:
                continue
            num_p += n * p
            den_p += n
            num_r += r["n_ref"] * rc
            den_r += r["n_ref"]
        P = num_p / den_p if den_p else 0.0
        R = num_r / den_r if den_r else 0.0
        out[t] = {"precision": round(P, 4), "recall": round(R, 4), "f": round(2 * P * R / (P + R), 4) if P + R else 0.0}
    return out


def contributions(rows, t="20"):
    """Per tile: change of the block F at tau when only this tile is switched from coarse to fine (linearised
    marginal value; per-tile F gains are not additive because the block metric is point-weighted)."""
    def terms(src):
        p = np.array([r.get(f"{src}_precision{t}") if r.get(f"{src}_precision{t}") is not None else np.nan for r in rows])
        rc = np.array([r.get(f"{src}_recall{t}") if r.get(f"{src}_recall{t}") is not None else np.nan for r in rows])
        n = np.array([r.get(f"n_{src}") or 0 for r in rows], dtype=float)
        nr = np.array([r["n_ref"] for r in rows], dtype=float)
        valid = np.isfinite(p) & np.isfinite(rc)
        return (np.where(valid, n * p, 0), np.where(valid, n, 0), np.where(valid, nr * rc, 0), np.where(valid, nr, 0))
    c, f = terms("coarse"), terms("fine")
    sums = [x.sum() for x in c]

    def F(np_, dp, nr, dr):
        P, R = np_ / np.maximum(dp, 1e-12), nr / np.maximum(dr, 1e-12)
        return np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0.0)
    base = F(*sums)
    swapped = F(*[s - ci + fi for s, ci, fi in zip(sums, c, f)])
    return swapped - base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--k", default="200")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fit", action="store_true", help="tuning set only: fit the gain model here")
    ap.add_argument("--target", default="contrib_f20", help="per-tile target: contrib_f20 (marginal block F) or gain_f20")
    ap.add_argument("--model", type=Path, help="fitted gain model (hold-out)")
    ap.add_argument("--model-out", type=Path)
    a = ap.parse_args()

    rows = json.loads((a.run / "tile_gains.json").read_text(encoding="utf-8"))["rows"]
    shots_by_tile = json.loads((a.run / "tile_shots.json").read_text(encoding="utf-8"))["k"][a.k]
    shots = [set(shots_by_tile.get(str(r["id"]), [])) for r in rows]
    total_images = len(set().union(*shots))
    fp = json.loads((a.run / "fine_pass" / "fine_pass.json").read_text(encoding="utf-8"))
    sec_per_image = fp["depthmaps"]["wall_s"] / fp["depthmaps"]["count"]
    sig = [{k: r.get(k) for k in SIGNALS} for r in rows]
    for t in TAUS:
        for r, v in zip(rows, contributions(rows, t)):
            r[f"contrib_f{t}"] = float(v)

    if a.fit:
        use = [i for i, r in enumerate(rows) if r["usable"]]
        model = schedule.fit([sig[i] for i in use], [rows[i][a.target] for i in use], SIGNALS)
        if a.model_out:
            a.model_out.write_text(json.dumps({"fitted_on": str(a.run), "target": a.target, "k": a.k,
                                               **model.to_dict()}, indent=1), encoding="utf-8")
    else:
        model = schedule.GainModel.from_dict(json.loads(a.model.read_text(encoding="utf-8")))
    pred = model.predict(sig)
    usable = np.array([r["usable"] for r in rows])
    true_gain = np.array([r[a.target] if r[a.target] is not None else np.nan for r in rows], dtype=float)
    ok = usable & np.isfinite(true_gain)
    from scipy.stats import spearmanr
    rho = float(spearmanr(pred[ok], true_gain[ok])[0]) if ok.sum() > 2 else None

    overlap = np.array([r["overlap"] if r["overlap"] is not None else np.nan for r in rows], dtype=float)
    static_score = 1.0 - np.where(np.isfinite(overlap), schedule.ranks(overlap), 0.5) + 1e-6
    oracle_score = np.where(np.isfinite(true_gain), true_gain, 0.0)
    pred_pos = pred - np.nanmin(pred) + 1e-6  # ranks only matter; keep every tile selectable

    def run(name, sel):
        mask = np.zeros(len(rows), bool)
        mask[sel.order] = True
        n_img = sel.images[-1] if sel.images else 0
        return {"method": name, "tiles": int(mask.sum()), "images": int(n_img),
                "fine_time_s": round(n_img * sec_per_image, 1), "scores": block_scores(rows, mask)}

    results = {"run": str(a.run), "k": a.k, "total_images": total_images, "sec_per_image_fine": round(sec_per_image, 2),
               "model": model.to_dict(), "spearman_pred_vs_true": rho,
               "coarse": block_scores(rows, np.zeros(len(rows), bool)),
               "fine_all": block_scores(rows, np.ones(len(rows), bool)), "budgets": {}}
    for b in a.budgets:
        B = int(round(b * total_images))
        res = [run("adaptive", schedule.select(pred_pos, shots, B)),
               run("static", schedule.select(static_score, shots, B)),
               run("oracle", schedule.select(np.clip(oracle_score, 0, None), shots, B))]
        rnd = [run(f"random_{s}", schedule.select(np.zeros(len(rows)), shots, B, cost_aware=False,
                                                  order=schedule.random_order(len(rows), s)))
               for s in range(a.seeds)]
        res += rnd
        results["budgets"][str(b)] = {"images": B, "methods": res}
    (a.run / "adaptive_replay.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"total images {total_images}; spearman(pred, true {a.target}) = {rho}")
    print(f"coarse F20 {results['coarse']['20']['f']:.4f}  fine-all F20 {results['fine_all']['20']['f']:.4f}")
    for b, v in results["budgets"].items():
        for m in v["methods"]:
            s = m["scores"]
            print(f"B={b:>4} {m['method']:<9} img {m['images']:>4} tiles {m['tiles']:>4} "
                  f"F10 {s['10']['f']:.4f} F20 {s['20']['f']:.4f} F50 {s['50']['f']:.4f} R20 {s['20']['recall']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
