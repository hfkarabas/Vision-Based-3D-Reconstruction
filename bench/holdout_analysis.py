"""Hypothesis test on the hold-out blocks (H1-H3 as pre-registered in the literature review; see
Documentation/research/ADAPTIVE_SCHEDULER.md). Reads the per-tile scores and image sets that
bench/adaptive_replay.py uses, rebuilds the same selections (frozen gain model, same rules) and adds:

  * block bootstrap: tiles are grouped into super-blocks of `--block` x `--block` tiles (default 4 x 4 = 100 m,
    against spatial autocorrelation), super-blocks are resampled with replacement and the block metrics
    recomputed for the fixed selections; 95 pct percentile intervals of the differences
      H1  adaptive - coarse     recall and precision at tau (beta = 0.5: recall >= +5 pp, precision >= -1 pp)
      H2  adaptive - static     F at tau (and adaptive - random, seed mean)
      H3  sign of the H1/H2 differences on every hold-out block
  * budget-quality curve (beta 0.05 .. 1.0) for every block given, and the end-to-end cost model
    (coarse pass + fine depth maps of the selected images + fusion scaled by the image share) against the
    full fine pass.

usage: python bench/holdout_analysis.py <run> [<run> ...] --model configs/gain_model.json
         [--tune <tuning run>] [--out outputs/holdout/hypothesis.json] [--figure paper/figures/butce_kalite.pdf]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_replay import SIGNALS  # noqa: E402

from uav3d import schedule  # noqa: E402

TAUS = ("10", "20", "50")
SEEDS = 3


def load(run: Path, k: str):
    rows = json.loads((run / "tile_gains.json").read_text(encoding="utf-8"))["rows"]
    by_tile = json.loads((run / "tile_shots.json").read_text(encoding="utf-8"))["k"][k]
    shots = [set(by_tile.get(str(r["id"]), [])) for r in rows]
    return rows, shots


def terms(rows, t):
    """Per-tile (n*P, n, n_ref*R, n_ref) for coarse and fine; invalid tiles contribute nothing."""
    out = {}
    for src in ("coarse", "fine"):
        p = np.array([r.get(f"{src}_precision{t}") for r in rows], dtype=float)
        rc = np.array([r.get(f"{src}_recall{t}") for r in rows], dtype=float)
        n = np.array([r.get(f"n_{src}") or 0 for r in rows], dtype=float)
        nr = np.array([r["n_ref"] for r in rows], dtype=float)
        ok = np.isfinite(p) & np.isfinite(rc)
        out[src] = np.stack([np.where(ok, n * p, 0), np.where(ok, n, 0), np.where(ok, nr * rc, 0), np.where(ok, nr, 0)])
    return out


def metrics(T, mask, w):
    """Block P, R, F for selection `mask` (fine inside) with tile weights `w` (bootstrap counts); w: (B, n)."""
    s = np.where(mask[None, :], 1.0, 0.0)
    comp = T["coarse"][None] * (1 - s)[:, None] + T["fine"][None] * s[:, None]  # (1, 4, n)
    tot = np.einsum("bn,kn->bk", w, comp[0])
    P = tot[:, 0] / np.maximum(tot[:, 1], 1e-12)
    R = tot[:, 2] / np.maximum(tot[:, 3], 1e-12)
    F = np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0.0)
    return P, R, F


def selections(rows, shots, model, budget_images):
    sig = [{k: r.get(k) for k in SIGNALS} for r in rows]
    pred = model.predict(sig)
    pred_pos = np.where(np.isfinite(pred), pred, 0.0)
    overlap = np.array([r["overlap"] if r["overlap"] is not None else np.nan for r in rows], dtype=float)
    static = 1.0 - np.where(np.isfinite(overlap), schedule.ranks(overlap), 0.5) + 1e-6
    true = np.array([r.get("contrib_f20", np.nan) for r in rows], dtype=float)
    n = len(rows)

    def mask(sel):
        m = np.zeros(n, bool)
        m[sel.order] = True
        return m, (sel.images[-1] if sel.images else 0)

    out = {"adaptive": mask(schedule.select(pred_pos, shots, budget_images)),
           "static": mask(schedule.select(static, shots, budget_images)),
           "oracle": mask(schedule.select(np.clip(np.where(np.isfinite(true), true, 0), 0, None), shots, budget_images))}
    for s in range(SEEDS):
        out[f"random_{s}"] = mask(schedule.select(np.zeros(n), shots, budget_images, cost_aware=False,
                                                  order=schedule.random_order(n, s)))
    return out


def contributions_f20(rows):
    from adaptive_replay import contributions
    for r, v in zip(rows, contributions(rows, "20")):
        r["contrib_f20"] = float(v)


def bootstrap_weights(rows, block, reps, seed=0):
    ix = np.array([r["ix"] for r in rows])
    iy = np.array([r["iy"] for r in rows])
    grp = (iy // block) * (ix.max() // block + 1) + ix // block
    ids, inv = np.unique(grp, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=reps)  # (reps, groups)
    return counts[:, inv].astype(float), len(ids)


def cost_model(run: Path, images: int, total_images: int):
    c = json.loads((run / "coarse_pass" / "fine_pass.json").read_text(encoding="utf-8"))
    f = json.loads((run / "fine_pass" / "fine_pass.json").read_text(encoding="utf-8"))

    def fusion(fp):
        return sum(v["fuse"]["wall_s"] + v["filter"]["wall_s"] for v in fp["subscenes"].values())
    coarse_s = c["depthmaps"]["wall_s"] + fusion(c)
    fine_dm = f["depthmaps"]["wall_s"]
    fine_s = fine_dm + fusion(f)
    share = images / f["depthmaps"]["count"]
    adaptive_s = coarse_s + share * fine_s
    return {"coarse_s": round(coarse_s, 1), "fine_full_s": round(fine_s, 1),
            "adaptive_s": round(adaptive_s, 1), "adaptive_vs_full": round(adaptive_s / fine_s, 3),
            "image_share": round(share, 3), "peak_rss_fine_gb": max(v["fuse"]["peak_rss_gb"] for v in f["subscenes"].values())}


def analyse(run, model, k, block, reps, tau):
    rows, shots = load(run, k)
    contributions_f20(rows)
    total_images = len(set().union(*shots))
    T = {t: terms(rows, t) for t in TAUS}
    W, groups = bootstrap_weights(rows, block, reps)
    one = np.ones((1, len(rows)))
    none = np.zeros(len(rows), bool)
    allm = np.ones(len(rows), bool)
    res = {"run": str(run), "tiles": len(rows), "superblocks": groups, "total_images": total_images,
           "coarse": {}, "fine_all": {}, "budgets": {}, "curve": {}}
    for t in TAUS:
        for name, m in (("coarse", none), ("fine_all", allm)):
            P, R, F = metrics(T[t], m, one)
            res[name][t] = {"precision": round(float(P[0]), 4), "recall": round(float(R[0]), 4), "f": round(float(F[0]), 4)}
    for beta in (0.25, 0.5, 0.75):
        B = int(round(beta * total_images))
        sel = selections(rows, shots, model, B)
        entry = {"images": B, "methods": {}, "tests": {}}
        for name, (m, n_img) in sel.items():
            entry["methods"][name] = {"tiles": int(m.sum()), "images": int(n_img), **{
                t: dict(zip(("precision", "recall", "f"), (round(float(x[0]), 4) for x in metrics(T[t], m, one))))
                for t in TAUS}}
        Ta = T[tau]
        Pa, Ra, Fa = metrics(Ta, sel["adaptive"][0], W)
        Pc, Rc, Fc = metrics(Ta, none, W)
        _, _, Fs = metrics(Ta, sel["static"][0], W)
        Fr = np.mean([metrics(Ta, sel[f"random_{s}"][0], W)[2] for s in range(SEEDS)], axis=0)

        def ci(d, point):
            lo, hi = np.percentile(d, [2.5, 97.5])
            return {"diff": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                    "p_le_0": round(float((d <= 0).mean()), 4)}
        pt = {n: entry["methods"][n][tau] for n in entry["methods"]}
        rnd_pt = np.mean([pt[f"random_{s}"]["f"] for s in range(SEEDS)])
        entry["tests"] = {
            "recall_adaptive_minus_coarse": ci(Ra - Rc, pt["adaptive"]["recall"] - res["coarse"][tau]["recall"]),
            "precision_adaptive_minus_coarse": ci(Pa - Pc, pt["adaptive"]["precision"] - res["coarse"][tau]["precision"]),
            "f_adaptive_minus_coarse": ci(Fa - Fc, pt["adaptive"]["f"] - res["coarse"][tau]["f"]),
            "f_adaptive_minus_static": ci(Fa - Fs, pt["adaptive"]["f"] - pt["static"]["f"]),
            "f_adaptive_minus_random": ci(Fa - Fr, pt["adaptive"]["f"] - rnd_pt),
        }
        entry["cost"] = cost_model(run, sel["adaptive"][1], total_images)
        res["budgets"][str(beta)] = entry
    for beta in np.round(np.arange(0.05, 1.0001, 0.05), 2):
        B = int(round(beta * total_images))
        sel = selections(rows, shots, model, B)
        f = {n: float(metrics(T[tau], m, one)[2][0]) for n, (m, _) in sel.items()}
        res["curve"][str(beta)] = {"adaptive": f["adaptive"], "static": f["static"], "oracle": f["oracle"],
                                   "random": float(np.mean([f[f"random_{s}"] for s in range(SEEDS)])),
                                   "adaptive_images": int(sel["adaptive"][1])}
    return res


def verdicts(results, tau):
    h = {}
    holdouts = [r for r in results if r["role"] == "holdout"]
    for r in holdouts:
        t = r["budgets"]["0.5"]["tests"]
        r["H1"] = bool(t["recall_adaptive_minus_coarse"]["diff"] >= 0.05
                       and t["precision_adaptive_minus_coarse"]["diff"] >= -0.01)
        # H2: feedback adds something only if the static selector's F is outside the interval
        r["H2_feedback_helps"] = bool(t["f_adaptive_minus_static"]["ci95"][0] > 0)
    h["H1"] = {"rule": "beta=0.5: recall(adaptive) - recall(coarse) >= +5 pp and precision drop <= 1 pp, tau=%s cm" % tau,
               "per_block": {Path(r["run"]).name: r["H1"] for r in holdouts}}
    h["H2"] = {"rule": "feedback helps if the 95 pct interval of F(adaptive) - F(static) at beta=0.5 excludes 0",
               "per_block": {Path(r["run"]).name: r["H2_feedback_helps"] for r in holdouts}}
    h["H3"] = {"rule": "adaptive beats coarse and static (F, beta=0.5) on every hold-out block",
               "per_block": {Path(r["run"]).name: bool(r["budgets"]["0.5"]["tests"]["f_adaptive_minus_coarse"]["diff"] > 0
                                                       and r["budgets"]["0.5"]["tests"]["f_adaptive_minus_static"]["diff"] > 0)
                             for r in holdouts}}
    h["H3"]["holds"] = all(h["H3"]["per_block"].values()) if holdouts else None
    return h


COLORS = {"adaptive": "#2a78d6", "static": "#eb6834", "random": "#1baf7a", "oracle": "#eda100"}
LABELS = {"adaptive": "Adaptive", "static": "Static", "random": "Random", "oracle": "Oracle"}
STYLE = {"adaptive": dict(ls="-", marker="o"), "static": dict(ls="--", marker="s"),
         "random": dict(ls=":", marker="^"), "oracle": dict(ls="-.", marker="D")}


TEXT = {"tr": {"coarse": "kaba", "fine": "tam ince", "x": r"Görüntü bütçesi $\beta$ [%]", "tune": " (ayar)"},
        "en": {"coarse": "coarse", "fine": "full fine", "x": r"Image budget $\beta$ [%]", "tune": " (tuning)"}}


def figure(results, out: Path, tau, lang: str = "tr"):
    t = TEXT[lang]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, len(results), figsize=(2.15 * len(results), 2.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results):
        betas = np.array([float(b) for b in r["curve"]])
        for m in ("oracle", "adaptive", "random", "static"):
            y = 100 * np.array([r["curve"][b][m] for b in r["curve"]])
            ax.plot(100 * betas, y, color=COLORS[m], lw=1.5, ms=3, markevery=[4, 9, 14], **STYLE[m], label=LABELS[m])
        c, f = 100 * r["coarse"][tau]["f"], 100 * r["fine_all"][tau]["f"]
        ax.axhline(c, color="#8a8984", lw=0.8)
        ax.axhline(f, color="#8a8984", lw=0.8, ls="--")
        ax.text(101, c, t["coarse"], va="center", fontsize=6.5, color="#52514e")
        ax.text(101, f, t["fine"], va="center", fontsize=6.5, color="#52514e")
        ax.set_title(r["label"].replace(" (ayar)", t["tune"]), fontsize=8)
        ax.set_xlabel(t["x"])
        ax.set_xlim(0, 100)
        ax.grid(axis="y", color="#e3e2de", lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"F$_{%s}$ [%%]" % tau)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tune", type=Path, help="tuning block: drawn in the curve, never tested")
    ap.add_argument("--k", default="50")
    ap.add_argument("--tau", default="20")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("outputs/holdout/hypothesis.json"))
    ap.add_argument("--figure", type=Path)
    ap.add_argument("--lang", choices=("tr", "en"), default="tr", help="figure labels")
    a = ap.parse_args()
    model = schedule.GainModel.from_dict(json.loads(a.model.read_text(encoding="utf-8")))
    results = []
    if a.tune:
        r = analyse(a.tune, model, a.k, a.block, a.reps, a.tau)
        r.update(role="tune", label=a.tune.name.replace("_unsplit", "").replace("usegeo_", "UseGeo-") + " (ayar)")
        results.append(r)
    for run in a.runs:
        r = analyse(run, model, a.k, a.block, a.reps, a.tau)
        r.update(role="holdout", label=run.name.replace("_unsplit", "").replace("usegeo_", "UseGeo-"))
        results.append(r)
    out = {"tau_cm": a.tau, "block_tiles": a.block, "reps": a.reps, "model": str(a.model),
           "hypotheses": verdicts(results, a.tau), "blocks": results}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    if a.figure:
        figure(results, a.figure, a.tau, a.lang)
    for r in results:
        print(f"== {r['label']}  tiles {r['tiles']}  images {r['total_images']}  "
              f"coarse F{a.tau} {r['coarse'][a.tau]['f']:.4f}  fine-all {r['fine_all'][a.tau]['f']:.4f}")
        for b, e in r["budgets"].items():
            ms = e["methods"]
            rnd = np.mean([ms[f"random_{s}"][a.tau]["f"] for s in range(SEEDS)])
            print(f"  beta {b}: adaptive {ms['adaptive'][a.tau]['f']:.4f} static {ms['static'][a.tau]['f']:.4f} "
                  f"random {rnd:.4f} oracle {ms['oracle'][a.tau]['f']:.4f}  cost {e['cost']['adaptive_vs_full']:.2f}x full")
            for k_, v in e["tests"].items():
                print(f"    {k_:<34} {v['diff']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    print(json.dumps(out["hypotheses"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
