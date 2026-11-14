"""Budgeted tile selection for regional re-densification (Documentation/research/ADAPTIVE_SCHEDULER.md).

Inputs per tile: reference-free signals (uav3d.signals) and the shots that observe it (refine.subsets).
Cost: OpenMVS computes a full depth map per image of a scene, so the cost of a set of tiles is the number of
distinct images it needs times the per-image cost of the fine level. A tile whose images are already in the
selection costs (almost) nothing; the greedy step therefore divides gain by the *marginal* image count.

Gain model: signals are turned into within-scene percentile ranks (scale free, so no dataset-specific
thresholds), oriented (+1: high value -> more gain, -1: low value -> more gain) and combined with non-negative
weights. Orientation and weights are fitted on the tuning set only and stored in configs/defaults.json.

This module must not import uav3d.evaluate (no reference data in the production path).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def ranks(values) -> np.ndarray:
    """Within-scene percentile rank in [0, 1] (average rank for ties); NaN stays NaN."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    n = int(ok.sum())
    if n == 0:
        return out
    if n == 1:
        out[ok] = 0.5
        return out
    from scipy.stats import rankdata
    out[ok] = (rankdata(v[ok]) - 1) / (n - 1)
    return out


@dataclass
class GainModel:
    """g(t) = bias + sum_k w_k * r_k(t), r_k = oriented percentile rank of signal k (missing -> 0.5)."""
    weights: dict[str, float] = field(default_factory=dict)
    orientation: dict[str, int] = field(default_factory=dict)
    bias: float = 0.0

    def features(self, signals: list[dict]) -> np.ndarray:
        cols = []
        for k in self.weights:
            r = ranks([s.get(k) if s.get(k) is not None else np.nan for s in signals])
            if self.orientation.get(k, 1) < 0:
                r = 1.0 - r
            cols.append(np.where(np.isfinite(r), r, 0.5))
        return np.column_stack(cols) if cols else np.zeros((len(signals), 0))

    def predict(self, signals: list[dict]) -> np.ndarray:
        X = self.features(signals)
        w = np.array([self.weights[k] for k in self.weights], dtype=float)
        return self.bias + (X @ w if len(w) else np.zeros(len(signals)))

    def to_dict(self) -> dict:
        return {"weights": self.weights, "orientation": self.orientation, "bias": self.bias}

    @classmethod
    def from_dict(cls, d: dict) -> "GainModel":
        return cls(dict(d.get("weights", {})), {k: int(v) for k, v in d.get("orientation", {}).items()},
                   float(d.get("bias", 0.0)))


def fit(signals: list[dict], gains, keys: list[str]) -> GainModel:
    """Tuning only. Orientation of each signal from the sign of its Spearman correlation with the gain;
    non-negative least squares of the gain on the oriented ranks (signals with no positive contribution
    get weight 0)."""
    from scipy.optimize import nnls
    from scipy.stats import spearmanr

    y = np.asarray(gains, dtype=float)
    orient = {}
    for k in keys:
        x = np.array([s.get(k) if s.get(k) is not None else np.nan for s in signals], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        rho = spearmanr(x[ok], y[ok])[0] if ok.sum() > 2 else 0.0
        orient[k] = 1 if (np.isfinite(rho) and rho >= 0) else -1
    model = GainModel({k: 1.0 for k in keys}, orient)
    X = model.features(signals)
    ok = np.isfinite(y)
    A = np.column_stack([X[ok], np.ones(ok.sum())])
    # nnls cannot take a free bias; centre y and X instead, then recover the bias
    Xm, ym = A[:, :-1].mean(0), y[ok].mean()
    w, _ = nnls(A[:, :-1] - Xm, y[ok] - ym)
    model.weights = {k: round(float(v), 6) for k, v in zip(keys, w)}
    model.bias = round(float(ym - Xm @ w), 6)
    return model


@dataclass
class Selection:
    order: list[int]            # selected tile indices, in selection order
    images: list[int]           # cumulative distinct images after each step
    gain: list[float]           # cumulative predicted gain after each step


def select(scores, shots: list[set], budget_images: int, *, cost_aware: bool = True,
           order=None) -> Selection:
    """Greedy selection under a budget of distinct images.

    cost_aware: pick the tile maximising score / (1 + new images) (lazy re-evaluation); otherwise take
    tiles in ``order`` (or by descending score) and keep each that still fits.
    Tiles with score <= 0 are never selected by the cost-aware rule.
    """
    s = np.asarray(scores, dtype=float)
    n = len(s)
    chosen: list[int] = []
    have: set = set()
    imgs, gains, total = [], [], 0.0
    if cost_aware:
        remaining = {i for i in range(n) if np.isfinite(s[i]) and s[i] > 0 and shots[i]}
        while remaining:
            best, best_ratio = None, -np.inf
            for i in remaining:
                ratio = s[i] / (1 + len(shots[i] - have))
                if ratio > best_ratio:
                    best, best_ratio = i, ratio
            remaining.discard(best)
            new = shots[best] - have
            if len(have) + len(new) > budget_images:
                # it does not fit; tiles needing fewer new images may still fit
                continue
            have |= new
            total += s[best]
            chosen.append(best)
            imgs.append(len(have))
            gains.append(total)
    else:
        seq = list(order) if order is not None else list(np.argsort(-s, kind="stable"))
        for i in seq:
            if not shots[i]:
                continue
            new = shots[i] - have
            if len(have) + len(new) > budget_images:
                continue
            have |= new
            total += float(s[i]) if np.isfinite(s[i]) else 0.0
            chosen.append(int(i))
            imgs.append(len(have))
            gains.append(total)
    return Selection(chosen, imgs, gains)


def random_order(n: int, seed: int) -> list[int]:
    return list(np.random.default_rng(seed).permutation(n))
