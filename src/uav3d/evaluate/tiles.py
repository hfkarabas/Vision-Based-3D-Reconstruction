"""Per-tile accuracy/completeness against a reference (tuning data for the adaptive scheduler).

Same definitions as uav3d.evaluate.geometry (Tanks and Temples precision/recall/F at tau), computed per
square tile of a fixed grid so that two reconstructions of the same block (e.g. coarse and fine) are
scored on identical regions. Points are voxel-thinned; at most ``per_tile_sample`` points per tile and
direction are queried against KD-trees of the full thinned other cloud.

Per tile: precision from our points in the tile, recall from reference points in the tile, plus the
2 m cell coverage of both clouds so that edge tiles can be excluded in the analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from uav3d.evaluate.geometry import _read, _thin

TAUS = (0.10, 0.20, 0.50)


def tile_index(xy: np.ndarray, origin: tuple[float, float], size: float, shape: tuple[int, int]) -> np.ndarray:
    """Flat tile id (iy * nx + ix) per point; -1 outside the grid."""
    ix = np.floor((xy[:, 0] - origin[0]) / size).astype(np.int64)
    iy = np.floor((xy[:, 1] - origin[1]) / size).astype(np.int64)
    ok = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    return np.where(ok, iy * shape[1] + ix, -1)


def _groups(ids: np.ndarray, n: int) -> list[np.ndarray]:
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    starts = np.searchsorted(sorted_ids, np.arange(n), side="left")
    ends = np.searchsorted(sorted_ids, np.arange(n), side="right")
    return [order[s:e] for s, e in zip(starts, ends)]


def _coverage(xy: np.ndarray, origin, size: float, cell: float) -> float:
    if len(xy) == 0:
        return 0.0
    k = int(round(size / cell))
    c = np.unique(np.floor((xy - np.asarray(origin)) / cell).astype(np.int64), axis=0)
    return round(len(c) / (k * k), 3)


def score_tiles(ours: np.ndarray, ref: np.ndarray, origin: tuple[float, float], size: float,
                shape: tuple[int, int], taus=TAUS, per_tile_sample: int = 20_000, cell: float = 2.0,
                seed: int = 0) -> list[dict]:
    """Per-tile precision/recall/F of `ours` against `ref` (both Nx3, same frame)."""
    n = shape[0] * shape[1]
    rng = np.random.default_rng(seed)
    t_ref, t_ours = cKDTree(ref), cKDTree(ours)
    g_ours = _groups(tile_index(ours[:, :2], origin, size, shape), n)
    g_ref = _groups(tile_index(ref[:, :2], origin, size, shape), n)
    out = []
    for tid in range(n):
        iy, ix = divmod(tid, shape[1])
        o0 = (origin[0] + ix * size, origin[1] + iy * size)
        io, ir = g_ours[tid], g_ref[tid]
        row = {"id": tid, "ix": ix, "iy": iy, "bounds": [o0[0], o0[1], o0[0] + size, o0[1] + size],
               "n_ours": int(len(io)), "n_ref": int(len(ir)),
               "cover_ours": _coverage(ours[io, :2], o0, size, cell), "cover_ref": _coverage(ref[ir, :2], o0, size, cell)}
        if len(io) and len(ir):
            so = io[rng.choice(len(io), min(per_tile_sample, len(io)), replace=False)]
            sr = ir[rng.choice(len(ir), min(per_tile_sample, len(ir)), replace=False)]
            d_o, _ = t_ref.query(ours[so], workers=-1)
            d_r, _ = t_ours.query(ref[sr], workers=-1)
            for t in taus:
                p, r = float(np.mean(d_o < t)), float(np.mean(d_r < t))
                row[f"tau_{int(round(t * 100))}cm"] = {"precision": round(p, 4), "recall": round(r, 4),
                                                       "f": round(2 * p * r / (p + r), 4) if p + r else 0.0}
        out.append(row)
    return out


def evaluate_clouds(clouds: dict[str, Path], ref_path: Path, size: float, z_offset: float,
                    bounds: tuple[float, float, float, float], voxel: float = 0.05, **kw) -> dict:
    """Score several reconstructions of one block on the same tile grid (anchored at bounds' corner)."""
    x0, y0, x1, y1 = bounds
    shape = (int(np.ceil((y1 - y0) / size)), int(np.ceil((x1 - x0) / size)))
    ref = _thin(_read(ref_path, drop_noise=False, bounds=bounds), voxel)
    result = {"tile_size": size, "z_offset": z_offset, "origin": [x0, y0], "shape": list(shape), "clouds": {}}
    for name, path in clouds.items():
        ours = _thin(_read(path, drop_noise=True, bounds=bounds), voxel)
        ours[:, 2] -= z_offset
        result["clouds"][name] = score_tiles(ours, ref, (x0, y0), size, shape, **kw)
        del ours
    return result


def main(argv=None) -> int:
    """python -m uav3d.evaluate.tiles <out.json> <reference.las> <size_m> <z_offset> <x0> <y0> <x1> <y1> name=cloud.laz ..."""
    import sys

    a = argv if argv is not None else sys.argv[1:]
    out, ref, size, z = Path(a[0]), Path(a[1]), float(a[2]), float(a[3])
    bounds = tuple(map(float, a[4:8]))
    clouds = dict(s.split("=", 1) for s in a[8:])
    res = evaluate_clouds({k: Path(v) for k, v in clouds.items()}, ref, size, z, bounds)
    out.write_text(json.dumps(res), encoding="utf-8")
    print(f"{len(res['clouds'])} clouds x {res['shape'][0] * res['shape'][1]} tiles -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
