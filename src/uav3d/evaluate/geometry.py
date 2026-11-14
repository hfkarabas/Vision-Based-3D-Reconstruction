"""Point-cloud accuracy and completeness against an independent reference (e.g. UAV LiDAR).

Definitions follow Tanks and Temples (Knapitsch et al. 2017):
  precision(t)  = share of reconstructed points within t of the reference,
  recall(t)     = share of reference points within t of the reconstruction,
  F(t)          = harmonic mean.
Taus are declared in DATASET_PROTOCOL.md (10, 20, 50 cm) and must not be tuned on results.

Both clouds are cropped to their common footprint (cells observed by both on a coarse XY
grid), voxel-thinned to a uniform density, and distances are measured on random samples
against KD-trees of the full thinned other cloud. Noise (class 7) is excluded from ours.
"""
from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

NOISE = 7


def _read(path: Path, drop_noise: bool, bounds=None, chunk: int = 20_000_000) -> np.ndarray:
    parts = []
    with laspy.open(path) as f:
        for pts in f.chunk_iterator(chunk):
            xyz = np.column_stack([pts.x, pts.y, pts.z])
            keep = np.ones(len(xyz), dtype=bool)
            if drop_noise:
                keep &= np.asarray(pts.classification) != NOISE
            if bounds is not None:
                x0, y0, x1, y1 = bounds
                keep &= (xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1)
            parts.append(xyz[keep])
    return np.vstack(parts) if parts else np.zeros((0, 3))


def _thin(xyz: np.ndarray, voxel: float) -> np.ndarray:
    _, first = np.unique(np.floor(xyz / voxel).astype(np.int64), axis=0, return_index=True)
    return xyz[first]


def _footprint(xyz: np.ndarray, x0: float, y0: float, cell: float, shape) -> np.ndarray:
    ix = ((xyz[:, 0] - x0) / cell).astype(np.int64).clip(0, shape[1] - 1)
    iy = ((xyz[:, 1] - y0) / cell).astype(np.int64).clip(0, shape[0] - 1)
    grid = np.zeros(shape, dtype=bool)
    grid[iy, ix] = True
    return grid


def flat_ground_offset(ours_path: Path, ref_path: Path, half: float = 100.0, radius: float = 0.5,
                       max_range: float = 0.15, sample: int = 200_000, seed: int = 0) -> dict:
    """Median vertical offset (ours - reference) on flat bare ground, in a central square.

    Our ground-class points are paired with reference points within ``radius`` (XY); a pair
    counts only where the reference is locally flat (z range < max_range), i.e. no vegetation,
    edges or walls. A systematic offset here is a datum/calibration bias, not a surface error."""
    with laspy.open(ours_path) as f:
        cls_pts = [p for p in f.chunk_iterator(20_000_000)]
    xyz = np.vstack([np.column_stack([p.x, p.y, p.z])[np.asarray(p.classification) == 2] for p in cls_pts])
    if len(xyz) == 0:  # unclassified cloud (e.g. a bare OpenMVS pass): the flat-reference test alone selects ground
        xyz = np.vstack([np.column_stack([p.x, p.y, p.z]) for p in cls_pts])
    cx, cy = np.median(xyz[:, 0]), np.median(xyz[:, 1])
    bounds = (cx - half, cy - half, cx + half, cy + half)
    xyz = xyz[(abs(xyz[:, 0] - cx) < half) & (abs(xyz[:, 1] - cy) < half)]
    ref = _read(ref_path, drop_noise=False, bounds=bounds)
    rng = np.random.default_rng(seed)
    xyz = xyz[rng.choice(len(xyz), min(sample, len(xyz)), replace=False)]
    tree = cKDTree(ref[:, :2])
    dz = []
    for p, ii in zip(xyz, tree.query_ball_point(xyz[:, :2], r=radius)):
        if len(ii) >= 5:
            z = ref[ii, 2]
            if z.max() - z.min() < max_range:
                dz.append(p[2] - np.median(z))
    dz = np.asarray(dz)
    if len(dz) < 100:
        return {"pairs": int(len(dz)), "offset_m": 0.0}
    return {"pairs": int(len(dz)), "offset_m": round(float(np.median(dz)), 3),
            "mad_m": round(float(np.median(np.abs(dz - np.median(dz)))), 3)}


def evaluate(ours_path: Path, ref_path: Path, taus=(0.10, 0.20, 0.50), voxel: float = 0.05,
             footprint_cell: float = 2.0, sample: int = 2_000_000, seed: int = 0, z_offset: float = 0.0) -> dict:
    ours = _thin(_read(ours_path, drop_noise=True), voxel)
    ours[:, 2] -= z_offset
    b = (ours[:, 0].min(), ours[:, 1].min(), ours[:, 0].max(), ours[:, 1].max())
    ref = _thin(_read(ref_path, drop_noise=False, bounds=b), voxel)
    if len(ref) == 0:
        raise ValueError("reference and reconstruction do not overlap")
    x0, y0 = min(b[0], ref[:, 0].min()), min(b[1], ref[:, 1].min())
    shape = (int((max(b[3], ref[:, 1].max()) - y0) / footprint_cell) + 1,
             int((max(b[2], ref[:, 0].max()) - x0) / footprint_cell) + 1)
    common = _footprint(ours, x0, y0, footprint_cell, shape) & _footprint(ref, x0, y0, footprint_cell, shape)

    def inside(xyz):
        ix = ((xyz[:, 0] - x0) / footprint_cell).astype(np.int64).clip(0, shape[1] - 1)
        iy = ((xyz[:, 1] - y0) / footprint_cell).astype(np.int64).clip(0, shape[0] - 1)
        return xyz[common[iy, ix]]

    ours, ref = inside(ours), inside(ref)
    rng = np.random.default_rng(seed)
    s_ours = ours[rng.choice(len(ours), min(sample, len(ours)), replace=False)]
    s_ref = ref[rng.choice(len(ref), min(sample, len(ref)), replace=False)]
    d_ours, _ = cKDTree(ref).query(s_ours, workers=-1)      # accuracy direction
    d_ref, _ = cKDTree(ours).query(s_ref, workers=-1)       # completeness direction
    # vertical bias on near-horizontal agreement: nearest-neighbour signed dz for close pairs
    _, j = cKDTree(ref[:, :2]).query(s_ours[:, :2], workers=-1)
    dz = s_ours[:, 2] - ref[j, 2]
    out = {"common_area_m2": round(float(common.sum()) * footprint_cell ** 2),
           "points_ours": int(len(ours)), "points_ref": int(len(ref)), "voxel_m": voxel,
           "median_dist_ours_to_ref_m": round(float(np.median(d_ours)), 3),
           "median_dist_ref_to_ours_m": round(float(np.median(d_ref)), 3),
           "median_vertical_offset_m": round(float(np.median(dz[np.abs(dz) < 2])), 3)}
    for t in taus:
        p, r = float(np.mean(d_ours < t)), float(np.mean(d_ref < t))
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[f"tau_{int(round(t * 100))}cm"] = {"precision": round(p, 3), "recall": round(r, 3), "f": round(f, 3)}
    return out


def main(argv=None) -> int:
    """python -m uav3d.evaluate.geometry <run_dir> <reference.las> [products key]"""
    import sys

    args = argv if argv is not None else sys.argv[1:]
    run_dir, ref = Path(args[0]), Path(args[1])
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    results = {}
    terrain = next((s for s in run.get("stages", []) if s["name"] == "terrain"), None)
    candidates = {"odm": run["products"]["point_cloud"]}
    if terrain:
        candidates["cleaned"] = terrain["products"]["point_cloud"]
    # one vertical bias for the run (measured on ODM's own ground), applied to every candidate
    bias = flat_ground_offset(Path(candidates.get("cleaned", candidates["odm"])), ref)
    results["vertical_bias"] = bias
    print("vertical bias", json.dumps(bias))
    for name, laz in candidates.items():
        results[name] = evaluate(Path(laz), ref)
        results[name + "_aligned"] = evaluate(Path(laz), ref, z_offset=bias["offset_m"])
        print(name, json.dumps(results[name]))
        print(name + "_aligned", json.dumps(results[name + "_aligned"]))
    (run_dir / "eval_geometry.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
