"""Reference-free per-tile quality signals from a coarse reconstruction (adaptive scheduler input).

All signals come from products of the coarse pass only; no reference data is read here.
  overlap    median number of images seeing the tile (ODM opensfm/stats/overlap.tif)
  void       share of DSM cells without data inside the tile
  density    coarse points per m2 in the tile, divided by the block median (1 = typical)
  roughness  RMS residual of a least-squares plane fitted to the tile's DSM [m]
  relief     p95 - p5 of the tile's DSM [m]
Tiles share the grid of uav3d.evaluate.tiles (origin, size, shape in the products' CRS).
"""
from __future__ import annotations

import laspy
import numpy as np
import rasterio
from rasterio.windows import from_bounds

from uav3d.refine import tile_index


def _window(ds, b):
    w = from_bounds(*b, transform=ds.transform).round_offsets().round_lengths()
    return ds.read(1, window=w, boundless=True, masked=True)


def plane_rms(z: np.ma.MaskedArray) -> float | None:
    """RMS residual of a plane fitted to the valid cells of a 2-D grid (None if < 10 valid cells)."""
    if z.count() < 10:
        return None
    iy, ix = np.nonzero(~np.ma.getmaskarray(z))
    v = z.compressed().astype(float)
    a = np.c_[ix, iy, np.ones(len(ix))]
    coef, *_ = np.linalg.lstsq(a, v, rcond=None)
    return float(np.sqrt(np.mean((v - a @ coef) ** 2)))


def point_counts(laz_path, origin, size, shape, chunk: int = 20_000_000) -> np.ndarray:
    n = np.zeros(shape[0] * shape[1], dtype=np.int64)
    with laspy.open(laz_path) as f:
        for pts in f.chunk_iterator(chunk):
            ids = tile_index(np.c_[pts.x, pts.y], origin, size, shape)
            n += np.bincount(ids[ids >= 0], minlength=len(n))
    return n


def tile_signals(dsm_path, overlap_path, laz_path, origin, size: float, shape) -> list[dict]:
    counts = point_counts(laz_path, origin, size, shape)
    dens = counts / size ** 2
    med = float(np.median(dens[dens > 0])) if (dens > 0).any() else 1.0
    rows = []
    with rasterio.open(dsm_path) as dsm, rasterio.open(overlap_path) as ov:
        for tid in range(shape[0] * shape[1]):
            iy, ix = divmod(tid, shape[1])
            b = (origin[0] + ix * size, origin[1] + iy * size, origin[0] + (ix + 1) * size, origin[1] + (iy + 1) * size)
            z = _window(dsm, b)
            o = _window(ov, b)
            ov_valid = o.compressed()
            ov_valid = ov_valid[ov_valid > 0]
            zc = z.compressed()
            rows.append({
                "id": tid, "ix": ix, "iy": iy, "bounds": list(b),
                "overlap": float(np.median(ov_valid)) if len(ov_valid) else 0.0,
                "void": round(1.0 - z.count() / z.size, 4) if z.size else 1.0,
                "density": round(float(dens[tid]) / med, 4),
                "roughness": None if (r := plane_rms(z)) is None else round(r, 3),
                "relief": round(float(np.percentile(zc, 95) - np.percentile(zc, 5)), 3) if len(zc) else None,
            })
    return rows
