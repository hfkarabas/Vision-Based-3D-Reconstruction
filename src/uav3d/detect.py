"""Image evidence for buildings: SAM 3 concept segmentation of the orthophoto.

The orthophoto is resampled to a fixed ground resolution (buildings must look like buildings
to an image model, whatever the flight GSD), cut into overlapping tiles, segmented by SAM 3 in
its own environment (external tool, like ODM and Roofer) and mosaicked into one raster on the
same grid: the highest instance score per cell (0-100, 0 = no building).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject

from uav3d.record import run_measured

RUNNER = Path(__file__).resolve().parent / "external" / "sam3_runner.py"


class DetectError(RuntimeError):
    pass


def tile_orthophoto(ortho: str, work: Path, res: float, tile_px: int, overlap: float) -> dict:
    """Resample the orthophoto to ``res`` and write overlapping PNG tiles; return the grid."""
    with rasterio.open(ortho) as ds:
        b, crs = ds.bounds, ds.crs
        width, height = int(np.ceil((b.right - b.left) / res)), int(np.ceil((b.top - b.bottom) / res))
        transform = Affine(res, 0, b.left, 0, -res, b.top)
        bands = [1, 2, 3, 4] if ds.count >= 4 else [1, 2, 3]
        img = np.zeros((len(bands), height, width), dtype="uint8")
        for i, band in enumerate(bands):
            reproject(rasterio.band(ds, band), img[i], dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.average)
    valid = img[3] > 0 if len(bands) == 4 else img[:3].max(axis=0) > 0
    step = int(tile_px * (1 - overlap))
    tiles = []
    work.mkdir(parents=True, exist_ok=True)
    for r0 in range(0, max(1, height - tile_px + step), step):
        for c0 in range(0, max(1, width - tile_px + step), step):
            r1, c1 = min(r0 + tile_px, height), min(c0 + tile_px, width)
            if valid[r0:r1, c0:c1].mean() < 0.05:
                continue
            png = work / f"t_{r0:05d}_{c0:05d}.png"
            Image.fromarray(np.moveaxis(img[:3, r0:r1, c0:c1], 0, -1)).save(png)
            tiles.append({"png": str(png), "npz": str(png.with_suffix(".npz")), "row": r0, "col": c0})
    return {"transform": transform, "crs": crs, "shape": (height, width), "valid": valid, "tiles": tiles}


def mosaic(grid: dict, min_score: float) -> np.ndarray:
    """Highest instance score (0-100) per cell; instances below min_score are ignored."""
    out = np.zeros(grid["shape"], dtype="uint8")
    for t in grid["tiles"]:
        d = np.load(t["npz"])
        for mask, score in zip(d["masks"], d["scores"]):
            if score < min_score:
                continue
            h, w = mask.shape
            view = out[t["row"]:t["row"] + h, t["col"]:t["col"] + w]
            np.maximum(view, np.where(mask, np.uint8(round(float(score) * 100)), np.uint8(0)), out=view)
    out[~grid["valid"]] = 0
    return out


def detect(python: Path, ortho: str, out_dir: Path, cfg: dict, log: Path) -> dict:
    dcfg = cfg["detection"]
    work = out_dir / "sam3_tiles"
    grid = tile_orthophoto(ortho, work, dcfg["tile_res_m"], dcfg["tile_px"], dcfg["tile_overlap"])
    if not grid["tiles"]:
        raise DetectError("orthophoto has no valid pixels to segment")
    manifest = work / "manifest.json"
    manifest.write_text(json.dumps({"prompts": dcfg["prompts"], "tiles": grid["tiles"]}), encoding="utf-8")
    info = run_measured([str(python), str(RUNNER), str(manifest)], log)
    if info["returncode"] != 0:
        raise DetectError(f"SAM 3 runner failed (exit {info['returncode']}); see {log}")
    scores = mosaic(grid, dcfg["min_score"])
    raster = out_dir / "building_score.tif"
    with rasterio.open(raster, "w", driver="GTiff", height=scores.shape[0], width=scores.shape[1], count=1,
                       dtype="uint8", crs=grid["crs"], transform=grid["transform"], nodata=0,
                       compress="deflate") as dst:
        dst.write(scores, 1)
    for t in grid["tiles"]:  # tiles are intermediate
        Path(t["png"]).unlink(missing_ok=True)
        Path(t["npz"]).unlink(missing_ok=True)
    info.update(raster=str(raster), tiles=len(grid["tiles"]), res_m=dcfg["tile_res_m"], prompts=dcfg["prompts"],
                building_share=round(float((scores > 0)[grid["valid"]].mean()), 4))
    return info
