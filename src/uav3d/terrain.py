"""Clean terrain: remove impossible low points, re-classify ground, rebuild DTM and DSM.

Photogrammetric clouds contain points below the true surface. Two kinds matter:
- isolated low noise (mismatches): removed with ELM (Chen et al. 2012) and a statistical
  outlier filter, as the SMRF authors recommend before ground filtering (Pingel et al. 2013);
- coherent "virtual" surfaces mirrored in water (ponds, flooded fields): dense and
  geometrically consistent, so ELM keeps them. They are removed with a coarse low-surface test:
  a point far below a robust local reference surface (median of 5 m cell medians over ~25 m),
  with a tolerance that grows with the local slope, cannot be a surface seen from above.
SMRF then re-classifies ground with ODM's own parameters; ODM's semantic classes (3 vegetation,
6 building) are kept for non-ground points.

The DTM is gridded at a coarse cell (terrain detail, not object detail), gap-filled inside the
surveyed area, median-smoothed and resampled to ODM's DEM grid, so later stages swap it in
unchanged. The DSM is regridded from the cleaned cloud on ODM's grid.
"""
from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np
import rasterio
from rasterio.fill import fillnodata
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from uav3d.record import run_measured

NOISE = 7


class TerrainError(RuntimeError):
    pass


def _grid(ref, cell: float | None = None) -> dict:
    with rasterio.open(ref) as ds:
        res = cell or ds.res[0]
        return {"origin_x": ds.bounds.left, "origin_y": ds.bounds.bottom, "resolution": res,
                "width": int(np.ceil((ds.bounds.right - ds.bounds.left) / res)),
                "height": int(np.ceil((ds.bounds.top - ds.bounds.bottom) / res))}


def _pdal(odm, pipeline: list, work: Path, name: str, log: Path) -> dict:
    path = work / f"{name}.pipeline.json"
    path.write_text(json.dumps(pipeline, indent=1), encoding="utf-8")
    info = run_measured([str(odm.exe("pdal")), "pipeline", str(path)], log, env=odm.env())
    if info["returncode"] != 0:
        raise TerrainError(f"PDAL {name} failed (exit {info['returncode']}); see {log}")
    return info


def _writer(path: Path, grid: dict, output_type: str) -> dict:
    return {"type": "writers.gdal", "filename": str(path), "output_type": output_type,
            "resolution": grid["resolution"], "origin_x": grid["origin_x"], "origin_y": grid["origin_y"],
            "width": grid["width"], "height": grid["height"], "data_type": "float32", "nodata": -9999,
            "gdaldriver": "GTiff", "gdalopts": "COMPRESS=DEFLATE,TILED=YES,BIGTIFF=IF_SAFER"}


def low_surface_mask(x, y, z, cell_m: float, window_cells: int, base_tolerance_m: float) -> np.ndarray:
    """True for points below a robust local reference surface by more than the tolerance.

    Reference: per-cell median z, then a median over window_cells x window_cells cells.
    Tolerance: base + local slope (from the reference) x half the window, so steep but real
    terrain is not cut."""
    ix = ((x - x.min()) / cell_m).astype(np.int64)
    iy = ((y - y.min()) / cell_m).astype(np.int64)
    shape = (int(iy.max()) + 1, int(ix.max()) + 1)
    key = iy * shape[1] + ix
    order = np.argsort(key, kind="stable")
    keys, starts = np.unique(key[order], return_index=True)
    zs = z[order]
    cell_med = np.full(shape, np.nan)
    cell_med.flat[keys] = [np.median(zs[s:e]) for s, e in zip(starts, np.append(starts[1:], len(zs)))]
    observed = ~np.isnan(cell_med)
    ref = ndimage.median_filter(np.where(observed, cell_med, np.nanmedian(cell_med)), size=window_cells)
    gy, gx = np.gradient(ref, cell_m)
    tolerance = base_tolerance_m + np.hypot(gx, gy) * cell_m * window_cells / 2
    return z < (ref - tolerance)[iy, ix]


def classify(odm, laz: str, work: Path, tcfg: dict, log: Path) -> tuple[Path, dict]:
    """ELM + statistical outliers (PDAL) -> low-surface test (numpy) -> SMRF (PDAL)."""
    stage1, cleaned = work / "stage1.laz", work / "cleaned.laz"
    info = {"denoise": _pdal(odm, [
        laz,
        {"type": "filters.ferry", "dimensions": "Classification=>OrigClass"},
        {"type": "filters.assign", "value": "Classification = 0"},
        {"type": "filters.elm", **tcfg["elm"]},
        {"type": "filters.outlier", **tcfg["outlier"]},
        {"type": "writers.las", "filename": str(stage1), "compression": "laszip", "forward": "all",
         "extra_dims": "OrigClass=uint8"},
    ], work, "denoise", log)}
    las = laspy.read(stage1)
    cls = np.asarray(las.classification).copy()
    keep = cls != NOISE
    low = np.zeros(len(cls), dtype=bool)
    lcfg = tcfg["low_surface"]
    low[keep] = low_surface_mask(np.asarray(las.x)[keep], np.asarray(las.y)[keep], np.asarray(las.z)[keep],
                                 lcfg["cell_m"], lcfg["window_cells"], lcfg["base_tolerance_m"])
    cls[low] = NOISE
    las.classification = cls
    las.write(stage1)
    info["points"] = int(len(cls))
    info["noise_elm_outlier"] = int((~keep).sum())
    info["noise_low_surface"] = int(low.sum())
    info["smrf"] = _pdal(odm, [
        str(stage1),
        {"type": "filters.smrf", "ignore": f"Classification[{NOISE}:{NOISE}]", **tcfg["smrf"]},
        # non-ground keeps ODM's semantic class, unless ODM had called it ground
        {"type": "filters.assign", "value": "Classification = OrigClass WHERE Classification == 1 && OrigClass != 2"},
        {"type": "writers.las", "filename": str(cleaned), "compression": "laszip"},
    ], work, "smrf", log)
    stage1.unlink(missing_ok=True)
    return cleaned, info


def build_dtm(odm, cleaned: Path, odm_dtm: str, odm_dsm: str, gsd_m: float | None, work: Path,
              tcfg: dict, log: Path) -> dict:
    target = _grid(odm_dtm)
    cell = float(np.clip((gsd_m or target["resolution"]) * tcfg["dtm_cell_gsd_factor"],
                         tcfg["dtm_cell_min_m"], tcfg["dtm_cell_max_m"]))
    coarse = _grid(odm_dtm, cell)
    raw = work / "dtm_coarse.tif"
    info = {"raster": _pdal(odm, [str(cleaned), {"type": "filters.range", "limits": "Classification[2:2]"},
                                  _writer(raw, coarse, "idw")], work, "dtm", log)}
    with rasterio.open(raw) as ds:
        data, profile = ds.read(1), ds.profile
    valid = data != -9999
    survey = _survey_mask(odm_dsm, profile)
    filled = fillnodata(data, mask=valid.astype(np.uint8),
                        max_search_distance=max(1, int(tcfg["fill_max_distance_m"] / cell)),
                        smoothing_iterations=tcfg["fill_smoothing_iterations"])
    size = max(3, int(round(tcfg["median_m"] / cell)) | 1)
    smooth = ndimage.median_filter(filled, size=size)
    smooth = np.where((filled != -9999) & survey, smooth, -9999).astype("float32")
    with rasterio.open(odm_dtm) as ref:
        out_profile = {**ref.profile, "dtype": "float32", "nodata": -9999, "compress": "deflate", "tiled": True}
        fine = np.full((ref.height, ref.width), -9999, dtype="float32")
        reproject(smooth, fine, src_transform=profile["transform"], src_crs=ref.crs, src_nodata=-9999,
                  dst_transform=ref.transform, dst_crs=ref.crs, dst_nodata=-9999, resampling=Resampling.bilinear)
    fine[~_survey_mask(odm_dsm, out_profile)] = -9999
    out = work / "dtm.tif"
    with rasterio.open(out, "w", **out_profile) as dst:
        dst.write(fine, 1)
    raw.unlink(missing_ok=True)
    info.update(cell_m=round(cell, 3), ground_coverage=round(float(valid[survey].mean()), 3) if survey.any() else 0.0,
                median_px=size)
    return info


def fill_small_gaps(path: Path, odm_dsm: str, max_distance_m: float) -> dict:
    """Fill cells without points (shadows, sparse matches) up to max_distance_m from data,
    inside the surveyed area only; ODM's own DSM is gap-filled the same way."""
    with rasterio.open(path) as ds:
        data, profile, res = ds.read(1), ds.profile, ds.res[0]
    valid = data != -9999
    survey = _survey_mask(odm_dsm, profile)
    filled = fillnodata(data, mask=valid.astype(np.uint8), max_search_distance=max(1, int(max_distance_m / res)),
                        smoothing_iterations=0)
    filled = np.where(survey & (filled != -9999), filled, -9999).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(filled, 1)
    return {"observed_share": round(float(valid[survey].mean()), 3) if survey.any() else 0.0,
            "filled_share": round(float((filled != -9999)[survey].mean()), 3) if survey.any() else 0.0}


def _survey_mask(odm_dsm: str, profile: dict) -> np.ndarray:
    """ODM DSM footprint resampled to the given grid: where the survey actually observed.
    (ODM's DSM and DTM grids can differ by a column, so always resample.)"""
    with rasterio.open(odm_dsm) as ds:
        src = (ds.read(1) != ds.nodata).astype("uint8") if ds.nodata is not None else np.ones(ds.shape, "uint8")
        out = np.zeros((profile["height"], profile["width"]), dtype="uint8")
        reproject(src, out, src_transform=ds.transform, src_crs=ds.crs, dst_transform=profile["transform"],
                  dst_crs=ds.crs, resampling=Resampling.max)
    return out.astype(bool)


def clean(odm, laz: str, odm_dtm: str, odm_dsm: str, gsd_m: float | None, cfg: dict, work: Path, log: Path) -> dict:
    """Write cleaned.laz, dtm.tif and dsm.tif into ``work``; return products + measurements."""
    tcfg = cfg["terrain"]
    work.mkdir(parents=True, exist_ok=True)
    cleaned, info = classify(odm, laz, work, tcfg, log)
    info["dtm"] = build_dtm(odm, cleaned, odm_dtm, odm_dsm, gsd_m, work, tcfg, log)
    dsm = work / "dsm.tif"
    info["dsm"] = _pdal(odm, [str(cleaned), {"type": "filters.range", "limits": f"Classification![{NOISE}:{NOISE}]"},
                              _writer(dsm, _grid(odm_dtm), "max")], work, "dsm", log)
    info["dsm"].update(fill_small_gaps(dsm, odm_dsm, tcfg["dsm_fill_max_distance_m"]))
    info["health_odm_dtm"] = health(odm_dtm, tcfg)
    info["health_dtm"] = health(work / "dtm.tif", tcfg)
    info["products"] = {"point_cloud": str(cleaned), "dtm": str(work / "dtm.tif"), "dsm": str(dsm)}
    return info


def overlap_summary(overlap_tif) -> dict:
    """Share of the surveyed area seen by n images (ODM's opensfm/stats/overlap.tif).

    Reference-free reliability signal: on the AirZoo rural tune set the DTM error grows
    steadily as overlap falls (median |error| 0.5 m at >= 8 images, 2.3 m at 2)."""
    with rasterio.open(overlap_tif) as ds:
        a = ds.read(1, masked=True).compressed()
    if a.size == 0:
        return {}
    bins = {"lt3": a < 3, "3to4": (a >= 3) & (a < 5), "5to7": (a >= 5) & (a < 8), "ge8": a >= 8}
    return {"share_" + k: round(float(v.mean()), 3) for k, v in bins.items()} | {"median": float(np.median(a))}


def health(dtm_path, tcfg: dict) -> dict:
    """Reference-free DTM sanity: share of cells more than pit_depth_m below (pits) or above
    (spikes) their local median over local_window_m, and the deepest pit. Decimated grid."""
    with rasterio.open(dtm_path) as ds:
        step = max(1, int(round(tcfg["health_cell_m"] / ds.res[0])))
        a = ds.read(1, masked=True)[::step, ::step]
        res = ds.res[0] * step
    if a.count() == 0:
        return {"pit_fraction": None}
    filled = a.filled(np.ma.median(a))
    win = max(3, int(round(tcfg["local_window_m"] / res)) | 1)
    dev = (filled - ndimage.median_filter(filled, size=win))[~np.ma.getmaskarray(a)]
    return {"pit_fraction": round(float((dev < -tcfg["pit_depth_m"]).mean()), 4),
            "spike_fraction": round(float((dev > tcfg["pit_depth_m"]).mean()), 4),
            "deepest_pit_m": round(float(dev.min()), 2), "cell_m": round(res, 3)}
