"""Derive a bounded, explicitly survey-assisted terrain candidate using PDAL and GDAL."""

import argparse
import json
from pathlib import Path
import subprocess

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.fill import fillnodata
from rasterio.vrt import WarpedVRT
from scipy.ndimage import distance_transform_edt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdal", type=Path, required=True)
    parser.add_argument("--max-fill-m", type=float, default=12)
    args = parser.parse_args()
    base, out = args.base.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    grid = json.loads((base / "grid.json").read_text(encoding="utf-8"))
    with rasterio.open(grid["dtm"]) as prior:
        old_dtm = prior.read(1, masked=True).filled(np.nan)
        profile = prior.profile
        west, south, _, _ = prior.bounds
    height, width = old_dtm.shape
    transform = profile["transform"]
    pixel = transform.a
    if pixel <= 0 or not np.isclose(pixel, -transform.e) or transform.b != 0 or transform.d != 0:
        raise ValueError("PDAL ground fallback requires a north-up DTM with square cells.")
    dsm_path = out / "dsm_on_terrain_grid.tif"
    vrt_options = dict(
        crs=profile["crs"], transform=transform, width=width, height=height, resampling=Resampling.average
    )
    with rasterio.open(base / "survey_dsm.tif") as src, WarpedVRT(src, **vrt_options) as vrt:
        surface = vrt.read(1, masked=True).filled(np.nan)
    with rasterio.open(dsm_path, "w", **profile) as raster:
        raster.write(np.where(np.isfinite(surface), surface, -9999).astype("float32"), 1)

    pipeline_path = out / "pdal_ground_pipeline.json"
    classified = out / "pdal_classified_ground.tif"
    pipeline = [
        dict(type="readers.gdal", filename=str(dsm_path), header="Z"),
        dict(type="filters.expression", expression="Z > -9998"),
        dict(type="filters.smrf", scalar=1.2, slope=0.2, threshold=0.45, window=18.0),
        dict(type="filters.expression", expression="Classification == 2"),
        dict(
            type="writers.gdal", filename=str(classified), output_type="min", data_type="float32",
            origin_x=west, origin_y=south, resolution=pixel, width=width, height=height,
            radius=pixel * 0.51, nodata=-9999, override_srs=grid["crs"],
        ),
    ]
    pipeline_path.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
    subprocess.run([str(args.pdal.resolve()), "pipeline", str(pipeline_path)], check=True)
    with rasterio.open(classified) as src:
        if src.shape != old_dtm.shape or src.transform != transform or src.crs != profile["crs"]:
            raise ValueError("PDAL output does not match the original terrain grid.")
        ground = src.read(1, masked=True).filled(np.nan)
    polygons = gpd.read_file(base / "geoai_polygons.gpkg").to_crs(profile["crs"])
    excluded = geometry_mask(
        polygons.geometry.buffer(1.0), out_shape=ground.shape, transform=transform, invert=True
    )
    observed = np.isfinite(ground) & ~excluded
    if not observed.any():
        raise ValueError("No classified ground samples remain outside the building polygons.")
    # GDAL interpolates between classified ground. Explicitly bound Euclidean
    # distance and keep remaining holes as NoData.
    distance = distance_transform_edt(~observed) * pixel
    filled = fillnodata(
        np.where(observed, ground, np.nan).astype("float32"),
        mask=observed.astype("uint8"),
        max_search_distance=args.max_fill_m / pixel,
        smoothing_iterations=0,
    )
    valid = np.isfinite(filled) & (distance <= args.max_fill_m)
    values = np.where(valid, filled, np.nan).astype("float32")
    for name, array, dtype, nodata in [
        ("dtm", np.where(valid, values, -9999), "float32", -9999),
        ("support", valid, "uint8", 0),
        ("observed_ground", observed, "uint8", 0),
        ("distance_to_ground", np.where(valid, distance, -9999), "float32", -9999),
    ]:
        with rasterio.open(out / f"{name}.tif", "w", **dict(profile, dtype=dtype, nodata=nodata)) as dst:
            dst.write(array.astype(dtype), 1)
    origin = grid["origin_enu"]
    np.savez_compressed(
        out / "blender_height_grid.npz",
        z=values,
        x=west + (np.arange(width) + 0.5) * pixel - origin[0],
        y=transform.f + (np.arange(height) + 0.5) * transform.e - origin[1],
    )
    overlap = observed & np.isfinite(old_dtm)
    differences = ground[overlap] - old_dtm[overlap]
    report = dict(
        method="AirZoo survey DSM -> PDAL SMRF -> detected roof exclusion -> bounded GDAL FillNodata",
        source_dsm=grid["dsm"],
        crs=grid["crs"],
        origin_enu=origin,
        pixel_m=pixel,
        cells_total=int(valid.size),
        observed_classified_ground_cells=int(observed.sum()),
        interpolated_cells=int((valid & ~observed).sum()),
        nodata_cells=int((~valid).sum()),
        maximum_fill_distance_m=float(distance[valid].max()),
        previous_uav_dtm_overlap_ground_cells=int(overlap.sum()),
        survey_minus_previous_uav_dtm_p10_p50_p90_m=np.percentile(differences, [10, 50, 90]).tolist(),
        caveat="Reference DSM contributes elevation. Ground under detected roofs is interpolated, and absolute vertical datum is unverified.",
    )
    (out / "survey_ground_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
