"""Prepare a bounded UAV RGB/height grid and turn GeoAI polygons into LOD1 inputs."""

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon, box, mapping
from shapely.geometry.polygon import orient


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def on_grid(source, profile, crs, bands, resampling, nodata):
    with rio.open(source) as src, WarpedVRT(
        src,
        src_crs=crs,
        crs=profile["crs"],
        transform=profile["transform"],
        width=profile["width"],
        height=profile["height"],
        nodata=nodata,
        dtype="float32" if math.isnan(nodata) else "uint8",
        resampling=resampling,
        warp_mem_limit=64,
    ) as vrt:
        return vrt.read(bands)


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with rio.open(args.dtm) as dtm:
        crs = dtm.crs
        bounds = dtm.bounds
        if not crs or not crs.is_projected or crs.linear_units_factor[1] != 1:
            raise ValueError("A projected metre-based DTM is required.")
        terrain_grid = dtm.crs, dtm.transform, dtm.shape
    for path in {args.support, args.observed_support or args.support}:
        with rio.open(path) as support:
            if (support.crs, support.transform, support.shape) != terrain_grid:
                raise ValueError(f"Terrain support grid must match the DTM: {path}")
    metadata_crs = (
        CRS.from_string(ET.parse(args.srs_metadata).getroot().findtext("SRS"))
        if args.srs_metadata
        else None
    )
    with rio.open(args.rgb) as rgb, rio.open(args.dsm) as dsm:
        if rgb.count < 3 or rgb.dtypes[:3] != ("uint8",) * 3:
            raise ValueError("The source must provide three visible 8-bit RGB bands.")
        if metadata_crs and not args.rgb_crs and not args.dsm_crs and (
            rgb.transform != dsm.transform or rgb.shape != dsm.shape
        ):
            raise ValueError("A shared CRS override requires co-registered source grids.")
        source_crs = args.rgb_crs or metadata_crs or rgb.crs
        dsm_crs = args.dsm_crs or metadata_crs or dsm.crs
        if not source_crs or not dsm_crs:
            raise ValueError("Authoritative CRS information is required for both rasters.")
        source_meta = dict(
            rgb_original_crs=str(rgb.crs),
            dsm_original_crs=str(dsm.crs),
            rgb_assigned_crs=str(source_crs),
            dsm_assigned_crs=str(dsm_crs),
            rgb_native_affine=list(rgb.transform)[:6],
            dsm_native_affine=list(dsm.transform)[:6],
            rgb_native_shape=list(rgb.shape),
        )
    if not math.isfinite(args.gsd) or args.gsd <= 0:
        raise ValueError("Ground sample distance must be finite and positive.")
    width = math.ceil((bounds.right - bounds.left) / args.gsd)
    height = math.ceil((bounds.top - bounds.bottom) / args.gsd)
    if width * height > args.max_pixels:
        raise ValueError("DTM region exceeds the inference grid limit. Supply a smaller DTM or coarser GSD.")
    profile = dict(
        driver="GTiff",
        width=width,
        height=height,
        transform=from_bounds(*bounds, width, height),
        crs=crs,
        compress="deflate",
        tiled=True,
    )
    rgb = on_grid(args.rgb, profile, source_crs, [1, 2, 3], Resampling.average, 0)
    if np.mean(np.any(rgb != 0, axis=0)) < 0.1:
        raise ValueError("RGB and DTM have insufficient overlap. Check the source CRS.")
    with rio.open(out / "inference_rgb.tif", "w", count=3, dtype="uint8", **profile) as target:
        target.write(rgb)
    from PIL import Image

    Image.fromarray(np.moveaxis(rgb, 0, -1)).save(out / "terrain_photo.png")
    for path, label, src_crs, method in [
        (args.dsm, "survey_dsm", dsm_crs, Resampling.average),
        (args.dtm, "terrain_dtm", crs, Resampling.bilinear),
        (args.support, "terrain_support", crs, Resampling.nearest),
        (args.observed_support or args.support, "observed_ground", crs, Resampling.nearest),
    ]:
        values = on_grid(path, profile, src_crs, [1], method, float("nan"))
        with rio.open(out / f"{label}.tif", "w", count=1, dtype="float32", nodata=np.nan, **profile) as target:
            target.write(values)
    write_json(
        out / "grid.json",
        dict(
            **source_meta,
            rgb=str(args.rgb.resolve()),
            dsm=str(args.dsm.resolve()),
            dtm=str(args.dtm.resolve()),
            rgb_role=args.rgb_role,
            dsm_role=args.dsm_role,
            dtm_role=args.dtm_role,
            terrain_support=str(args.support.resolve()),
            observed_ground=str((args.observed_support or args.support).resolve()),
            source_crs_metadata=str(args.srs_metadata.resolve()) if args.srs_metadata else None,
            crs=crs.to_string(),
            bounds=list(bounds),
            gsd_m=[abs(profile["transform"].a), abs(profile["transform"].e)],
            origin_enu=[(bounds.left + bounds.right) / 2, (bounds.top + bounds.bottom) / 2, 0],
        ),
    )


def values(data, mask):
    return data[mask & np.isfinite(data)]


def fraction(data, mask):
    return float(np.count_nonzero(mask & np.isfinite(data)) / max(1, mask.sum()))


def elevation(poly, reference, dsm, dtm, support, observed, all_roofs, probability, config):
    ring_geom = poly.buffer(7)
    try:
        window = geometry_window(reference, [mapping(ring_geom)])
    except rio.errors.WindowError:
        return {"status": "outside_grid"}
    r, c = window.toslices()
    shape = (int(window.height), int(window.width))
    transform = reference.window_transform(window)

    def mask(geom):
        return geometry_mask([mapping(geom)], out_shape=shape, transform=transform, invert=True)

    footprint = mask(poly)
    core = poly.buffer(-0.9)
    roof = mask(core) if not core.is_empty else footprint
    ring = mask(ring_geom.difference(poly.buffer(1.5))) & ~all_roofs[r, c]
    confidence = values(probability[r, c], roof)
    ground = values(dtm[r, c], footprint)
    roof_samples = values(dsm[r, c], roof)
    surround = values(dsm[r, c], ring)
    paired = ring & np.isfinite(dtm[r, c]) & np.isfinite(dsm[r, c]) & (observed[r, c] > 0)
    result = dict(
        mean_building_probability=float(confidence.mean()) if len(confidence) else None,
        dtm_fraction=fraction(dtm[r, c], footprint),
        supported_fraction=float(np.count_nonzero(footprint & (support[r, c] > 0)) / max(1, footprint.sum())),
        roof_sample_count=len(roof_samples),
        ground_annulus_sample_count=len(surround),
        paired_ground_sample_count=int(paired.sum()),
    )
    if result["dtm_fraction"] < 0.98 or result["supported_fraction"] < 0.95:
        return dict(result, status="insufficient_terrain_support")
    if min(len(roof_samples), len(surround), paired.sum()) < 15:
        return dict(result, status="insufficient_height_samples")
    # Use the lower part of the annulus to avoid roofs/vegetation. Estimate a
    # *local* vertical offset on those same ground cells, never an absolute datum.
    ground_level = float(np.percentile(surround, 30))
    samples = paired & (dsm[r, c] <= ground_level + 0.8)
    offsets = (dsm[r, c] - dtm[r, c])[samples]
    if len(offsets) < 15:
        return dict(result, status="insufficient_paired_ground")
    offset = float(np.median(offsets))
    spread = float(np.percentile(offsets, 90) - np.percentile(offsets, 10))
    roof_level = float(np.median(roof_samples))
    base = float(np.median(ground))
    roof_z = roof_level - offset
    result.update(
        dsm_roof_m=roof_level,
        dsm_annulus_p30_m=ground_level,
        dsm_minus_dtm_local_ground_m=offset,
        ground_offset_p90_p10_m=spread,
        dsm_roof_ground_relief_m=roof_level - ground_level,
        dsm_roof_p90_p10_m=float(np.percentile(roof_samples, 90) - np.percentile(roof_samples, 10)),
        dtm_footprint_median_m=base,
        dtm_footprint_p95_p05_m=float(np.percentile(ground, 95) - np.percentile(ground, 5)),
        roof_in_dtm_frame_m=roof_z,
        lod1_height_m=roof_z - base,
    )
    if spread > config.max_ground_spread or result["dtm_footprint_p95_p05_m"] > config.max_ground_spread:
        return dict(result, status="unstable_ground")
    if min(result["lod1_height_m"], result["dsm_roof_ground_relief_m"]) < config.min_height:
        return dict(result, status="insufficient_positive_roof_relief")
    if max(result["lod1_height_m"], result["dsm_roof_ground_relief_m"]) > config.max_height:
        return dict(result, status="height_out_of_bounds")
    return dict(result, status="eligible")


def assess(args):
    import geopandas as gpd

    out = args.output
    provenance = json.loads((out / "grid.json").read_text(encoding="utf-8"))
    polygons = gpd.read_file(args.polygons)
    with rio.open(out / "inference_rgb.tif") as ref:
        if polygons.crs != ref.crs:
            polygons = polygons.to_crs(ref.crs)
        planes = []
        for name in ("survey_dsm", "terrain_dtm", "terrain_support", "observed_ground", "building_probabilities"):
            with rio.open(out / f"{name}.tif") as src:
                if (src.transform, src.shape, src.crs) != (ref.transform, ref.shape, ref.crs):
                    raise ValueError(f"Grid mismatch: {name}")
                planes.append(src.read(1, masked=True).astype("float32").filled(np.nan))
        dsm, dtm, support, observed, probabilities = planes
        with rio.open(out / "building_masks.tif") as src:
            roofs = src.read(1) == 1
        features = []
        complete = box(*ref.bounds).buffer(-max(ref.res))
        for index, geom in enumerate(polygons.geometry, start=1):
            if not isinstance(geom, Polygon) or not geom.is_valid or geom.is_empty:
                raise ValueError(f"Expected valid polygon from GeoAI, row {index}")
            # Strip duplicate and collinear raster staircase vertices at a
            # tolerance well below the raster resolution.
            geom = orient(geom.simplify(0.02, preserve_topology=True), sign=1.0)
            props = (
                elevation(geom, ref, dsm, dtm, support, observed, roofs, probabilities, args)
                if complete.covers(geom)
                else {"status": "footprint_clipped_by_inference_extent"}
            )
            props.update(
                building_id=f"Building_{index:04}",
                area_m2=geom.area,
                height_provenance=(
                    f"{provenance['dsm_role']} DSM roof, matched to {provenance['dtm_role']} DTM "
                    "through paired local ground samples"
                ),
                footprint_provenance="semantic segmentation of orthophoto roof; proxy for ground footprint",
            )
            if props["status"] == "eligible":
                triangles = constrained_delaunay_triangles(geom)
                if not np.isclose(sum(t.area for t in triangles.geoms), geom.area, rtol=1e-6):
                    raise ValueError(f"Roof triangulation did not preserve the footprint, row {index}")
                props["foundation_triangles_xy"] = [list(t.exterior.coords)[:-1] for t in triangles.geoms]
            features.append(dict(type="Feature", properties=props, geometry=mapping(geom)))
    write_json(out / "building_footprints_projected.json", dict(type="FeatureCollection", features=features))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    for name in ("rgb", "dsm", "dtm", "support", "output"):
        prep.add_argument(f"--{name}", type=Path, required=True)
    prep.add_argument("--srs-metadata", type=Path)
    prep.add_argument("--rgb-crs", type=CRS.from_string)
    prep.add_argument("--dsm-crs", type=CRS.from_string)
    for name in ("rgb", "dsm", "dtm"):
        prep.add_argument(f"--{name}-role", choices=("reconstruction", "reference"), required=True)
    prep.add_argument("--observed-support", type=Path)
    prep.add_argument("--gsd", type=float, default=0.3)
    prep.add_argument("--max-pixels", type=int, default=8_000_000)
    prep.set_defaults(run=prepare)
    check = modes.add_parser("assess")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--polygons", type=Path, required=True)
    check.add_argument("--min-height", type=float, default=2.0)
    check.add_argument("--max-height", type=float, default=120.0)
    check.add_argument("--max-ground-spread", type=float, default=4.0)
    check.set_defaults(run=assess)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
