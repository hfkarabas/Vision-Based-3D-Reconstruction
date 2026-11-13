"""Connect classified ground to GDAL FillNodata beneath automatic roof proposals."""

import argparse
import json
from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import xy
from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt, generate_binary_structure, iterate_structure
from shapely.geometry import MultiPoint

from omega_terrain_cleanup import aligned, interpolate, raster, save_raster


def prepare(ground, dense, polygons, output, radius, enclosed_gap_m=0, closing_cells=0, surrounded=False):
    if not 0 < radius <= 8:
        raise ValueError("Foundation interpolation must be bounded to at most 8 m.")
    base, profile = raster(ground / "dtm.tif")
    classified, source = raster(ground / "classified_ground.tif")
    counts, support = raster(dense)
    if not aligned(profile, source) or not aligned(profile, support):
        raise ValueError("Classified ground, density and DTM must share a grid.")
    features = gpd.read_file(polygons).to_crs(profile["crs"])
    if features.empty:
        raise ValueError("No automatic building proposals are available.")
    shape = base.shape
    transform = profile["transform"]
    target = geometry_mask(features.geometry.buffer(1.5), out_shape=shape, transform=transform, invert=True)
    anchors = np.isfinite(classified) & (counts > 0) & ~target
    interpolated, distance = interpolate(classified, anchors, radius, transform.a)
    measured_support = counts > 0
    support = measured_support
    if enclosed_gap_m:
        if not 0 < enclosed_gap_m <= 1:
            raise ValueError("An enclosed dense-support gap must be bounded to at most 1 m.")
        internal = binary_fill_holes(measured_support)
        if closing_cells:
            if closing_cells not in (1, 2):
                raise ValueError("Limit the closing radius to one or two terrain cells.")
            structure = iterate_structure(generate_binary_structure(2, 2), closing_cells)
            internal |= binary_closing(measured_support, structure=structure)
        support = internal & (distance_transform_edt(~measured_support) * transform.a <= enclosed_gap_m)
    if surrounded:
        # Foundation ground is inferred under the roof, where camera rays cannot
        # measure ground directly. Limit this trial to interpolation inside the
        # convex hull of nearby observed ground, with no extrapolation at edges.
        support = measured_support.copy()
        for polygon in features.geometry:
            nearby = geometry_mask([polygon.buffer(radius)], out_shape=shape, transform=transform, invert=True)
            rows, cols = np.nonzero(anchors & nearby)
            if len(rows) < 15:
                continue
            east, north = xy(transform, rows, cols)
            hull = MultiPoint(np.column_stack([east, north])).convex_hull
            if hull.geom_type == "Polygon":
                support |= geometry_mask([hull], out_shape=shape, transform=transform, invert=True)
    inferred = target & support & np.isfinite(interpolated)
    # Keep holes where the bounded interpolation cannot establish a foundation.
    result = np.where(target, np.where(inferred, interpolated, np.nan), base)
    output.mkdir(parents=True, exist_ok=False)
    save_raster(output / "dtm.tif", result, profile)
    save_raster(output / "observed.tif", anchors & np.isfinite(result), profile, mask=True)
    save_raster(output / "inferred_foundation.tif", inferred, profile, mask=True)
    origin = json.loads((ground / "complete.json").read_text(encoding="utf-8"))["origin"]
    with np.load(ground / "terrain.npz") as terrain:
        np.savez_compressed(output / "terrain.npz", x=terrain["x"], y=terrain["y"], z=result)
    report = dict(origin=origin, interpolation="GDAL FillNodata from PDAL classified-ground Z",
                  radius_m=radius, automatic_footprint_buffer_m=1.5,
                  inferred_cells=int(inferred.sum()),
                  inferred_enclosed_gap_cells=int((inferred & ~measured_support).sum()),
                  enclosed_gap_max_m=enclosed_gap_m,
                  closing_cells=closing_cells,
                  require_surrounding_anchor_hull=surrounded,
                  maximum_distance_m=float(distance[inferred].max()) if inferred.any() else None,
                  new_cells=int((inferred & ~np.isfinite(base)).sum()),
                  observed_ground_under_roof=False)
    (output / "complete.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def assemble(args):
    from types import SimpleNamespace

    from omega_uav_pipeline import measured
    from omega_uav_products import building_heights

    args.output.mkdir(parents=True, exist_ok=False)
    prepared = args.output / "ground"
    prepare(args.ground, args.dense, args.polygons, prepared, args.radius,
            args.enclosed_gap_m, args.closing_cells, args.surrounded)
    out = args.output / "buildings"
    out.mkdir()
    for name in ("terrain_photo.png", "grid.json"):
        shutil.copy2(args.asset_template / name, out / name)
    shutil.copy2(args.polygons, out / "geoai_polygons.gpkg")
    shutil.copy2(args.inference, out / "inference_provenance.json")
    meta = json.loads((out / "grid.json").read_text(encoding="utf-8"))
    adapter = SimpleNamespace(output=args.output, extent=meta["bounds"], crs=meta["crs"], pdal=args.pdal)
    statuses = building_heights(adapter, out, args.orthophoto, measured)
    script = Path(__file__).with_name("omega_building_blender.py").resolve()
    measured(out, "blender_assembly", [
        args.blender, "-b", "-t", 4, "--python-exit-code", 1, "--python", script,
        "--", "--output", out, "--height-grid", prepared / "terrain.npz", "--allow-terrain-only",
    ])
    records = json.loads((out / "reports/blender_buildings.json").read_text(encoding="utf-8"))
    result = dict(height_gate=statuses, geometry=records,
                  valid_lod1=sum(record["status"] == "built" for record in records))
    (args.output / "complete.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ground", "dense", "polygons", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=6)
    parser.add_argument("--enclosed-gap-m", type=float, default=0)
    parser.add_argument("--closing-cells", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--surrounded", action="store_true")
    for name in ("asset-template", "orthophoto", "inference", "pdal", "blender"):
        parser.add_argument(f"--{name}", type=lambda path: Path(path).resolve())
    args = parser.parse_args()
    assembly = [args.asset_template, args.orthophoto, args.inference, args.pdal, args.blender]
    if any(assembly):
        if not all(assembly):
            parser.error("Assembly requires the asset template, orthophoto, inference manifest, PDAL and Blender.")
        args.output = args.output.resolve()
        assemble(args)
    else:
        prepare(args.ground, args.dense, args.polygons, args.output, args.radius,
                args.enclosed_gap_m, args.closing_cells, args.surrounded)
