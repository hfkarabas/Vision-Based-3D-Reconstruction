"""Use PDAL to rasterize a supported COLMAP cloud and prepare the existing terrain workflow."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from rasterio.fill import fillnodata

import omega_terrain_cleanup as terrain


def dump(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def execute(pdal, directory, name, pipeline, *options):
    recipe = directory / f"{name}.json"
    dump(recipe, pipeline)
    started = time.monotonic()
    with (directory / f"{name}.log").open("wb") as log:
        subprocess.run([str(pdal), "pipeline", str(recipe), *map(str, options)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    return time.monotonic() - started


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with args.settings.open(encoding="utf-8") as stream:
        settings = json.load(stream)["pilot"]
    translation = np.eye(4)
    translation[:3, 3] = args.origin
    if args.transform:
        matrix = np.loadtxt(args.transform)
        if matrix.shape == (3, 4):
            matrix = np.vstack([matrix, [0, 0, 0, 1]])
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Expected a finite COLMAP 3x4 or 4x4 transform.")
        translation = translation @ matrix
    pdal = args.pdal.resolve()
    reader = dict(type="readers.ply", filename=str(args.cloud.resolve()))
    steps = [reader, dict(type="filters.transformation", matrix=" ".join(map(str, translation.flat)))]
    meta = out / "stats-meta.json"
    times = {"stats": execute(pdal, out, "stats", [*steps, dict(type="filters.stats", dimensions="X,Y,Z"),
                                                 dict(type="writers.null")], "--metadata", meta, "--stream")}
    statistics = json.loads(meta.read_text(encoding="utf-8"))["stages"]["filters.stats"]["statistic"]
    dimensions = {row["name"]: row for row in statistics}
    if min(dimensions[axis]["count"] for axis in "XYZ") < 3:
        raise ValueError("The COLMAP cloud contains too few fused points for terrain rasterization.")
    cell = settings["terrain_resolution_m"]
    gsd = settings["rgb_resolution_m"]
    left, bottom = [float(np.floor(dimensions[axis]["minimum"] / cell) * cell) for axis in "XY"]
    right, top = [float(np.ceil(dimensions[axis]["maximum"] / cell) * cell) for axis in "XY"]
    width, height = round((right - left) / cell), round((top - bottom) / cell)
    rgb_width, rgb_height = int(np.ceil((right - left) / gsd)), int(np.ceil((top - bottom) / gsd))
    if min(width, height) < 2 or width * height > 500_000 or rgb_width * rgb_height > 8_000_000:
        raise ValueError("Observed cloud exceeds the bounded pilot raster extent.")
    crs = CRS.from_user_input(args.crs)
    if not crs.is_projected or crs.linear_units_factor[1] != 1:
        raise ValueError("The source cloud must have a documented metre-based projected frame.")

    def writer(name, resolution, w, h, dimension, operation):
        return dict(type="writers.gdal", filename=str(out / f"{name}.tif"), resolution=resolution,
                    radius=resolution * 0.65, output_type=operation, nodata=-9999, data_type="float32",
                    origin_x=left, origin_y=bottom, width=w, height=h, dimension=dimension,
                    override_srs=crs.to_wkt())

    times["dsm"] = execute(pdal, out, "dsm", [
        *steps, writer("dsm", cell, width, height, "Z", "max")
    ], "--stream")
    for color in ("Red", "Green", "Blue"):
        times[color] = execute(pdal, out, color, [
            *steps, writer(color, gsd, rgb_width, rgb_height, color, "mean")
        ], "--stream")
    bands = []
    for color in ("Red", "Green", "Blue"):
        with rasterio.open(out / f"{color}.tif") as src:
            data = src.read(1, masked=True)
            bands.append(data)
            profile = src.profile
    observed = ~np.any([np.ma.getmaskarray(band) for band in bands], axis=0)
    if not np.any(observed):
        raise ValueError("The RGB projection has no measured color coverage.")
    with rasterio.open(out / "rgb.tif", "w", **dict(profile, count=3, dtype="uint8",
                                                    nodata=None, compress="deflate")) as dst:
        for index, band in enumerate(bands, 1):
            dense = fillnodata(band.filled(0), mask=observed.astype(np.uint8), max_search_distance=2)
            dst.write(np.clip(np.rint(dense), 0, 255).astype("uint8"), index)
    with rasterio.open(out / "rgb_observed.tif", "w", **dict(profile, count=1, dtype="uint8", nodata=None)) as dst:
        dst.write(observed.astype("uint8"), 1)
    with rasterio.open(out / "dsm.tif") as src:
        supported = int(src.read(1, masked=True).count())
    dump(out / "provenance.json", dict(
        source_cloud=str(args.cloud.resolve()), crs=crs.to_wkt(), cloud_to_projected=translation.tolist(),
        bounds_m=[left, bottom, right, top], dsm_cells=width * height, dsm_supported_cells=supported,
        rgb_observed_cells=int(observed.sum()), rgb_cells=observed.size, seconds=times,
        rgb_role="top-down projection of reconstructed UAV RGB points; no camera orthorectification",
        rgb_fill_max_pixels=2, dsm_role="maximum elevation of corroborated COLMAP depth fusion",
        independent_reference_used=False,
    ))


def support_roofs(args):
    """Interpolate classified ground locally below detected roofs on observed DSM cells."""
    import geopandas as gpd

    if not 0 < args.radius <= 8:
        raise ValueError("The beneath-roof interpolation limit must be in (0, 8] metres.")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source, profile = terrain.raster(args.dsm)
    base, base_profile = terrain.raster(args.dtm)
    observed, observed_profile = terrain.raster(args.anchors)
    if not all(terrain.aligned(profile, other) for other in (base_profile, observed_profile)):
        raise ValueError("Use DSM, DTM and ground-anchor inputs on the same grid.")
    aff = profile["transform"]
    shapes = gpd.read_file(args.polygons).to_crs(profile["crs"])
    target = geometry_mask(shapes.geometry.buffer(1.5), out_shape=source.shape,
                           transform=aff, invert=True) if len(shapes) else np.zeros_like(source, dtype=bool)
    anchors = np.isfinite(source) & (observed > 0) & ~target
    proposal, distance = terrain.interpolate(source, anchors, args.radius, aff.a)
    edited = target & np.isfinite(source) & np.isfinite(proposal)
    surface = np.where(edited, proposal, base)
    # Covered means observed ground or an explicitly bounded interpolation under
    # a detected roof. It never implies that ground beneath the roof was seen.
    terrain.save_raster(out / "dtm.tif", surface, profile)
    terrain.save_raster(out / "support.tif", anchors | edited, profile, mask=True)
    terrain.save_raster(out / "observed_ground.tif", anchors, profile, mask=True)
    terrain.save_raster(out / "inferred_ground.tif", edited, profile, mask=True)
    width, height = profile["width"], profile["height"]
    x = (np.arange(width) + 0.5 - width / 2) * aff.a
    y = (np.arange(height) + 0.5 - height / 2) * aff.e
    np.savez_compressed(out / "terrain.npz", x=x, y=y, z=surface)
    dump(out / "ground_provenance.json", dict(
        observed_ground=int(anchors.sum()), inferred_ground=int(edited.sum()),
        filled_nodata_cells=int((edited & ~np.isfinite(base)).sum()),
        maximum_anchor_distance_m=float(distance[edited].max()) if edited.any() else None,
        radius_m=args.radius, beyond_observed_dsm_changes=0,
        source_role="UAV COLMAP fused depth only",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    rasterize = commands.add_parser("rasterize")
    for name in ("cloud", "pdal", "settings", "output"):
        rasterize.add_argument(f"--{name}", type=Path, required=True)
    rasterize.add_argument("--crs", required=True)
    rasterize.add_argument("--origin", nargs=3, type=float, default=[0, 0, 0])
    rasterize.add_argument("--transform", type=Path)
    rasterize.set_defaults(run=run)
    support = commands.add_parser("support-roofs")
    for name in ("dsm", "dtm", "anchors", "polygons", "output"):
        support.add_argument(f"--{name}", type=Path, required=True)
    support.add_argument("--radius", type=float, default=6)
    support.set_defaults(run=support_roofs)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
