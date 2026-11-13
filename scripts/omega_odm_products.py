"""Transfer georeferenced OpenDroneMap orthophoto and DEM products into the asset stages."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from omega_uav_products import as_grid, grid_for, largest_component, raster


def observed_products(args):
    """Rasterize direct point occupancy and class-2 observations with PDAL."""
    from omega_uav_pipeline import measured
    from omega_uav_products import pdal, writer

    config = json.loads(args.settings.read_text(encoding="utf-8"))["surface"]
    grid = grid_for(args, config["terrain_cell_m"], config["terrain_max_cells"])
    args.output.mkdir(parents=True, exist_ok=False)
    reader = [dict(type="readers.las", filename=str(args.cloud)),
              dict(type="filters.reprojection", out_srs=args.crs)]
    pdal(args, args.output, "direct_dense", [
        *reader, writer(args.output / "dense_count.tif", args, grid, "count"),
    ], measured)
    pdal(args, args.output, "direct_ground", [
        *reader, dict(type="filters.expression", expression="Classification == 2"),
        writer(args.output / "classified_ground.tif", args, grid, "min"),
    ], measured)


def transfer(args):
    config = json.loads(args.settings.read_text(encoding="utf-8"))["surface"]
    photo_grid = grid_for(args, config["ortho_cell_m"], config["orthophoto_max_pixels"])
    terrain_grid = grid_for(args, config["terrain_cell_m"], config["terrain_max_cells"])
    ortho = args.output / "orthophoto"
    ground = args.output / "ground"
    for path in (ortho, ground):
        path.mkdir(parents=True, exist_ok=False)

    shape = photo_grid["height"], photo_grid["width"]
    rgb = np.zeros((3, *shape), dtype=np.uint8)
    alpha = np.zeros(shape, dtype=np.uint8)
    with rasterio.open(args.rgb) as src:
        if src.count < 3:
            raise ValueError("An ODM orthophoto must contain three RGB bands")
        projection = dict(src_transform=src.transform, src_crs=src.crs,
                          dst_transform=photo_grid["transform"], dst_crs=photo_grid["crs"])
        reproject(src.dataset_mask(), alpha, **projection)
        for band in range(1, 4):
            reproject(rasterio.band(src, band), rgb[band - 1], **projection,
                      resampling=Resampling.bilinear)
    valid_rgb = alpha > 0
    raster(ortho / "orthophoto.tif", rgb, photo_grid, valid=valid_rgb)
    dsm = as_grid(args.dsm, photo_grid)
    raster(ortho / "dsm.tif", dsm, photo_grid, nodata=np.nan)
    Image.fromarray(np.moveaxis(rgb, 0, -1)).save(ortho / "supported_photo.png")

    surface = as_grid(args.dtm, terrain_grid)
    classified = as_grid(args.observed, terrain_grid)
    mask = np.zeros(surface.shape, dtype=np.uint8)
    reproject(alpha, mask, src_transform=photo_grid["transform"], src_crs=photo_grid["crs"],
              dst_transform=terrain_grid["transform"], dst_crs=terrain_grid["crs"])
    native_valid = np.isfinite(surface)
    valid = native_valid & (mask > 0)
    observed = np.isfinite(classified) & valid
    surface[~valid] = np.nan
    raster(ground / "dtm.tif", surface, terrain_grid, nodata=np.nan)
    raster(ground / "classified_ground.tif", classified, terrain_grid, nodata=np.nan)
    raster(ground / "observed.tif", observed.astype(np.uint8), terrain_grid)
    raster(ground / "inferred.tif", (valid & ~observed).astype(np.uint8), terrain_grid)
    raster(ground / "unsupported.tif", (~valid).astype(np.uint8), terrain_grid)
    west, south, east, north = args.extent
    origin = [(west + east) / 2, (south + north) / 2, 0]
    aff = terrain_grid["transform"]
    x = aff.c + (np.arange(terrain_grid["width"]) + 0.5) * aff.a - origin[0]
    y = aff.f + (np.arange(terrain_grid["height"]) + 0.5) * aff.e - origin[1]
    np.savez_compressed(ground / "terrain.npz", x=x, y=y, z=surface)
    report = dict(origin=origin, native_dtm_cells=int(native_valid.sum()),
                  observed_cells=int(observed.sum()), inferred_cells=int((valid & ~observed).sum()),
                  unsupported_cells=int((~valid).sum()), largest_component_fraction=largest_component(valid),
                  source_rgb=str(args.rgb), source_dsm=str(args.dsm), source_dtm=str(args.dtm),
                  source_ground_observations=str(args.observed),
                  sampling="Nearest DEM samples, bounded to native phototexture mask",
                  texture_label="OpenDroneMap camera orthophoto")
    (ground / "complete.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(required=True)
    for name, keys, operation in [
        ("transfer", ("rgb", "dsm", "dtm", "observed"), transfer),
        ("observed", ("cloud", "pdal"), observed_products),
    ]:
        command = operations.add_parser(name)
        for key in (*keys, "settings", "output"):
            command.add_argument(f"--{key}", required=True, type=lambda path: Path(path).resolve())
        command.add_argument("--extent", required=True, type=float, nargs=4)
        command.add_argument("--crs", required=True)
        command.set_defaults(operation=operation)
    args = parser.parse_args()
    args.operation(args)
