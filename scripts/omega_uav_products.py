"""Connect upstream photogrammetry, GIS, segmentation and Blender products."""

from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.ndimage import label

from omega_colmap_pilot import save, stage
from omega_terrain_cleanup import interpolate


SCRIPTS = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def raster(path, array, grid, nodata=None, valid=None):
    array = np.asarray(array)
    bands = array if array.ndim == 3 else array[None, :, :]
    with rio.open(path, "w", driver="GTiff", count=len(bands), dtype=str(array.dtype),
                  nodata=nodata, compress="deflate", **grid) as dst:
        dst.write(bands)
        if valid is not None:
            dst.write_mask(np.asarray(valid, dtype=np.uint8) * 255)


def grid_for(args, cell, limit):
    west, south, east, north = args.extent
    shape = np.array([east - west, north - south]) / cell
    width, height = np.rint(shape).astype(int)
    if min(width, height) < 2 or not np.allclose(shape, [width, height], atol=1e-5, rtol=0):
        raise ValueError("Extent dimensions must be positive integer multiples of the grid cell.")
    if width * height > limit:
        raise ValueError("Requested projected grid exceeds the configured resource bound.")
    crs = rio.crs.CRS.from_string(args.crs)
    if not crs.is_projected or crs.linear_units_factor[1] != 1:
        raise ValueError("The photo workflow requires a projected metre-based CRS.")
    return dict(width=int(width), height=int(height), crs=crs,
                transform=from_bounds(*args.extent, width, height))


def as_grid(source, target, method=Resampling.nearest):
    with rio.open(source) as src:
        result = np.full((target["height"], target["width"]), np.nan, dtype=np.float32)
        reproject(rio.band(src, 1), result, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata, dst_transform=target["transform"], dst_crs=target["crs"],
                  dst_nodata=np.nan, resampling=method)
    return result


def writer(path, args, grid, operation):
    return dict(type="writers.gdal", filename=str(path), resolution=grid["transform"].a,
                output_type=operation, data_type="float32", nodata=-9999, binmode=True,
                origin_x=args.extent[0], origin_y=args.extent[1],
                width=grid["width"], height=grid["height"], override_srs=args.crs)


def projected_cloud(args, colmap, buffer):
    frame = read(colmap / "frame.json")
    matrix = np.eye(4)
    matrix[:3] = np.loadtxt(frame["transform"])
    matrix[:3, 3] += frame["origin_projected_m"]
    west, south, east, north = args.extent
    return [
        dict(type="readers.ply", filename=str(colmap / "dense" / "fused.ply")),
        dict(type="filters.transformation", matrix=" ".join(map(str, matrix.ravel()))),
        dict(type="filters.crop", bounds=f"([{west-buffer},{east+buffer}],[{south-buffer},{north+buffer}])"),
    ]


def pdal(args, output, name, pipeline, measured):
    recipe = output / f"{name}_pipeline.json"
    save(recipe, pipeline)
    measured(output, name, [args.pdal, "pipeline", recipe])


def orthophoto(args, settings, colmap, measured):
    out = args.output / "orthophoto"
    out.mkdir(exist_ok=False)
    cfg = settings["surface"]
    ortho_grid = grid_for(args, cfg["ortho_cell_m"], cfg["orthophoto_max_pixels"])
    ground_grid = grid_for(args, cfg["terrain_cell_m"], cfg["terrain_max_cells"])
    density = out / "dense_count.tif"
    pdal(args, out, "dense_support", [
        *projected_cloud(args, colmap, 0), writer(density, args, ground_grid, "count"),
    ], measured)
    counts = as_grid(density, ground_grid)
    if np.count_nonzero(counts > 0) < 12:
        raise ValueError("Dense reconstruction barely intersects the requested projected extent.")
    dense = colmap / "dense"
    mesh = out / "surface.ply"
    if cfg["surface_mesh"] == "delaunay":
        command = ["delaunay_mesher", "--input_path", dense, "--output_path", mesh,
                   "--DelaunayMeshing.num_threads", 4]
    elif cfg["surface_mesh"] == "poisson":
        command = ["poisson_mesher", "--input_path", dense / "fused.ply", "--output_path", mesh,
                   "--PoissonMeshing.depth", 10, "--PoissonMeshing.num_threads", 4]
    else:
        raise ValueError("Select the documented COLMAP delaunay or poisson mesher.")
    stage(out, "mesher", args.colmap, command)
    textures = out / "textured"
    textures.mkdir()
    stage(out, "texture", args.colmap, [
        "mesh_texturer", "--workspace_path", dense, "--input_path", mesh, "--output_path", textures,
        "--MeshTextureMapping.num_threads", 4,
    ])
    measured(out, "blender_render", [
        args.blender, "-b", "-t", 4, "--python-exit-code", 1, "--python",
        SCRIPTS / "omega_uav_ortho_blender.py", "--", "--mesh", textures / "mesh.ply",
        "--frame", colmap / "frame.json", "--output", out,
        "--extent", *args.extent, "--size", ortho_grid["width"], ortho_grid["height"],
    ])
    with np.load(out / "surface.npz") as image:
        height = image["elevation"]
        alpha = image["alpha"]
    rgb = np.asarray(Image.open(out / "orthophoto.png").convert("RGBA"))
    support = as_grid(density, ortho_grid) > 0
    valid = (rgb[:, :, 3] >= 252) & (alpha >= 0.985) & np.isfinite(height) & support
    raster(out / "orthophoto.tif", np.where(valid, rgb[:, :, :3].transpose(2, 0, 1), 0).astype("uint8"),
           ortho_grid, valid=valid)
    raster(out / "dsm.tif", np.where(valid, height, np.nan).astype("float32"), ortho_grid, nodata=np.nan)
    Image.fromarray(np.where(valid[:, :, None], rgb[:, :, :3], 0).astype("uint8")).save(out / "supported_photo.png")
    save(out / "complete.json", dict(dense_cell_fraction=float(np.mean(counts > 0)),
                                    supported_camera_rgb_dsm_fraction=float(valid.mean())))


def largest_component(valid):
    regions, _ = label(valid)
    sizes = np.bincount(regions.ravel())
    return float(sizes[1:].max(initial=0) / valid.size)


def ground(args, settings, colmap, measured):
    out = args.output / "ground"
    out.mkdir(exist_ok=False)
    cfg = settings["surface"]
    grid = grid_for(args, cfg["terrain_cell_m"], cfg["terrain_max_cells"])
    ortho = args.output / "orthophoto"
    classifiers = {
        "smrf": dict(type="filters.smrf", cell=grid["transform"].a, slope=0.2,
                     scalar=1.2, threshold=0.45, window=18),
        "csf": dict(type="filters.csf", resolution=grid["transform"].a, rigidness=3,
                    iterations=500, threshold=0.5),
    }
    classified = out / "classified_ground.tif"
    pdal(args, out, "classifier", [
        *projected_cloud(args, colmap, 18), classifiers[cfg["ground_classifier"]],
        dict(type="filters.expression", expression="Classification == 2"),
        writer(classified, args, grid, "min"),
    ], measured)
    observed_values = as_grid(classified, grid)
    support = as_grid(ortho / "dense_count.tif", grid) > 0
    photographed = np.isfinite(as_grid(ortho / "dsm.tif", grid))
    observed = np.isfinite(observed_values) & support
    candidate, _ = interpolate(observed_values, observed, cfg["maximum_ground_fill_m"], grid["transform"].a)
    valid = np.isfinite(candidate) & support & photographed
    observed &= valid
    surface = np.where(valid, candidate, np.nan).astype("float32")
    raster(out / "dtm.tif", surface, grid, nodata=np.nan)
    raster(out / "observed.tif", observed.astype("uint8"), grid)
    raster(out / "inferred.tif", (valid & ~observed).astype("uint8"), grid)
    raster(out / "unsupported.tif", (~valid).astype("uint8"), grid)
    origin = [(args.extent[0] + args.extent[2]) / 2, (args.extent[1] + args.extent[3]) / 2, 0]
    aff = grid["transform"]
    x = (np.arange(grid["width"]) + 0.5) * aff.a + aff.c - origin[0]
    y = (np.arange(grid["height"]) + 0.5) * aff.e + aff.f - origin[1]
    np.savez_compressed(out / "terrain.npz", x=x, y=y, z=surface)
    save(out / "complete.json", dict(origin=origin, observed_cells=int(observed.sum()),
                                    inferred_cells=int((valid & ~observed).sum()), unsupported_cells=int((~valid).sum()),
                                    largest_component_fraction=largest_component(valid)))


def building_heights(args, out, ortho, measured):
    import geopandas as gpd
    from rasterstats import zonal_stats
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import box, mapping
    from shapely.geometry.polygon import orient

    with rio.open(ortho / "dsm.tif") as src:
        grid = dict(width=src.width, height=src.height, crs=src.crs, transform=src.transform)
    ground_dir = args.output / "ground"
    dtm = out / "aligned_ground.tif"
    ground_z = as_grid(ground_dir / "dtm.tif", grid, Resampling.bilinear)
    raster(dtm, ground_z, grid, nodata=np.nan)
    ndsm = out / "ndsm.tif"
    measured(out, "gdal_height_difference", [
        args.pdal.with_name("gdal.exe"), "raster", "calc", "--input", f"A={ortho / 'dsm.tif'}",
        "--input", f"B={dtm}", "--output", ndsm, "--calc", "A-B",
        "--propagate-nodata", "--nodata", "-9999", "--ot", "Float32",
    ])
    geometry = gpd.read_file(out / "geoai_polygons.gpkg").to_crs(grid["crs"])
    valid_extent = box(*args.extent).buffer(-grid["transform"].a)
    with rio.open(ortho / "dsm.tif") as src:
        dsm = src.read(1)
    with rio.open(ndsm) as src:
        relief = src.read(1)
    nodata = -9999
    affine = grid["transform"]

    def stats(polygons, values, operations):
        return zonal_stats(polygons, np.where(np.isfinite(values), values, nodata), affine=affine,
                           nodata=nodata, stats=operations)

    shapes = [orient(poly, sign=1.0) for poly in geometry.geometry]
    roof_cores = [poly.buffer(-0.6) for poly in shapes]
    cores = [poly if core.is_empty else core for poly, core in zip(shapes, roof_cores, strict=True)]
    annuli = [poly.buffer(5).difference(poly.buffer(1)) for poly in shapes]
    with rio.open(ground_dir / "observed.tif") as src:
        native = src.read(1)
        anchor_counts = zonal_stats(annuli, native, affine=src.transform, nodata=0, stats="count")
    estimates = zip(
        shapes, stats(cores, dsm, "median count"), stats(cores, relief, "median count"),
        stats(shapes, ground_z, "median count percentile_95 percentile_5"),
        stats(cores, np.ones_like(dsm), "count"), stats(shapes, np.ones_like(dsm), "count"), anchor_counts, strict=True,
    )
    features = []
    for index, (shape, roof, heights, base, roof_pixels, base_pixels, anchors) in enumerate(estimates, 1):
        props = dict(building_id=f"Building_{index:04d}", area_m2=shape.area,
                     roof_samples=roof["count"], paired_height_samples=heights["count"],
                     ground_samples=base["count"], surrounding_observed_ground_cells=anchors["count"],
                     roof_core_fraction=heights["count"] / max(1, roof_pixels["count"]),
                     foundation_fraction=base["count"] / max(1, base_pixels["count"]),
                     lod1_height_m=heights["median"], roof_in_dtm_frame_m=roof["median"],
                     height_provenance="GDAL DSM minus DTM; rasterstats median of roof core",
                     footprint_provenance="GeoAI roof outline; proxy for ground footprint")
        if not valid_extent.covers(shape):
            status = "clipped_footprint"
        elif props["roof_core_fraction"] < 0.90 or props["foundation_fraction"] < 0.98:
            status = "insufficient_raster_support"
        elif anchors["count"] < 15 or heights["count"] < 15:
            status = "insufficient_ground_or_roof_samples"
        elif not 2 <= heights["median"] <= 120 or base["percentile_95"] - base["percentile_5"] > 4:
            status = "invalid_roof_relief_or_foundation"
        else:
            status = "eligible"
            triangles = constrained_delaunay_triangles(shape)
            if not np.isclose(sum(t.area for t in triangles.geoms), shape.area, rtol=1e-6):
                raise ValueError("GEOS triangulation did not preserve the detected footprint.")
            props["foundation_triangles_xy"] = [list(t.exterior.coords)[:-1] for t in triangles.geoms]
        features.append(dict(type="Feature", properties=dict(props, status=status), geometry=mapping(shape)))
    save(out / "building_footprints_projected.json", dict(type="FeatureCollection", features=features))
    return Counter(feature["properties"]["status"] for feature in features)


def buildings(args, settings, measured):
    out = args.output / "buildings"
    out.mkdir(exist_ok=False)
    ortho = args.output / "orthophoto"
    ground_dir = args.output / "ground"
    cfg = settings["segmentation"]
    measured(out, "segmentation", [
        sys.executable, SCRIPTS / "omega_building_inference.py", "--rgb", ortho / "orthophoto.tif",
        "--hub-cache", args.hub_cache, "--model", cfg["repo"], "--threshold", cfg["threshold"],
        "--minimum-area", cfg["minimum_area_m2"], "--output", out,
    ])
    inference = read(out / "inference.json")
    if (inference["model_revision"], inference["model_checkpoint_sha256"]) != (cfg["revision"], cfg["sha256"]):
        raise ValueError("GeoAI checkpoint does not match the configured revision and digest.")
    statuses = building_heights(args, out, ortho, measured)
    measured(out, "overlay", [
        sys.executable, SCRIPTS / "omega_building_review.py", "overlay", "--rgb", ortho / "orthophoto.tif",
        "--primary", out / "geoai_polygons.gpkg", "--output", out / "building_detection_overlay.png",
        "--projected-assessment", out / "building_footprints_projected.json",
    ])
    shutil.copyfile(ortho / "supported_photo.png", out / "terrain_photo.png")
    save(out / "grid.json", dict(crs=args.crs, bounds=args.extent, origin_enu=read(ground_dir / "complete.json")["origin"],
                                 texture_label="COLMAP camera-textured orthophoto"))
    measured(out, "blender_assembly", [
        args.blender, "-b", "-t", 4, "--python-exit-code", 1, "--python", SCRIPTS / "omega_building_blender.py",
        "--", "--output", out, "--height-grid", ground_dir / "terrain.npz", "--allow-terrain-only",
    ])
    solids = read(out / "reports" / "blender_buildings.json")
    save(out / "complete.json", dict(raster_candidates=dict(statuses),
                                    solid_candidates=dict(Counter(row["status"] for row in solids)),
                                    valid_lod1_volumes=sum(row["status"] == "built" for row in solids)))


def products(args, settings, through, measured):
    colmap = args.output / "colmap"
    for index, (name, operation) in enumerate([
        ("orthophoto", lambda: orthophoto(args, settings, colmap, measured)),
        ("ground", lambda: ground(args, settings, colmap, measured)),
        ("buildings", lambda: buildings(args, settings, measured)),
    ], 1):
        if index > through:
            break
        # Completed phases are reused only in their original validated run. Failed
        # phase directories are retained for diagnosis and require a fresh run.
        if not (args.output / name / "complete.json").exists():
            operation()
