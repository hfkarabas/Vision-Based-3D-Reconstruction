"""Audit registered photo footprints and terrain stages on common projected samples."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import label
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

from omega_uav_pipeline import measured
from omega_uav_products import as_grid, raster


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def raster_difference(first, second, regions):
    common = np.isfinite(first) & np.isfinite(second)
    delta = second - first
    result = {}
    for name, mask in regions.items():
        values = delta[mask & common]
        result[name] = dict(
            samples=len(values),
            signed_median_m=float(np.median(values)) if len(values) else None,
            abs_p95_after_median_m=(
                float(np.percentile(np.abs(values - np.median(values)), 95)) if len(values) else None
            ),
        )
    return result


def camera_footprints(workspace, module_path, elevation):
    sys.path.insert(0, str(module_path))
    import pycolmap

    frame = read(workspace / "frame.json")
    transform = np.loadtxt(frame["transform"])
    linear, translation = transform[:, :3], transform[:, 3] + frame["origin_projected_m"]
    reconstruction = pycolmap.Reconstruction(workspace / "dense/sparse")
    features = []

    for image in reconstruction.images.values():
        camera = reconstruction.cameras[image.camera_id]
        pose = image.cam_from_world().inverse()
        border = np.array([[0.5, 0.5], [camera.width - 0.5, 0.5],
                           [camera.width - 0.5, camera.height - 0.5], [0.5, camera.height - 0.5]])
        rays = camera.cam_from_img(border)
        origin = np.asarray(pose.translation) @ linear.T + translation
        direction = np.column_stack([rays, np.ones(4)]) @ pose.rotation.matrix().T @ linear.T
        if np.any(direction[:, 2] >= -1e-8) or origin[2] <= elevation:
            continue
        points = origin + direction * ((elevation - origin[2]) / direction[:, 2])[:, None]
        polygon = Polygon(points[:, :2])
        if polygon.is_valid and polygon.area > 0:
            features.append({"type": "Feature", "geometry": mapping(polygon),
                             "properties": {"image": image.name}})
    return frame["crs"], features, len(reconstruction.images)


def statistics(mask, region):
    inside = mask & region
    denominator = int(region.sum())
    components, number = label(inside)
    counts = np.bincount(components[inside])
    return dict(pixels=int(inside.sum()), fraction=float(inside.sum() / denominator),
                components=int(number), largest_fraction=float(counts.max(initial=0) / denominator))


def texture_mask(path, grid):
    with rasterio.open(path) as src:
        support = np.zeros((grid["height"], grid["width"]), dtype=np.uint8)
        reproject(src.dataset_mask(), support, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=grid["transform"], dst_crs=grid["crs"])
    return support > 0


def main(args):
    spec = read(args.config)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cell = float(spec["sample_m"])
    west, south, east, north = spec["extent"]
    width, height = round((east - west) / cell), round((north - south) / cell)
    if not 0 < width * height <= 2_000_000:
        raise ValueError("Review sample count outside the configured bound")
    transform = from_origin(west, north, cell, cell)
    grid = dict(crs=spec["crs"], transform=transform, height=height, width=width)
    xy = np.column_stack([axis.ravel() for axis in np.meshgrid(
        west + (np.arange(width) + 0.5) * cell, north - (np.arange(height) + 0.5) * cell)])
    geometry = {name: shape(item) for name, item in spec["regions"].items()}
    geometry["site"] = shape(dict(type="Polygon", coordinates=[[
        (west, south), (east, south), (east, north), (west, north), (west, south),
    ]]))
    regions = {name: geometry_mask([poly], (height, width), transform, invert=True)
               for name, poly in geometry.items()}
    if any(not region.any() for region in regions.values()):
        raise ValueError("A review region lies outside the selected sample extent")
    report = {}
    masks = {}
    extent = (west, east, south, north)
    elevation = {}
    for run, files in spec["runs"].items():
        products = {}
        for key in ("dense", "dsm", "classified", "dtm", "inferred", "unmasked_render"):
            if key in files:
                array = as_grid(Path(files[key]), grid)
                products[key] = array > 0 if key in ("dense", "inferred") else np.isfinite(array)
                if key in ("dsm", "dtm"):
                    elevation[run, key] = array
        if files.get("camera_render_dir"):
            from omega_uav_coverage_audit import scene_data

            render_dir = Path(files["camera_render_dir"])
            _, _, before_mask, _, _ = scene_data(render_dir)
            with rasterio.open(render_dir / "dsm.tif") as src:
                before = np.zeros((height, width), dtype=np.uint8)
                reproject(before_mask.astype(np.uint8), before, src_transform=src.transform,
                          src_crs=src.crs, dst_transform=transform, dst_crs=grid["crs"])
            products["unmasked_render"] = before > 0
            products["dense_but_no_camera_render"] = products["dense"] & ~products["unmasked_render"]
            products["camera_render_without_dense"] = products["unmasked_render"] & ~products["dense"]
        if files.get("rgb"):
            products["rgb"] = texture_mask(Path(files["rgb"]), grid)
        if "dsm" in products and "dense" in products:
            products["dsm_without_dense_cell"] = products["dsm"] & ~products["dense"]
        if files.get("mesh"):
            scene = out / f"{run}_mesh.npy"
            sample = out / f"{run}_samples.npz"
            np.savez(sample, points=xy - files["origin"][:2])
            measured(out, f"{run}_bvh", [
                args.blender, "-b", "-t", 4, "--python-exit-code", 1, "--python",
                Path(__file__).with_name("omega_uav_ground_mesh_audit.py"),
                "--", "--scene", Path(files["mesh"]).resolve(),
                "--samples", sample, "--output", scene,
            ])
            z = np.load(scene).reshape(height, width) + files["origin"][2]
            products["mesh"] = np.isfinite(z)
            if "rgb" in products:
                products["textured_mesh"] = products["mesh"] & products["rgb"]
        metrics = {}
        for name, region in regions.items():
            metrics[name] = {k: statistics(v, region) for k, v in products.items()}
            if "dtm" in products and "mesh" in products:
                overlap = products["dtm"] & products["mesh"] & region
                dtm = as_grid(Path(files["dtm"]), grid)
                error = np.abs(z[overlap] - dtm[overlap])
                metrics[name]["transfer"] = dict(
                    dtm_without_mesh_m2=float((region & products["dtm"] & ~products["mesh"]).sum() * cell**2),
                    mesh_over_nodata_m2=float((region & ~products["dtm"] & products["mesh"]).sum() * cell**2),
                    z_abs_median_m=float(np.median(error)) if error.size else None,
                    z_abs_p95_m=float(np.percentile(error, 95)) if error.size else None,
                )
        report[run] = metrics
        masks[run] = products

    crs, photos, registered = camera_footprints(args.colmap, args.pycolmap, spec["footprint_plane_z_m"])
    if crs != spec["crs"]:
        raise ValueError("Camera footprints and coverage audit require the same CRS")
    footprint = unary_union([shape(item["geometry"]) for item in photos])
    (out / "expected_photo_footprints_projected.json").write_text(json.dumps(
        dict(type="FeatureCollection", projected_crs=crs, features=photos), indent=2), encoding="utf-8")
    expected = geometry_mask([footprint], (height, width), transform, invert=True)
    common = masks[spec["comparison"]]
    if "inferred" in common:
        raster(out / "inferred_dtm_review.tif", common["inferred"].astype(np.uint8), grid)
        raster(out / "unsupported_dtm_review.tif", (~common["dtm"]).astype(np.uint8), grid)
        fig, ax = plt.subplots(figsize=(11, 8), layout="constrained")
        classes = np.where(common["dtm"], 1, 0)
        classes[common["inferred"]] = 2
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        palette = ["#e5e5e5", "#236a57", "#de842c"]
        ax.imshow(classes, extent=extent, cmap=ListedColormap(palette), interpolation="none",
                  vmin=0, vmax=2)
        ax.set(xlabel="UTM easting (m)", ylabel="UTM northing (m)", title="DTM evidence and interpolation")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.legend(handles=[Patch(color=color, label=label_text) for color, label_text in zip(
            palette, ("Unsupported", "Class-2 point in cell", "Inferred / no class-2 point in cell"),
            strict=True)], loc="upper right")
        fig.savefig(out / "dtm_evidence.png", dpi=140)
        plt.close(fig)
    report["photo_coverage"] = dict(
        registered=registered, downward_footprints=len(photos), bounds=list(footprint.bounds),
        reference_plane_z_m=spec["footprint_plane_z_m"], planimetric_upper_bound_m2=footprint.area,
        inside_site_m2=float(expected.sum() * cell**2),
        mesh_inside_photo_bounds_m2=float((expected & common["mesh"]).sum() * cell**2),
        mesh_total_m2=float(common["mesh"].sum() * cell**2),
        photo_bounds_without_mesh_m2=float(footprint.area - (expected & common["mesh"]).sum() * cell**2),
    )
    if pairs := spec.get("elevation_comparisons"):
        report["elevation_comparisons"] = {
            f"{first}_vs_{second}_{key}": raster_difference(
                elevation[first, key], elevation[second, key], regions,
            ) for first, second, key in pairs
        }
    (out / "coverage.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    with rasterio.open(spec["runs"][spec["comparison"]]["rgb"]) as src:
        from rasterio.plot import plotting_extent
        axs[0, 0].imshow(src.read([1, 2, 3]).transpose(1, 2, 0), extent=plotting_extent(src))
    axs[0, 0].set_title("Camera-derived orthophoto and review regions")
    for name, poly in geometry.items():
        if name != "site":
            axs[0, 0].plot(*poly.exterior.xy, color="gold", lw=1.1)
            axs[0, 0].annotate(name, poly.centroid.coords[0], color="white",
                               fontsize=8, bbox=dict(facecolor="#323232", alpha=.8, pad=2))
    for ax, title, mask in zip(axs.flat[1:],
                              ("Fused points", "Camera DSM", "DTM", "Blender surface", "DSM-to-mesh loss"),
                              (common["dense"], common["dsm"], common["dtm"], common["mesh"],
                               common["dsm"] & ~common["mesh"]), strict=True):
        ax.imshow(mask, extent=extent, cmap="Blues", interpolation="none")
        ax.set_title(title)
    for ax in axs.flat:
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.set(xlabel="UTM easting (m)", ylabel="UTM northing (m)", xlim=(west, east), ylim=(south, north))
    fig.savefig(out / "stage_coverage.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 9), layout="constrained")
    polygons = [footprint] if footprint.geom_type == "Polygon" else footprint.geoms
    for index, poly in enumerate(polygons):
        ax.fill(*poly.exterior.xy, facecolor="#e7e7e7", edgecolor="#555", linewidth=1,
                label="Registered image footprint union (reference plane)" if index == 0 else None)
    masked = np.ma.masked_where(~common["mesh"], common["mesh"])
    ax.imshow(masked, extent=extent, cmap="Blues", vmin=0, vmax=1,
              interpolation="none", zorder=2)
    ax.plot(*geometry["site"].exterior.xy, color="#e47820", lw=1.2, label="Full dense processing extent")
    ax.scatter([], [], marker="s", color="#08306b", label="Verified Blender terrain")
    ax.autoscale_view()
    bounds = footprint.bounds
    ax.set_xlim(min(west, bounds[0])-6, max(east, bounds[2])+6)
    ax.set_ylim(min(south, bounds[1])-6, max(north, bounds[3])+6)
    ax.set(xlabel="UTM easting (m)", ylabel="UTM northing (m)", title="Sheffield Cross full flight and output coverage")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.legend(loc="upper right")
    fig.savefig(out / "flight_vs_mesh.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "colmap", "pycolmap", "blender", "output"):
        parser.add_argument(f"--{key}", type=lambda p: Path(p).resolve(), required=True)
    main(parser.parse_args())
