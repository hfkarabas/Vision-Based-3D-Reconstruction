"""Locate coverage losses in existing COLMAP imagery, depth and fused point outputs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.transform import rowcol, xy

from omega_uav_coverage_audit import labels


def depth_map(path):
    with path.open("rb") as source:
        dims = []
        for _ in range(3):
            value = bytearray()
            while (char := source.read(1)) != b"&":
                if not char or not char.isdigit() or len(value) >= 8:
                    raise ValueError(f"Invalid COLMAP depth header: {path}")
                value.extend(char)
            dims.append(int(value))
        width, height, channels = dims
        values = np.fromfile(source, dtype="<f4")
    if channels != 1 or values.size != width * height:
        raise ValueError(f"Invalid COLMAP depth raster: {path}")
    return values.reshape(height, width)


def projected_points(camera, pose, pixels, depth):
    # COLMAP's depth is camera-space Z. Use the supplied undistorted camera and pose.
    rays = camera.cam_from_img(pixels)
    camera_points = np.column_stack([rays, np.ones(len(rays))]) * depth[:, None]
    return pose.inverse() * camera_points


def audit(args):
    sys.path.insert(0, str(args.pycolmap.resolve()))
    import pycolmap

    regions = json.loads(args.review.read_text(encoding="utf-8"))["datasets"][args.dataset]["regions"]
    regions = [item for item in regions if item["class"] in {"road_or_path", "open_ground"}]
    root = args.pilot
    workspace = args.workspace or root / "colmap" / "dense"
    with rasterio.open(root / "orthophoto" / "dsm.tif") as dsm:
        fine_grid = dsm.transform
        shape = dsm.shape
    with rasterio.open(args.counts or root / "orthophoto" / "dense_count.tif") as counts:
        dense = counts.read(1, masked=True).filled(0) > 0
        grid = counts.transform
    with np.load(root / "orthophoto" / "surface.npz") as image:
        elevation = np.where(image["alpha"] >= 0.985, image["elevation"], np.nan)
    transform = np.loadtxt(root / "colmap" / "dense_to_local.txt")
    offset = np.asarray(json.loads((root / "colmap" / "frame.json").read_text())["origin_projected_m"])
    linear, translation = transform[:, :3], transform[:, 3] + offset
    scale = np.linalg.norm(linear[:, 0])
    masks = labels(regions, shape)
    annotated = np.logical_or.reduce(list(masks.values()))
    points_z = elevation.copy()
    fallback = annotated & ~np.isfinite(points_z)
    # This is evaluation only: bracket the frustum estimate using local representative
    # elevations where the rendered geometry gives no height at all.
    for region in masks.values():
        samples = points_z[region & np.isfinite(points_z)]
        if samples.size == 0:
            raise ValueError("No render-derived elevation in a reviewed region.")
        points_z[region & fallback] = np.median(samples)
    rr, cc = np.nonzero(annotated)
    east, north = xy(fine_grid, rr, cc)
    target = np.column_stack([east, north, points_z[rr, cc]])
    target_sfm = (target - translation) @ np.linalg.inv(linear).T
    reconstruction = pycolmap.Reconstruction(workspace / "sparse")
    hits = {key: np.zeros(len(target), dtype=np.int32) for key in
            ("frustum", "photo_z_agrees", "geo_z_agrees", "geo_depth_in_front")}
    occupancy = {key: np.zeros(dense.shape, dtype=np.int32) for key in ("photometric", "geometric")}
    summaries = []
    for image in sorted(reconstruction.images.values(), key=lambda item: item.name):
        pose = image.cam_from_world()
        camera = reconstruction.cameras[image.camera_id]
        camera_xyz = pose * target_sfm
        visible = camera_xyz[:, 2] > 0
        uv = np.full((len(target), 2), -1.0)
        uv[visible] = camera.img_from_cam(camera_xyz[visible])
        visible &= ((uv[:, 0] >= 0) & (uv[:, 0] < camera.width) &
                    (uv[:, 1] >= 0) & (uv[:, 1] < camera.height))
        index = np.flatnonzero(visible)
        hits["frustum"][index] += 1
        summary = dict(image=image.name, review_pixels_in_frustum=len(index))
        for name, kind in (("photo", "photometric"), ("geo", "geometric")):
            depth = depth_map(workspace / "stereo" / "depth_maps" / f"{image.name}.{kind}.bin")
            if depth.shape != (camera.height, camera.width):
                raise ValueError("Depth dimensions differ from the undistorted COLMAP camera.")
            col, row = np.floor(uv[index]).astype(int).T
            sampled = depth[row, col]
            delta = (sampled - camera_xyz[index, 2]) * scale
            real = np.isfinite(sampled) & (sampled > 0)
            hits[f"{name}_z_agrees"][index[real & (np.abs(delta) <= args.tolerance)]] += 1
            if name == "geo":
                hits["geo_depth_in_front"][index[real & (delta < -args.tolerance)]] += 1
            dy, dx = np.nonzero(np.isfinite(depth) & (depth > 0))
            world = projected_points(camera, pose, np.column_stack([dx + 0.5, dy + 0.5]), depth[dy, dx])
            world = world @ linear.T + translation
            sample_row, sample_col = map(np.asarray, rowcol(fine_grid, world[:, 0], world[:, 1]))
            inside = (sample_row >= 0) & (sample_row < shape[0]) & (sample_col >= 0) & (sample_col < shape[1])
            sr, sc = sample_row[inside], sample_col[inside]
            supported = annotated[sr, sc] & (np.abs(world[inside, 2] - points_z[sr, sc]) <= args.tolerance)
            bx, by = xy(fine_grid, sr[supported], sc[supported])
            br, bc = map(np.asarray, rowcol(grid, bx, by))
            if len(br):
                locations = np.unique(np.ravel_multi_index((br, bc), dense.shape))
                occupancy[kind].ravel()[locations] += 1
            summary[kind + "_valid_pixels"] = len(dy)
            summary[kind + "_review_depth_samples_at_height"] = int(supported.sum())
        summaries.append(summary)
    gr, gc = map(np.asarray, rowcol(grid, east, north))
    fields = {
        "registered_camera_frusta": hits["frustum"], "photometric_z_agreement": hits["photo_z_agrees"],
        "geometric_z_agreement": hits["geo_z_agrees"], "geometric_depth_in_front": hits["geo_depth_in_front"],
        "backprojected_photo_maps": occupancy["photometric"][gr, gc],
        "backprojected_geo_maps": occupancy["geometric"][gr, gc],
        "fused_cell": dense[gr, gc].astype(int),
    }
    masks["visible_ground"] = annotated
    data = {}
    for name, mask in masks.items():
        chosen = mask[rr, cc]
        data[name] = dict(
            pixels=int(chosen.sum()),
            fallback_height_fraction=float(fallback[mask].mean()),
            metrics={key: dict(at_least_one=float(np.mean(values[chosen] >= 1)),
                               at_least_three=float(np.mean(values[chosen] >= 3)),
                               median=float(np.median(values[chosen]))) for key, values in fields.items()},
            depth_without_fused=float(np.mean((fields["backprojected_geo_maps"][chosen] > 0) &
                                              (fields["fused_cell"][chosen] == 0))),
            three_depth_without_fused=float(np.mean((fields["backprojected_geo_maps"][chosen] >= 3) &
                                                    (fields["fused_cell"][chosen] == 0))),
        )
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(dataset=args.dataset, registered=reconstruction.num_reg_images(), z_tolerance_m=args.tolerance,
                  evaluation_height="Unmasked camera mesh Z, per-region median fallback only where render has no Z",
                  projection_note="Frustum is upper-bound camera coverage. Depth in front can indicate occlusion or depth error. "
                                  "Backprojections check height and XY bins; they do not replicate stereo_fusion correspondence tests.",
                  depth_workspace=str(workspace.resolve()), per_image=summaries, regions=data)
    (args.output / "depth_coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    keys = ["registered_camera_frusta", "backprojected_photo_maps", "backprojected_geo_maps", "fused_cell"]
    canvas = Image.new("RGB", (shape[1] * 2, (shape[0] + 23) * 2))
    draw = ImageDraw.Draw(canvas)
    for i, key in enumerate(keys):
        heat = np.full((*shape, 3), (29, 33, 38), dtype=np.uint8)
        heat[rr, cc] = (114, 38, 56)
        active = fields[key] > 0
        heat[rr[active], cc[active]] = (49, 156, 149)
        if key != "fused_cell":
            high = fields[key] >= 3
            heat[rr[high], cc[high]] = (104, 217, 140)
        left, top = i % 2 * shape[1], i // 2 * (shape[0] + 23)
        canvas.paste(Image.fromarray(heat), (left, top + 23))
        draw.text((left + 6, top + 5), key.replace("_", " ") + " | red=0, teal=1+, green=3+", fill="white")
    canvas.save(args.output / "depth_coverage.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "review", "pycolmap", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--counts", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.5)
    audit(parser.parse_args())
