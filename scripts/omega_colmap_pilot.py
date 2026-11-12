"""Run a bounded COLMAP pilot from UAV photos or an existing registered camera model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import psutil
from pyproj import CRS, Transformer


def save(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")


def stage(work, name, colmap, flags):
    command = [str(colmap), *map(str, flags)]
    started = time.monotonic()
    peak = 0
    stop_reason = None
    with (work / f"{name}.log").open("wb") as log, subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT,
        env=dict(os.environ, PATH=str(colmap.parent) + os.pathsep + os.environ["PATH"]),
    ) as process:
        observer = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                peak = max(peak, observer.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            if peak > 3 * 1024**3 or time.monotonic() - started > 900:
                stop_reason = "per-stage 3 GiB RSS or 900 second pilot limit"
                process.kill()
                process.wait()
                break
            time.sleep(0.2)
    save(work / f"{name}.json", dict(command=command, seconds=time.monotonic() - started,
                                   peak_process_rss_bytes=peak, returncode=process.returncode,
                                   stop_reason=stop_reason))
    if process.returncode != 0:
        raise RuntimeError(f"COLMAP stage {name} failed. See {work / (name + '.log')}.")


def degrees(values, reference, negative):
    decimal = sum(float(value) / divisor for value, divisor in zip(values, [1, 60, 3600], strict=True))
    return -decimal if reference == negative else decimal


def gps_alignment(images, paths, work, colmap, model, target_crs, pycolmap):
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    positions = {}
    for path in paths:
        with Image.open(path) as image:
            gps = image.getexif().get_ifd(34853)
        if not all(key in gps for key in (1, 2, 3, 4, 6)):
            continue
        lat = degrees(gps[2], gps[1], "S")
        lon = degrees(gps[4], gps[3], "W")
        height = float(gps[6]) * (-1 if gps.get(5) in (1, b"\x01") else 1)
        positions[path.relative_to(images).as_posix()] = transformer.transform(lon, lat, height)
    if len(positions) < 4:
        raise ValueError("Four or more camera GPS positions are required for the projected pilot.")
    origin = np.round(np.median(list(positions.values()), axis=0), decimals=0)
    references = work / "gps_positions_local.txt"
    references.write_text("".join(
        name + " " + " ".join(map(str, np.array(xyz) - origin)) + "\n" for name, xyz in positions.items()
    ), encoding="utf-8")
    aligned = work / "gps_aligned"
    aligned.mkdir()
    transform = work / "colmap_similarity.txt"
    stage(work, "gps_alignment", colmap, [
        "model_aligner", "--input_path", model, "--output_path", aligned,
        "--ref_images_path", references, "--ref_is_gps", 0,
        "--alignment_type", "custom", "--alignment_max_error", 8,
        "--transform_path", transform,
    ])
    source, destination = [pycolmap.Reconstruction(path) for path in (model, aligned)]
    centers = [
        (image.projection_center(), destination.images[image_id].projection_center())
        for image_id, image in source.images.items()
        if image_id in destination.images and image.has_pose and destination.images[image_id].has_pose
    ]
    matrix = work / "dense_to_local.txt"
    src, dst = np.array(centers).transpose(1, 0, 2)
    transform = pycolmap.estimate_sim3d(src, dst)
    if transform is None or np.max(np.linalg.norm((transform * src) - dst, axis=1)) > 0.01:
        raise ValueError("Could not confirm the exported COLMAP similarity transform.")
    np.savetxt(matrix, transform.matrix())
    save(work / "frame.json", dict(crs=target_crs, origin_projected_m=origin.tolist(),
                                  camera_positions=len(positions), transform=str(matrix),
                                  height_reference="Camera EXIF GPS; vertical datum unverified",
                                  checkpoints_used_for_alignment=False))


def analyze_database(database, work):
    stride = 2147483647
    with sqlite3.connect(database) as db:
        names = dict(db.execute("SELECT image_id, name FROM images"))
        rows = list(db.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"))
    pairs = [dict(first=names[pair // stride], second=names[pair % stride], inliers=count)
             for pair, count in rows]
    save(work / "verified_pairs.json", pairs)


def triangulate_fixed(colmap, model, images, work, pycolmap):
    """Supply real covisibility tracks to COLMAP fusion while keeping the input poses fixed."""
    original = pycolmap.Reconstruction(model)
    if len(original.cameras) != 1:
        raise ValueError("The fixed-pose triangulation adapter requires a single supplied camera.")
    camera = next(iter(original.cameras.values()))
    database = work / "database.db"
    stage(work, "features", colmap, [
        "feature_extractor", "--database_path", database, "--image_path", images,
        "--ImageReader.single_camera", 1, "--ImageReader.camera_model", camera.model.name,
        "--ImageReader.camera_params", ",".join(map(str, camera.params)),
        "--FeatureExtraction.max_image_size", 2000, "--FeatureExtraction.num_threads", 4,
        "--FeatureExtraction.use_gpu", 1,
    ])
    stage(work, "matches", colmap, [
        "exhaustive_matcher", "--database_path", database, "--FeatureMatching.num_threads", 4,
        "--FeatureMatching.use_gpu", 1,
    ])
    analyze_database(database, work)
    by_name = {image.name: image for image in original.images.values()}
    with sqlite3.connect(database) as db:
        rows = list(db.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id"))
    camera_ids = {row[2] for row in rows}
    if {row[1] for row in rows} != set(by_name) or len(camera_ids) != 1:
        raise ValueError("COLMAP feature image IDs do not describe the supplied pose images.")
    mapped = work / "fixed_database_ids"
    mapped.mkdir()
    (mapped / "cameras.txt").write_text(
        " ".join(map(str, [camera_ids.pop(), camera.model.name, camera.width, camera.height, *camera.params])) + "\n",
        encoding="utf-8",
    )
    image_lines = []
    for image_id, name, camera_id in rows:
        pose = by_name[name].cam_from_world()
        wxyz = np.roll(pose.rotation.quat, 1)
        image_lines.append(" ".join(map(str, [image_id, *wxyz, *pose.translation, camera_id, name])))
    (mapped / "images.txt").write_text("\n\n".join(image_lines) + "\n\n", encoding="utf-8")
    (mapped / "points3D.txt").write_text("# Initially empty.\n", encoding="utf-8")
    triangulated = work / "triangulated"
    triangulated.mkdir()
    stage(work, "triangulation", colmap, [
        "point_triangulator", "--database_path", database, "--image_path", images,
        "--input_path", mapped, "--output_path", triangulated, "--refine_intrinsics", 0,
        "--Mapper.fix_existing_frames", 1, "--Mapper.num_threads", 4,
    ])
    after = pycolmap.Reconstruction(triangulated)
    if after.num_reg_images() != len(by_name) or not after.num_points3D():
        raise ValueError("Fixed-pose triangulation produced no usable covisibility tracks.")
    pose_error = max(np.max(np.abs(image.cam_from_world().matrix() - by_name[image.name].cam_from_world().matrix()))
                     for image in after.images.values())
    if pose_error > 1e-8 or not all(
        np.allclose(candidate.params, camera.params, atol=1e-8, rtol=0) for candidate in after.cameras.values()
    ):
        raise ValueError("Triangulation unexpectedly changed the supplied camera geometry.")
    save(work / "fixed_pose_verification.json", dict(
        triangulated_points=after.num_points3D(), maximum_world_to_camera_difference=float(pose_error),
        original_intrinsics_preserved=True, method="COLMAP feature matching and fixed-pose point triangulator",
    ))
    return triangulated


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    work = args.output.resolve()
    images = args.images.resolve()
    colmap = args.colmap.resolve()
    paths = sorted(path for path in images.rglob("*") if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not 3 <= len(paths) <= 180:
        raise ValueError("Supply 3–180 supported images in the selected pilot input directory.")
    with args.settings.open(encoding="utf-8") as config:
        settings = json.load(config)["pilot"]
    inventory = []
    for path in paths:
        with Image.open(path) as image:
            size = list(image.size)
            image.verify()
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        inventory.append(dict(name=path.relative_to(images).as_posix(), dimensions_px=size, sha256=digest))
    save(work / "inputs.json", dict(images=inventory, settings=settings, fixed_poses=bool(args.fixed_model)))
    sys.path.insert(0, str(args.pycolmap.resolve()))
    import pycolmap
    if args.align_existing:
        gps_alignment(images, paths, work, colmap, args.align_existing.resolve(), args.gps_crs, pycolmap)
        return
    if args.fixed_model:
        model = args.fixed_model.resolve()
        if args.triangulate_fixed:
            model = triangulate_fixed(colmap, model, images, work, pycolmap)
    else:
        models = work / "models"
        models.mkdir()
        database = work / "database.db"
        stage(work, "features", colmap, [
            "feature_extractor", "--database_path", database, "--image_path", images,
            "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "SIMPLE_RADIAL",
            "--FeatureExtraction.max_image_size", 2000, "--FeatureExtraction.num_threads", 4,
            "--FeatureExtraction.use_gpu", 1,
        ])
        stage(work, "matches", colmap, [
            "exhaustive_matcher", "--database_path", database, "--FeatureMatching.num_threads", 4,
            "--FeatureMatching.use_gpu", 1,
        ])
        analyze_database(database, work)
        stage(work, "mapping", colmap, [
            "mapper", "--database_path", database, "--image_path", images, "--output_path", models,
            "--Mapper.num_threads", 4,
        ])
        candidates = [(pycolmap.Reconstruction(path).num_reg_images(), path) for path in models.iterdir()
                      if (path / "cameras.bin").is_file()]
        if not candidates:
            raise RuntimeError("No usable sparse model. Inspect the mapping log and verified graph.")
        _, model = max(candidates, key=lambda row: row[0])
        gps_alignment(images, paths, work, colmap, model, args.gps_crs, pycolmap)
    reconstruction = pycolmap.Reconstruction(model)
    registered = reconstruction.num_reg_images()
    save(work / "sparse.json", dict(model=str(model), image_count=len(paths), registered_images=registered,
                                   points3d=reconstruction.num_points3D(), fixed_poses=bool(args.fixed_model)))
    if registered < 3:
        raise ValueError("Dense fusion requires at least three registered images.")
    workspace = work / "dense"
    size = settings["dense_max_image_size"]
    cache = settings["stereo_cache_gib"]
    stage(work, "undistort", colmap, [
        "image_undistorter", "--image_path", images, "--input_path", model, "--output_path", workspace,
        "--output_type", "COLMAP", "--max_image_size", size, "--num_threads", 4,
        "--num_patch_match_src_images", settings["dense_source_views"],
    ])
    if args.stereo_pairs:
        # The supplied fixed-pose model can have camera records without sparse tracks.
        # A measured acquisition-neighbor list is then provided as native COLMAP configuration.
        names = {image.name for image in reconstruction.images.values()}
        pair_lines = args.stereo_pairs.read_text(encoding="utf-8").splitlines()
        if len(pair_lines) % 2 or any(
            target not in names or any(source.strip() not in names for source in sources.split(","))
            for target, sources in zip(pair_lines[::2], pair_lines[1::2], strict=True)
        ):
            raise ValueError("Stereo neighbor list contains an image outside the registered model.")
        (workspace / "stereo" / "patch-match.cfg").write_text("\n".join(pair_lines) + "\n", encoding="utf-8")
    depth = [] if not args.depth_range else [
        "--PatchMatchStereo.depth_min", args.depth_range[0],
        "--PatchMatchStereo.depth_max", args.depth_range[1],
    ]
    stage(work, "stereo", colmap, [
        "patch_match_stereo", "--workspace_path", workspace, "--PatchMatchStereo.gpu_index", 0,
        "--PatchMatchStereo.geom_consistency", 1, "--PatchMatchStereo.max_image_size", size,
        "--PatchMatchStereo.cache_size", cache, "--PatchMatchStereo.num_iterations", settings["stereo_iterations"],
        *depth,
    ])
    stage(work, "fusion", colmap, [
        "stereo_fusion", "--workspace_path", workspace, "--input_type", "geometric",
        "--output_path", workspace / "fused.ply", "--StereoFusion.num_threads", 4,
        "--StereoFusion.max_image_size", size, "--StereoFusion.cache_size", cache,
        "--StereoFusion.min_num_pixels", settings["minimum_fused_pixels"],
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("colmap", "pycolmap", "images", "settings", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    alignment = parser.add_mutually_exclusive_group(required=True)
    alignment.add_argument("--gps-crs", type=CRS.from_user_input)
    alignment.add_argument("--fixed-model", type=Path)
    parser.add_argument("--align-existing", type=Path)
    parser.add_argument("--triangulate-fixed", action="store_true")
    parser.add_argument("--stereo-pairs", type=Path)
    parser.add_argument("--depth-range", nargs=2, type=float)
    args = parser.parse_args()
    if args.gps_crs:
        if not args.gps_crs.is_projected or args.gps_crs.axis_info[0].unit_conversion_factor != 1:
            parser.error("--gps-crs must specify a projected CRS with metre units")
        args.gps_crs = args.gps_crs.to_string()
    if args.align_existing and not args.gps_crs:
        parser.error("--align-existing requires --gps-crs")
    if args.triangulate_fixed and not args.fixed_model:
        parser.error("--triangulate-fixed requires --fixed-model")
    if args.depth_range and not 0 < args.depth_range[0] < args.depth_range[1] < np.inf:
        parser.error("--depth-range requires positive, increasing, finite values")
    run(args)


if __name__ == "__main__":
    main()
