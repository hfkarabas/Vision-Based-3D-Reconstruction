"""Run independent CPU COLMAP SfM on the fixed AirZoo images and compare common poses."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pycolmap as colmap

from analyze_airzoo_geometry import percentiles, read, station_metadata
from vision3d.geometry.camera import parse_camera
from vision3d.io import load_image
from vision3d.reconstruction.pair import source_hash


def similarity(source, target):
    """Fit a proper 3D similarity with the common camera centers."""
    a, b = source - source.mean(axis=0), target - target.mean(axis=0)
    u, singular, vh = np.linalg.svd(b.T @ a)
    handedness = np.ones(3)
    handedness[-1] = np.sign(np.linalg.det(u @ vh))
    rotation = (u * handedness) @ vh
    scale = np.dot(singular, handedness) / np.square(a).sum()
    return scale, rotation, target.mean(axis=0) - scale * (rotation @ source.mean(axis=0))


def pose_agreement(ours, external, ours_rotations, external_rotations):
    def angles(matrices):
        return np.degrees(np.arccos(np.clip((np.trace(matrices, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))

    scale, rotation, shift = similarity(external, ours)
    errors = np.linalg.norm(scale * external @ rotation.T + shift - ours, axis=1)
    center_aligned = external_rotations @ rotation.T @ ours_rotations.transpose(0, 2, 1)
    left, right = np.triu_indices(len(ours), 1)
    relative_ours = ours_rotations[right] @ ours_rotations[left].transpose(0, 2, 1)
    relative_other = external_rotations[right] @ external_rotations[left].transpose(0, 2, 1)
    relative_error = angles(relative_other @ relative_ours.transpose(0, 2, 1))

    votes = ours_rotations.transpose(0, 2, 1) @ external_rotations
    u, _, vh = np.linalg.svd(votes.sum(axis=0))
    orientation = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vh))]) @ vh
    a = (external - external.mean(axis=0)) @ orientation.T
    b = ours - ours.mean(axis=0)
    orientation_scale = np.sum(a * b) / np.sum(a * a)
    singular = np.linalg.svd(b, compute_uv=False)
    return {"aligned_center_difference_sfm_units": percentiles(errors),
            "center_fit_aligned_rotation_difference_deg": percentiles(angles(center_aligned)),
            "relative_rotation_difference_deg": percentiles(relative_error),
            "orientation_fit_center_difference_sfm_units": percentiles(np.linalg.norm(orientation_scale * a - b, axis=1)),
            "orientation_fit_rotation_difference_deg": percentiles(
                angles(external_rotations @ orientation.T @ ours_rotations.transpose(0, 2, 1))),
            "center_distribution_singular_value_ratios": (singular / singular[0]).tolist(),
            "aligned_center_difference_by_image": errors.tolist()}


def run(sparse, output, threads):
    start = perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    images, models = output / "images", output / "models"
    images.mkdir()
    models.mkdir()
    inputs, local, stations, *_ = station_metadata(sparse)
    intrinsics = read(sparse / "intrinsics.json")
    cameras = {row["image_id"]: parse_camera(row["processed"]) for row in intrinsics}
    keys = {tuple([c.width, c.height, c.fx, c.fy, c.cx, c.cy]) for c in cameras.values()}
    if len(keys) != 1 or any(len(c.distortion) for c in cameras.values()):
        raise ValueError("This comparison uses one fixed pinhole calibration for all input images")
    width, height, *params = keys.pop()
    # COLMAP uses half-integer pixel centers. The saved OpenCV cameras use integer centers.
    params[2] += .5
    params[3] += .5
    for entry in inputs:
        if source_hash(entry["source"]) != entry["sha256"]:
            raise ValueError(f"Benchmark source changed: {entry['source']}")
        image = load_image(entry["source"]).image
        pixels = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(images / f"{entry['image_id']:04d}.png"), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise ValueError(f"Could not write lossless input for {entry['image_id']}")
    database = output / "database.db"
    stage = perf_counter()
    extraction = colmap.FeatureExtractionOptions(max_image_size=max(width, height), num_threads=threads)
    extraction.sift.max_num_features = 6000
    reader = colmap.ImageReaderOptions(camera_model="PINHOLE", camera_params=",".join(map(str, params)))
    colmap.extract_features(database, images, camera_mode=colmap.CameraMode.SINGLE,
                            reader_options=reader, extraction_options=extraction, device=colmap.Device.cpu)
    extraction_seconds = perf_counter() - stage
    stage = perf_counter()
    colmap.match_exhaustive(database, matching_options=colmap.FeatureMatchingOptions(num_threads=threads),
                            device=colmap.Device.cpu)
    matching_seconds = perf_counter() - stage
    stage = perf_counter()
    pipeline = colmap.IncrementalPipelineOptions(num_threads=threads, random_seed=0,
                                                  ba_refine_focal_length=False, ba_refine_extra_params=False)
    reconstructions = colmap.incremental_mapping(database, images, models, pipeline)
    mapping_seconds = perf_counter() - stage
    result = {"version": colmap.__version__, "cuda": colmap.has_cuda, "input_count": len(inputs),
              "camera_params": params, "image_size": [width, height], "threads": threads, "seed": 0,
              "seconds": {"features": extraction_seconds, "matching": matching_seconds, "mapping": mapping_seconds,
                          "total": perf_counter() - start},
              "models": [{"id": i, "registered": model.num_reg_images(), "landmarks": model.num_points3D()}
                         for i, model in reconstructions.items()]}
    if reconstructions:
        best, model = max(reconstructions.items(), key=lambda row: row[1].num_reg_images())
        by_input = {int(Path(image.name).stem): image for image in model.images.values() if image.has_pose}
        common = sorted(set(local) & set(by_input))
        result.update(best_model=best, registered=sorted(by_input), common_registered=common,
                      registered_by_station=dict(Counter(stations[i] for i in by_input)),
                      mean_reprojection_px=model.compute_mean_reprojection_error())
        if len(common) >= 3:
            ours = np.stack([local[i].pose.center for i in common])
            external = np.stack([by_input[i].projection_center() for i in common])
            result.update(pose_agreement(ours, external, np.stack([local[i].pose.rotation for i in common]),
                                         np.stack([by_input[i].cam_from_world().rotation.matrix() for i in common])))
        poses = []
        for entry in inputs:
            i = entry["image_id"]
            row = {"image_id": i, "status": "registered" if i in by_input else "not_registered"}
            if i in by_input:
                row["world_to_camera"] = np.vstack([by_input[i].cam_from_world().matrix(), [0, 0, 0, 1]]).tolist()
            poses.append(row)
        landmarks = []
        id_to_input = {image.image_id: i for i, image in by_input.items()}
        for id_, point in model.points3D.items():
            observations = {str(id_to_input[e.image_id]): e.point2D_idx for e in point.track.elements
                            if e.image_id in id_to_input}
            landmarks.append({"id": id_, "xyz": point.xyz.tolist(), "observations": observations})
        export = output / "for-project-dense"
        export.mkdir()
        for name, data in (("manifest", {"inputs": inputs}), ("poses", poses),
                           ("intrinsics", intrinsics), ("landmarks", landmarks)):
            (export / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def compare_saved(sparse, external, output):
    _, local, *_ = station_metadata(sparse)
    poses = {row["image_id"]: np.asarray(row["world_to_camera"]) for row in read(external / "poses.json")
             if row["status"] == "registered"}
    common = sorted(set(local) & set(poses))
    ours = np.stack([local[i].pose.center for i in common])
    ext_r = np.stack([poses[i][:3, :3] for i in common])
    ext_t = np.stack([poses[i][:3, 3] for i in common])
    external_centers = -np.einsum("nji,nj->ni", ext_r, ext_t)
    result = pose_agreement(ours, external_centers, np.stack([local[i].pose.rotation for i in common]), ext_r)
    result["common_registered"] = common
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


def export_project_model(sparse, output):
    """Write our exact poses, observations and cameras as a COLMAP binary model."""
    _, views, *_ = station_metadata(sparse)
    cameras = {row["image_id"]: parse_camera(row["processed"]) for row in read(sparse / "intrinsics.json")}
    model = colmap.Reconstruction()
    with np.load(sparse / "features.npz") as data:
        for i, view in views.items():
            camera = cameras[i]
            if len(camera.distortion):
                raise ValueError("This fixed benchmark adapter expects PINHOLE cameras")
            model.add_camera_with_trivial_rig(colmap.Camera(
                camera_id=i + 1, model="PINHOLE", width=camera.width, height=camera.height,
                params=[camera.fx, camera.fy, camera.cx + .5, camera.cy + .5]))
            image = colmap.Image(name=f"{i:04d}.png", keypoints=data[f"image_{i}_points"] + .5,
                                 camera_id=i + 1, image_id=i + 1)
            pose = colmap.Rigid3d(view.pose.matrix[:3])
            model.add_image_with_trivial_frame(image, pose)
    for point in read(sparse / "landmarks.json"):
        entries = [colmap.TrackElement(int(i) + 1, feature) for i, feature in point["observations"].items()
                   if int(i) in views]
        if len(entries) < 2:
            continue
        color = np.rint(np.asarray(point["color"]) * 255).clip(0, 255).astype(np.uint8)
        model.add_point3D(point["xyz"], colmap.Track(entries), color)
    output.mkdir(parents=True, exist_ok=False)
    model.write(output)
    reloaded = colmap.Reconstruction(output)
    if (reloaded.num_reg_images() != len(views) or reloaded.num_points3D() != model.num_points3D()
            or any(not np.allclose(reloaded.images[i + 1].cam_from_world().matrix(), view.pose.matrix[:3])
                   for i, view in views.items())):
        raise ValueError("COLMAP model round-trip changed the fixed reconstruction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--compare-saved", type=Path, help="Only compare previously exported poses, without running SfM")
    modes.add_argument("--export-project-model", action="store_true", help="Export fixed project poses to a COLMAP model")
    args = parser.parse_args()
    if args.export_project_model:
        export_project_model(args.sparse, args.output)
    elif args.compare_saved is not None:
        compare_saved(args.sparse, args.compare_saved, args.output)
    else:
        run(args.sparse, args.output, args.threads)
