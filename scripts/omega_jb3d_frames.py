"""Decode an exact JB3D frame subset while preserving supplied corrected COLMAP poses."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from pyproj import Transformer
from scipy.spatial.transform import Rotation


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    poses = {}
    for line in (args.corrected / "images_30fps.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        row = line.split()
        if len(row) != 10 or not row[-1].endswith(".png"):
            raise ValueError("Expected the supplied JB3D per-frame pose table with empty point observations.")
        index = int(Path(row[-1]).stem)
        if index in poses:
            raise ValueError(f"Duplicate video frame {index}.")
        poses[index] = row
    indices = list(range(args.start, args.end + 1, args.step))
    if len(indices) < 7 or set(indices) - poses.keys():
        raise ValueError("Select at least seven frames, all with supplied camera poses.")
    metadata = json.loads(subprocess.check_output([
        str(args.ffmpeg.with_name("ffprobe.exe")), "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-show_format", "-of", "json", str(args.video.resolve()),
    ]))
    stream = metadata["streams"][0]
    source_cameras = (args.source / "cameras.txt").read_text(encoding="utf-8")
    camera = next(line.split() for line in source_cameras.splitlines() if line and not line.startswith("#"))
    if [stream["width"], stream["height"]] != [int(camera[2]), int(camera[3])]:
        raise ValueError("The source video resolution disagrees with the supplied COLMAP intrinsics.")
    if int(stream["nb_frames"]) != max(poses) + 1 or min(poses) != 0:
        raise ValueError("The video frame count disagrees with the complete zero-based pose table.")
    images = out / "images"
    images.mkdir()
    selection = "+".join(f"eq(n\\,{index})" for index in indices)
    command = [str(args.ffmpeg), "-hide_banner", "-nostdin", "-v", "warning", "-threads", "2",
               "-i", str(args.video.resolve()), "-filter_threads", "1", "-vf", "select=" + selection,
               "-fps_mode", "vfr", "-threads", "2", "-start_number", "0", str(images / "%08d.tmp.png")]
    with (out / "decode.log").open("wb") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
    decoded = sorted(images.glob("*.tmp.png"))
    if len(decoded) != len(indices):
        raise ValueError("FFmpeg did not decode the complete requested subset.")
    for path, index in zip(decoded, indices, strict=True):
        path.rename(images / poses[index][-1])

    model = out / "fixed_model"
    model.mkdir()
    shutil.copyfile(args.source / "cameras.txt", model / "cameras.txt")
    (model / "images.txt").write_text("".join(" ".join(poses[index]) + "\n\n" for index in indices),
                                      encoding="utf-8")
    (model / "points3D.txt").write_text("# Fixed supplied cameras, no triangulated pilot points.\n", encoding="utf-8")
    neighbors = []
    for index in indices:
        near = sorted((other for other in indices if other != index), key=lambda other: (abs(index - other), other))[:4]
        neighbors.extend([poses[index][-1], ", ".join(poses[other][-1] for other in near)])
    (out / "stereo_pairs.txt").write_text("\n".join(neighbors) + "\n", encoding="utf-8")

    points = np.loadtxt(args.corrected / "points3D.txt", usecols=(1, 2, 3))
    centers, distances = [], []
    for index in indices:
        pose = np.array(poses[index][1:8], dtype=float)
        rotation = Rotation.from_quat(np.roll(pose[:4], -1))
        centers.append(-rotation.inv().apply(pose[4:]))
        camera_points = rotation.apply(points) + pose[4:]
        depth = camera_points[:, 2]
        # Use a loose frustum only to initialize PatchMatch search bounds from
        # the authors' RGB-derived sparse geometry in the same metric frame.
        valid = (depth > 0) & (np.abs(camera_points[:, 0]) < depth * 0.75) & (
            np.abs(camera_points[:, 1]) < depth * 0.45)
        distances.extend(depth[valid])
    lower, upper = np.quantile(distances, [0.01, 0.99])
    longitude, latitude, _ = Transformer.from_crs(4978, 4979, always_xy=True).transform(
        *np.loadtxt(args.observer))
    local_crs = f"+proj=aeqd +lat_0={latitude} +lon_0={longitude} +datum=WGS84 +units=m +type=crs"
    positions = np.array(centers)
    (out / "inputs.json").write_text(json.dumps(dict(
        source_metadata=metadata, source_video=str(args.video.resolve()), decode=command,
        frame_indices=indices, capture_span_m=float(np.linalg.norm(positions[-1] - positions[0])),
        median_adjacent_baseline_m=float(np.median(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
        depth_range_m=[float(lower * 0.85), float(upper * 1.15)],
        frame_crs_for_raster=local_crs, original_geometry_frame="corrected metric ENU",
        alignment_source="Author-supplied corrected per-frame COLMAP poses",
        lidar_used_in_reconstruction=False,
    ), indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "corrected", "video", "observer", "ffmpeg", "output"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve(), required=True)
    for name in ("start", "end", "step"):
        parser.add_argument(f"--{name}", type=int, required=True)
    args = parser.parse_args()
    if args.start < 0 or args.end < args.start or args.step < 1:
        parser.error("Expected a positive step and a valid nonnegative frame interval.")
    run(args)


if __name__ == "__main__":
    main()
