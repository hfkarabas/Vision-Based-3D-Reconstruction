"""Streaming diagnostics command line workflow."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
import platform
import importlib.metadata

import cv2
import numpy as np
from vision3d.io import iter_frames
from vision3d.preprocessing import preprocess
from vision3d.features import extract
from vision3d.features.matching import match


def write_image(path, image):
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError(f"Cannot encode diagnostic: {path}")
    encoded.tofile(path)


def analyze(input_path, output, detector="sift", max_dimension=1600, max_features=4000,
            interval=1, max_frames=None, ratio=0.75, mutual=False, threshold=1.5,
            confidence=0.99, seed=0, threads=1, score_blur=True):
    start = perf_counter()
    if max_dimension < 1 or max_features < 1 or interval < 1 or threads < 1:
        raise ValueError("Dimensions, feature limit, interval and threads must be positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("Maximum frame count must be positive")
    if not 0 < ratio < 1 or not 0 < confidence < 1 or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid ratio or RANSAC configuration")
    cv2.setRNGSeed(seed)
    cv2.setNumThreads(threads)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("frames", "keypoints", "matches"):
        (output / folder).mkdir()
    config = dict(detector=detector, max_dimension=max_dimension, max_features=max_features,
                  interval=interval, max_frames=max_frames, ratio=ratio, mutual=mutual,
                  threshold=threshold, confidence=confidence, seed=seed, threads=threads, score_blur=score_blur)
    images, pairs, errors = [], [], []
    previous = None
    for index, frame in enumerate(iter_frames(input_path, interval, max_frames, errors)):
        name = f"frame_{index:06d}"
        image, stats = preprocess(frame.image, max_dimension, score_blur)
        features = extract(image, detector, max_features)
        record = {"id": name, **frame.metadata, **stats, "keypoints": features.count,
                  "feature_seconds": features.seconds, "detector_config": features.config}
        if "frame_index" in frame.metadata:
            frame_name = f"frames/source_{frame.metadata['frame_index']:09d}.jpg"
            write_image(output / frame_name, image)
            record["extracted_frame"] = frame_name
        write_image(output / "keypoints" / f"{name}.jpg",
                    cv2.drawKeypoints(image, features.keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
        images.append(record)
        if previous is not None:
            old_name, old_image, old_features = previous
            result = match(old_features, features, ratio, mutual, threshold, confidence)
            pair_name = f"{old_name}__{name}"
            pairs.append({"first": old_name, "second": name, **result.metrics()})
            for stage, matches in (("raw", result.raw), ("ratio", result.ratio), ("inliers", result.inliers)):
                canvas = cv2.drawMatches(old_image, old_features.keypoints, image, features.keypoints,
                                         matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                write_image(output / "matches" / f"{pair_name}_{stage}.jpg", canvas)
        previous = name, image, features
    metrics = {"images": images, "pairs": pairs, "images_processed": len(images),
               "total_seconds": perf_counter() - start, "peak_memory_bytes": None,
               "memory_note": "Peak process memory unavailable on this platform; input is streamed."}
    # psutil is supplied by Open3D. Windows exposes the lifetime peak working set.
    try:
        import psutil
        metrics["peak_memory_bytes"] = getattr(psutil.Process().memory_info(), "peak_wset", None)
        if metrics["peak_memory_bytes"] is not None:
            metrics["memory_note"] = "Process lifetime peak working set; includes interpreter/imports, excludes child processes."
    except (ImportError, OSError):
        pass
    manifest = {"schema_version": 1, "input": str(Path(input_path).resolve()), "config": config,
                "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "versions": {name: importlib.metadata.version(name) for name in
                             ("vision3d", "numpy", "opencv-contrib-python", "Pillow")},
                "skipped_inputs": errors, "pair_policy": "consecutive processed images",
                "coordinate_system": "EXIF-oriented image, then recorded pixel-center resize transform",
                "warning": "Fundamental matrix inliers do not establish reconstructability."}
    for name, value in (("manifest.json", manifest), ("metrics.json", metrics)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    fields = ["first", "second", "raw_matches", "ratio_matches", "consistent_matches", "ransac_inliers",
              "inlier_ratio", "geometry_status", "matching_seconds"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU image preprocessing and matching diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("analyze")
    command.add_argument("--input", required=True, dest="input_path")
    command.add_argument("--output", default=None)
    command.add_argument("--detector", choices=["sift", "orb"], default="sift")
    for name, default in (("max-dimension", 1600), ("max-features", 4000), ("interval", 1), ("threads", 1), ("seed", 0)):
        command.add_argument(f"--{name}", type=int, default=default)
    command.add_argument("--max-frames", type=int)
    command.add_argument("--ratio", type=float, default=0.75)
    command.add_argument("--threshold", type=float, default=1.5, help="RANSAC distance in processed pixels")
    command.add_argument("--confidence", type=float, default=0.99)
    command.add_argument("--mutual", action="store_true")
    command.add_argument("--no-blur", action="store_false", dest="score_blur")
    pair = sub.add_parser("reconstruct-pair", help="Calibrated sparse two-view reconstruction")
    pair.add_argument("--image1", required=True)
    pair.add_argument("--image2", required=True)
    calibration = pair.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--camera")
    calibration.add_argument("--approximate-intrinsics", action="store_true", dest="approximate")
    assumption = pair.add_mutually_exclusive_group()
    assumption.add_argument("--focal-px", type=float)
    assumption.add_argument("--fov-degrees", type=float)
    pair.add_argument("--output", required=True)
    pair.add_argument("--detector", choices=["sift", "orb"], default="sift")
    pair.add_argument("--max-dimension", type=int, default=1600)
    pair.add_argument("--max-features", type=int, default=6000)
    pair.add_argument("--ratio", type=float, default=0.75)
    pair.add_argument("--seed", type=int, default=0)
    pair.add_argument("--preview", action="store_true", help="Save a static colored point cloud preview")
    sequence = sub.add_parser("reconstruct-sequence", help="Track-based incremental multi-view SfM")
    sequence.add_argument("--input", required=True, dest="input_path")
    sequence.add_argument("--output", required=True)
    calibration = sequence.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--camera")
    calibration.add_argument("--approximate-intrinsics", action="store_true", dest="approximate")
    assumption = sequence.add_mutually_exclusive_group()
    assumption.add_argument("--focal-px", type=float)
    assumption.add_argument("--fov-degrees", type=float)
    sequence.add_argument("--detector", choices=["sift", "orb"], default="sift")
    for name, default in (("max-dimension", 1600), ("max-features", 6000), ("window", 2),
                          ("long-gap", 4), ("max-images", 30), ("min-registered", 5), ("seed", 0)):
        sequence.add_argument(f"--{name}", type=int, default=default)
    sequence.add_argument("--ratio", type=float, default=0.75)
    sequence.add_argument("--local-ba", action="store_true")
    sequence.add_argument("--quality-config", help="JSON geometry, registration and bundle thresholds")
    args = vars(parser.parse_args(argv))
    selected = args.pop("command")
    if selected == "reconstruct-sequence":
        from vision3d.geometry.two_view import ReconstructionRejected
        from vision3d.reconstruction.sequence import reconstruct_sequence
        try:
            result = reconstruct_sequence(**args)
        except ReconstructionRejected as exc:
            parser.exit(3, f"reconstruction rejected: {exc}; diagnostics: {args['output']}\n")
        except (ValueError, OSError, cv2.error) as exc:
            parser.exit(2, f"error: {exc}\n")
        print(json.dumps({"output": str(args["output"]), "registered_images": result["registered_images"],
                          "valid_landmarks": result["valid_landmarks"], "status": result["status"]}))
        if result["status"] != "success":
            parser.exit(3, "partial reconstruction: camera or landmark quality target not reached\n")
        return
    if selected == "reconstruct-pair":
        from vision3d.geometry.two_view import ReconstructionRejected
        from vision3d.reconstruction.pair import reconstruct_pair
        try:
            result = reconstruct_pair(**args)
        except ReconstructionRejected as exc:
            parser.exit(3, f"reconstruction rejected: {exc}; diagnostics: {args['output']}\n")
        except (ValueError, OSError, cv2.error) as exc:
            parser.exit(2, f"error: {exc}\n")
        print(json.dumps({"output": str(args["output"]), "valid_points": result["valid_points"],
                          "median_reprojection_error_px": result["median_reprojection_error_px"]}))
        return
    if args["output"] is None:
        args["output"] = Path("outputs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        if args["threads"] < 1:
            raise ValueError("Threads must be positive")
        result = analyze(**args)
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"output": str(args["output"]), "images_processed": result["images_processed"],
                      "seconds": result["total_seconds"]}))
