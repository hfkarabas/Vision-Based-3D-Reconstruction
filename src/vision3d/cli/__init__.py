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
    calibration.add_argument("--auto-intrinsics", action="store_true")
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
    inputs = sequence.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", dest="input_path")
    inputs.add_argument("--images", nargs="+", dest="input_path", metavar="IMAGE")
    sequence.add_argument("--output", required=True)
    calibration = sequence.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--auto-intrinsics", action="store_true")
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
    sequence.add_argument("--exhaustive", action="store_true", help="Test every pair (at most 30 images)")
    sequence.add_argument("--pair-cache", help="Cache reciprocal descriptor matches; rerun geometry on every run")
    sequence.add_argument("--verifier", choices=("ransac", "magsac"), default="ransac")
    sequence.add_argument("--probe-seeds", action="store_true", help="Probe valid seeds for a third pose using unchanged PnP gates")
    sequence.add_argument("--continuation-seed", action="store_true", help="Prefer a valid seed with third-view track support")
    sequence.add_argument("--quality-config", help="JSON geometry, registration and bundle thresholds")
    sequence.add_argument("--feature-cache", help="Content/configuration keyed numeric feature cache directory")
    sequence.add_argument('--recovery-pairs', type=int, default=0, help='Additional verified appearance-ranked pairs, budget 0..30')
    sequence.add_argument('--map-recovery-pairs', type=int, default=0,
                          help='Additional pairs to match against registered views when SfM stalls, budget 0..100')
    sequence.add_argument('--map-recovery-rounds', type=int, default=3, help='Maximum stalled-map matching rounds, 1..10')
    sequence.add_argument("--seed-search", type=int, default=0, help="Maximum seed hypotheses to grow before initialization")
    sequence.add_argument("--seed-search-views", type=int, default=8, help="Additional cameras per seed trial")
    sequence.add_argument("--position-hints", help="Optional JSON mapping filenames to local metric [east, north, up] hints")
    sequence.add_argument("--station-radius", type=float, default=15., help="Horizontal station clustering radius in meters")
    sequence.add_argument("--quality-pnp", action="store_true", help="Try an additional landmark-quality PnP hypothesis")
    sequence.add_argument("--final-retry", type=int, default=0, help="Maximum additional cameras after final refinement")
    sequence.add_argument("--station-recovery", action="store_true", help="Reserve recovery pairs for unmapped GPS stations")
    sequence.add_argument("--bridge-pairs", type=int, default=0, help="Budget for additional pre-initialization station bridges")
    sequence.add_argument("--initial-pair", nargs=2, metavar="FILENAME",
                          help="Replay a specific verified seed by filename, with the normal two-view gates")
    sequence.add_argument('--joint-ba', action='store_true', help='Final bounded joint map refinement')
    sequence.add_argument('--refine-points', action='store_true', help='Fixed-camera refinement of verified multiview landmarks')
    keyframes = sub.add_parser("keyframes", help="Extract motion- and quality-selected MP4 frames")
    keyframes.add_argument("--video", required=True)
    keyframes.add_argument("--output", required=True)
    keyframes.add_argument("--sample-hz", type=float, default=4)
    keyframes.add_argument("--max-keyframes", type=int, default=40)
    keyframes.add_argument("--min-sharpness", type=float, default=18)
    keyframes.add_argument("--min-features", type=int, default=45)
    keyframes.add_argument("--strategy", choices=("time-uniform", "overlap-aware"), default="time-uniform",
                           help="Method for reducing accepted candidates to the keyframe budget")
    keyframe_calibration = keyframes.add_mutually_exclusive_group()
    keyframe_calibration.add_argument("--fov-degrees", type=float)
    keyframe_calibration.add_argument("--camera", help="Shared calibration at decoded video resolution")
    dense = sub.add_parser("dense-pair", help="Experimental CPU dense stereo from a calibrated pair result")
    dense.add_argument("--pair-dir", required=True)
    dense.add_argument("--output", required=True)
    dense.add_argument("--mesh", action="store_true", help="Export an open vertex-colored stereo surface")
    multiview = sub.add_parser("dense-sequence", help="Fuse independently checked multi-view stereo depth")
    multiview.add_argument("--sparse-dir", required=True)
    multiview.add_argument("--output", required=True)
    multiview.add_argument("--profile", choices=("fast", "balanced", "high", "uav"), default="balanced")
    multiview.add_argument("--mesh", action="store_true")
    multiview.add_argument("--object-masks", help="JSON mapping registered filenames to normalized object polygons")
    multiview.add_argument("--glb", action="store_true", help="Also export a validated vertex-colored GLB mesh")
    multiview.add_argument("--obj", action="store_true", help="Also export validated OBJ geometry and normals")
    multiview.add_argument("--target-faces", type=int)
    multiview.add_argument("--depth-diagnostics", action="store_true", help="Save rectified depth maps and sampled fusion provenance")
    multiview.add_argument("--mesh-common-witnesses", action="store_true",
                           help="Require all three mesh corners to share the UAV depth witnesses")
    asset = sub.add_parser("surface-asset", help="Validate and optimize a completed dense surface")
    asset.add_argument("--dense-dir", required=True)
    asset.add_argument("--sparse-dir", help="Relocated sparse artifact directory, if necessary")
    asset.add_argument("--output", required=True)
    asset.add_argument("--target-faces", type=int, default=450000)
    asset.add_argument("--max-error", type=float, help="Maximum sampled P99 surface deviation in SfM units")
    asset.add_argument("--geometry-weighted-winding", action="store_true",
                       help="Preserve geometrically coherent edge ties when resolving winding")
    texture = sub.add_parser("texture-asset", help="Project source photographs onto a validated surface and export GLB")
    texture.add_argument("--surface-dir", required=True)
    texture.add_argument("--output", required=True)
    texture.add_argument("--atlas-size", type=int, default=4096)
    texture.add_argument("--holdout-stride", type=int, default=0, help="Reserve every Nth photo for appearance validation")
    texture.add_argument("--units-per-world", type=float, default=1., help="Output units per SfM unit, supplied by the user")
    texture.add_argument("--front-faces-only", action="store_true", help="Exclude visible backfaces from the photographic material")
    texture.add_argument("--refine-visible-faces", type=int, default=0,
                         help="Maximum partially visible faces to subdivide before photographic reassignment")
    texture.add_argument("--obj", action="store_true", help="Also export OBJ and its photographic MTL")
    args = vars(parser.parse_args(argv))
    selected = args.pop("command")
    if selected == "texture-asset":
        from vision3d.reconstruction.texture_asset import build_texture_asset
        try:
            result = build_texture_asset(**args, stage=lambda name: print("Stage: " + name, flush=True))
        except (ValueError, OSError, KeyError, TypeError, RuntimeError, cv2.error) as exc:
            parser.exit(2, f"error: {exc}\n")
        print(json.dumps(result))
        return
    if selected == "surface-asset":
        from vision3d.reconstruction.surface_asset import build_surface_asset
        try:
            result = build_surface_asset(**args, stage=lambda name: print("Stage: " + name, flush=True))
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            parser.exit(2, f"error: {exc}\n")
        print(json.dumps(result))
        return
    if selected == "keyframes":
        from vision3d.io.video import KeyframeOptions, extract_keyframes
        try:
            options = KeyframeOptions(**{name: args.pop(name) for name in
                                        ("sample_hz", "max_keyframes", "min_sharpness", "min_features", "strategy")})
            result = extract_keyframes(**args, options=options,
                                       stage=lambda message: print("Stage: " + message, flush=True))
        except (ValueError, OSError, cv2.error) as exc:
            parser.exit(2, f"error: {exc}\n")
        print(json.dumps({"output": args["output"], "selected": len(result["selected"]),
                          "decoded": result["decoded_frames"]}))
        return
    if selected == "dense-sequence":
        from vision3d.reconstruction.multiview import reconstruct_multiview
        try:
            result = reconstruct_multiview(**args, stage=lambda name: print("Stage: " + name, flush=True))
        except (ValueError, OSError, KeyError, TypeError, cv2.error) as exc:
            parser.exit(2, f"error: {exc}\n")
        print(json.dumps(result))
        return
    if selected == "dense-pair":
        from vision3d.reconstruction.stereo import reconstruct_dense_pair
        try:
            print(json.dumps(reconstruct_dense_pair(**args)))
        except (ValueError, OSError, KeyError, TypeError, cv2.error) as exc:
            parser.exit(2, f"error: {exc}\n")
        return
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
