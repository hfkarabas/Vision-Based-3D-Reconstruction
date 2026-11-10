"""Bounded MP4 keyframe extraction for the existing photo-based reconstruction."""
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from vision3d.features import extract
from vision3d.features.matching import match
from vision3d.geometry import verify
from vision3d.geometry.camera import Camera, load_cameras
from vision3d.preprocessing import blur_score, grayscale
from vision3d.reconstruction.pair import source_hash, write_json


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int

    @property
    def seconds(self):
        return self.frames / self.fps


@dataclass(frozen=True)
class KeyframeOptions:
    sample_hz: float = 4.0
    max_keyframes: int = 40
    min_sharpness: float = 18.0
    min_features: int = 45

    def __post_init__(self):
        if not np.isfinite(self.sample_hz) or not 0.5 <= self.sample_hz <= 12:
            raise ValueError("Sampling rate must be between 0.5 and 12 frames/s")
        if type(self.max_keyframes) is not int or not 2 <= self.max_keyframes <= 50:
            raise ValueError("Keyframe limit must be an integer from 2 to 50")
        if not np.isfinite(self.min_sharpness) or self.min_sharpness < 0 or not 8 <= self.min_features <= 700:
            raise ValueError("Invalid minimum sharpness or feature count")


def open_video(path):
    path = Path(path)
    if path.suffix.lower() != ".mp4" or not path.is_file():
        raise ValueError("Select an existing MP4 video")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("OpenCV could not open this MP4; check the file and codec")
        # Read the decoded orientation and dimensions, not the encoded header alone.
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite([fps, count]).all() or not 0.5 <= fps <= 240 or count < 2:
            raise ValueError("MP4 needs readable FPS and at least two frames")
        if count / fps > 600:
            raise ValueError("Video is longer than 10 minutes; trim it before reconstruction")
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise ValueError("The first MP4 frame could not be decoded")
        height, width = frame.shape[:2]
        if width * height > 25_000_000 or not 64 <= min(width, height) <= max(width, height) <= 8192:
            raise ValueError("Video resolution is outside the supported 64–8192 px per side / 25 megapixel limit")
        return capture, VideoInfo(width, height, float(fps), int(count)), frame
    except BaseException:
        capture.release()
        raise


def inspect_video(path):
    try:
        capture, info, _ = open_video(path)
    except cv2.error as exc:
        raise ValueError("OpenCV could not inspect this MP4; check the file and codec") from exc
    capture.release()
    return info


def video_camera(info, fov_degrees=None):
    if fov_degrees is not None and (not np.isfinite(fov_degrees) or not 10 <= fov_degrees <= 150):
        raise ValueError("Video horizontal FOV must be between 10 and 150 degrees")
    focal = (np.hypot(info.width, info.height) if fov_degrees is None
             else info.width / (2 * np.tan(np.deg2rad(fov_degrees) / 2)))
    prior = "focal equals decoded image diagonal" if fov_degrees is None else f"assumed horizontal FOV {fov_degrees:g} degrees"
    return Camera(info.width, info.height, float(focal), float(focal), (info.width-1)/2, (info.height-1)/2,
                  source="video focal-length prior", approximate=True,
                  assumptions=f"{prior}; centered principal point, square pixels, zero lens distortion",
                  confidence="fallback")


def feature_coverage(features):
    width, height = features.dimensions
    cells = np.floor(features.points / [width, height] * 4).astype(int)
    return len(np.unique(cells, axis=0))


def overlap(previous, current):
    matches = match(previous, current, ratio=.80, mutual=True, geometric=False).consistent
    a = previous.points[[m.queryIdx for m in matches]]
    b = current.points[[m.trainIdx for m in matches]]
    _, mask, _ = verify(a, b, threshold=1.5, confidence=.995)
    support = int(mask.sum())
    # Exact or near-stationary frames may have no nondegenerate F. Use their
    # descriptor displacement only for duplicate detection, never as geometry.
    shift = float(np.median(np.linalg.norm(a-b, axis=1)) / np.hypot(*current.dimensions)) if matches else None
    return {"mutual": len(matches), "inliers": support, "shift": shift,
            "inlier_ratio": support / len(matches) if matches else 0.0}


def choose_candidates(path, options, stage, cancelled):
    capture, info, frame = open_video(path)
    stride = max(1, round(info.fps / options.sample_hz), (info.frames + 1199) // 1200)
    accepted, decisions = [], []
    previous = lost = None
    lost_count = segment = 0
    decoded = 0
    # Bound processing even when the container understates its duration.
    limit = int(info.fps * 600) + 1
    try:
        for index in range(limit):
            if cancelled():
                raise InterruptedError("Keyframe selection cancelled")
            if index and not capture.grab():
                break
            decoded = index + 1
            if index % stride:
                continue
            if len(decisions) >= 1200:
                raise ValueError("Decoded frame count exceeds the bounded sampling budget; trim or re-encode the video")
            if index:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    raise ValueError(f"Cannot decode sampled frame {index}")
            if frame.shape[:2] != (info.height, info.width):
                raise ValueError("Decoded video dimensions changed mid-stream")
            scale = min(1, 640 / max(info.width, info.height))
            gray = grayscale(cv2.resize(frame, (round(info.width*scale), round(info.height*scale)),
                                        interpolation=cv2.INTER_AREA))
            sharpness = blur_score(gray)
            row = {"frame_index": index, "seconds": index / info.fps, "sharpness": sharpness}
            decisions.append(row)
            if len(decisions) % 10 == 1:
                stage(f"Scoring video at {index/info.fps:.1f}/{info.seconds:.1f} s; {len(accepted)} candidates retained")
            if sharpness < options.min_sharpness:
                row["decision"] = "blur"
                continue
            features = extract(gray, "sift", 700)
            coverage = feature_coverage(features)
            row.update(features=features.count, grid_cells=coverage)
            if features.count < options.min_features or coverage < 5:
                row["decision"] = "weak_features"
                continue
            if previous is not None:
                evidence = overlap(previous, features)
                row.update(evidence)
                if evidence["mutual"] >= 12 and evidence["shift"] < .006:
                    row["decision"] = "near_duplicate"
                    lost, lost_count = None, 0
                    continue
                if evidence["inliers"] < 12 or evidence["inlier_ratio"] < .25:
                    # Restart only when a second nearby candidate corroborates a
                    # new segment. Single abrupt cut/blur samples are discarded.
                    link = overlap(lost, features) if lost is not None else None
                    lost_count += 1
                    lost = features
                    if (lost_count < 3 or index / info.fps - accepted[-1]["seconds"] < 1 or
                            link is None or link["mutual"] < 12 or
                            (link["inliers"] < 12 and link["shift"] >= .006)):
                        row["decision"] = "low_overlap"
                        continue
                    segment += 1
            row.update(decision="candidate", segment=segment)
            accepted.append(row)
            previous, lost, lost_count = features, None, 0
        else:
            raise ValueError("Video exceeds the 10 minute decode limit")
    finally:
        capture.release()
    if decoded < info.frames - max(2, int(info.frames * .05)):
        raise ValueError(f"Video ended unexpectedly after {decoded} of {info.frames} declared frames")
    if len(accepted) > options.max_keyframes:
        keep = set(np.linspace(0, len(accepted)-1, options.max_keyframes).round().astype(int))
        for i, row in enumerate(accepted):
            if i not in keep:
                row["decision"] = "budget"
        accepted = [row for i, row in enumerate(accepted) if i in keep]
    return info, decoded, stride, accepted, decisions


def extract_keyframes(video, output, options=KeyframeOptions(), fov_degrees=None,
                      stage=lambda message: None, cancelled=lambda: False, camera=None):
    """Save selected native-resolution PNGs and a shared camera model."""
    start = perf_counter()
    output, video = Path(output), Path(video).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new keyframe output directory: {output}")
    if cancelled():
        raise InterruptedError("Keyframe selection cancelled")
    fingerprint = video.stat() if video.is_file() else None
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    stage("Inspecting MP4 and scoring sampled frames")
    info, decoded, stride, accepted, decisions = choose_candidates(video, options, stage, cancelled)
    if camera is not None:
        if fov_degrees is not None:
            raise ValueError("Choose either explicit video calibration or an assumed FOV")
        calibration, second = load_cameras(camera)
        if calibration != second or (calibration.width, calibration.height) != (info.width, info.height):
            raise ValueError("Video calibration must be shared and match the decoded video dimensions")
    else:
        calibration = video_camera(info, fov_degrees)
    report = {"source": str(video), "video": {**asdict(info), "seconds": info.seconds},
              "timestamp_basis": "frame_index/fps (nominal)", "decoded_frames": decoded, "sample_stride": stride,
              "options": asdict(options), "selected": accepted, "decisions": decisions,
              "intrinsics": calibration.to_dict(),
              "intrinsics_basis": "provided calibration file" if camera is not None else "estimated focal prior"}
    output.mkdir(parents=True, exist_ok=False)
    if len(accepted) < 2:
        report.update(status="rejected", runtime_seconds=perf_counter() - start,
                      decision_counts=dict(Counter(row["decision"] for row in decisions)))
        write_json(output / "keyframes.json", report)
        raise ValueError(f"Only {len(accepted)} usable viewpoints; try a sharp video with more camera movement and overlap")
    folder = output / "frames"
    folder.mkdir()
    capture, actual_info, first = open_video(video)
    try:
        if actual_info != info:
            raise ValueError("Video metadata changed between selection and extraction")
        index = 0
        for count, row in enumerate(accepted, 1):
            target = row["frame_index"]
            if cancelled():
                raise InterruptedError("Keyframe extraction cancelled")
            while index < target:
                if cancelled():
                    raise InterruptedError("Keyframe extraction cancelled")
                if not capture.grab():
                    raise ValueError(f"Video ended before selected frame {target}")
                index += 1
            ok, frame = capture.retrieve() if index else (True, first)
            name = f"frame_{target:08d}.png"
            if not ok or not cv2.imwrite(str(folder / name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f"Could not write selected frame {target}")
            row.update(filename=name, decision="selected")
            stage(f"Writing keyframe {count}/{len(accepted)}")
    finally:
        capture.release()
    latest = video.stat()
    if fingerprint is None or (latest.st_size, latest.st_mtime_ns) != (fingerprint.st_size, fingerprint.st_mtime_ns):
        raise ValueError("Video changed during keyframe extraction")
    write_json(output / "camera.json", calibration.to_dict())
    if cancelled():
        raise InterruptedError("Keyframe extraction cancelled")
    report.update(status="success", source_sha256=source_hash(video),
                  decision_counts=dict(Counter(row["decision"] for row in decisions)),
                  runtime_seconds=perf_counter() - start)
    write_json(output / "keyframes.json", report)
    return report
