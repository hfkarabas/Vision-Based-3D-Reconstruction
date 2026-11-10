from collections import Counter
import json
from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

from vision3d.demo import support
from vision3d.io.video import (KeyframeOptions, VideoInfo, extract_keyframes, inspect_video, select_by_overlap,
                               video_camera)


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "development.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (320, 240))
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 encoder unavailable")
    rng = np.random.default_rng(27)
    patches = [(x, y, rng.integers(3, 11), rng.integers(0, 256, (24, 24, 3), dtype=np.uint8))
               for y in range(15, 210, 35) for x in range(60, 295, 35)]
    try:
        for view in range(5):
            image = np.zeros((240, 320, 3), np.uint8)
            for x, y, dx, patch in patches:
                x -= view * dx
                image[y:y + 24, x:x + 24] = patch
            for _ in range(3):
                writer.write(image)
            if view == 1:
                writer.write(np.zeros_like(image))
    finally:
        writer.release()
    return path


@pytest.mark.parametrize("strategy", ["time-uniform", "overlap-aware"])
def test_video_keyframes_score_overlap_blur_and_duplicates(clip, tmp_path, strategy):
    assert inspect_video(clip) == VideoInfo(320, 240, 8, 16)
    output = tmp_path / "selected"
    report = extract_keyframes(clip, output, KeyframeOptions(sample_hz=8, max_keyframes=4, strategy=strategy))
    decisions = Counter(item["decision"] for item in report["decisions"])
    assert decisions["near_duplicate"] >= 5
    assert decisions["blur"] == 1
    assert decisions["budget"] >= 1
    selected = report["selected"]
    assert len(selected) == 4
    assert selected[0]["frame_index"] == 0 and selected[-1]["frame_index"] >= 13
    assert [f["frame_index"] for f in selected] == sorted(f["frame_index"] for f in selected)
    assert any(f.get("inliers", 0) >= 12 for f in selected)
    diagnostics = report["selection"]
    assert diagnostics["verified_neighbor_pairs"] + diagnostics["weak_neighbor_transitions"] == 3
    assert sum(diagnostics["source_frame_interval_counts"].values()) == 3
    assert diagnostics["temporal_coverage_fraction"] == pytest.approx(selected[-1]["frame_index"] / 15)
    assert diagnostics["near_duplicate_neighbor_pairs"] == 0
    if strategy == "time-uniform":
        assert diagnostics["evaluated_overlap_pairs"] == 3
    assert cv2.imread(str(output / "frames" / selected[0]["filename"])).shape[:2] == (240, 320)
    calibration = json.loads((output / "camera.json").read_text())
    assert calibration["approximate"] and calibration["source"] == "video focal-length prior"
    assert calibration["fx"] == pytest.approx(400)


def test_overlap_selection_keeps_bridges_without_choosing_near_duplicates():
    candidates = [{"frame_index": 10 * i} for i in range(13)]
    uniform = [0, 3, 6, 9, 12]
    supported = {(0, 3), (3, 5), (5, 8), (8, 12)}
    near_duplicate = {(0, 1), (1, 2), (2, 3), (3, 12)}

    def measure(i, j):
        if (i, j) in near_duplicate:
            return {"inliers": 100, "inlier_ratio": 1.0, "shift": .001}
        if (i, j) in supported:
            return {"inliers": 30, "inlier_ratio": .5, "shift": .02}
        return {"inliers": 5, "inlier_ratio": .1, "shift": .03}

    assert select_by_overlap(candidates, uniform, measure) == [0, 3, 5, 8, 12]


def test_overlap_search_is_bounded_and_preserves_endpoints():
    count = 1200
    candidates = [{"frame_index": i} for i in range(count)]
    uniform = np.linspace(0, count - 1, 40).round().astype(int).tolist()
    calls = 0

    def measure(i, j):
        nonlocal calls
        calls += 1
        return {"inliers": 50, "inlier_ratio": .5, "shift": .02}

    selected = select_by_overlap(candidates, uniform, measure)
    assert selected == sorted(set(selected))
    assert (len(selected), selected[0], selected[-1]) == (40, 0, count - 1)
    assert calls <= 15 * count


def test_video_intrinsics_scale_and_explicit_calibration(clip, tmp_path):
    from vision3d.geometry.camera import Camera

    original = video_camera(inspect_video(clip), 60)
    resized = original.resized(160, 120)
    assert resized.fx == pytest.approx(original.fx / 2)
    assert (resized.cx, resized.cy) == (79.5, 59.5)
    source = tmp_path / "provided.json"
    camera = Camera(320, 240, 300, 301, 159.5, 119.5)
    source.write_text(json.dumps(camera.to_dict()))
    result = extract_keyframes(clip, tmp_path / "explicit", camera=source)
    assert result["intrinsics"] == camera.to_dict()
    assert result["intrinsics_basis"] == "provided calibration file"


def test_video_corruption_and_cancellation(clip, tmp_path):
    corrupt = tmp_path / "broken.mp4"
    corrupt.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="could not open"):
        inspect_video(corrupt)
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls > 5

    output = tmp_path / "cancelled"
    with pytest.raises(InterruptedError, match="cancelled"):
        extract_keyframes(clip, output, cancelled=stop)
    assert not (output / "keyframes.json").exists()


def test_temporal_schedule_uses_bounded_discovery_above_exhaustive_limit(tmp_path):
    for i in range(31):
        (tmp_path / f"{i:03d}.png").touch()
    standard = support.reconstruction_command(tmp_path, None, tmp_path / "out", video=True)
    enhanced = support.reconstruction_command(tmp_path, None, tmp_path / "out", video=True, enhanced=True)
    for command in (standard, enhanced):
        assert command[command.index("--window") + 1] == "4"
        assert command[command.index("--max-images") + 1] == "50"
        assert "--exhaustive" not in command
    assert standard[standard.index("--recovery-pairs") + 1] == "12"
    assert enhanced[enhanced.index("--recovery-pairs") + 1] == "30"
    assert "--probe-seeds" in enhanced


def test_desktop_video_worker_passes_selected_frames_and_calibration(clip, tmp_path, monkeypatch):
    events = []
    commands = []

    def child(command, log, stage, emit, stop):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        if "keyframes" in command:
            extract_keyframes(clip, output, KeyframeOptions(sample_hz=8))
        else:
            assert "--images" in command and "--camera" in command
            calibration = Path(command[command.index("--camera") + 1])
            assert json.loads(calibration.read_text())["source"] == "video focal-length prior"
            output.mkdir()
            (output / "sparse.ply").write_bytes(b"worker fixture")
            (output / "metrics.json").write_text(json.dumps(
                dict(status="success", registered_images=5, valid_landmarks=20, runtime_seconds=1)))
        return 0

    monkeypatch.setattr(support, "run_child", child)
    support.run_job(clip, "", tmp_path / "runs", lambda *event: events.append(event),
                    threading.Event(), video_options={"sample_hz": 8, "strategy": "overlap-aware"})
    assert len(commands) == 2
    assert commands[0][commands[0].index("--strategy") + 1] == "overlap-aware"
    assert any(kind == "result" for kind, _ in events), events
    assert any(kind == "validated" and value["video"] for kind, value in events)
