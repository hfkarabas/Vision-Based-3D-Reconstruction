"""Desktop integration boundaries; the existing SfM math tests remain unchanged."""
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import pytest

from vision3d.demo import support
from vision3d.demo.viewer import load_cloud


def images(folder, count=2):
    folder.mkdir(exist_ok=True, parents=True)
    for i in range(count):
        Image.new("RGB", (40, 30), (100, 150, 200)).save(folder / f"{i}.PNG")
    return folder


def camera(path):
    path.write_text(json.dumps(dict(width=40, height=30, fx=50, fy=50, cx=20, cy=15)))
    return path


def metrics(folder, **extra):
    folder.mkdir(parents=True, exist_ok=True)
    data = dict(status="success", registered_images=6, valid_landmarks=100, runtime_seconds=2.5)
    data.update(extra)
    (folder / "metrics.json").write_text(json.dumps(data))
    (folder / "sparse.ply").write_bytes(b"not loaded by metadata reader")
    return folder


def test_supported_detection_and_order(tmp_path):
    images(tmp_path)
    Image.new("RGB", (40, 30)).save(tmp_path / "a.JpG")
    (tmp_path / "ignore.txt").write_text("not an image")
    (tmp_path / "directory.png").mkdir()
    assert [x.name for x in support.supported_images(tmp_path)] == ["0.PNG", "1.PNG", "a.JpG"]


def test_missing_folder(tmp_path):
    with pytest.raises(ValueError, match="existing image folder"):
        support.supported_images(tmp_path / "missing")


def test_validation_and_corrupt_input(tmp_path):
    folder = images(tmp_path / "images")
    (folder / "broken.jpg").write_bytes(b"bad image")
    result = support.validate_inputs(folder, camera(tmp_path / "camera.json"))
    assert result["usable"] == 2 and result["supported"] == 3
    assert "broken.jpg" in result["skipped"][0]


def test_insufficient_readable_images(tmp_path):
    folder = images(tmp_path / "images", 1)
    (folder / "bad.png").write_bytes(b"bad")
    with pytest.raises(ValueError, match="two readable"):
        support.validate_inputs(folder, camera(tmp_path / "camera.json"))


def test_missing_calibration(tmp_path):
    with pytest.raises(ValueError, match="calibration JSON"):
        support.validate_inputs(images(tmp_path), "")


def test_invalid_calibration(tmp_path):
    c = tmp_path / "bad.json"
    c.write_text("{")
    with pytest.raises(ValueError, match="calibration"):
        support.validate_inputs(images(tmp_path / "images"), c)


def test_calibration_dimensions_and_approximation(tmp_path):
    folder = images(tmp_path / "images")
    c = camera(tmp_path / "camera.json")
    data = json.loads(c.read_text())
    data["width"] = 60
    c.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="dimensions"):
        support.validate_inputs(folder, c)
    data.update(width=40, approximate=True, assumptions="assumed FOV")
    c.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="approximate mode"):
        support.validate_inputs(folder, c)


def test_per_image_calibration(tmp_path):
    folder = images(tmp_path / "images")
    c = camera(tmp_path / "camera.json")
    data = json.loads(c.read_text())
    c.write_text(json.dumps({"images": {"0.PNG": data}}))
    with pytest.raises(ValueError, match="Missing calibration for 1.PNG"):
        support.validate_inputs(folder, c)
    c.write_text(json.dumps({"images": {"0.PNG": data, "1.PNG": data}}))
    assert support.validate_inputs(folder, c)["usable"] == 2


def test_castle_resolution_without_absolute_paths(tmp_path):
    folder = images(tmp_path / "datasets/sceaux/images", 6)
    (tmp_path / "configs").mkdir()
    c = camera(tmp_path / "configs/sceaux.json")
    assert support.resolve_castle(tmp_path) == (folder.resolve(), c.resolve())
    (folder / "0.PNG").unlink()
    with pytest.raises(ValueError, match="six-photo"):
        support.resolve_castle(tmp_path)


def test_castle_missing_does_not_download(tmp_path):
    with pytest.raises(ValueError, match="no download"):
        support.resolve_castle(tmp_path)


def test_unique_output_reservation(tmp_path):
    a, b = support.allocate_run(tmp_path), support.allocate_run(tmp_path)
    assert a != b and a.is_dir() and b.is_dir()
    assert not (a / "reconstruction").exists()
    (a / "keep.txt").write_text("original")
    support.allocate_run(tmp_path)
    assert (a / "keep.txt").read_text() == "original"


def test_stable_cli_configuration(tmp_path):
    folder = images(tmp_path / "images", 6)
    command = support.reconstruction_command(folder, tmp_path / "camera.json", tmp_path / "out")
    assert "reconstruct-sequence" in command and "--local-ba" in command
    assert "--recovery-pairs" not in command and "--joint-ba" not in command
    assert command[-1] == "5"
    assert command[command.index("--input") + 1] == str(folder.resolve())


def test_result_metadata(tmp_path):
    result = support.read_result(metrics(tmp_path))
    assert (result.registered, result.landmarks, result.seconds) == (6, 100, 2.5)
    assert result.ply == tmp_path / "sparse.ply"


@pytest.mark.parametrize("extra", [dict(registered_images=True), dict(valid_landmarks=-1),
    dict(runtime_seconds=float("nan")), dict(status="rejected", reason="low parallax")])
def test_invalid_result_metadata(tmp_path, extra):
    with pytest.raises(ValueError):
        support.read_result(metrics(tmp_path, **extra))


def test_partial_and_missing_ply(tmp_path):
    metrics(tmp_path, status="partial")
    assert support.read_result(tmp_path).status == "partial"
    (tmp_path / "sparse.ply").unlink()
    with pytest.raises(ValueError, match="sparse.ply"):
        support.read_result(tmp_path)


def test_corrupt_result(tmp_path):
    (tmp_path / "metrics.json").write_text("{")
    with pytest.raises(ValueError, match="Cannot read"):
        support.read_result(tmp_path)


def test_verified_stage_transitions(tmp_path):
    assert "loading" in support.artifact_stage(tmp_path)
    (tmp_path / "manifest.json").write_text("{")
    assert "loading" in support.artifact_stage(tmp_path)
    (tmp_path / "manifest.json").write_text("{}")
    assert "Features prepared" in support.artifact_stage(tmp_path)
    (tmp_path / "pairs.json").write_text("[]")
    assert "Geometric matches saved" in support.artifact_stage(tmp_path)
    (tmp_path / "initial_candidates.json").write_text("[]")
    assert "Initial pair selected" in support.artifact_stage(tmp_path)


def test_worker_validation_failure_emits_done(tmp_path):
    events = []
    support.run_job(tmp_path / "missing", "", tmp_path / "outputs",
                    lambda *event: events.append(event), threading.Event())
    assert events[-1] == ("done", None)
    assert any(e[0] == "error" for e in events)


def test_worker_launch_failure_keeps_log(tmp_path, monkeypatch):
    folder = images(tmp_path / "images")
    c = camera(tmp_path / "camera.json")
    def fail(*args, **kwargs):
        raise OSError("could not launch")
    monkeypatch.setattr(support.subprocess, "Popen", fail)
    events = []
    support.run_job(folder, c, tmp_path / "outputs", lambda *e: events.append(e), threading.Event())
    assert any(e[0] == "error" and "could not launch" in e[1] for e in events)
    assert events[-1] == ("done", None)
    assert len(list((tmp_path / "outputs").glob("*/reconstruction.log"))) == 1


def test_worker_completed_and_rejected_process(tmp_path, monkeypatch):
    folder = images(tmp_path / "images")
    c = camera(tmp_path / "camera.json")
    class Process:
        returncode = 0
        def __init__(self, command, **kwargs):
            self.output = Path(command[command.index("--output") + 1])
            metrics(self.output)
        def poll(self):
            return self.returncode
    monkeypatch.setattr(support.subprocess, "Popen", Process)
    events = []
    support.run_job(folder, c, tmp_path / "outputs", lambda *e: events.append(e), threading.Event())
    assert any(e[0] == "result" for e in events)
    Process.returncode = 2
    events.clear()
    support.run_job(folder, c, tmp_path / "outputs", lambda *e: events.append(e), threading.Event())
    assert any(e[0] == "error" and "code 2" in e[1] for e in events)


def test_worker_shutdown_reaps_child(tmp_path, monkeypatch):
    folder = images(tmp_path / "images")
    c = camera(tmp_path / "camera.json")
    stop = threading.Event()
    actions = []
    class Process:
        returncode = None
        def __init__(self, *args, **kwargs):
            stop.set()
        def poll(self):
            return self.returncode
        def terminate(self):
            actions.append("terminate")
            self.returncode = -1
        def wait(self, **kwargs):
            actions.append("wait")
    monkeypatch.setattr(support.subprocess, "Popen", Process)
    support.run_job(folder, c, tmp_path / "outputs", lambda *e: None, stop)
    assert actions == ["terminate", "wait"]


def test_viewer_rejects_missing_and_corrupt_ply(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_cloud(tmp_path / "missing.ply")
    (tmp_path / "bad.ply").write_bytes(b"not PLY")
    with pytest.raises(ValueError, match="no usable"):
        load_cloud(tmp_path / "bad.ply")


def test_viewer_loads_actual_colors(tmp_path):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector([[0, 0, 1], [1, 2, 3]])
    cloud.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0]])
    path = tmp_path / "colored.ply"
    assert o3d.io.write_point_cloud(str(path), cloud)
    loaded = load_cloud(path)
    assert np.allclose(loaded.colors, cloud.colors)
    assert np.allclose(loaded.points, cloud.points)


def test_open_output_folder_validation(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="does not exist"):
        support.open_directory(tmp_path / "missing")
    if support.sys.platform == "win32":
        calls = []
        monkeypatch.setattr(support.os, "startfile", calls.append)
        support.open_directory(tmp_path)
        assert calls == [str(tmp_path.resolve())]
