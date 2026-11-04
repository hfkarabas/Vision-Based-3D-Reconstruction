import json
import numpy as np
from PIL import Image
import pytest

from vision3d.cli import main
from vision3d.geometry.camera import Camera


def test_five_view_cli_and_exports(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    source = tmp_path / "images"
    source.mkdir()
    rng = np.random.default_rng(21)
    images = [np.zeros((480, 800, 3), np.uint8) for _ in range(5)]
    for y in range(40, 420, 75):
        for x in range(190, 750, 80):
            disparity = int(rng.integers(8, 23))
            patch = rng.integers(0, 256, (39, 39, 3), dtype=np.uint8)
            for i, image in enumerate(images):
                start = x - i * disparity
                image[y:y + 39, start:start + 39] = patch
    for i, image in enumerate(images):
        Image.fromarray(image).save(source / f"{i:02d}.png")
    Image.new("RGB", (800, 480)).save(source / "05.png")
    camera = tmp_path / "camera.json"
    calibration = Camera(800, 480, 600, 600, 399.5, 239.5).to_dict()
    camera.write_text(json.dumps({"images": {f"{i:02d}.png": calibration for i in range(6)}}))
    output = tmp_path / "result"
    main(["reconstruct-sequence", "--input", str(source), "--camera", str(camera),
          "--output", str(output), "--max-features", "2000", "--max-dimension", "800", "--local-ba",
          "--feature-cache", str(tmp_path / "cache")])
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["registered_images"] == 5
    assert metrics["input_images"] == 6 and metrics["rejected_images"] == [5]
    assert metrics["valid_landmarks"] > 100
    assert metrics["after_final_ba"]["invalid_observations"] == 0
    assert metrics["bundle_adjustment"]
    import open3d as o3d
    cloud = o3d.io.read_point_cloud(str(output / "sparse.ply"))
    assert len(cloud.points) == metrics["valid_landmarks"] and cloud.has_colors()
    tracks = json.loads((output / "tracks.json").read_text())
    assert len({t["landmark_id"] for t in tracks if t["landmark_id"] is not None}) == metrics["valid_landmarks"]
    poses = json.loads((output / "poses.json").read_text())
    assert len(poses) == 6 and poses[-1]["world_to_camera"] is None
    second = tmp_path / "warm"
    main(["reconstruct-sequence", "--input", str(source), "--camera", str(camera),
          "--output", str(second), "--max-features", "2000", "--max-dimension", "800", "--local-ba",
          "--feature-cache", str(tmp_path / "cache")])
    warm = json.loads((second / "metrics.json").read_text())
    assert warm["feature_cache_hits"] == 6
    assert warm["registration_order"] == metrics["registration_order"]
    assert warm["valid_landmarks"] == metrics["valid_landmarks"]
    warm_poses = json.loads((second / "poses.json").read_text())
    assert [p["world_to_camera"] for p in poses] == [p["world_to_camera"] for p in warm_poses]
    assert [l["xyz"] for l in json.loads((output / "landmarks.json").read_text())] == [l["xyz"] for l in json.loads((second / "landmarks.json").read_text())]


def test_sequence_rejection_cli(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for i in range(3):
        Image.new("RGB", (100, 100)).save(images / f"{i}.png")
    with pytest.warns(UserWarning), pytest.raises(SystemExit) as exc:
        main(["reconstruct-sequence", "--input", str(images), "--approximate-intrinsics", "--fov-degrees", "60",
              "--output", str(tmp_path / "rejected")])
    assert exc.value.code == 3
    assert not (tmp_path / "rejected/sparse.ply").exists()


def test_invalid_quality_configuration(tmp_path):
    from vision3d.reconstruction.sequence import load_quality
    for index, content in enumerate(["[]", '{"unknown": {}}', '{"bundle": {"max_points": 100000}}']):
        path = tmp_path / f"{index}.json"
        path.write_text(content)
        with pytest.raises(ValueError):
            load_quality(path)
