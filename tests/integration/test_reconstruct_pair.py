import json
import numpy as np
from PIL import Image
import pytest

from vision3d.cli import main
from vision3d.geometry.camera import Camera
from vision3d.reconstruction.pair import reconstruct_pair


def photographs(tmp_path):
    """Rasterized frontoparallel textured patches at different known depths."""
    rng = np.random.default_rng(21)
    left = np.zeros((600, 800, 3), np.uint8)
    right = left.copy()
    for y in range(70, 550, 90):
        for x in range(130, 750, 90):
            disparity = int(rng.integers(22, 55))
            patch = rng.integers(0, 256, (45, 45, 3), dtype=np.uint8)
            left[y:y + 45, x:x + 45] = patch
            right[y:y + 45, x - disparity:x - disparity + 45] = patch
    paths = [tmp_path / "left.png", tmp_path / "right.png"]
    for path, image in zip(paths, (left, right)):
        Image.fromarray(image).save(path)
    camera_path = tmp_path / "camera.json"
    camera_path.write_text(json.dumps(Camera(800, 600, 600, 600, 399.5, 299.5).to_dict()))
    return paths, camera_path


def test_complete_reconstruction_cli(tmp_path, capsys):
    paths, camera = photographs(tmp_path)
    output = tmp_path / "reconstruction"
    main(["reconstruct-pair", "--image1", str(paths[0]), "--image2", str(paths[1]),
          "--camera", str(camera), "--output", str(output), "--max-dimension", "600"])
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["status"] == "success" and metrics["valid_points"] > 100
    assert metrics["median_reprojection_error_px"] < 0.5
    tracks = np.load(output / "correspondences.npz")
    assert len(tracks["ply_observation_indices"]) == metrics["valid_points"]
    assert tracks["positive_depth"][tracks["valid_pose_points"]].all()
    assert (output / "sparse.ply").stat().st_size > 100
    assert "valid_points" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        reconstruct_pair(*paths, output, camera=camera)


def test_rejected_run_has_no_cloud(tmp_path):
    path = tmp_path / "blank.png"
    Image.new("RGB", (100, 100)).save(path)
    output = tmp_path / "rejected"
    with pytest.warns(UserWarning), pytest.raises(SystemExit) as error:
        main(["reconstruct-pair", "--image1", str(path), "--image2", str(path),
              "--approximate-intrinsics", "--fov-degrees", "60", "--output", str(output)])
    assert error.value.code == 3
    assert not (output / "sparse.ply").exists()
    assert json.loads((output / "metrics.json").read_text())["reason"] == "insufficient_correspondences"


def test_missing_or_wrong_calibration(tmp_path):
    paths, camera = photographs(tmp_path)
    with pytest.raises(ValueError, match="Choose"):
        reconstruct_pair(*paths, tmp_path / "invalid")
    camera.write_text(json.dumps(Camera(400, 300, 300, 300, 199, 149).to_dict()))
    with pytest.raises(ValueError, match="dimensions"):
        reconstruct_pair(*paths, tmp_path / "invalid", camera=camera)
