import json

import numpy as np
from PIL import Image
import pytest

from vision3d.geometry.camera import infer_camera, load_calibration
from vision3d.io import image_paths, load_image


def test_jpeg_exif_and_portrait_resize(tmp_path):
    path = tmp_path / "portrait.jpg"
    exif = Image.Exif()
    exif[274] = 6
    exif[41989] = 35
    Image.new("RGB", (120, 80)).save(path, exif=exif)
    frame = load_image(path)
    height, width = frame.image.shape[:2]
    assert (width, height) == (80, 120)
    camera = infer_camera(width, height, frame.metadata)
    assert camera.approximate and camera.confidence == "exif_estimate"
    assert camera.fx == pytest.approx(np.hypot(120, 80) * 35 / np.hypot(36, 24))
    small = camera.resized(40, 60)
    assert (small.cx, small.cy, small.fx) == pytest.approx((19.5, 29.5, camera.fx / 2))


def test_focal_plane_axes_follow_orientation():
    tags = {"37386": 5, "41486": 180, "41487": 190, "41488": 4, "40962": 1000, "40963": 600}
    metadata = {"source_size": [1000, 600], "exif_orientation": 8, "exif_focal": tags}
    camera = infer_camera(600, 1000, metadata)
    assert (camera.fx, camera.fy) == (950, 900)
    tags["41989"] = 80
    assert infer_camera(600, 1000, metadata).confidence == "fallback"
    del tags["41989"]
    tags["40962"] = 2000
    assert infer_camera(600, 1000, metadata).confidence == "fallback"


@pytest.mark.parametrize("tags", [{}, {"41989": 2}, {"37386": 5}, {"41989": 10000}])
def test_fallback_scales_each_image_independently(tags):
    small = infer_camera(300, 400, {"exif_focal": tags})
    large = infer_camera(800, 600, {})
    assert (small.fx, large.fx) == (500, 1000)
    assert small.confidence == large.confidence == "fallback"


def test_pair_specific_calibration_and_cross_folder_selection(tmp_path):
    paths = []
    for name, width in [("b", 100), ("a", 120)]:
        folder = tmp_path / name
        folder.mkdir()
        path = folder / f"{name}.jpg"
        Image.new("RGB", (width, 80)).save(path)
        paths.append(path)
    assert image_paths(paths) == paths[::-1]
    with pytest.raises(ValueError, match="more than once"):
        image_paths([paths[0], paths[0]])
    calibration = tmp_path / "pair.json"
    calibration.write_text(json.dumps({
        "camera1": dict(width=120, height=80, fx=70, fy=71, cx=60, cy=40),
        "camera2": dict(width=100, height=80, fx=66, fy=67, cx=50, cy=40),
    }))
    _, pair = load_calibration(calibration, pair=True)
    assert [c.width for c in pair] == [120, 100]
    with pytest.raises(ValueError, match="not a two-camera pair"):
        load_calibration(calibration)
