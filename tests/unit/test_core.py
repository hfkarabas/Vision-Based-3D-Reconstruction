from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytest
from vision3d.io import load_image, iter_frames
from vision3d.preprocessing import preprocess, grayscale, blur_score
from vision3d.features import extract
from vision3d.features.matching import match, ratio_filter
from vision3d.geometry import verify


@pytest.fixture
def texture():
    return np.random.default_rng(42).integers(0, 256, (240, 320, 3), dtype=np.uint8)


def test_image_loading(tmp_path, texture):
    for name in ("b.png", "a.jpg"):
        Image.fromarray(texture).save(tmp_path / name)
    (tmp_path / "bad.png").write_text("bad")
    errors = []
    with pytest.warns(UserWarning):
        frames = list(iter_frames(tmp_path, errors=errors))
    assert [Path(f.metadata["source"]).name for f in frames] == ["a.jpg", "b.png"]
    assert len(errors) == 1
    assert load_image(tmp_path / "b.png").image.shape == texture.shape


def test_orientation(tmp_path):
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (40, 20)).save(tmp_path / "a.jpg", exif=exif)
    assert load_image(tmp_path / "a.jpg").image.shape[:2] == (40, 20)


def test_invalid(tmp_path):
    for path in (tmp_path, tmp_path / "missing", tmp_path / "bad.txt"):
        with pytest.raises(ValueError):
            list(iter_frames(path))
    with pytest.raises(ValueError):
        preprocess(np.array([]))
    with pytest.raises(ValueError):
        list(iter_frames(tmp_path, interval=0))


def test_preprocess(texture):
    resized, stats = preprocess(texture, 160)
    assert resized.shape == (120, 160, 3)
    assert stats["scale_x"] == stats["scale_y"] == 0.5
    assert grayscale(resized).shape == (120, 160)
    assert blur_score(texture) > blur_score(cv2.GaussianBlur(texture, (15, 15), 3))
    assert preprocess(texture, 640)[0].shape == texture.shape


@pytest.mark.parametrize("detector,dtype", [("sift", np.float32), ("orb", np.uint8)])
def test_features_matching(texture, detector, dtype):
    first = extract(texture, detector, 500)
    second = extract(np.roll(texture, 5, axis=1), detector, 500)
    assert first.count > 20
    assert first.descriptors.dtype == dtype
    result = match(first, second, mutual=True)
    assert len(result.raw) >= len(result.ratio) >= len(result.consistent) >= len(result.inliers)
    assert len(result.ratio) > 10


def test_ratio():
    good = [cv2.DMatch(0, 0, 1.0), cv2.DMatch(0, 1, 3.0)]
    bad = [cv2.DMatch(1, 0, 2.8), cv2.DMatch(1, 1, 3.0)]
    assert ratio_filter([good, bad, good[:1]], 0.75) == [good[0]]
    with pytest.raises(ValueError):
        ratio_filter([], 1)


@pytest.mark.parametrize("detector", ["sift", "orb"])
def test_empty_features(detector):
    empty = extract(np.zeros((100, 100), np.uint8), detector)
    assert empty.count == 0
    assert match(empty, empty).status == "insufficient_correspondences"


def test_geometry_failure():
    points = np.zeros((12, 2), np.float32)
    matrix, mask, status = verify(points, points)
    assert matrix is None and not mask.any() and status == "degenerate_correspondences"
    assert verify(points[:3], points[:3])[2] == "insufficient_correspondences"


def test_geometry_nonplanar():
    rng = np.random.default_rng(2)
    xyz = rng.uniform(-1, 1, (100, 3))
    xyz[:, 2] += 5
    p1 = (xyz[:, :2] / xyz[:, 2:] * 500 + 300).astype(np.float32)
    shifted = xyz + [0.4, 0.05, 0.1]
    p2 = (shifted[:, :2] / shifted[:, 2:] * 500 + 300).astype(np.float32)
    matrix, mask, status = verify(p1, p2)
    assert status == "estimated" and mask.sum() >= 90 and matrix.shape == (3, 3)


def test_single_descriptor(texture):
    from dataclasses import replace
    features = extract(texture, "orb", 100)
    single = replace(features, descriptors=features.descriptors[:1])
    assert not match(features, single).raw
    assert not match(single, features, mutual=True).consistent
