import numpy as np
import pytest

from vision3d.reconstruction.stereo import disparity_range, stereo_points, surface_triangles


def test_range_is_bounded_and_ignores_isolated_outliers():
    samples = [20] * 80 + [25] * 20 + [3000]
    minimum, count = disparity_range(samples, 160)
    assert minimum < 20 <= 25 < minimum + count
    assert count % 16 == 0
    with pytest.raises(ValueError, match="bounded"):
        disparity_range(np.linspace(-200, 200, 100), 300)


def test_dense_stereo_reconstructs_known_pixel_shift():
    left = np.random.default_rng(17).integers(0, 256, (80, 192, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-12] = left[:, 12:]
    valid = np.ones(left.shape[:2], bool)
    # f=120 px and baseline=1 imply depth=10 at 12 px disparity.
    q = np.array([[1, 0, 0, -96], [0, 1, 0, -40], [0, 0, 0, 120], [0, 0, 1, 0]])
    xyz, rgb, pixels = stereo_points(left, right, valid, valid, 0, 32, q, np.eye(3), (8, 12))
    assert len(xyz) > 4000
    assert np.median(xyz[:, 2]) == pytest.approx(10, abs=0.1)
    assert rgb.shape == xyz.shape and np.all((rgb >= 0) & (rgb <= 1))
    assert len(surface_triangles(pixels, (192, 80), xyz[:, 2])) > 4000


def test_mesh_does_not_bridge_invalid_pixels_or_depth_discontinuities():
    pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]])
    triangles = surface_triangles(pixels, (3, 2), np.array([10, 10, 10, 30, 30]))
    assert triangles.tolist() == [[0, 2, 1]]
