from dataclasses import replace
import json
import cv2
import numpy as np
import pytest

from vision3d.geometry.camera import Camera, approximate_camera, load_cameras
from vision3d.geometry.two_view import Quality, ReconstructionRejected, project, reconstruct, triangulate
from vision3d.reconstruction.pair import export_cloud, sample_colors


@pytest.fixture
def scene():
    rng = np.random.default_rng(123)
    xyz = rng.uniform([-2, -1.5, 4], [2, 1.5, 10], (250, 3))
    camera1 = Camera(800, 600, 650, 660, 399.5, 299.5, distortion=(-0.08, 0.01, 0.001, -0.001, 0))
    camera2 = Camera(900, 700, 710, 700, 449.5, 349.5, distortion=(-0.04, 0.01, 0, 0, 0))
    rotation = cv2.Rodrigues(np.array([0.01, 0.04, -0.02]))[0]
    translation = np.array([-1.0, 0.05, 0.02])
    translation /= np.linalg.norm(translation)
    return xyz, camera1, camera2, rotation, translation


def test_camera_scaling_and_normalization(scene, tmp_path):
    xyz, c1, _, _, _ = scene
    assert np.array_equal(c1.K, [[650, 0, 399.5], [0, 660, 299.5], [0, 0, 1]])
    small = c1.resized(400, 300)
    assert small.fx == 325 and small.cx == 199.5 and small.distortion == c1.distortion
    pixels = project(xyz, c1)
    assert np.allclose(c1.normalized(pixels), xyz[:, :2] / xyz[:, 2:], atol=1e-7)
    assert np.allclose(small.normalized((pixels + 0.5) / 2 - 0.5), c1.normalized(pixels))
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(c1.to_dict()))
    assert load_cameras(path)[0].fx == c1.fx
    data = c1.to_dict()
    data["K"][0][0] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="conflicts"):
        load_cameras(path)


@pytest.mark.parametrize("changes", [dict(fx=0), dict(fy=float("nan")), dict(width=0),
    dict(cx=9999), dict(distortion=(1, 2, 3)), dict(coordinates="raw"), dict(approximate=True),
    dict(fx=True), dict(fx="650"), dict(distortion=(1, 2, 3, float("inf"))), dict(distortion="bad")])
def test_invalid_camera(scene, changes):
    with pytest.raises(ValueError):
        replace(scene[1], **changes)


def test_approximation():
    with pytest.warns(UserWarning):
        camera = approximate_camera(800, 600, fov_degrees=90)
    assert camera.approximate and camera.fx == pytest.approx(400)
    assert "90" in camera.assumptions
    for kwargs in ({}, dict(focal_px=300, fov_degrees=60), dict(fov_degrees=180), dict(focal_px=-1)):
        with pytest.raises(ValueError):
            approximate_camera(800, 600, **kwargs)


@pytest.mark.parametrize("data", ["{", "{}", '{"camera1": {}}', '{"width": "800"}'])
def test_invalid_configuration_file(tmp_path, data):
    path = tmp_path / "invalid.json"
    path.write_text(data)
    with pytest.raises(ValueError):
        load_cameras(path)


def test_triangulation_and_reprojection(scene):
    xyz, c1, c2, r, t = scene
    p1, p2 = project(xyz, c1), project(xyz, c2, r, t)
    result = triangulate(p1, p2, c1, c2, r, t)
    assert result["valid"].all()
    assert np.allclose(result["xyz"], xyz, atol=1e-5)
    assert result["errors_px"].max() < 1e-4
    # The wrong translation sign triangulates behind both cameras.
    wrong = triangulate(p1, p2, c1, c2, r, -t)
    assert not wrong["positive_depth"].any()
    disturbed = p2.copy()
    disturbed[0, 1] += 20
    bad = triangulate(p1, disturbed, c1, c2, r, t)
    assert not bad["valid"][0] and bad["errors_px"][0].max() > 2
    bounded = triangulate(p1, p2, c1, c2, r, t, Quality(max_distance=2))
    assert not bounded["valid"].any()


def test_pose_recovery_outliers_and_scale(scene):
    xyz, c1, c2, r, t = scene
    rng = np.random.default_rng(12)
    p1, p2 = project(xyz, c1), project(xyz, c2, r, t)
    p1 += rng.normal(0, 0.1, p1.shape)
    p2 += rng.normal(0, 0.1, p2.shape)
    p2[:30] = rng.uniform([10, 10], [890, 690], (30, 2))
    cv2.setRNGSeed(0)
    result = reconstruct(p1, p2, c1, c2)
    rotation_error = np.rad2deg(np.arccos(np.clip((np.trace(result["rotation"] @ r.T) - 1) / 2, -1, 1)))
    direction_error = np.rad2deg(np.arccos(np.clip(result["translation"] @ t, -1, 1)))
    assert rotation_error < 1 and direction_error < 2
    assert result["metrics"]["valid_points"] > 180
    assert np.linalg.norm(result["translation"]) == pytest.approx(1)
    assert result["translation"] @ t > 0  # cheirality selects sign; magnitude remains unknown
    assert result["metrics"]["median_reprojection_error_px"] < 1


@pytest.mark.parametrize("mode", ["pure_rotation", "low_parallax", "planar", "duplicate", "insufficient"])
def test_degeneracy(scene, mode):
    xyz, c1, c2, r, t = scene
    xyz = xyz.copy()
    if mode == "pure_rotation":
        t = np.zeros(3)
    elif mode == "low_parallax":
        t = t * 0.001
    elif mode == "planar":
        xyz[:, 2] = 6
    elif mode == "duplicate":
        xyz[:] = xyz[0]
    elif mode == "insufficient":
        xyz = xyz[:7]
    cv2.setRNGSeed(0)
    with pytest.raises(ReconstructionRejected):
        reconstruct(project(xyz, c1), project(xyz, c2, r, t), c1, c2)


def test_invalid_geometry(scene):
    xyz, c1, c2, r, t = scene
    p1, p2 = project(xyz, c1), project(xyz, c2, r, t)
    invalid = p1.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        reconstruct(invalid, p2, c1, c2)
    with pytest.raises(ValueError):
        reconstruct(p1, p2[:3], c1, c2)
    with pytest.raises(ValueError):
        triangulate(p1, p2, c1, c2, r, np.zeros(3))
    with pytest.raises(ValueError):
        Quality(essential_threshold_px=float("nan"))
    infinity = triangulate(p1, p1, c1, c1, np.eye(3), [-1, 0, 0])
    assert not infinity["valid"].any()


def test_color_mapping_and_ply(tmp_path):
    image = np.zeros((8, 10, 3), np.uint8)
    image[3, 5] = [30, 60, 90]
    color = sample_colors(image, np.array([[(5.5) * 0.5 - 0.5, (3.5) * 0.5 - 0.5]]),
                          {"scale_x": 0.5, "scale_y": 0.5})
    assert np.allclose(color[0], [90, 60, 30] / np.array(255))
    path = tmp_path / "points.ply"
    export_cloud(path, [[1.0, 2.0, 3.0]], color)
    import open3d as o3d
    cloud = o3d.io.read_point_cloud(str(path))
    assert np.allclose(np.asarray(cloud.points), [[1, 2, 3]])
    assert np.allclose(np.asarray(cloud.colors), color, atol=1 / 255)
    with pytest.raises(ValueError):
        export_cloud(path, [[np.nan, 0, 1]], color)
