from types import SimpleNamespace

import numpy as np
import pytest

from vision3d.geometry.camera import Camera
from vision3d.reconstruction import dense_quality
from vision3d.reconstruction.dense_quality import landmark_visibility, round_trip_depth, spread_references
from vision3d.reconstruction.multiview import DepthMap, plan_depth, recenter_rectification
from vision3d.reconstruction.state import Pose
from vision3d.reconstruction.surface import build_surface


def test_view_budget_rewards_uncovered_valid_landmarks():
    pose = Pose(np.eye(3), np.zeros(3))
    landmarks = [{"xyz": [0, 0, 5], "observations": {"0": 1, "1": 2}, "reprojection_errors_px": {"1": 3}},
                 {"xyz": [0, 0, -5], "observations": {"2": 5}},
                 {"xyz": [0, 0, 5], "observations": {"0": 3, "1": 4}},
                 {"xyz": [0, 0, 5], "observations": {"2": 8}}]
    visibility = landmark_visibility(landmarks, dict.fromkeys(range(3), pose), 2.5)
    assert visibility == {0: {0, 2}, 1: {2}, 2: {3}}
    assert spread_references({0: [(10,)], 1: [(9,)], 2: [(5,)]}, visibility, 2) == [0, 2]


def test_depth_round_trip_uses_camera_translation_and_refuses_depth_edges():
    k = np.array([[50, 0, 10], [0, 50, 8], [0, 0, 1]])
    image = np.full((20, 24, 3), 128, np.uint8)
    depths = np.full(image.shape[:2], 10, np.float32)
    sample = DepthMap(0, 1, np.eye(3), np.zeros(3), k, depths, image,
                      np.array([[8, 10]]), np.array([[0, 0, 10.]]), np.full((1, 3), 128 / 255))
    witness = DepthMap(2, 3, np.eye(3), np.array([-1.1, 0, 0]), k, depths.copy(), image,
                       np.empty((0, 2)), np.empty((0, 3)), np.empty((0, 3)))
    rows, _, errors = round_trip_depth(sample, witness, .02, 1.5)
    assert rows.tolist() == [0]
    assert np.allclose(errors, 0)
    witness.depth[8, 5] = 0
    assert not len(round_trip_depth(sample, witness, .02, 1.5)[0])
    witness.depth[8, 5] = 11
    assert not len(round_trip_depth(sample, witness, .02, 1.5)[0])


def test_return_pixel_gate_rejects_plausible_depth_with_wide_baseline():
    k = np.array([[100, 0, 120], [0, 100, 20], [0, 0, 1]])
    image = np.full((40, 240, 3), 128, np.uint8)
    sample = SimpleNamespace(xyz=np.array([[0, 0, 10.]]), rotation=np.eye(3), translation=np.zeros(3),
                             intrinsic=k, pixels=np.array([[20, 120]]), colors=np.full((1, 3), 128 / 255))
    witness = SimpleNamespace(rotation=np.eye(3), translation=np.array([-10, 0, 0]), intrinsic=k,
                              depth=np.full((40, 240), 10.19), image=image)
    assert len(round_trip_depth(sample, witness, .02, 2)[0]) == 1
    assert not len(round_trip_depth(sample, witness, .02, 1.5)[0])


def test_rectified_depth_preflight_preserves_vertical_disparity():
    camera = Camera(256, 192, 120, 120, 127.5, 95.5)
    plan = plan_depth(Pose(np.eye(3), [0, 0, 0]), Pose(np.eye(3), [0, -1, 0]),
                      camera, camera, np.tile([0, 0, 12], (40, 1)))
    assert plan.vertical
    assert plan.minimum <= 10 < plan.minimum + plan.count


def test_open_mesh_cleanup_removes_long_and_nonmanifold_faces():
    xyz = np.array([[x, y, 0] for y in range(6) for x in range(6)] + [[100, 1, 0], [1.6, .9, .1]])
    faces = []
    for y in range(5):
        for x in range(5):
            a = y * 6 + x
            faces.extend([[a, a + 6, a + 1], [a + 1, a + 6, a + 7]])
    faces.extend([[1, 7, 36], [1, 7, 37]])
    mesh, metrics = build_surface(xyz, np.full_like(xyz, .5), faces, max_edge=2.)
    assert len(mesh.triangles) == 50
    assert metrics["long_faces_removed"] == 1
    assert metrics["nonmanifold_faces_removed"] == 1
    assert metrics["open_manifold_edges"] and not metrics["closed_manifold_edges"]


@pytest.mark.parametrize("axis, baseline", [(0, -4), (1, 4)])
def test_recentered_stereo_reprojects_both_disparity_axes(axis, baseline):
    rng = np.random.default_rng(9)
    a = rng.uniform([2, 1, 7], [6, 5, 11], (100, 3))
    b = a.copy()
    b[:, axis] += baseline
    projection = np.column_stack((np.diag([600, 600, 1]), np.zeros(3)))
    other = projection.copy()
    other[axis, 3] = 600 * baseline
    p1, p2, q, offset = recenter_rectification(a, b, projection, other, (640, 480), axis)
    ua, ub = a @ p1[:, :3].T, b @ p2[:, :3].T
    ua, ub = ua[:, :2] / ua[:, 2:3], ub[:, :2] / ub[:, 2:3]
    assert offset != 0 and np.allclose(ua[:, 1 - axis], ub[:, 1 - axis])
    homogeneous = np.column_stack((ua, (ua - ub)[:, axis], np.ones(len(ua)))) @ q.T
    assert np.allclose(homogeneous[:, :3] / homogeneous[:, 3:], a)


def test_recentered_depth_requires_two_distinct_supporting_references(monkeypatch):
    def depth(reference, source, recentered=False):
        return SimpleNamespace(reference=reference, source=source, recentered=recentered,
                               xyz=np.ones((1, 3)), colors=np.ones((1, 3)))

    monkeypatch.setattr(dense_quality, "round_trip_depth", lambda *args: (np.array([0]), np.ones((1, 3)), np.zeros(1)))
    sample, same_edge, witness = depth(0, 1, True), depth(1, 0), depth(2, 0)
    accepted, report = dense_quality.supported_samples([sample, same_edge, witness], .02, 1.5, 2)
    assert all(depth.reference != 0 for depth, _, _ in accepted)
    assert report[0]["witness_histogram"] == [0, 1]
    accepted, _ = dense_quality.supported_samples([sample, same_edge, witness, depth(3, 1)], .02, 1.5, 2)
    assert accepted[0][0] is sample
