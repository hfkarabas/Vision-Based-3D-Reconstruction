from dataclasses import replace
import cv2
import numpy as np
import pytest

from vision3d.features import Features
from vision3d.geometry.camera import Camera
from vision3d.geometry.two_view import project
from vision3d.reconstruction.bundle import BundleOptions, local_bundle_adjust
from vision3d.reconstruction.incremental import (estimate_pnp, extend_landmarks, initialize,
    register, select_initial, validate_landmarks)
from vision3d.reconstruction.state import Landmark, Pose, State, Track, View, build_tracks


@pytest.fixture
def scene():
    rng = np.random.default_rng(77)
    xyz = rng.uniform([-2, -1.5, 7], [2, 1.5, 14], (100, 3))
    camera = Camera(1000, 800, 750, 750, 499.5, 399.5, distortion=(-0.03, 0.001, 0, 0, 0))
    views, poses = {}, {}
    for i, x in enumerate([0, 1, 0.5, -0.5, 1.5]):
        r = cv2.Rodrigues(np.array([0.0, 0.01 * i, 0.0]))[0]
        pose = Pose(r, -r @ np.array([x, 0, 0]))
        poses[i] = pose
        points = project(xyz, camera, r, pose.translation).astype(np.float32)
        features = Features(tuple(cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in points), points,
                            None, (1000, 800), {"detector": "sift"}, 0)
        views[i] = View(i, f"{i}.png", camera, features, np.full((len(xyz), 3), 0.5))
    tracks = {i: Track(i, {view: i for view in views}) for i in range(len(xyz))}
    return State(views, tracks), xyz, poses


def test_track_merges_conflicts_duplicates_disconnected():
    edges = [((0, 1), (1, 2)), ((1, 2), (2, 3)), ((0, 1), (2, 3)),
             ((0, 9), (2, 3)), ((3, 0), (4, 0)), ((4, 0), (4, 0))]
    tracks, metrics = build_tracks(edges)
    assert [t.observations for t in tracks.values()] == [{0: 1, 1: 2, 2: 3}, {3: 0, 4: 0}]
    assert metrics["conflicting_edges_rejected"] == 1
    assert metrics["duplicate_or_cycle_edges"] == 2
    assert metrics["length_distribution"] == {2: 1, 3: 1}
    assert all(len(t.observations) == len(set(t.observations)) for t in tracks.values())


def test_initial_selection_and_track_landmarks(scene):
    state, _, _ = scene
    state.views[2].features = replace(state.views[2].features, points=state.views[0].features.points.copy())
    pairs = {(0, 1): np.c_[np.arange(80), np.arange(80)],
             (0, 2): np.c_[np.arange(100), np.arange(100)]}
    initial, metrics = select_initial(state.views, pairs)
    initialize(state, initial)
    assert state.anchors == (0, 1) and len(state.landmarks) > 70
    assert metrics[0]["status"] == "accepted"
    assert metrics[1]["status"] == "rejected"  # greater match count, zero baseline
    assert np.allclose(state.poses[0].matrix, np.eye(4))
    assert np.linalg.norm(state.poses[1].center) == pytest.approx(1)


def test_pnp_registration_arbitrary_order_and_deduplication(scene):
    state, xyz, poses = scene
    state.poses = {0: poses[0], 1: poses[1]}
    state.anchors = (0, 1)
    state.registration_order = [0, 1]
    assert extend_landmarks(state)["new_landmarks"] == 100
    assert extend_landmarks(state)["new_landmarks"] == 0
    for image in (4, 3, 2):
        accepted, metrics = register(state, image)
        assert accepted and metrics["inliers"] == 100
        assert np.allclose(state.poses[image].center, poses[image].center, atol=1e-4)
        extend_landmarks(state)
    assert len(state.landmarks) == 100
    assert not register(state, 4)[0]
    assert len(state.registration_order) == 5
    assert all(len(l.observations) == 5 for l in state.landmarks.values())
    assert np.allclose([l.xyz for l in state.landmarks.values()], xyz, atol=1e-4)
    assert validate_landmarks(state)["invalid_observations"] == 0


def test_pnp_outliers_failure_and_pose_convention(scene):
    state, xyz, poses = scene
    pixels = state.views[4].features.points.copy()
    pixels[:20] = np.random.default_rng(3).uniform(0, 500, (20, 2))
    recovered, mask, metrics = estimate_pnp(xyz, pixels, state.views[4].camera)
    assert recovered is not None and 75 <= mask.sum() <= 82
    assert np.allclose(recovered.center, poses[4].center, atol=1e-3)
    assert np.allclose(recovered.matrix[:3, :3] @ recovered.center + recovered.translation, 0)
    assert estimate_pnp(xyz[:3], pixels[:3], state.views[4].camera)[0] is None
    random_pixels = np.random.default_rng(4).uniform(0, 800, pixels.shape)
    assert estimate_pnp(xyz, random_pixels, state.views[4].camera)[0] is None
    with pytest.raises(ValueError):
        estimate_pnp(xyz * np.nan, pixels, state.views[4].camera)


def prepared_bundle(scene):
    state, xyz, poses = scene
    state.poses = {i: poses[i] for i in (0, 1, 2)}
    state.anchors = (0, 1)
    state.registration_order = [0, 1, 2]
    rng = np.random.default_rng(5)
    state.landmarks = {i: Landmark(i, p + rng.normal(0, 0.015, 3), np.full(3, 0.5), {0: i, 1: i, 2: i}) for i, p in enumerate(xyz)}
    state.poses[2] = Pose(poses[2].rotation, poses[2].translation + [0.02, -0.01, 0.01])
    return state


def test_bundle_convergence_and_fixed_gauge(scene):
    state = prepared_bundle(scene)
    anchors = [state.poses[i].matrix.copy() for i in state.anchors]
    result = local_bundle_adjust(state, BundleOptions(max_points=100, max_evaluations=50))
    assert result["status"] == "accepted"
    assert result["after"]["rms_error_px"] < result["before"]["rms_error_px"] * 0.1
    for i, matrix in zip(state.anchors, anchors):
        assert np.array_equal(state.poses[i].matrix, matrix)


def test_bundle_rollback_preserves_state(scene, monkeypatch):
    state = prepared_bundle(scene)
    xyz = np.array([l.xyz.copy() for l in state.landmarks.values()])
    pose = state.poses[2].matrix.copy()
    def fail(*args, **kwargs):
        raise RuntimeError("injected numerical solver failure")
    monkeypatch.setattr("vision3d.reconstruction.bundle.least_squares", fail)
    result = local_bundle_adjust(state)
    assert result["status"] == "rolled_back"
    assert np.array_equal(state.poses[2].matrix, pose)
    assert np.array_equal([l.xyz for l in state.landmarks.values()], xyz)


def test_landmark_outlier_observation_pruning(scene):
    state, xyz, poses = scene
    state.poses = poses
    state.landmarks[0] = Landmark(0, xyz[0], np.full(3, 0.5), {0: 0, 1: 0, 2: 5})
    assert validate_landmarks(state)["invalid_observations"] == 1
    validate_landmarks(state, prune=True)
    assert state.landmarks[0].observations == {0: 0, 1: 0}


def test_deferred_image_registers_after_new_tracks_triangulate(scene):
    state, xyz, poses = scene
    state.poses = {0: poses[0], 1: poses[1]}
    state.anchors = (0, 1)
    state.registration_order = [0, 1]
    for i in range(30):
        state.tracks[i].observations = {0: i, 1: i, 2: i}
        state.landmarks[i] = Landmark(i, xyz[i], np.full(3, 0.5), {0: i, 1: i})
    for i in range(30, 100):
        state.tracks[i].observations = {1: i, 2: i, 3: i}
    assert not register(state, 3)[0] and 3 not in state.poses
    assert register(state, 2)[0]
    assert extend_landmarks(state)["new_landmarks"] == 70
    assert register(state, 3)[0]


def test_bundle_nonfinite_solution_rolls_back(scene, monkeypatch):
    from types import SimpleNamespace
    state = prepared_bundle(scene)
    original = state.poses[2].matrix.copy()
    def nonfinite(fun, initial, **kwargs):
        return SimpleNamespace(status=1, x=initial * np.nan, fun=np.array([np.nan]))
    monkeypatch.setattr("vision3d.reconstruction.bundle.least_squares", nonfinite)
    assert local_bundle_adjust(state)["status"] == "rolled_back"
    assert np.array_equal(state.poses[2].matrix, original)


def test_invalid_pose_convention():
    with pytest.raises(ValueError):
        Pose(np.eye(3) * 2, np.zeros(3))
    with pytest.raises(ValueError):
        Pose(np.diag([1, 1, -1]), np.zeros(3))
    with pytest.raises(ValueError):
        Pose(np.eye(3), [np.nan, 0, 0])
