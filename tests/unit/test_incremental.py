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


def test_bundle_uses_fixed_covisible_boundary_after_leaving_anchors(scene):
    state, xyz, poses = scene
    state.poses = dict(poses)
    state.anchors = (0, 1)
    state.registration_order = [0, 1, 2, 3, 4]
    rng = np.random.default_rng(22)
    state.landmarks = {i: Landmark(i, x + rng.normal(0, 0.01, 3), np.full(3, 0.5), {2: i, 3: i, 4: i}) for i, x in enumerate(xyz)}
    state.poses[4] = Pose(poses[4].rotation, poses[4].translation + [0.02, 0.01, 0])
    fixed_before = {i: state.poses[i].matrix.copy() for i in (0, 1, 2, 3)}
    result = local_bundle_adjust(state, BundleOptions(max_cameras=3, max_points=100, max_evaluations=50))
    assert result["status"] == "accepted" and set(result["fixed_cameras"]) == {2, 3}
    assert result["local_same_observation_rms_px"]["after"] < result["local_same_observation_rms_px"]["before"]
    for i, value in fixed_before.items():
        assert np.array_equal(state.poses[i].matrix, value)


def test_bundle_checks_invalid_observation_identity_not_only_count(scene, monkeypatch):
    state = prepared_bundle(scene)
    before = state.poses[2].matrix.copy()
    outcomes = iter([{(0, 0, 0)}, {(1, 1, 1)}])
    monkeypatch.setattr("vision3d.reconstruction.bundle.invalid_observations", lambda *args: next(outcomes))
    result = local_bundle_adjust(state, BundleOptions(max_points=100, max_evaluations=50))
    assert result["status"] == "rolled_back" and result["new_invalid_observations"] == 1
    assert np.array_equal(state.poses[2].matrix, before)


def test_bundle_pruning_failure_is_atomic(scene, monkeypatch):
    import vision3d.reconstruction.bundle as bundle
    state = prepared_bundle(scene)
    before = state.poses[2].matrix.copy()
    xyz = np.array([l.xyz.copy() for l in state.landmarks.values()])
    original = bundle.validate_landmarks
    def prune_failure(*args, **kwargs):
        if kwargs.get("prune"):
            raise RuntimeError("injected pruning failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(bundle, "validate_landmarks", prune_failure)
    assert local_bundle_adjust(state, BundleOptions(max_points=100, max_evaluations=50))["status"] == "rolled_back"
    assert np.array_equal(state.poses[2].matrix, before)
    assert np.array_equal([l.xyz for l in state.landmarks.values()], xyz)


def test_pnp_rejects_clustered_support(scene):
    state, _, poses = scene
    xyz = np.random.default_rng(9).uniform([-0.02, -0.02, 10], [0.02, 0.02, 11], (100, 3))
    pose = poses[4]
    pixels = project(xyz, state.views[4].camera, pose.rotation, pose.translation)
    recovered, _, metrics = estimate_pnp(xyz, pixels, state.views[4].camera)
    assert recovered is None and metrics["reason"] == "pnp_spatial_support_too_small"


def test_duplicate_2d_to_multiple_landmark_associations_are_excluded(scene):
    from vision3d.reconstruction.incremental import pnp_correspondences
    state, xyz, _ = scene
    state.landmarks = {i: Landmark(i, xyz[i], np.full(3, 0.5), {0: i, 1: i}) for i in range(30)}
    state.tracks[0].observations[4] = 10
    state.tracks[1].observations[4] = 10
    ids, _, pixels = pnp_correspondences(state, 4)
    assert 0 not in ids and 1 not in ids and 10 not in ids  # all three share feature 10
    assert len(pixels) == 27


def test_initial_selection_accounts_for_track_conflicts(scene):
    state, _, _ = scene
    tracks = {i: Track(i, {0: i, 2: i}) for i in range(100)}
    pairs = {(0, 1): np.c_[np.arange(100), np.arange(100)], (0, 2): np.c_[np.arange(100), np.arange(100)]}
    initial, metrics = select_initial(state.views, pairs, tracks=tracks)
    assert initial[1:3] == (0, 2)
    assert metrics[0]["reason"] == "too_few_conflict_free_tracks"


def test_alternative_triangulation_pair_rejects_contaminating_observation(scene):
    state, xyz, poses = scene
    state.poses = {i: poses[i] for i in range(4)}
    state.tracks = {0: Track(0, {i: 0 for i in range(4)})}
    state.views[3].features.points[0] += [0, 15]
    metrics = extend_landmarks(state)
    assert metrics["new_landmarks"] == 1
    assert 3 not in state.landmarks[0].observations
    assert np.allclose(state.landmarks[0].xyz, xyz[0], atol=1e-4)


def test_fifteen_camera_execution_and_final_geometry(scene):
    state, xyz, poses = scene
    for i in range(5, 15):
        rotation = cv2.Rodrigues(np.array([0, 0.01 * i, 0]))[0]
        center = np.array([0.1 * i, 0.15 * np.sin(i / 3), 0])
        pose = Pose(rotation, -rotation @ center)
        points = project(xyz, state.views[0].camera, rotation, pose.translation).astype(np.float32)
        state.views[i] = replace(state.views[0], id=i, name=f"{i}.png", features=replace(state.views[0].features, points=points))
        poses[i] = pose
        for tid, track in state.tracks.items():
            track.observations[i] = tid
    state.poses = {0: poses[0], 1: poses[1]}
    state.anchors = (0, 1)
    state.registration_order = [0, 1]
    extend_landmarks(state)
    for i in range(14, 1, -1):
        assert register(state, i)[0]
        extend_landmarks(state)
    metrics = validate_landmarks(state)
    assert len(state.poses) == 15 and len(state.landmarks) == 100
    assert metrics["observations"] == 1500 and metrics["invalid_observations"] == 0
    assert metrics["rms_error_px"] < 1e-3


def test_bundle_nonfinite_initial_state_is_rejected_without_mutation(scene):
    state = prepared_bundle(scene)
    state.landmarks[0].xyz[0] = np.nan
    before = state.poses[2].matrix.copy()
    result = local_bundle_adjust(state)
    assert result["status"] == "rolled_back" and result["reason"] == "nonfinite_initial_state"
    assert np.array_equal(before, state.poses[2].matrix)


def test_bounded_joint_bundle_preserves_gauge_and_observation_set(scene):
    state = prepared_bundle(scene)
    anchors = {i: state.poses[i].matrix.copy() for i in state.anchors}
    observations = {i: dict(l.observations) for i, l in state.landmarks.items()}
    result = local_bundle_adjust(state, BundleOptions(max_evaluations=50), joint=True)
    assert result['status'] == 'accepted' and result['scope'] == 'bounded_joint'
    assert result['after']['rms_error_px'] < result['before']['rms_error_px']
    assert result['new_invalid_observations'] == 0
    assert {i: l.observations for i, l in state.landmarks.items()} == observations
    for i, pose in anchors.items():
        assert np.array_equal(pose, state.poses[i].matrix)


def test_joint_bundle_camera_limit_is_explicit(scene):
    state, _, poses = scene
    state.poses = {i: poses[0] for i in range(31)}
    assert local_bundle_adjust(state, joint=True)['reason'] == 'joint_camera_limit_30'


def test_joint_bundle_requires_actual_anchor_observations(scene):
    state, xyz, poses = scene
    state.poses = poses
    state.anchors = (0, 1)
    state.registration_order = [0, 1, 2, 3, 4]
    state.landmarks = {i: Landmark(i, p, np.full(3, .5), {2:i, 3:i, 4:i}) for i,p in enumerate(xyz)}
    assert local_bundle_adjust(state, joint=True)['reason'] == 'joint_window_not_anchored'



def test_seed_probe_accepts_a_third_pose_without_mutating_live_state(scene):
    from vision3d.reconstruction.incremental import RegistrationQuality

    state, _, _ = scene
    pairs = {(0, 1): np.c_[np.arange(100), np.arange(100)]}
    initial, diagnostics = select_initial(state.views, pairs, tracks=state.tracks,
                                          continuation_min=20, probe_quality=RegistrationQuality())
    assert initial[1:3] == (0, 1)
    assert diagnostics[0]["best_third_view_support"] > 80
    assert diagnostics[0]["continuation_probes"][-1]["reason"] == "accepted"
    assert not state.poses and not state.landmarks
