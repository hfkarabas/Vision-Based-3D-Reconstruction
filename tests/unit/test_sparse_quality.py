"""Fixed-camera refinement correctness, rollback and viewer-only geometry."""
import copy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision3d.demo.view_geometry import camera_view, camera_frustum
from vision3d.evaluation.sparse import quality_mask, percentile
from vision3d.features import Features
from vision3d.geometry.camera import Camera
from vision3d.geometry.two_view import project
from vision3d.reconstruction.state import State, Pose, View, Landmark, Track
from vision3d.reconstruction.refinement import refine_landmarks


def noisy_scene():
    rng = np.random.default_rng(32)
    xyz = rng.uniform([-1, -.5, 5], [1, .5, 9], size=(30, 3))
    camera = Camera(800, 600, 700, 700, 400, 300, distortion=(-.01, .001, 0, 0, 0))
    views, poses = {}, {}
    for i in range(4):
        rotation = cv2.Rodrigues(np.array([0, .01 * i, 0]))[0]
        pose = Pose(rotation, -rotation @ np.array([i * .5, 0, 0])); poses[i] = pose
        pixels = project(xyz, camera, rotation, pose.translation) + rng.normal(0, .05, (len(xyz), 2))
        features = Features((), pixels.astype(np.float32), None, (800, 600), {}, 0)
        views[i] = View(i, str(i), camera, features, np.full((len(xyz), 3), .5))
    tracks = {i: Track(i, {j: i for j in views}) for i in range(len(xyz))}
    landmarks = {i: Landmark(i, p + rng.normal(0, .003, 3), np.full(3, .5), dict(tracks[i].observations)) for i, p in enumerate(xyz)}
    return State(views, tracks, poses=poses, landmarks=landmarks, anchors=(0, 1)), xyz


def test_refinement_reduces_geometric_noise_and_keeps_poses_tracks():
    state, truth = noisy_scene()
    poses = {i: p.matrix.copy() for i, p in state.poses.items()}
    before = np.array([l.xyz.copy() for l in state.landmarks.values()])
    obs = copy.deepcopy({i: l.observations for i, l in state.landmarks.items()})
    result = refine_landmarks(state)
    after = np.array([l.xyz for l in state.landmarks.values()])
    assert result['accepted_points'] >= 25
    assert result['same_observation_rms_px']['after'] < result['same_observation_rms_px']['before']
    assert np.linalg.norm(after - truth) < np.linalg.norm(before - truth)
    assert {i: l.observations for i, l in state.landmarks.items()} == obs
    for i, pose in state.poses.items(): assert np.array_equal(pose.matrix, poses[i])
    assert all(np.array_equal(l.color, [.5, .5, .5]) for l in state.landmarks.values())


def test_two_view_points_unchanged():
    state, _ = noisy_scene()
    for l in state.landmarks.values(): l.observations = {0: l.id, 1: l.id}
    original = np.array([l.xyz.copy() for l in state.landmarks.values()])
    result = refine_landmarks(state)
    assert result['accepted_points'] == 0
    assert result['unchanged_reasons']['two_view_unchanged'] == 30
    assert np.array_equal(original, [l.xyz for l in state.landmarks.values()])


@pytest.mark.parametrize('failure', ['exception', 'nan', 'extreme'])
def test_refinement_failed_proposals_are_rolled_back(monkeypatch, failure):
    state, _ = noisy_scene()
    original = np.array([l.xyz.copy() for l in state.landmarks.values()])
    def fake(fun, initial, **kwargs):
        if failure == 'exception': raise RuntimeError('solver failed')
        x = np.full(3, np.nan) if failure == 'nan' else initial + [100, 0, 100]
        return SimpleNamespace(x=x, fun=np.zeros(8), status=1)
    monkeypatch.setattr('vision3d.reconstruction.refinement.least_squares', fake)
    result = refine_landmarks(state)
    assert result['accepted_points'] == 0
    assert np.array_equal(original, [l.xyz for l in state.landmarks.values()])


def test_refinement_nonfinite_initial_and_bad_limits():
    state, _ = noisy_scene(); state.landmarks[0].xyz[:] = np.nan
    result = refine_landmarks(state)
    assert result['unchanged_reasons']['nonfinite_initial'] == 1
    with pytest.raises(ValueError): refine_landmarks(state, max_evaluations=0)


def test_camera_aligned_view_preserves_points_and_conventions():
    points = np.array([[-1., -1, 5], [1, 1, 5]])
    original = points.copy()
    view = camera_view(points, [np.eye(4)])
    assert np.allclose(view['lookat'], [0, 0, 5])
    assert np.allclose(view['front'], [0, 0, -1])
    assert np.allclose(view['up'], [0, -1, 0])
    assert abs(view['front'] @ view['up']) < 1e-9
    assert np.array_equal(original, points)
    with pytest.raises(ValueError): camera_view(points, [])
    with pytest.raises(ValueError): camera_view(points, [np.eye(3)])
    with pytest.raises(ValueError): camera_view([[0, 0, np.nan]], [np.eye(4)])


def test_true_camera_frustum_center_and_projection():
    rotation = cv2.Rodrigues(np.array([0., .2, 0]))[0]
    pose = Pose(rotation, -rotation @ np.array([2., 1, 0]))
    k = np.array([[600., 0, 400], [0, 600, 300], [0, 0, 1]])
    vertices, lines = camera_frustum(pose.matrix, k, 800, 600, .5)
    assert np.allclose(vertices[0], pose.center)
    local = vertices[1:] @ rotation.T + pose.translation
    uv = local @ k.T; uv = uv[:, :2] / uv[:, 2:]
    assert np.allclose(uv, [[0, 0], [799, 0], [799, 599], [0, 599]])
    assert lines.shape == (8, 2)
    with pytest.raises(ValueError): camera_frustum(pose.matrix, k, 0, 600, .5)


def test_quality_subset_is_explicit_and_does_not_modify_geometry():
    a = {'xyz': [0, 0, 5], 'observations': {'0': 1, '1': 2, '2': 3},
         'reprojection_errors_px': {'0': .2, '1': .3, '2': .1},
         'depths': {'0': 5, '1': 5, '2': 5}, 'max_triangulation_angle_degrees': 10}
    b = copy.deepcopy(a); b['observations'].pop('2')
    c = copy.deepcopy(a); c['depths']['0'] = -1
    points = [a, b, c]; before = copy.deepcopy(points)
    assert quality_mask(points).tolist() == [True, False, False]
    assert points == before
    assert percentile([]) == {}
    assert percentile([1, 2, 3])['50'] == 2


@pytest.mark.parametrize('flag,expected', [([], False), (['--refine-points'], True)])
def test_cli_point_refinement_is_explicit(monkeypatch, flag, expected):
    from vision3d.cli import main
    seen = {}
    def reconstruct(**kwargs):
        seen.update(kwargs)
        return {'registered_images': 6, 'valid_landmarks': 100, 'status': 'success'}
    monkeypatch.setattr('vision3d.reconstruction.sequence.reconstruct_sequence', reconstruct)
    main(['reconstruct-sequence', '--input', 'images', '--camera', 'camera.json', '--output', 'new', *flag])
    assert seen['refine_points'] is expected
    assert seen['recovery_pairs'] == 0 and seen['joint_ba'] is False
