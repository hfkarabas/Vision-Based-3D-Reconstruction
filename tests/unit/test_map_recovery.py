from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from vision3d.geometry.camera import Camera
from vision3d.geometry.two_view import Quality
from vision3d.reconstruction.incremental import pnp_correspondences
from vision3d.reconstruction.map_recovery import MapRecovery, frontier_pairs, station_balanced_pairs
from vision3d.reconstruction.state import Landmark, Pose, State, Track, View


@pytest.fixture
def state():
    camera = Camera(100, 100, 100, 100, 50, 50)
    features = SimpleNamespace(points=np.array([[50, 50], [90, 90], [60, 50]], dtype=float),
                               descriptors=np.ones((3, 2)), config={"detector": "sift"})
    views = {i: View(i, str(i), camera, features, np.zeros((3, 3))) for i in range(5)}
    poses = {i: Pose(np.eye(3), np.zeros(3)) for i in range(2)}
    tracks = {1: Track(1, {0: 0, 1: 0, 2: 0}), 2: Track(2, {0: 2, 1: 2}),
              3: Track(3, {3: 0, 4: 0}), 4: Track(4, {1: 1, 3: 1})}
    landmarks = {1: Landmark(1, np.array([0., 0., 5.]), np.zeros(3), {0: 0, 1: 0}),
                 2: Landmark(2, np.array([.5, 0., 5.]), np.zeros(3), {0: 2, 1: 2})}
    return State(views, tracks, poses=poses, landmarks=landmarks)


def test_frontier_prefers_existing_map_support_and_skips_tested_pairs(state):
    pending = {2, 3, 4}
    support = {i: set(pnp_correspondences(state, i)[0]) for i in pending}
    signatures = {i: np.array([1., 0.]) for i in range(5)}
    ranked = frontier_pairs(state, pending, {(0, 2)}, {(0, 2): None}, {i: [] for i in pending},
                            20, signatures, support)
    assert ranked[0] == (2, 1, (1, 2))
    assert all(pair != (0, 2) for _, _, pair in ranked)
    assert [candidate for candidate, _, _ in ranked[1:]] == [3, 3, 4, 4]


def test_merge_preserves_mapped_ids_and_rejects_conflicts(state):
    recovery = MapRecovery(state, [], 4, 2)
    original = state.landmarks[2]
    counts, aliases = recovery.integrate(state, [
        (1., (0, 2), (4, 0)),  # Absorb an unmapped track into landmark 2.
        (2., (0, 2), (3, 1)),  # Different feature in the same image.
    ], 2.)
    assert counts == {"integrated_edges": 1, "image_conflict": 1}
    assert aliases == {3: 2}
    assert state.landmarks[2] is original
    assert state.tracks[2].observations == {0: 2, 1: 2, 3: 0, 4: 0}
    assert original.observations == {0: 2, 1: 2}
    assert pnp_correspondences(state, 3)[0] == [2]


def test_recovery_never_merges_two_existing_landmarks(state):
    state.landmarks[3] = Landmark(3, np.array([0., 0., 5.]), np.zeros(3), {3: 0, 4: 0})
    counts, aliases = MapRecovery(state, [], 2, 1).integrate(state, [(1., (0, 2), (3, 0))], 2.)
    assert counts == {"two_landmarks": 1}
    assert not aliases and set(state.landmarks) == {1, 2, 3}


def test_merge_rejects_inconsistent_registered_observation(state):
    state.tracks[2].observations.pop(1)
    state.landmarks[2].observations.pop(1)
    counts, aliases = MapRecovery(state, [], 2, 1).integrate(state, [(1., (0, 2), (3, 1))], 2.)
    assert counts == {"mapped_reprojection_rejected": 1}
    assert not aliases
    assert state.tracks[2].observations == {0: 2}
    assert state.tracks[4].observations == {1: 1, 3: 1}


@pytest.mark.parametrize("budget, rounds, tried, reason", [
    (3, 2, 3, "pair_budget"), (20, 3, 5, "no_untested_map_pairs"),
    (20, 1, 5, "round_budget"), (0, 2, 0, "pair_budget"),
])
def test_recovery_bounds_and_no_repeated_pair(state, budget, rounds, tried, reason):
    calls = []
    pending = {2, 3, 4}
    rows = [{"pair": [0, 2]}]
    recovery = MapRecovery(state, rows, budget, rounds)
    original = deepcopy(state.tracks)

    def evaluate(i, j, schedule):
        calls.append((i, j))
        assert schedule == "map_recovery"
        return {"pair": [i, j], "edge_status": "below_minimum_inliers"}, None, []

    for _ in range(rounds + 1):
        changed = recovery.run_round(state, pending, {}, rows, {i: [] for i in pending}, evaluate, Quality(), 20)
        if changed is None:
            break
        assert not changed
    assert recovery.stop_reason == reason
    assert recovery.pair_attempts == len(calls) == len(set(calls)) == tried
    assert (0, 2) not in calls and state.tracks == original


def test_recovery_retries_only_cameras_with_new_map_support(state):
    pending = {2, 3, 4}
    rows = [{"pair": [0, 2]}]
    recovery = MapRecovery(state, rows, 1, 1)

    def evaluate(i, j, schedule):
        assert (i, j) == (1, 2)
        return {"pair": [i, j], "edge_status": "verified"}, np.array([[2, 2]]), [(1., (1, 2), (2, 2))]

    changed = recovery.run_round(state, pending, {}, rows, {i: [] for i in pending}, evaluate, Quality(), 20)
    assert changed == {2}
    assert pnp_correspondences(state, 2)[0] == [1, 2]
    assert recovery.rounds[0]["frontier"][0]["before"] == 1
    assert recovery.rounds[0]["frontier"][0]["after"] == 2


def test_station_reservation_spreads_pairs_and_preserves_budget():
    ranked = [(i, j, tuple(sorted((i, j)))) for i in (2, 3, 4, 5) for j in (0, 1)]
    labels = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
    result = station_balanced_pairs(ranked, 5, {0, 1}, labels)
    assert [r[0] for r in result] == [3, 5, 3, 2, 2]
    assert len(result) == len(set(result)) == 5
    assert station_balanced_pairs(ranked, 5, {0, 3, 5}, labels) == ranked[:5]
