import numpy as np
from vision3d.reconstruction.graph import components, discovery_candidate
from vision3d.reconstruction.sequence import match_sequence
from vision3d.reconstruction.state import View
from vision3d.features import extract
from vision3d.geometry.camera import Camera
from vision3d.geometry.two_view import Quality


def test_discovery_prioritizes_disconnected_components_and_respects_caps():
    views = dict.fromkeys(range(4))
    signatures = {0: np.array([1., 0]), 1: np.array([1., 0]), 2: np.array([0.99, 0.01]), 3: np.array([0., 1.])}
    pairs = {(0, 1): None, (2, 3): None}
    assert components(views, pairs) == [[0, 1], [2, 3]]
    candidate = discovery_candidate(views, pairs, set(pairs), {}, signatures)
    assert candidate[:2] == (0, 2)
    assert discovery_candidate(views, pairs, set(pairs), {0: 3, 1: 3}, signatures) is None
    pairs[(0, 2)] = None
    assert components(views, pairs) == [[0, 1, 2, 3]]


def test_descriptor_similarity_never_creates_unverified_tracks(monkeypatch):
    image = np.random.default_rng(3).integers(0, 256, (240, 320, 3), np.uint8)
    features = extract(image, 'sift', 200)
    camera = Camera(320, 240, 250, 250, 159.5, 119.5)
    views = {i: View(i, str(i), camera, features, np.zeros((features.count, 3))) for i in range(5)}
    monkeypatch.setattr('vision3d.reconstruction.sequence.verify',
        lambda a, b, **kw: (None, np.zeros(len(a), bool), 'estimation_failed'))
    pairs, tracks, metrics, statistics = match_sequence(views, 1, 0, Quality(), 0.75, recovery_pairs=3)
    assert not pairs and not tracks
    assert len(metrics) == 7  # four scheduled plus bounded three discovery attempts
    assert statistics['graph']['components_after'] == [[0], [1], [2], [3], [4]]
    assert statistics['graph']['discovery_attempts'] == 3
