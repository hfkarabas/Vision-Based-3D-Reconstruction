from types import SimpleNamespace

import numpy as np

from vision3d.reconstruction import depth_recovery


def test_recovery_requires_two_independent_witnesses_and_preserves_original_colors(monkeypatch):
    def depth(ref, source, bits):
        return SimpleNamespace(reference=ref, source=source, witness_bits=np.asarray(bits, np.uint64),
                               xyz=np.array([[0, 0, 2.], [0, 0, 2.]]), focal_baseline=80.,
                               pixels=np.array([[0, 0], [0, 1]]), depth=np.full((1, 2), 2.),
                               colors=np.ones((2, 3)), recentered=ref == 0)

    sample = depth(0, 1, [0, 6])
    primary = [sample, depth(2, 1, [0, 0]), depth(4, 1, [0, 0])]
    mask = np.array([False, True])
    accepted = [(sample, mask, np.zeros((1, 3)))]
    accepted.extend((d, np.ones(2, bool), np.zeros((2, 3))) for d in primary[1:])
    centers = {i: np.array([.2 * i, 0, 0]) for i in range(7)}
    neighbors = {2: [(1, 3, None), (1, 6, None)], 4: [(1, 5, None)]}

    def vote(sample, witness, tolerance, return_px, *, sample_rows, witness_max_depth=None):
        return sample_rows, None, None

    monkeypatch.setattr(depth_recovery, "round_trip_depth", vote)
    args = (primary, accepted, centers, neighbors,
            lambda i, j, xyz: (depth(i, j, [0, 0]), None))
    result, limited, _ = depth_recovery.recover_depth(*args, 2, .02, 1.5, 28, 2, lambda _: None)
    assert limited["recovered_pixels"] == 0
    assert result[0][1].tolist() == [False, True]
    result, report, provenance = depth_recovery.recover_depth(*args, 3, .02, 1.5, 28, 2, lambda _: None)
    assert report["recovered_pixels"] == 1
    assert [a["new_votes"] for a in report["attempts"]] == [1, 0, 1]
    assert result[0][1].all()
    assert np.array_equal(result[0][2], [[1, 1, 1], [0, 0, 0]])
    assert [(ref, src, rows.tolist()) for ref, src, rows in provenance[0]] == [(2, 3, [0]), (4, 5, [0])]
