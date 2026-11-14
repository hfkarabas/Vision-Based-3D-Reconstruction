import numpy as np

from uav3d.evaluate import tiles


def test_tile_index_and_outside():
    xy = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [-0.1, 0.0], [2.0, 0.0]])
    ids = tiles.tile_index(xy, (0.0, 0.0), 1.0, (2, 2))
    assert ids.tolist() == [0, 1, 2, -1, -1]


def test_score_tiles_perfect_and_missing_half():
    g = np.stack(np.meshgrid(np.arange(0, 20, 0.5), np.arange(0, 10, 0.5)), -1).reshape(-1, 2)
    ref = np.c_[g, np.zeros(len(g))]
    ours = ref[ref[:, 0] < 15]  # tile 0 (x<10) complete, tile 1 (10..20) half missing
    rows = tiles.score_tiles(ours, ref, (0.0, 0.0), 10.0, (1, 2), taus=(0.1,), per_tile_sample=10_000)
    assert rows[0]["tau_10cm"] == {"precision": 1.0, "recall": 1.0, "f": 1.0}
    assert rows[1]["tau_10cm"]["precision"] == 1.0
    assert abs(rows[1]["tau_10cm"]["recall"] - 0.5) < 0.03
    assert rows[1]["cover_ours"] < rows[1]["cover_ref"]


def test_score_tiles_offset_lowers_precision():
    g = np.stack(np.meshgrid(np.arange(0, 10, 0.5), np.arange(0, 10, 0.5)), -1).reshape(-1, 2)
    ref = np.c_[g, np.zeros(len(g))]
    ours = ref + [0, 0, 0.3]
    rows = tiles.score_tiles(ours, ref, (0.0, 0.0), 10.0, (1, 1), taus=(0.1, 0.5))
    assert rows[0]["tau_10cm"]["f"] == 0.0 and rows[0]["tau_50cm"]["f"] == 1.0
