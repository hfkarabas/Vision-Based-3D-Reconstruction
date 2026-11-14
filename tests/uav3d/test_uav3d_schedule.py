import numpy as np

from uav3d import schedule


def test_ranks_are_scale_free_and_keep_nan():
    r = schedule.ranks([10.0, np.nan, 30.0, 20.0])
    assert np.allclose(r[[0, 2, 3]], [0.0, 1.0, 0.5]) and np.isnan(r[1])
    assert np.allclose(schedule.ranks([1, 2, 3]), schedule.ranks([100, 2000, 30000]))


def test_fit_recovers_orientation_and_drops_useless_signal():
    rng = np.random.default_rng(0)
    n = 200
    overlap = rng.integers(2, 14, n).astype(float)
    rough = rng.random(n)
    noise = rng.random(n)
    gain = 0.5 * (1 - schedule.ranks(overlap)) + 0.3 * schedule.ranks(rough) + 0.01 * rng.standard_normal(n)
    sig = [{"overlap": o, "roughness": r, "noise": z} for o, r, z in zip(overlap, rough, noise)]
    m = schedule.fit(sig, gain, ["overlap", "roughness", "noise"])
    assert m.orientation["overlap"] == -1 and m.orientation["roughness"] == 1
    assert m.weights["overlap"] > m.weights["roughness"] > 5 * m.weights["noise"]
    assert np.corrcoef(m.predict(sig), gain)[0, 1] > 0.95
    again = schedule.GainModel.from_dict(m.to_dict())
    assert np.allclose(again.predict(sig), m.predict(sig))


def test_select_respects_budget_and_prefers_shared_images():
    shots = [{1, 2, 3}, {1, 2, 3}, {7, 8, 9, 10, 11}, {4}]
    scores = [1.0, 0.9, 1.5, 0.1]
    sel = schedule.select(scores, shots, budget_images=4)
    # tile 0 (3 images), then tile 1 is free; tile 2 would exceed the budget; tile 3 fits
    assert sel.order == [0, 1, 3] and sel.images[-1] == 4
    assert all(i <= 4 for i in sel.images)


def test_select_in_given_order_without_cost_awareness():
    shots = [{1, 2}, {3, 4, 5}, {6}]
    sel = schedule.select([0, 0, 0], shots, budget_images=4, cost_aware=False, order=[1, 0, 2])
    assert sel.order == [1, 2]


def test_select_skips_tiles_without_images_or_gain():
    sel = schedule.select([1.0, -1.0, 2.0], [set(), {1}, {2}], budget_images=10)
    assert sel.order == [2]
