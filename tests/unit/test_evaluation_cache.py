import cv2
import numpy as np
import pytest

from vision3d.evaluation.poses import evaluate_poses, similarity_alignment
from vision3d.features.cache import cached_extract


@pytest.mark.parametrize("detector,dtype", [("sift", np.float32), ("orb", np.uint8)])
def test_numeric_cache_reproducibility_and_invalidation(tmp_path, detector, dtype):
    image = np.random.default_rng(9).integers(0, 256, (240, 320, 3), np.uint8)
    first, hit = cached_extract(image, "source1", 320, detector, 200, tmp_path)
    assert not hit
    second, hit = cached_extract(image, "source1", 320, detector, 200, tmp_path)
    assert hit and second.seconds == 0
    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.descriptors, second.descriptors) and second.descriptors.dtype == dtype
    assert not cached_extract(image, "source2", 320, detector, 200, tmp_path)[1]
    assert not cached_extract(image, "source1", 320, detector, 100, tmp_path)[1]


def test_corrupt_cache_recomputed(tmp_path):
    image = np.random.default_rng(2).integers(0, 256, (100, 100, 3), np.uint8)
    first, _ = cached_extract(image, "image", 100, "sift", 100, tmp_path)
    next(tmp_path.glob("*.npz")).write_bytes(b"bad cache")
    with pytest.warns(UserWarning):
        second, hit = cached_extract(image, "image", 100, "sift", 100, tmp_path)
    assert not hit and np.array_equal(first.points, second.points)


def test_similarity_pose_evaluation_rotation_translation_and_scale():
    centers = np.random.default_rng(5).normal(size=(10, 3))
    q = cv2.Rodrigues(np.array([0.1, 0.2, -0.15]))[0]
    scale, shift = 3.2, np.array([4, -2, 7])
    truth_centers = scale * centers @ q.T + shift
    s, r, t = similarity_alignment(centers, truth_centers)
    assert s == pytest.approx(scale) and np.allclose(r, q) and np.allclose(t, shift)
    estimated, reference = [], {}
    for i, (c, target) in enumerate(zip(centers, truth_centers)):
        re = cv2.Rodrigues(np.array([0.01 * i, 0, 0]))[0]
        rt = re @ q.T
        e = np.r_[np.c_[re, -re @ c], [[0, 0, 0, 1]]]
        ref = np.r_[np.c_[rt, -rt @ target], [[0, 0, 0, 1]]]
        estimated.append({"image_id": i, "name": str(i), "world_to_camera": e.tolist()})
        reference[str(i)] = ref.tolist()
    result = evaluate_poses(estimated, reference)
    assert result["center_rmse_reference_units"] < 1e-10
    assert result["median_aligned_rotation_degrees"] < 1e-5
    assert result["median_translation_direction_degrees"] < 1e-5
    assert not result["geometric_accuracy_warning"]
    reference["5"][0][3] += 1
    assert evaluate_poses(estimated, reference)["center_rmse_reference_units"] > 0.01


def test_alignment_degenerate_and_nonfinite():
    line = np.c_[np.arange(5), np.zeros((5, 2))]
    with pytest.raises(ValueError, match="collinear"):
        similarity_alignment(line, line)
    with pytest.raises(ValueError):
        similarity_alignment(line * np.nan, line)
