import numpy as np
from PIL import Image
import pytest

from scripts.validate_blender_assets import compare_render, replay_passed


def test_render_regression_detects_color_and_silhouette_changes(tmp_path):
    pixels = np.zeros((20, 20, 4), dtype=np.uint8)
    pixels[4:16, 4:16] = [40, 100, 60, 255]
    reference, comparison = tmp_path / "reference.png", tmp_path / "comparison.png"
    Image.fromarray(pixels).save(reference)
    same = compare_render(reference, reference)
    assert replay_passed(same) and same["pixel_identical"]
    pixels[5:10, 5:10, :3] = [95, 30, 80]
    Image.fromarray(pixels).save(comparison)
    color = compare_render(reference, comparison)
    assert color["silhouette_iou"] == 1
    assert not replay_passed(color)
    pixels[4:6, 4:16, 3] = 0
    Image.fromarray(pixels).save(comparison)
    assert compare_render(reference, comparison)["silhouette_iou"] < .995
    pixels[:, :, 3] = 0
    Image.fromarray(pixels).save(comparison)
    with pytest.raises(ValueError, match="empty"):
        compare_render(reference, comparison)
