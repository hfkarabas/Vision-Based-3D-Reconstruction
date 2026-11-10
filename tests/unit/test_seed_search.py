import json

import numpy as np
import pytest
from PIL import Image

from vision3d.io import load_image
from vision3d.reconstruction.seed_search import load_position_hints, shortlist, station_labels


def test_position_hints_keep_missing_views_unlabelled_and_cluster_revisits(tmp_path):
    path = tmp_path / "hints.json"
    path.write_text(json.dumps({"a.jpg": [0, 0, 0], "b.jpg": [40, 0, 0], "c.jpg": [0, 2, 20]}))
    data = load_position_hints(path)
    assert station_labels({0: data["a.jpg"], 2: data["b.jpg"], 4: data["c.jpg"]}, 10) == {0: 0, 4: 0, 2: 1}
    for invalid in ([], {"a.jpg": [0, 2]}, {"a.jpg": [True, 1, 1]}, {"a.jpg": [0, 1, float("inf")]}):
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            load_position_hints(path)


def test_shortlist_reserves_slots_for_distinct_station_bridges():
    ranked = [(20, 0, 1), (19, 0, 2), (18, 1, 2), (16, 0, 3), (15, 2, 3)]
    labels = {0: 0, 1: 0, 2: 1, 3: 2}
    assert shortlist(ranked[0], ranked, labels, 4) == [ranked[i] for i in (0, 1, 3, 4)]
    assert shortlist(ranked[0], ranked, {}, 2) == ranked[:2]


def test_mpo_loads_primary_jpeg_and_exif_without_changing_source(tmp_path):
    path = tmp_path / "DJI.JPG"
    primary = Image.new("RGB", (80, 64), (200, 25, 10))
    secondary = Image.new("RGB", (16, 16), (10, 25, 200))
    exif = Image.Exif()
    exif[306] = "2025:10:16 12:00:00"
    primary.save(path, format="MPO", save_all=True, append_images=[secondary], exif=exif)
    before = path.read_bytes()
    frame = load_image(path)
    assert frame.image.shape == (64, 80, 3)
    assert np.allclose(frame.image[0, 0], [10, 25, 200], atol=3)
    assert frame.metadata["timestamp"] == "2025:10:16 12:00:00"
    assert path.read_bytes() == before
