import json
import cv2
import numpy as np
from PIL import Image
import pytest
from vision3d.cli import analyze, main
from vision3d.io import iter_frames


@pytest.mark.parametrize("detector", ["sift", "orb"])
def test_pipeline(tmp_path, detector):
    source = tmp_path / "images"
    source.mkdir()
    image = np.random.default_rng(4).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    Image.fromarray(image).save(source / "a.png")
    Image.fromarray(np.roll(image, 8, axis=1)).save(source / "b.png")
    output = tmp_path / "result"
    result = analyze(source, output, detector=detector)
    assert result["images_processed"] == 2
    assert result["pairs"][0]["ratio_matches"] > 10
    assert result["pairs"][0]["geometry_status"] == "estimated"
    assert result["pairs"][0]["ransac_inliers"] >= 8
    assert json.loads((output / "manifest.json").read_text())["config"]["detector"] == detector
    assert (output / "summary.csv").exists()
    assert len(list((output / "matches").glob("*.jpg"))) == 3
    assert not list((output / "frames").iterdir())


def test_video(tmp_path):
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened(), "MP4 encoder required for integration validation"
    try:
        for i in range(8):
            writer.write(np.full((120, 160, 3), i * 20, np.uint8))
    finally:
        writer.release()
    frames = list(iter_frames(path, interval=2, max_frames=3))
    assert [f.metadata["frame_index"] for f in frames] == [0, 2, 4]
    assert frames[-1].metadata["timestamp_seconds"] == pytest.approx(0.4)
    main(["analyze", "--input", str(path), "--output", str(tmp_path / "video_out"),
          "--interval", "2", "--max-frames", "3"])
    assert len(list((tmp_path / "video_out" / "frames").glob("*.jpg"))) == 3


def test_unreadable_video(tmp_path):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="Cannot open video"):
        list(iter_frames(path))
