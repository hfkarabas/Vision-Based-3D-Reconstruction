"""Encode the six local Castle photos as a development MP4 montage."""
import argparse
from pathlib import Path

import cv2

from vision3d.geometry.camera import load_cameras
from vision3d.io import load_image
from vision3d.reconstruction.pair import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    files = sorted(Path("datasets/sceaux/images").glob("*.JPG"))
    if len(files) != 6:
        parser.error("Expected the six local Castle JPG photographs")
    camera, _ = load_cameras("configs/sceaux.json")
    size = (1600, 1202)
    args.output.mkdir(parents=True, exist_ok=False)
    writer = cv2.VideoWriter(str(args.output / "castle-photo-montage.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 12, size)
    try:
        if not writer.isOpened():
            raise OSError("The local OpenCV build could not encode MP4")
        for path in files:
            image = cv2.resize(load_image(path).image, size, interpolation=cv2.INTER_AREA)
            for _ in range(12):
                writer.write(image)
    finally:
        writer.release()
    write_json(args.output / "castle-video-calibration.json", camera.resized(*size).to_dict())


if __name__ == "__main__":
    main()
