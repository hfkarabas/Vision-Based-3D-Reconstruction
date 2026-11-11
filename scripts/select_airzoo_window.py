"""Choose a contiguous expanded AirZoo subset with measured local overlap."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from vision3d.features import extract
from vision3d.io.video import overlap, verified_overlap


def trigger(path):
    return int(path.stem.rsplit("_", 2)[1])


def select_window(anchor, count):
    paths = [Path(item["source"]) for item in anchor["inputs"]]
    folder = paths[0].parent
    marker, = folder.glob("*.MRK")
    ppk = {}
    for line in marker.read_text(encoding="utf-8-sig").splitlines():
        columns = line.split("\t")
        ppk[int(columns[0])] = (float(columns[1]), *(float(x.split(",")[0]) for x in columns[6:9]))
    images = sorted((p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}), key=trigger)
    if len({trigger(image) for image in images}) != len(images):
        raise ValueError("Duplicate image trigger")
    left, right = min(images.index(p) for p in paths), max(images.index(p) for p in paths)
    starts = range(max(0, right - count + 1), min(left + 1, len(images) - count + 1))
    if not starts:
        raise ValueError("No complete contiguous window of the requested size contains the anchor")
    offset = starts.start
    images = images[offset:starts.stop - 1 + count]
    ppk = np.asarray([ppk[trigger(image)] for image in images])
    if (np.diff(ppk[:, 0]) <= 0).any():
        raise ValueError("PPK trigger times must be increasing")
    local = (ppk[:, [2, 1, 3]] - ppk[0, [2, 1, 3]]) * [111195 * np.cos(np.radians(ppk[:, 1].mean())), 111195, 1]
    step = np.linalg.norm(np.diff(local[:, :2], axis=0), axis=1)
    features = []
    cv2.setNumThreads(1)
    for path in images:
        with Image.open(path) as source:
            source.draft("L", (960, 960))
            gray = np.asarray(ImageOps.exif_transpose(source).convert("L"))
        factor = min(1., 800 / max(gray.shape))
        size = tuple(round(dimension * factor) for dimension in gray.shape[::-1])
        features.append(extract(cv2.resize(gray, size, interpolation=cv2.INTER_AREA), "sift", 1000))
    evidence = {}
    for gap in (1, 2):
        for i in range(len(images) - gap):
            cv2.setRNGSeed(i * 1009 + (i + gap) * 9176)
            evidence[i, gap] = overlap(features[i], features[i + gap])
    options = []
    for i in range(len(starts)):
        adjacent = [evidence[j, 1] for j in range(i, i + count - 1)]
        skips = [evidence[j, 2] for j in range(i, i + count - 2)]
        # Keep the fixed anchor. Prefer continuous flight and measured two-step
        # support across the additional frames, then the weaker adjacent matches.
        jumps = int((step[i:i + count - 1] > max(10, 3 * np.median(step))).sum())
        rank = (-jumps, sum(verified_overlap(x) for x in adjacent), sum(verified_overlap(x) for x in skips),
                float(np.percentile([x["inliers"] for x in adjacent], 10)), -i)
        options.append(rank)
    rank = max(options)
    start = -rank[-1]
    selected = images[start:start + count]
    return {"images": [str(path) for path in selected], "triggers": [trigger(path) for path in selected],
            "anchor_images": len(paths), "competing_windows": len(starts),
            "large_position_steps": -rank[0], "verified_adjacent": rank[1], "verified_two_step": rank[2],
            "adjacent_inliers_p10": rank[3], "position_hints": {
                path.name: position.tolist() for path, position in zip(selected, local[start:start + count], strict=True)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=range(3, 101), required=True, metavar="3–100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_window(json.loads(args.anchor.read_text(encoding="utf-8")), args.count)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
