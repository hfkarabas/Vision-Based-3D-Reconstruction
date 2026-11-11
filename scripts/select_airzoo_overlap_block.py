"""Nominate a coherent AirZoo block from cross-station image evidence before SfM."""
import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.sparse.csgraph import connected_components

from select_airzoo_window import trigger
from vision3d.features import extract
from vision3d.features.matching import match
from vision3d.geometry import verify


def footprint(center, yaw, pitch, altitude):
    # DJI azimuth is measured from north. The approximate target plane is used
    # only to schedule overlap checks. It does not enter reconstruction.
    y, p = np.radians([yaw, pitch])
    forward = np.array([np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), np.sin(p)])
    right = np.array([np.cos(y), -np.sin(y), 0])
    down = np.cross(forward, right)
    rays = forward + np.array([-.72, .72, .72, -.72])[:, None] * right + np.array(
        [-.54, -.54, .54, .54])[:, None] * down
    scale = (center[2] - altitude) / np.maximum(-rays[:, 2], .16)
    return cv2.convexHull((center[:2] + scale[:, None] * rays[:, :2]).astype(np.float32))


def evidence(a, b):
    pairs = match(a, b, ratio=.8, mutual=True, geometric=False).consistent
    aa = a.points[[m.queryIdx for m in pairs]]
    bb = b.points[[m.trainIdx for m in pairs]]
    _, inliers, _ = verify(aa, bb, threshold=1.5, confidence=.995)
    count = int(inliers.sum())
    if count < 16 or count / len(pairs) < .35:
        return None
    area = min(cv2.contourArea(cv2.convexHull(points[inliers].astype(np.float32))) / np.prod(features.dimensions)
               for points, features in ((aa, a), (bb, b)))
    if area < .025:
        return None
    return {"inliers": count, "inlier_ratio": count / len(pairs), "image_hull_fraction": area,
            "weight": count * np.sqrt(area)}


def choose(folder, count):
    files = sorted(folder.glob("*.JPG"), key=trigger)
    marker, = folder.glob("*.MRK")
    marks = {int(row[0]): [float(v.split(",")[0]) for v in row[6:9]]
             for row in (line.split("\t") for line in marker.read_text(encoding="utf-8-sig").splitlines())}
    lla = np.asarray([marks[trigger(p)] for p in files])
    xyz = (lla[:, [1, 0, 2]] - lla[0, [1, 0, 2]]) * [111195 * np.cos(np.radians(lla[0, 0])), 111195, 1]
    baselines = np.linalg.norm(xyz[:, None, :2] - xyz[None, :, :2], axis=2)
    _, stations = connected_components(baselines < 15, directed=False)
    plane = xyz[:, 2].min() - 35
    polygons, features, quality = [], [], []
    cv2.setNumThreads(1)
    for path, center in zip(files, xyz, strict=True):
        with path.open("rb") as stream:
            header = stream.read(256 * 1024)
        attitude = [float(re.search(rb"Gimbal" + key + rb'Degree="([+0-9.\-]+)"', header)[1])
                    for key in (b"Yaw", b"Pitch")]
        polygons.append(footprint(center, *attitude, plane))
        with Image.open(path) as source:
            source.draft("L", (800, 800))
            image = np.asarray(ImageOps.exif_transpose(source).convert("L"))
        resized = cv2.resize(image, tuple(round(n * 750 / max(image.shape)) for n in image.shape[::-1]),
                             interpolation=cv2.INTER_AREA)
        features.append(extract(resized, "sift", 1200))
        quality.append([float(cv2.Laplacian(resized, cv2.CV_32F).var()),
                        float(((resized < 4) | (resized > 251)).mean())])
    candidates = set()
    areas = [cv2.contourArea(p) for p in polygons]
    for i, poly in enumerate(polygons):
        scored = []
        for j in range(len(files)):
            if stations[i] == stations[j] or not 18 <= baselines[i, j] <= 220:
                continue
            area, _ = cv2.intersectConvexConvex(poly, polygons[j])
            if area > .10 * min(areas[i], areas[j]):
                scored.append((area / max(areas[i], areas[j]), j))
        for _, j in sorted(scored, reverse=True)[:12]:
            candidates.add(tuple(sorted((i, j))))
        for j in range(i + 1, min(i + 4, len(files))):
            if stations[i] == stations[j]:
                candidates.add((i, j))
    links = np.zeros((len(files), len(files)))
    records = []
    for i, j in sorted(candidates):
        cv2.setRNGSeed(i * 1009 + j * 9176)
        overlap = evidence(features[i], features[j])
        if overlap:
            links[i, j] = links[j, i] = overlap["weight"]
            records.append({"images": [trigger(files[i]), trigger(files[j])], "stations": [int(stations[i]), int(stations[j])],
                            "baseline_m": baselines[i, j], **overlap})
    cross = stations[:, None] != stations
    cross_links = links * cross
    # Try each verified cross-station edge as an anchor. Grow along supported
    # edges, favoring support from other stations and at least two connections.
    best = None
    for i, j in zip(*np.nonzero(np.triu(cross_links, 1)), strict=True):
        selected = [i, j]
        while len(selected) < count:
            cross_weight = np.minimum(cross_links[:, selected], 45).sum(axis=1)
            local_weight = np.minimum(links[:, selected] * ~cross[:, selected], 45).sum(axis=1)
            score = cross_weight + .35 * local_weight
            score[selected] = 0
            k = int(np.argmax(score))
            if not score[k]:
                break
            selected.append(k)
        if len(selected) < count or len(set(stations[selected])) < 2:
            continue
        common = cross_links[np.ix_(selected, selected)]
        rank = (int(np.count_nonzero(common.any(axis=1))), float(np.minimum(common, 45).sum()), -i, -j)
        if best is None or rank > best[0]:
            best = rank, sorted(selected)
    if best is None:
        raise ValueError("No coherent graph block of the requested size")
    ids = best[1]
    return {"images": [str(files[i].resolve()) for i in ids], "triggers": [trigger(files[i]) for i in ids],
            "stations": {files[i].name: int(stations[i]) for i in ids},
            "position_hints": {files[i].name: xyz[i].tolist() for i in ids},
            "quality_sharpness_saturation": {files[i].name: quality[i] for i in ids},
            "candidate_pairs": len(candidates), "verified_pairs": records,
            "selected_cross_station_supported_images": best[0][0],
            "selection": "Cross-station PPK and gimbal overlap prior, verified SIFT inliers and image hull support"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, choices=range(20, 61), default=28, metavar="20–60")
    args = parser.parse_args()
    selected = choose(args.images, args.count)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(selected, stream, indent=2)
        stream.write("\n")
