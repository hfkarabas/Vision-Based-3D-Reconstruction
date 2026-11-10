"""Bounded appearance-ranked pair discovery; only verified edges connect views."""
import cv2
import numpy as np

from vision3d.features import Features
from vision3d.features.matching import match
from vision3d.geometry import verify


def components(image_ids, pairs):
    adjacency = {i: set() for i in image_ids}
    for a, b in pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining, groups = set(adjacency), []
    while remaining:
        seen, stack = set(), [min(remaining)]
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            stack.extend(adjacency[i] - seen)
        groups.append(sorted(seen))
        remaining -= seen
    return groups


def appearance_signatures(views):
    signatures = {}
    for i, view in views.items():
        descriptors = view.features.descriptors
        if descriptors is None or len(descriptors) < 2:
            continue
        values = descriptors.astype(np.float64)
        if view.features.config['detector'] == 'orb':
            values = np.unpackbits(descriptors, axis=1).astype(float)
        signature = np.r_[values.mean(axis=0), values.std(axis=0)]
        signatures[i] = signature / max(np.linalg.norm(signature), 1e-12)
    return signatures


def discovery_candidate(views, verified_pairs, attempted, counts, signatures):
    groups = components(views, verified_pairs)
    labels = {image: k for k, group in enumerate(groups) for image in group}
    candidates = []
    for i in sorted(signatures):
        for j in sorted(signatures):
            if j <= i or (i, j) in attempted or counts.get(i, 0) >= 3 or counts.get(j, 0) >= 3:
                continue
            distance = float(1 - signatures[i] @ signatures[j])
            candidates.append((labels[i] == labels[j], distance, i, j))
    if not candidates:
        return None
    _, distance, i, j = min(candidates)
    return i, j, distance


def bridge_schedule(views, labels, attempted, budget, ratio, threshold, confidence, seed):
    """Rank untested station bridges cheaply before full descriptor verification."""
    coarse = {}
    for i in sorted(labels):
        features = views[i].features
        if features.descriptors is None:
            continue
        ids = sorted(range(features.count), key=lambda k: (-features.keypoints[k].response, k))[:600]
        coarse[i] = Features(tuple(features.keypoints[k] for k in ids), features.points[ids],
                             features.descriptors[ids], features.dimensions, features.config, 0.0)
    groups = {}
    for i in coarse:
        for j in coarse:
            if j <= i or (i, j) in attempted or labels[i] == labels[j]:
                continue
            a, b = coarse[i], coarse[j]
            matches = match(a, b, ratio=ratio, mutual=True, geometric=False).consistent
            p = a.points[[m.queryIdx for m in matches]]
            q = b.points[[m.trainIdx for m in matches]]
            focal = min(views[i].camera.fx, views[i].camera.fy, views[j].camera.fx, views[j].camera.fy)
            cv2.setRNGSeed((seed + i * 1009 + j * 9176) % 2147483647)
            _, mask, _ = verify(views[i].camera.normalized(p) * focal, views[j].camera.normalized(q) * focal,
                                threshold, confidence, method=cv2.USAC_MAGSAC)
            support = int(mask.sum())
            if support:
                group = tuple(sorted((labels[i], labels[j])))
                groups.setdefault(group, []).append((-support, i, j))
    for candidates in groups.values():
        candidates.sort()
    selected = []
    for rank in range(budget):
        available = sorted(g[rank] for g in groups.values() if rank < len(g))
        if not available:
            break
        selected.extend((i, j, -score) for score, i, j in available[:budget-len(selected)])
        if len(selected) == budget:
            break
    return selected
