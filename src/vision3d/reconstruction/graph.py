"""Bounded appearance-ranked pair discovery; only verified edges connect views."""
import numpy as np


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
