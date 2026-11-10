"""Bounded competition between independent, geometry-verified seed maps."""
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .graph import components
from .incremental import extend_landmarks, initialize, register, registration_priority
from .state import State


def load_position_hints(path):
    """Read image-name keyed metric [east, north, up] positions for ranking."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("Position hints require an object keyed by image filename")
    for name, value in data.items():
        if not isinstance(value, list) or len(value) != 3 or any(type(x) not in (int, float) for x in value):
            raise ValueError("Position hints must be finite metric [east, north, up] vectors")
        try:
            position = np.array(value, dtype=float)
        except OverflowError as exc:
            raise ValueError("Position hints must be finite metric [east, north, up] vectors") from exc
        if not np.isfinite(position).all():
            raise ValueError("Position hints must be finite metric [east, north, up] vectors")
        data[name] = position
    return data


def station_labels(positions, radius):
    links = {(i, j) for i in positions for j in positions if i < j
             and np.linalg.norm((positions[i] - positions[j])[:2]) <= radius}
    return {i: station for station, group in enumerate(components(positions, links)) for i in group}


def shortlist(baseline, ranked, labels, budget):
    """Distribute the alternative budget among geometrically accepted station bridges."""
    if not labels:
        return [baseline, *[c for c in ranked if c[1:3] != baseline[1:3]][:budget-1]]
    selected = [baseline]
    bridges = {}
    for c in ranked:
        i, j = c[1:3]
        if (c[1:3] == baseline[1:3] or i not in labels or j not in labels or labels[i] == labels[j]):
            continue
        key = tuple(sorted((labels[i], labels[j])))
        bridges.setdefault(key, []).append(c)
    while bridges and len(selected) < budget:
        # Each station pair gets one opportunity before another from that pair.
        for key in sorted(bridges, key=lambda k: (-bridges[k][0][0], k)):
            selected.append(bridges[key].pop(0))
            if not bridges[key]:
                del bridges[key]
            if len(selected) == budget:
                break
    chosen = {c[1:3] for c in selected}
    return selected + [c for c in ranked if c[1:3] not in chosen][:budget-len(selected)]


def evaluate_seed(state, labels, minimum_points):
    counts = Counter(labels[i] for i in state.poses if i in labels)
    bridge_support = Counter()
    for landmark in state.landmarks.values():
        # Require a third geometrically accepted observation beyond a seed pair.
        if len(landmark.observations) >= 3:
            groups = {labels[i] for i in landmark.observations if i in labels}
            if len(groups) > 1:
                bridge_support.update(groups)
    supported = [s for s in sorted(counts) if counts[s] >= 2 and bridge_support[s] >= minimum_points]
    rank = (len(supported), len(state.poses), sum(bridge_support.values()), len(state.landmarks))
    return rank, {"station_counts": dict(counts), "supported_stations": supported,
                  "bridge_landmarks_by_station": dict(bridge_support), "registered": list(state.registration_order),
                  "landmarks": len(state.landmarks)}


def search_seeds(baseline, ranked, diagnostics, views, tracks, geometry, registration,
                 seed, positions, radius, budget, extra_views):
    labels = station_labels(positions, radius)
    chosen = baseline
    best_rank = None
    records = {tuple(d["pair"]): d for d in diagnostics if d["status"] == "accepted"}
    for candidate in shortlist(baseline, ranked, labels, budget):
        state = State(views, tracks)
        initialize(state, candidate, geometry)
        attempts = []
        while len(state.poses) < extra_views + 2:
            pending = [(registration_priority(state, i), i) for i in views if i not in state.poses]
            for priority, image in sorted(pending, reverse=True)[:8]:
                if priority[1] < registration.min_inliers:
                    continue
                cv2.setRNGSeed((seed + image * 1009 + len(state.poses) * 9176) % 2147483647)
                accepted, metrics = register(state, image, registration, geometry)
                attempts.append({"image_id": image, **metrics})
                if accepted:
                    extend_landmarks(state, geometry)
                    break
            else:
                break
        rank, report = evaluate_seed(state, labels, geometry.min_points)
        a, b = candidate[1:3]
        if a in positions and b in positions:
            report["position_baseline_m"] = float(np.linalg.norm(positions[a] - positions[b]))
        records[a, b]["seed_search"] = {**report, "attempts": attempts, "rank": rank}
        if best_rank is None or rank > best_rank:
            best_rank, chosen = rank, candidate
    records[chosen[1:3]]["seed_search"]["selected"] = True
    return chosen
