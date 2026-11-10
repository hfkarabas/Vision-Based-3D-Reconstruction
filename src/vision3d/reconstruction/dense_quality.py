"""View coverage and geometric depth checks for UAV dense reconstruction."""
from collections import defaultdict

import numpy as np


def landmark_visibility(landmarks, poses, max_reprojection_px):
    """Use only finite points and positive-depth, locally consistent observations."""
    visibility = defaultdict(set)
    for index, landmark in enumerate(landmarks):
        xyz = np.asarray(landmark["xyz"], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            continue
        errors = landmark.get("reprojection_errors_px", {})
        for key in landmark["observations"]:
            image = int(key)
            if image not in poses or not 0 <= errors.get(str(key), 0) <= max_reprojection_px:
                continue
            pose = poses[image]
            if pose.rotation[2] @ xyz + pose.translation[2] > 0:
                visibility[image].add(index)
    return visibility


def spread_references(neighbors, visibility, budget):
    """Prefer useful stereo edges while spending the budget on distinct scene support."""
    scores = {i: pairs[0][0] for i, pairs in neighbors.items()}
    selected, covered = [], set()
    while scores and len(selected) < budget:
        reference = max(scores, key=lambda i: (
            scores[i] * (.25 + .75 * len(visibility[i] - covered) / len(visibility[i])), -i))
        selected.append(reference)
        covered.update(visibility[reference])
        del scores[reference]
    return selected


def round_trip_depth(sample, witness, tolerance, max_return_px, rgb_tolerance=.25):
    """Interpolate a coherent witness footprint, then project it back to the source pixel."""
    local = sample.xyz @ witness.rotation.T + witness.translation
    rows = np.flatnonzero(np.isfinite(local).all(axis=1) & (local[:, 2] > 0))
    projected = local[rows] @ witness.intrinsic.T
    uv = projected[:, :2] / projected[:, 2:3]
    h, w = witness.depth.shape
    inside = ((uv >= 0) & (uv < [w - 1, h - 1])).all(axis=1)
    rows, uv = rows[inside], uv[inside]
    x, y = np.floor(uv).astype(int).T
    fx, fy = (uv - np.column_stack((x, y))).T
    weights = np.column_stack(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy))
    footprint = witness.depth[y[:, None] + [0, 0, 1, 1], x[:, None] + [0, 1, 0, 1]]
    near, far = footprint.min(axis=1), footprint.max(axis=1)
    consistent = (near > 0) & (far - near <= tolerance * near)
    rows, uv, weights, footprint = rows[consistent], uv[consistent], weights[consistent], footprint[consistent]
    depth = 1 / (weights / footprint).sum(axis=1)
    depth_ok = np.abs(local[rows, 2] - depth) <= tolerance * depth
    rows, uv, depth = rows[depth_ok], uv[depth_ok], depth[depth_ok]
    rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(witness.intrinsic).T
    world = (rays * depth[:, None] - witness.translation) @ witness.rotation
    returned = (world @ sample.rotation.T + sample.translation) @ sample.intrinsic.T
    positive = returned[:, 2] > 0
    rows, uv, returned = rows[positive], uv[positive], returned[positive]
    error = np.linalg.norm(returned[:, :2] / returned[:, 2:3] - sample.pixels[rows, ::-1], axis=1)
    x, y = np.rint(uv).astype(int).T
    rgb = witness.image[y, x, ::-1] / 255.0
    accepted = (error <= max_return_px) & (np.abs(rgb - sample.colors[rows]).mean(axis=1) <= rgb_tolerance)
    return rows[accepted], rgb[accepted], error[accepted]


def supported_samples(maps, tolerance, max_return_px, recentered_witnesses=1):
    """Count corroboration once per witness reference and weight it by pixel agreement."""
    if len({depth.reference for depth in maps}) != len(maps):
        raise ValueError("Depth corroboration needs one map per reference camera")
    accepted, diagnostics = [], []
    for sample in maps:
        counts = np.zeros(len(sample.xyz), np.int32)
        weights = np.ones(len(sample.xyz))
        colors = sample.colors.copy()
        best = np.full(len(sample.xyz), np.inf)
        for witness in maps:
            if (witness.reference == sample.reference
                    or {witness.reference, witness.source} == {sample.reference, sample.source}):
                continue
            rows, rgb, error = round_trip_depth(sample, witness, tolerance, max_return_px)
            weight = 1 / (1 + (error / max_return_px) ** 2)
            counts[rows] += 1
            weights[rows] += weight
            colors[rows] += rgb * weight[:, None]
            best[rows] = np.minimum(best[rows], error)
        minimum = recentered_witnesses if sample.recentered else 1
        mask = counts >= minimum
        diagnostics.append({"reference": sample.reference, "source": sample.source,
                            "minimum_witnesses": minimum,
                            "raw_samples": len(sample.xyz), "checked_samples": int(mask.sum()),
                            "witness_histogram": np.bincount(counts).tolist(),
                            "best_return_px_p50_p95": np.percentile(best[mask], [50, 95]).tolist() if mask.any() else None})
        if mask.any():
            accepted.append((sample, mask, colors[mask] / weights[mask, None]))
    return accepted, diagnostics
