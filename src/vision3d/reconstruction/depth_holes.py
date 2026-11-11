"""Experimental recovery of empty primary pixels from corroborated alternate depth."""
import numpy as np

from .dense_quality import round_trip_depth
from .depth_recovery import _distinct_rays


def primary_holes(primary, alternate, disparity):
    """Require an empty 3x3 neighborhood in the original rectified depth map."""
    z = alternate.depth[tuple(alternate.pixels.T)]
    rows = np.flatnonzero((z > 0) & (z <= alternate.focal_baseline / disparity))
    local = alternate.xyz[rows] @ primary.rotation.T + primary.translation
    positive = np.isfinite(local).all(axis=1) & (local[:, 2] > 0)
    rows, local = rows[positive], local[positive]
    projected = local @ primary.intrinsic.T
    uv = projected[:, :2] / projected[:, 2:3]
    h, w = primary.depth.shape
    inside = ((uv >= 1) & (uv <= [w - 2, h - 2])).all(axis=1)
    rows, uv, local = rows[inside], uv[inside], local[inside]
    xy = np.rint(uv).astype(int)
    x, y = xy.T
    empty = np.ones(len(rows), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            empty &= primary.depth[y + dy, x + dx] == 0
    return rows[empty], xy[empty, ::-1], local[empty, 2], np.linalg.norm((uv - xy)[empty], axis=1)


def supported_holes(primary, alternate, maps, centers, disparity, tolerance, return_px):
    """Use two camera-disjoint round-trip witnesses and at most one point per primary pixel."""
    rows, pixels, depths, distance = primary_holes(primary, alternate, disparity)
    remaining = np.ones(len(rows), dtype=bool)
    first = np.full(len(rows), -1, dtype=int)
    second = first.copy()
    edges = [{d.reference, d.source} for d in maps]
    candidate = {alternate.reference, alternate.source}
    for index, witness in enumerate(maps):
        if candidate & edges[index] or not remaining.any():
            continue
        matches, _, _ = round_trip_depth(alternate, witness, tolerance, return_px,
                                          sample_rows=rows[remaining],
                                          witness_max_depth=witness.focal_baseline / disparity)
        points = alternate.xyz[matches]
        separated = (_distinct_rays(points, centers, alternate.reference, witness.reference) &
                     _distinct_rays(points, centers, witness.reference, witness.source))
        ids = np.searchsorted(rows, matches[separated])
        earlier = first[ids]
        for previous in np.unique(earlier[earlier >= 0]):
            if not edges[previous] & edges[index]:
                selected = ids[earlier == previous]
                coherent = _distinct_rays(alternate.xyz[rows[selected]], centers,
                                           maps[previous].reference, witness.reference)
                selected = selected[coherent]
                second[selected] = index
                remaining[selected] = False
        first[ids[earlier < 0]] = index
    good = np.flatnonzero(second >= 0)
    if not len(good):
        return np.empty((0, 5), dtype=int)
    pixel_ids = np.ravel_multi_index(pixels[good].T, primary.depth.shape)
    order = np.lexsort((distance[good], pixel_ids))
    good, pixel_ids = good[order], pixel_ids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(pixel_ids)) + 1]
    low, high = np.minimum.reduceat(depths[good], starts), np.maximum.reduceat(depths[good], starts)
    keep = good[starts[high - low <= tolerance * low]]
    return np.column_stack((rows[keep], pixels[keep], first[keep], second[keep]))
