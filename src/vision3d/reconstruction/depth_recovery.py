"""Recover conditioned primary depth with bounded, independent stereo witnesses."""
from dataclasses import dataclass, field

import numpy as np

from .dense_quality import round_trip_depth


@dataclass
class PendingDepth:
    depth: object
    mask: np.ndarray
    colors: np.ndarray
    eligible: np.ndarray
    votes: list = field(default_factory=list)


def _distinct_rays(points, centers, first, second):
    a, b = points - centers[first], points - centers[second]
    cosine = np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-15)
    return cosine <= np.cos(np.deg2rad(1))


def _add_vote(state, rows, reference, source, maps, centers):
    """Consume each camera at most once in supplemental evidence for a pixel."""
    if not len(rows):
        return rows
    sample = state.depth
    points = sample.xyz[rows]
    valid = (_distinct_rays(points, centers, sample.reference, reference) &
             _distinct_rays(points, centers, reference, source))
    for index, primary in enumerate(maps):
        if {primary.reference, primary.source} & {reference, source}:
            valid &= (sample.witness_bits[rows] & (np.uint64(1) << np.uint64(index))) == 0
    for other_ref, other_source, earlier in state.votes:
        if {other_ref, other_source} & {reference, source}:
            valid &= ~np.isin(rows, earlier)
    return rows[valid]


def recover_depth(maps, accepted, centers, alternatives, build, budget, tolerance, return_px, disparity,
                  recentered_witnesses, stage):
    """Add independent witnesses to previously plausible primary samples, leaving accepted depth untouched."""
    by_ref = {d.reference: (mask, rgb) for d, mask, rgb in accepted}
    states, priorities = {}, {}
    for d in maps:
        mask, colors = by_ref.get(d.reference, (np.zeros(len(d.xyz), bool), np.empty((0, 3))))
        z = d.depth[tuple(d.pixels.T)]
        eligible = np.flatnonzero(~mask & (z > 0) & (z <= d.focal_baseline / disparity))
        states[d.reference] = PendingDepth(d, mask.copy(), colors, eligible)
        if not len(eligible):
            continue
        scored = eligible[::max(1, (len(eligible) + 11999) // 12000)]
        for witness in maps:
            if witness.reference == d.reference or {witness.reference, witness.source} == {d.reference, d.source}:
                continue
            plausible, _, _ = round_trip_depth(d, witness, tolerance, return_px, sample_rows=scored)
            if len(plausible):
                priorities[witness.reference] = priorities.get(witness.reference, 0) + len(plausible)

    attempts, visited = [], set()
    order = sorted(priorities, key=lambda ref: (-priorities[ref], ref))
    primary_edges = {frozenset((d.reference, d.source)) for d in maps}
    for reference in order:
        candidates = [(source, xyz) for _, source, xyz in alternatives[reference]
                      if frozenset((reference, source)) not in primary_edges][:4]
        for source, xyz in candidates:
            edge = frozenset((reference, source))
            if edge in visited or len(attempts) == budget:
                continue
            visited.add(edge)
            stage(f"Checking alternate stereo witness {len(attempts) + 1}/{budget}")
            witness, reason = build(reference, source, xyz)
            record = {"reference": reference, "source": source, "prior_support": priorities[reference],
                      "new_votes": 0, "rejection": reason}
            attempts.append(record)
            if witness is None:
                continue
            record["recentered"] = witness.recentered
            for state in states.values():
                d = state.depth
                if not len(state.eligible) or {reference, source} & {d.reference, d.source}:
                    continue
                rows, _, _ = round_trip_depth(d, witness, tolerance, return_px, sample_rows=state.eligible,
                                              witness_max_depth=witness.focal_baseline / disparity)
                rows = _add_vote(state, rows, reference, source, maps, centers)
                if len(rows):
                    state.votes.append((reference, source, rows))
                    record["new_votes"] += len(rows)
        if len(attempts) == budget:
            break

    result, recovered, provenance = [], {}, {}
    bit_index = {d.reference: i for i, d in enumerate(maps)}
    for state in states.values():
        d, old = state.depth, state.mask
        counts = np.bitwise_count(d.witness_bits)
        for _, _, rows in state.votes:
            counts[rows] += 1
        minimum = recentered_witnesses if d.recentered else 1
        added = state.eligible[counts[state.eligible] >= minimum]
        mask = old.copy()
        mask[added] = True
        recovered[d.reference] = len(added)
        records = []
        for reference, source, rows in state.votes:
            kept = np.intersect1d(added, rows)
            if len(kept):
                d.witness_bits[kept] |= np.uint64(1) << np.uint64(bit_index[reference])
                records.append((reference, source, kept))
        if records:
            provenance[d.reference] = records
        if mask.any():
            rgb = d.colors[mask].copy()
            rgb[old[mask]] = state.colors
            result.append((d, mask, rgb))
    return result, {"attempts": attempts, "recovered_by_reference": recovered,
                    "recovered_pixels": sum(recovered.values())}, provenance
