"""Bounded matching against the existing map when pose registration stalls."""
from collections import Counter
from time import perf_counter

import numpy as np

from .graph import appearance_signatures
from .incremental import extend_landmarks, observation_quality, pnp_correspondences
from .state import Track, conflicting_images


def frontier_pairs(state, pending, attempted, pairs, attempts, minimum, signatures, support):
    registered = set(state.poses)
    shared = Counter()
    covisibility = Counter()
    for track in state.tracks.values():
        images = track.observations.keys()
        for image in images & pending:
            if images & registered:
                shared[image] += 1
            covisibility.update((image, view) for view in images & registered)
    direct = Counter()
    for i, j in pairs:
        if i in registered and j in pending:
            direct[j] += 1
        if j in registered and i in pending:
            direct[i] += 1
    visible = {i: {lid for lid, point in state.landmarks.items() if i in point.observations}
               for i in registered}
    ranked = []
    for image in pending & signatures.keys():
        last = attempts[image][-1] if attempts[image] else {}
        count = len(support[image])
        quality = last.get("ransac_inliers", min(count, minimum - 1))
        for view in registered & signatures.keys():
            pair = tuple(sorted((image, view)))
            complementary = visible[view] - support[image]
            if pair in attempted or not complementary:
                continue
            source = state.views[view]
            pixels = source.features.points[[state.landmarks[lid].observations[view] for lid in sorted(complementary)]]
            cells = np.unique(np.clip(np.floor(pixels / [source.camera.width, source.camera.height] * 4), 0, 3), axis=0)
            similarity = float(signatures[image] @ signatures[view])
            rank = (bool(count or shared[image]), quality, count, direct[image], shared[image], -image,
                    covisibility[image, view] > 0, similarity, len(cells), len(complementary), -view)
            ranked.append((rank, image, view, pair))
    return [(image, view, pair) for _, image, view, pair in sorted(ranked, reverse=True)]


def station_balanced_pairs(ranked, limit, registered, labels):
    """Reserve up to half the budget for currently unmapped stations."""
    covered = {labels[i] for i in registered if i in labels}
    groups = {}
    for item in ranked:
        image = item[0]
        if image in labels and labels[image] not in covered:
            groups.setdefault(labels[image], []).append(item)
    reserved = []
    for level in range((limit + 1) // 2):
        for group in groups.values():
            if level < len(group) and len(reserved) < (limit + 1) // 2:
                reserved.append(group[level])
    chosen = set(reserved)
    return reserved + [item for item in ranked if item not in chosen][:limit-len(reserved)]


class MapRecovery:
    def __init__(self, state, pair_metrics, budget, rounds, station_labels=None):
        self.attempted = {tuple(row["pair"]) for row in pair_metrics}
        self.budget = budget
        self.max_rounds = rounds
        self.signatures = appearance_signatures(state.views)
        self.next_track_id = max(state.tracks, default=-1) + 1
        self.rounds = []
        self.pair_attempts = self.verified_pairs = 0
        self.stop_reason = None
        self.station_labels = station_labels

    def integrate(self, state, edges, max_error):
        """Extend tracks without renumbering mapped tracks or merging landmarks."""
        lookup = {(i, f): tid for tid, track in state.tracks.items() for i, f in track.observations.items()}
        counts = Counter()
        aliases = {}
        for _, left, right in sorted(edges):
            a, b = lookup.get(left), lookup.get(right)
            if a is not None and a == b:
                counts["duplicate_or_cycle"] += 1
                continue
            oa = state.tracks[a].observations if a is not None else dict([left])
            ob = state.tracks[b].observations if b is not None else dict([right])
            if conflicting_images(oa, ob):
                counts["image_conflict"] += 1
                continue
            if a in state.landmarks and b in state.landmarks:
                counts["two_landmarks"] += 1
                continue
            mapped = a if a in state.landmarks else b if b in state.landmarks else None
            merged = oa | ob
            if mapped is not None:
                xyz = state.landmarks[mapped].xyz
                unvalidated = merged.keys() - state.landmarks[mapped].observations.keys()
                newly_linked = merged.keys() - state.tracks[mapped].observations.keys()
                check = state.poses.keys() & unvalidated & (newly_linked | {left[0], right[0]})
                if any(not observation_quality(state, xyz, i, merged[i], max_error)[0] for i in check):
                    counts["mapped_reprojection_rejected"] += 1
                    continue
            ids = {tid for tid in (a, b) if tid is not None}
            keep = mapped if mapped is not None else min(ids, default=self.next_track_id)
            if not ids:
                self.next_track_id += 1
            for retired in ids - {keep}:
                aliases[retired] = keep
                del state.tracks[retired]
            state.tracks[keep] = Track(keep, dict(sorted(merged.items())))
            lookup.update(((i, f), keep) for i, f in merged.items())
            counts["integrated_edges"] += 1
        return dict(counts), aliases

    def run_round(self, state, pending, pairs, pair_metrics, attempts, evaluate, geometry, minimum):
        remaining = self.budget - self.pair_attempts
        rounds_left = self.max_rounds - len(self.rounds)
        if remaining <= 0 or rounds_left <= 0:
            self.stop_reason = "pair_budget" if remaining <= 0 else "round_budget"
            return None
        start = perf_counter()
        before = {i: set(pnp_correspondences(state, i)[0]) for i in pending}
        ranked = frontier_pairs(state, pending, self.attempted, pairs, attempts, minimum, self.signatures, before)
        if not ranked:
            self.stop_reason = "no_untested_map_pairs"
            return None
        limit = (remaining + rounds_left - 1) // rounds_left
        rows, edges = [], []
        selected = (station_balanced_pairs(ranked, limit, state.poses, self.station_labels)
                    if self.station_labels else ranked[:limit])
        for image, view, pair in selected:
            row, correspondences, verified_edges = evaluate(*pair, "map_recovery")
            row.update(recovery_round=len(self.rounds) + 1, frontier_image=image, reference_image=view)
            rows.append(row)
            self.attempted.add(pair)
            if row["edge_status"] == "verified":
                pairs[pair] = correspondences
                edges.extend(verified_edges)
        pair_metrics.extend(rows)
        counts, aliases = self.integrate(state, edges, geometry.max_reprojection_px)
        extension = extend_landmarks(state, geometry) if counts.get("integrated_edges") else {}
        after = {i: set(pnp_correspondences(state, i)[0]) for i in pending}
        changed = {i for i in pending if after[i] - before[i]}
        self.pair_attempts += len(rows)
        verified = sum(row["edge_status"] == "verified" for row in rows)
        self.verified_pairs += verified
        self.rounds.append({"registered_before": sorted(state.poses), "pairs_attempted": len(rows),
                            "verified_pairs": verified, "track_updates": counts, "retired_track_aliases": aliases,
                            "landmark_extension": extension, "runtime_seconds": perf_counter() - start,
                            "frontier": [{"image_id": i, "before": len(before[i]), "after": len(after[i]),
                                          "previous_ransac_inliers": attempts[i][-1].get("ransac_inliers") if attempts[i] else None}
                                         for i in sorted(changed | {row["frontier_image"] for row in rows})]})
        return changed
