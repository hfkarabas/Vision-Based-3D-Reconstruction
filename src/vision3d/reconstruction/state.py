"""Explicit observations, conflict-free tracks and world-to-camera state."""
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

from vision3d.features import Features
from vision3d.geometry.camera import Camera


@dataclass
class View:
    id: int
    name: str
    camera: Camera
    features: Features
    colors: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class Pose:
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        self.translation = np.asarray(self.translation, dtype=np.float64).reshape(-1)
        if (self.rotation.shape != (3, 3) or self.translation.shape != (3,)
                or not np.isfinite(self.rotation).all() or not np.isfinite(self.translation).all()
                or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(self.rotation), 1.0, atol=1e-6)):
            raise ValueError("Pose requires a finite proper rotation and a finite 3-vector translation")

    @property
    def center(self):
        return -self.rotation.T @ self.translation

    @property
    def matrix(self):
        return np.r_[np.c_[self.rotation, self.translation], [[0, 0, 0, 1]]]


@dataclass
class Track:
    id: int
    observations: dict[int, int]


@dataclass
class Landmark:
    id: int  # same immutable ID as its owning track
    xyz: np.ndarray
    color: np.ndarray
    observations: dict[int, int]


@dataclass
class State:
    views: dict[int, View]
    tracks: dict[int, Track]
    poses: dict[int, Pose] = field(default_factory=dict)
    landmarks: dict[int, Landmark] = field(default_factory=dict)
    anchors: tuple = ()
    registration_order: list = field(default_factory=list)
    history: list = field(default_factory=list)


def conflicting_images(left, right):
    """Return images where two tracks assign different features."""
    return {image for image in left.keys() & right.keys() if left[image] != right[image]}


def track_counts(tracks):
    lengths = Counter(len(t.observations) for t in tracks.values())
    return {"tracks": len(tracks), "observations": sum(length * count for length, count in lengths.items()),
            "length_distribution": dict(sorted(lengths.items())), "two_view_tracks": lengths[2]}


def build_tracks(edges):
    """Edges must already be geometrically verified, in deterministic confidence order.

    A conflicting edge is rejected before union; existing components are untouched.
    IDs are assigned by sorted minimum observation, independent of union root choice.
    """
    parent, members = {}, {}
    conflicts = duplicates = 0
    rejected = []
    def root(obs):
        if obs not in parent:
            parent[obs] = obs
            members[obs] = {obs[0]: obs[1]}
        while parent[obs] != obs:
            parent[obs] = parent[parent[obs]]
            obs = parent[obs]
        return obs
    for left, right in edges:
        if left[0] == right[0]:
            if left == right:
                duplicates += 1
            else:
                conflicts += 1
            continue
        a, b = root(left), root(right)
        if a == b:
            duplicates += 1
            continue
        overlap = conflicting_images(members[a], members[b])
        if overlap:
            conflicts += 1
            if len(rejected) < 1000:
                rejected.append({"left": left, "right": right, "reason": "duplicate_image_in_component",
                                 "conflicting_images": sorted(overlap)})
            continue
        if b < a:
            a, b = b, a
        parent[b] = a
        members[a].update(members.pop(b))
    groups = sorted((m for m in members.values() if len(m) >= 2), key=lambda m: min(m.items()))
    tracks = {i: Track(i, dict(sorted(m.items()))) for i, m in enumerate(groups)}
    return tracks, {**track_counts(tracks), "conflicting_edges_rejected": conflicts,
                    "duplicate_or_cycle_edges": duplicates, "conflict_examples": rejected,
                    "conflict_examples_truncated": conflicts > len(rejected)}
