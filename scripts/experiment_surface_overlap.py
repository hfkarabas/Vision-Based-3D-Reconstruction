"""Test conservative removal of occluded, geometrically redundant intersecting faces."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from audit_surface_topology import local_neighborhoods, triangle_patch
from vision3d.reconstruction.surface_validation import (face_geometry, mesh_arrays, pinhole_projection, ray_scene,
                                                        surface_views, visible_points)


def visibility_counts(mesh, points, normals, views):
    counts = np.zeros(len(points), dtype=int)
    scene = ray_scene(mesh)
    for view in views:
        uv, z = pinhole_projection(points, view)
        ray = view.pose.center - points
        incidence = np.abs(np.sum(ray * normals, axis=1)) / np.maximum(np.linalg.norm(ray, axis=1), 1e-30)
        valid = (z > 0) & (incidence >= .25) & ((uv >= 1) & (
            uv < [view.camera.width - 1, view.camera.height - 1])).all(axis=1)
        rows = np.flatnonzero(valid)
        counts[rows[visible_points(scene, points[rows], view.pose.center)]] += 1
    return counts


def fully_overlaps(candidate, support, tolerance):
    """Check whether a candidate projects inside the better-observed triangle."""
    a, b = support[:, 1] - support[:, 0], support[:, 2] - support[:, 0]
    normal = np.cross(a, b)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    offsets = candidate - support[:, :1]
    height = np.sum(offsets * normal[:, None], axis=2)
    aa, ab, bb = (a*a).sum(axis=1), (a*b).sum(axis=1), (b*b).sum(axis=1)
    pa, pb = np.sum(offsets * a[:, None], axis=2), np.sum(offsets * b[:, None], axis=2)
    det = (aa * bb - ab ** 2)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (bb[:, None] * pa - ab[:, None] * pb) / det
        v = (aa[:, None] * pb - ab[:, None] * pa) / det
    return ((np.abs(height) <= tolerance) & (u >= 0) & (v >= 0) & (u + v <= 1)).all(axis=1)


def experiment(surface_dir):
    data = json.loads((surface_dir / "metrics.json").read_text(encoding="utf-8"))
    mesh = o3d.io.read_triangle_mesh(str(surface_dir / "mesh.ply"))
    xyz, faces = mesh_arrays(mesh)
    pairs = []
    for _, rows in local_neighborhoods(xyz, faces):
        collisions = np.asarray(triangle_patch(xyz, faces[rows]).get_self_intersecting_triangles())
        if len(collisions):
            pairs.append(rows[collisions])
    if not pairs:
        return {"tested_intersections": 0, "redundant_faces": 0}
    pairs = np.unique(np.sort(np.concatenate(pairs), axis=1), axis=0)
    points, cross, area = face_geometry(xyz, faces)
    normal = cross / np.maximum(2 * area[:, None], 1e-30)
    first, second = pairs.T
    nearly_parallel = np.abs(np.sum(normal[first] * normal[second], axis=1)) > .96
    pairs = pairs[nearly_parallel]
    rows = np.unique(pairs)
    counts = np.zeros(len(faces), dtype=int)
    counts[rows] = visibility_counts(mesh, points[rows], normal[rows], surface_views(data["sparse_dir"]))
    candidate, keeper = np.concatenate([pairs, pairs[:, ::-1]]).T
    eligible = (counts[candidate] == 0) & (counts[keeper] >= 2)
    candidate, keeper = candidate[eligible], keeper[eligible]
    close = fully_overlaps(xyz[faces[candidate]], xyz[faces[keeper]], data["voxel_size"] / 2)
    redundant = np.unique(candidate[close])
    return {"tested_intersections": len(nearly_parallel), "nearly_parallel_intersections": len(pairs),
            "visibility_asymmetric_candidates": len(candidate), "redundant_faces": len(redundant),
            "redundant_face_fraction": len(redundant) / len(faces),
            "maximum_plane_separation": data["voxel_size"] / 2,
            "scope": "eight compact neighborhoods; candidate centroid unseen, keeper seen by at least two cameras"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = experiment(args.surface_dir)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
