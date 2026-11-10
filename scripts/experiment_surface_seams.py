"""Join matching open boundary edges with sub-voxel, normal-compatible support."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from vision3d.reconstruction.surface_validation import face_geometry, mesh_arrays, topology


def weld_boundary_seams(mesh, tolerance):
    """Join existing triangles across close reciprocal boundary edges without adding faces."""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Seam tolerance must be positive and finite")
    xyz, faces = mesh_arrays(mesh)
    a, b = faces.ravel(), faces[:, [1, 2, 0]].ravel()
    keys = np.minimum(a, b).astype(np.int64) * len(xyz) + np.maximum(a, b)
    _, first, count = np.unique(keys, return_index=True, return_counts=True)
    edges = first[count == 1]
    if len(edges) < 2:
        return mesh, {"joined_edge_pairs": 0, "merged_vertices": 0}
    # Compare boundary edges from distinct edge-connected components.
    labels, _, _ = mesh.cluster_connected_triangles()
    component = np.asarray(labels)[edges // 3]
    start, end = xyz[a[edges]], xyz[b[edges]]
    centers = (start + end) / 2
    direction = end - start
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    _, normal, areas = face_geometry(xyz, faces[edges // 3])
    normal /= np.maximum(2 * areas[:, None], 1e-30)
    _, neighbors = cKDTree(centers).query(centers, k=min(len(edges), 8), distance_upper_bound=tolerance)
    best, costs = np.full(len(edges), -1), np.full(len(edges), np.inf)
    for candidates in neighbors.T:
        source = np.flatnonzero(candidates < len(edges))
        target = candidates[source]
        dx, dy = start[source] - end[target], end[source] - start[target]
        cost = np.maximum(np.linalg.norm(dx, axis=1), np.linalg.norm(dy, axis=1))
        valid = (component[source] != component[target]) & (cost <= tolerance)
        valid &= np.sum(direction[source] * direction[target], axis=1) < -.96
        valid &= np.sum(normal[source] * normal[target], axis=1) > .96
        valid &= np.maximum(np.abs(np.sum(dx * normal[source], axis=1)),
                            np.abs(np.sum(dy * normal[source], axis=1))) < .2 * tolerance
        valid &= cost < costs[source]
        best[source[valid]], costs[source[valid]] = target[valid], cost[valid]
    reciprocal = np.flatnonzero((best > np.arange(len(edges))) & (best[best] == np.arange(len(edges))))
    if not len(reciprocal):
        return mesh, {"joined_edge_pairs": 0, "merged_vertices": 0}

    mate = np.full(len(xyz), -1, np.int32)
    joined = 0
    for i in reciprocal[np.argsort(costs[reciprocal], kind="stable")]:
        e, f = edges[i], edges[best[i]]
        left, right = np.array([a[e], b[e]]), np.array([b[f], a[f]])
        if np.any((mate[left] != -1) & (mate[left] != right)) or np.any((mate[right] != -1) & (mate[right] != left)):
            continue
        mate[left], mate[right] = right, left
        joined += 1
    merged = np.flatnonzero((mate >= 0) & (mate < np.arange(len(xyz))))
    kept = mate[merged]
    index = np.arange(len(xyz), dtype=np.int32)
    index[merged] = kept
    positions, colors = xyz.copy(), np.asarray(mesh.vertex_colors).copy()
    positions[kept] = (positions[kept] + positions[merged]) / 2
    colors[kept] = (colors[kept] + colors[merged]) / 2
    candidate = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(positions), o3d.utility.Vector3iVector(index[faces]))
    candidate.vertex_colors = o3d.utility.Vector3dVector(colors)
    candidate.remove_unreferenced_vertices()
    candidate.compute_vertex_normals()
    _, report = topology(candidate)
    if (report["nonmanifold_edges"] or report["winding_conflicts"] or report["degenerate_triangles"]
            or report["invalid_vertex_normals"]):
        return mesh, {"joined_edge_pairs": 0, "merged_vertices": 0, "rejected_edge_pairs": joined}
    return candidate, {"joined_edge_pairs": joined, "merged_vertices": len(merged)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = o3d.io.read_triangle_mesh(str(args.mesh))
    result, metrics = weld_boundary_seams(original, args.tolerance)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"seams": metrics, "before": topology(original)[1], "after": topology(result)[1]}, stream, indent=2)
        stream.write("\n")
