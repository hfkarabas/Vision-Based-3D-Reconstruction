"""Partition selected triangles along shared edges without changing their surface."""
import numpy as np
import open3d as o3d

from .surface_validation import face_geometry, mesh_arrays, update_surface_normals


def subdivide_visible_candidates(mesh, candidates, budget):
    xyz, faces = mesh_arrays(mesh)
    _, _, area = face_geometry(xyz, faces)
    rows = np.flatnonzero(candidates)
    selected = rows[np.argsort(-area[rows], kind="stable")[:budget]]
    if not len(selected):
        return mesh, 0
    a, b = faces, faces[:, [1, 2, 0]]
    keys = np.minimum(a, b).astype(np.int64) * len(xyz) + np.maximum(a, b)
    edges = np.unique(keys[selected])
    midpoints = np.searchsorted(edges, keys)
    active = (midpoints < len(edges)) & (edges[np.minimum(midpoints, len(edges) - 1)] == keys)
    midpoints += len(xyz)
    colors = np.asarray(mesh.vertex_colors)
    u, v = edges // len(xyz), edges % len(xyz)
    vertices = np.concatenate((xyz, (xyz[u] + xyz[v]) / 2))
    colors = np.concatenate((colors, (colors[u] + colors[v]) / 2))
    face_masks = active @ [1, 2, 4]
    pieces = [faces[face_masks == 0]]
    for edge in range(3):
        next_edge, opposite = (edge + 1) % 3, (edge + 2) % 3
        single = face_masks == (1 << edge)
        a, b, c = faces[single][:, [edge, next_edge, opposite]].T
        m = midpoints[single, edge]
        pieces.extend((np.column_stack((a, m, c)), np.column_stack((m, b, c))))
        double = face_masks == ((1 << edge) | (1 << next_edge))
        a, b, c = faces[double][:, [edge, next_edge, opposite]].T
        m, n = midpoints[double][:, [edge, next_edge]].T
        pieces.extend((np.column_stack((a, m, c)), np.column_stack((m, n, c)), np.column_stack((m, b, n))))
    full = face_masks == 7
    a, b, c = faces[full].T
    m, n, k = midpoints[full].T
    pieces.extend((np.column_stack((a, m, k)), np.column_stack((m, b, n)), np.column_stack((k, n, c)),
                   np.column_stack((m, n, k))))
    refined = np.concatenate(pieces)
    _, _, new_area = face_geometry(vertices, refined)
    if (new_area <= 0).any() or not np.isclose(new_area.sum(), area.sum(), rtol=1e-10, atol=1e-12):
        raise ValueError("Subdividing candidate faces changed the geometric surface")
    result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(refined))
    result.vertex_colors = o3d.utility.Vector3dVector(colors)
    update_surface_normals(result)
    return result, len(selected)
