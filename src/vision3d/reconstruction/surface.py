"""Open surfaces from validated disparity samples and round-trip mesh exports."""
import numpy as np


def footprint_edge_gate(points, faces, camera_center, focal, voxel, shared_support):
    """Allow longer image-grid edges only on well-supported, non-grazing footprints."""
    a, b, c = points[faces].transpose(1, 0, 2)
    ab, ac = b - a, c - a
    longest = np.sqrt(np.maximum.reduce([np.square(edge).sum(axis=1) for edge in (ab, ac, c - b)]))
    base = longest <= 6 * voxel
    candidates = np.flatnonzero(~base & (longest <= 12 * voxel) & shared_support)
    aa, bb, cc = a[candidates], b[candidates], c[candidates]
    ray = (aa + bb + cc) / 3 - camera_center
    distance = np.linalg.norm(ray, axis=1)
    normal = np.cross(bb - aa, cc - aa)
    incidence = abs(np.sum(normal * ray, axis=1)) / np.maximum(np.linalg.norm(normal, axis=1) * distance, 1e-15)
    # No holes are filled. Faces already come from adjacent pixels and meet
    # the existing local depth-jump check. A grazing face remains inadmissible.
    base[candidates] = (incidence >= .55) & (longest[candidates] <= 3 * distance / focal)
    return base


def build_surface(xyz, colors, triangles, target_faces=None, *, max_edge=None):
    import open3d as o3d

    triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    sorted_faces = np.sort(triangles, axis=1)
    valid = (np.diff(sorted_faces, axis=1) > 0).all(axis=1)
    _, first = np.unique(sorted_faces[valid], axis=0, return_index=True)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz),
                                     o3d.utility.Vector3iVector(triangles[valid][first]))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    if not mesh.has_triangles():
        raise ValueError("No continuous corroborated depth supports a mesh")
    edges = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    area = np.linalg.norm(np.cross(edges[:, 1] - edges[:, 0], edges[:, 2] - edges[:, 0]), axis=1)
    invalid = area <= 1e-14
    removed_long = 0
    if max_edge is not None:
        lengths = np.linalg.norm(edges - np.roll(edges, 1, axis=1), axis=2)
        long = (lengths > max_edge).any(axis=1)
        removed_long = int((long & ~invalid).sum())
        invalid |= long
    mesh.remove_triangles_by_mask(invalid)
    if not mesh.has_triangles():
        raise ValueError("Surface triangles have negligible area")
    labels, sizes, _ = mesh.cluster_connected_triangles()
    minimum = max(32, int(max(sizes) * .0002))
    mesh.remove_triangles_by_mask(np.asarray(sizes)[labels] < minimum)
    mesh.remove_unreferenced_vertices()
    if not mesh.has_triangles():
        raise ValueError("No connected surface with at least 32 triangles; try better overlap or a denser profile")
    removed_nonmanifold = 0
    if max_edge is not None:
        before = len(mesh.triangles)
        mesh.remove_non_manifold_edges()
        removed_nonmanifold += before - len(mesh.triangles)
    if target_faces and len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)
        if max_edge is not None:
            edges = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
            long = (np.linalg.norm(edges - np.roll(edges, 1, axis=1), axis=2) > max_edge).any(axis=1)
            mesh.remove_triangles_by_mask(long)
            removed_long += int(long.sum())
            before = len(mesh.triangles)
            mesh.remove_non_manifold_edges()
            removed_nonmanifold += before - len(mesh.triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    if max_edge is not None and mesh.has_triangles():
        labels, sizes, _ = mesh.cluster_connected_triangles()
        mesh.remove_triangles_by_mask(np.asarray(sizes)[labels] < minimum)
        mesh.remove_unreferenced_vertices()
        if mesh.has_triangles():
            mesh.orient_triangles()
    mesh.compute_vertex_normals()
    if not mesh.has_triangles() or not np.isfinite(np.asarray(mesh.vertex_normals)).all():
        raise ValueError("Mesh cleanup did not produce a finite surface")
    return mesh, {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
                  "input_faces": len(triangles), "component_min_faces": minimum,
                  "max_edge_world_units": max_edge, "long_faces_removed": removed_long,
                  "nonmanifold_faces_removed": removed_nonmanifold,
                  "open_manifold_edges": bool(mesh.is_edge_manifold(allow_boundary_edges=True)),
                  "closed_manifold_edges": bool(mesh.is_edge_manifold(allow_boundary_edges=False))}


def write_mesh_ply(path, mesh):
    import open3d as o3d

    if not o3d.io.write_triangle_mesh(str(path), mesh):
        raise OSError(f"Cannot write mesh PLY: {path}")
    loaded = o3d.io.read_triangle_mesh(str(path))
    if (not loaded.has_vertex_colors() or not loaded.has_vertex_normals()
            or not np.array_equal(np.asarray(mesh.triangles), np.asarray(loaded.triangles))
            or not np.allclose(np.asarray(mesh.vertices), np.asarray(loaded.vertices))
            or not np.allclose(np.asarray(mesh.vertex_colors), np.asarray(loaded.vertex_colors), atol=1/255 + 1e-9)
            or not np.allclose(np.asarray(mesh.vertex_normals), np.asarray(loaded.vertex_normals))):
        raise OSError("Mesh PLY round-trip validation failed")
