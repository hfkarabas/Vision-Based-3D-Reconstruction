"""Open surfaces from validated disparity samples and round-trip mesh exports."""
import numpy as np


def build_surface(xyz, colors, triangles, target_faces=None):
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
    mesh.remove_triangles_by_mask(area <= 1e-14)
    if not mesh.has_triangles():
        raise ValueError("Surface triangles have negligible area")
    labels, sizes, _ = mesh.cluster_connected_triangles()
    minimum = max(32, int(max(sizes) * .0002))
    mesh.remove_triangles_by_mask(np.asarray(sizes)[labels] < minimum)
    mesh.remove_unreferenced_vertices()
    if not mesh.has_triangles():
        raise ValueError("No connected surface with at least 32 triangles; try better overlap or a denser profile")
    if target_faces and len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if not mesh.has_triangles() or not np.isfinite(np.asarray(mesh.vertex_normals)).all():
        raise ValueError("Mesh cleanup did not produce a finite surface")
    return mesh, {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
                  "input_faces": len(triangles), "component_min_faces": minimum,
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
