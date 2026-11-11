"""Topology, sampled geometric deviation and camera-facing mesh orientation."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from vision3d.geometry.camera import Camera, parse_camera
from .state import Pose


@dataclass(frozen=True)
class SurfaceView:
    id: int
    pose: Pose
    camera: Camera
    source: Path
    sha256: str


def surface_views(sparse_dir):
    def read(name):
        return json.loads((Path(sparse_dir) / name).read_text(encoding="utf-8"))

    sources = {entry["image_id"]: entry for entry in read("manifest.json")["inputs"]}
    cameras = {entry["image_id"]: parse_camera(entry["original"]) for entry in read("intrinsics.json")}
    views = []
    for entry in read("poses.json"):
        if entry["status"] != "registered":
            continue
        i = entry["image_id"]
        matrix = np.asarray(entry["world_to_camera"], dtype=float)
        if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("Invalid world-to-camera matrix")
        source = sources[i]
        path = Path(source["source"])
        views.append(SurfaceView(i, Pose(matrix[:3, :3], matrix[:3, 3]), cameras[i], path, source["sha256"]))
    if not views:
        raise ValueError("The sparse reconstruction has no registered cameras")
    return views


def mesh_arrays(mesh):
    xyz, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if (not len(xyz) or not len(faces) or not np.isfinite(xyz).all()
            or faces.min() < 0 or faces.max() >= len(xyz)):
        raise ValueError("Surface must contain finite vertices and valid indexed triangles")
    return xyz, faces


def face_geometry(xyz, faces):
    a, b, c = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    return (a + b + c) / 3, cross, np.linalg.norm(cross, axis=1) / 2


def topology(mesh):
    xyz, faces = mesh_arrays(mesh)
    n = len(xyz)
    edge_keys = np.empty(faces.size, dtype=np.int64)
    for i in range(3):
        a, b = faces[:, i], faces[:, (i + 1) % 3]
        # The low bit records orientation relative to ascending vertex IDs.
        edge_keys[i * len(faces):(i + 1) * len(faces)] = (
            2 * (np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)) + (a > b))
    edge_keys.sort()
    _, first, count = np.unique(edge_keys // 2, return_index=True, return_counts=True)
    shared = first[count == 2]
    _, _, area = face_geometry(xyz, faces)
    labels, sizes, component_areas = mesh.cluster_connected_triangles()
    normals = np.asarray(mesh.vertex_normals)
    norm = np.linalg.norm(normals, axis=1)
    return np.asarray(labels), {
        "vertices": n, "triangles": len(faces), "components": len(sizes),
        "largest_component_triangles": int(max(sizes)), "area": float(sum(component_areas)),
        "boundary_edges": int((count == 1).sum()), "nonmanifold_edges": int((count > 2).sum()),
        "winding_conflicts": int((edge_keys[shared] == edge_keys[shared + 1]).sum()),
        "degenerate_triangles": int((area <= 0).sum()),
        "invalid_vertex_normals": int((~np.isfinite(norm) | (norm < 1e-7)).sum()) if len(norm) == n else n,
    }


def ray_scene(mesh):
    xyz, faces = mesh_arrays(mesh)
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.core.Tensor(xyz, dtype=o3d.core.Dtype.Float32),
                        o3d.core.Tensor(faces, dtype=o3d.core.Dtype.UInt32))
    return scene


def visible_points(scene, points, center):
    """A ray aimed at a surface point should first hit at unit ray parameter."""
    if not len(points):
        return np.empty(0, dtype=bool)
    rays = np.empty((len(points), 6), np.float32)
    rays[:, :3] = center
    rays[:, 3:] = points - center
    hits = scene.cast_rays(o3d.core.Tensor(rays), nthreads=2)["t_hit"].numpy()
    return np.isfinite(hits) & (np.abs(hits - 1) < 2e-4)


def pinhole_projection(points, view):
    local = points @ view.pose.rotation.T + view.pose.translation
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = local[:, :2] / local[:, 2:3] * [view.camera.fx, view.camera.fy] + [view.camera.cx, view.camera.cy]
    return uv, local[:, 2]


def orient_to_cameras(mesh, views):
    xyz, faces = mesh_arrays(mesh)
    labels, sizes, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    # Sample by index within each component, keeping even small islands eligible.
    ordered = np.argsort(labels, kind="stable")
    ends = np.cumsum(sizes)
    rows = np.concatenate([ordered[end - size:end:max(1, int(np.ceil(size / 96)))]
                           for end, size in zip(ends, sizes)])
    centers, cross, area = face_geometry(xyz, faces[rows])
    normals = cross / np.maximum(2 * area[:, None], 1e-30)
    scores = np.zeros(len(sizes))
    support = np.zeros(len(sizes))
    scene = ray_scene(mesh)
    for view in views:
        uv, z = pinhole_projection(centers, view)
        ray = view.pose.center - centers
        dot = np.sum(ray * normals, axis=1) / np.maximum(np.linalg.norm(ray, axis=1), 1e-30)
        inside = ((uv >= 0) & (uv < [view.camera.width - 1, view.camera.height - 1])).all(axis=1)
        rows_in_view = np.flatnonzero((z > 0) & inside & (np.abs(dot) >= .25))
        visible = rows_in_view[visible_points(scene, centers[rows_in_view], view.pose.center)]
        np.add.at(scores, labels[rows[visible]], dot[visible])
        np.add.at(support, labels[rows[visible]], np.abs(dot[visible]))
    resolved = (support >= 2) & (np.abs(scores) >= support / 3)
    flip = resolved & (scores < 0)
    faces[flip[labels]] = faces[flip[labels]][:, [0, 2, 1]]
    cancelled = update_surface_normals(mesh)
    return {"camera_resolved_components": int(resolved.sum()), "camera_flipped_components": int(flip.sum()),
            "faces_with_camera_resolved_orientation": int(resolved[labels].sum()),
            "faces_flipped": int(flip[labels].sum()), "cancelled_vertex_normals_recovered": cancelled}


def update_surface_normals(mesh):
    xyz, faces = mesh_arrays(mesh)
    mesh.compute_vertex_normals()
    vertex_normals = np.asarray(mesh.vertex_normals)
    cancelled = np.flatnonzero(np.linalg.norm(vertex_normals, axis=1) < 1e-7)
    for vertex in cancelled:
        incident = faces[(faces == vertex).any(axis=1)]
        _, cross, area = face_geometry(xyz, incident)
        largest = np.argmax(area)
        vertex_normals[vertex] = cross[largest] / (2 * area[largest])
    return len(cancelled)


def sampled_deviation(source, destination, samples=30000):
    xyz, faces = mesh_arrays(source)
    _, cross, area = face_geometry(xyz, faces)
    if not np.any(area > 0):
        raise ValueError("Surface deviation requires positive triangle area")
    rng = np.random.default_rng(0)
    rows = rng.choice(len(faces), size=samples, p=area / area.sum())
    a, b = rng.random((2, samples))
    a, b = np.where(a + b > 1, 1 - a, a), np.where(a + b > 1, 1 - b, b)
    vertices = xyz[faces[rows]]
    points = vertices[:, 0] + a[:, None] * (vertices[:, 1] - vertices[:, 0]) + b[:, None] * (vertices[:, 2] - vertices[:, 0])
    hit = ray_scene(destination).compute_closest_points(o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32), nthreads=2)
    delta = np.linalg.norm(points - hit["points"].numpy(), axis=1)
    cosine = np.abs(np.sum(cross[rows] / (2 * area[rows, None]) * hit["primitive_normals"].numpy(), axis=1))
    return {"samples": samples, "distance_p50_p95_p99": np.percentile(delta, [50, 95, 99]).tolist(),
            "distance_max": float(delta.max()),
            "unsigned_normal_angle_p50_p95_deg": np.degrees(np.arccos(np.percentile(cosine.clip(0, 1), [50, 5]))).tolist()}


def paired_edges(faces, vertex_count):
    """Return directed edge endpoints and the slots for each exactly two-face edge."""
    a, b = faces.ravel(), faces[:, [1, 2, 0]].ravel()
    keys = np.minimum(a, b).astype(np.int64) * vertex_count + np.maximum(a, b)
    order = np.argsort(keys)
    _, first, count = np.unique(keys[order], return_index=True, return_counts=True)
    first = first[count == 2]
    return a, b, order[first], order[first + 1]


def repair_winding(mesh, geometry_weighted=False):
    """Solve shared-edge winding constraints and cut faces on contradictory cycles."""
    xyz, faces = mesh_arrays(mesh)
    a, b, left, right = paired_edges(faces, len(xyz))
    parity = (a[left] > b[left]) == (a[right] > b[right])
    face_a, face_b = left // 3, right // 3
    _, cross, areas = face_geometry(xyz, faces)
    forest = np.arange(len(left))
    if geometry_weighted:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree

        normals = cross / np.maximum(2 * areas[:, None], 1e-30)
        agreement = np.sum(normals[face_a] * normals[face_b], axis=1) * np.where(parity, -1, 1)
        length = np.linalg.norm(xyz[a[left]] - xyz[b[left]], axis=1)
        # Use the strongest geometric ties first, with low priority for folds.
        weight = length * (.05 + .95 * ((agreement.clip(-1, 1) + 1) / 2) ** 2)
        graph = csr_matrix((-weight, (face_a, face_b)), shape=(len(faces), len(faces)))
        tree = minimum_spanning_tree(graph).tocoo()
        edge_id = face_a.astype(np.int64) * len(faces) + face_b
        tree_id = tree.row.astype(np.int64) * len(faces) + tree.col
        forest = np.flatnonzero(np.isin(edge_id, tree_id))
    adjacent = np.full(faces.size, -1, np.int32)
    opposite = np.zeros(faces.size, dtype=bool)
    first, last = left[forest], right[forest]
    adjacent[first], adjacent[last] = last // 3, first // 3
    opposite[first], opposite[last] = parity[forest], parity[forest]
    state = np.full(len(faces), -1, np.int8)
    for root in range(len(faces)):
        if state[root] >= 0:
            continue
        state[root] = 0
        queue = [root]
        for current in queue:
            for edge in range(3 * current, 3 * current + 3):
                neighbor = adjacent[edge]
                if neighbor >= 0 and state[neighbor] < 0:
                    state[neighbor] = int(state[current]) ^ int(opposite[edge])
                    queue.append(int(neighbor))
    broken = (state[face_a] != state[face_b]) != parity
    ends = np.column_stack((face_a[broken], face_b[broken]))
    cut_cost = areas[ends]
    if geometry_weighted:
        degree = np.bincount(ends.ravel(), minlength=len(faces))
        cut_cost /= degree[ends]
    cut = np.unique(ends[np.arange(len(ends)), np.argmin(cut_cost, axis=1)])
    flip = state == 1
    faces[flip] = faces[flip][:, [0, 2, 1]]
    if len(cut):
        remove = np.zeros(len(faces), dtype=bool)
        remove[cut] = True
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()
    return {"winding_faces_flipped": int(flip.sum()), "contradictory_faces_cut": len(cut),
            "contradictory_area_removed": float(areas[cut].sum()),
            "geometry_weighted": geometry_weighted, "contradictory_edges": int(broken.sum())}


def optimize_surface(mesh, views, target_faces, max_error, geometry_weighted=False):
    """Try quadric decimation and accept only bounded, topology-checked surfaces."""
    if target_faces < 100 or not np.isfinite(max_error) or max_error <= 0:
        raise ValueError("Expected at least 100 target faces and positive maximum deviation")
    _, before = topology(mesh)
    if before["triangles"] > target_faces:
        candidate = mesh.simplify_quadric_decimation(target_faces)
    else:
        candidate = o3d.geometry.TriangleMesh(mesh)
    candidate.remove_degenerate_triangles()
    candidate.remove_duplicated_triangles()
    candidate.remove_unreferenced_vertices()
    winding = repair_winding(candidate, geometry_weighted=geometry_weighted)
    orientation = {**winding, **orient_to_cameras(candidate, views)}
    _, after = topology(candidate)
    forward, reverse = sampled_deviation(mesh, candidate), sampled_deviation(candidate, mesh)
    if (after["nonmanifold_edges"] > 0 or after["degenerate_triangles"] > 0
            or after["winding_conflicts"] > 0 or after["invalid_vertex_normals"] > 0):
        raise ValueError(f"Optimized surface failed topology validation: {after}")
    if any(r["distance_p50_p95_p99"][2] > max_error or r["distance_max"] > 5 * max_error for r in (forward, reverse)):
        raise ValueError(f"Optimized surface exceeds deviation bound {max_error:g}: {forward}, {reverse}")
    return candidate, {"input": before, "output": after, "orientation": orientation,
                       "input_to_output": forward, "output_to_input": reverse,
                       "max_error_p99": max_error, "max_sample_error": 5 * max_error}
