"""Test whether hidden intersecting faces are covered by observed neighboring geometry."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from audit_surface_topology import local_neighborhoods, triangle_patch
from vision3d.reconstruction.surface_validation import (face_geometry, mesh_arrays, pinhole_projection, ray_scene,
                                                        sampled_deviation, surface_views, topology, visible_points)


def face_visibility(mesh, xyz, faces, rows, views):
    centers, _, _ = face_geometry(xyz, faces[rows])
    samples = np.concatenate((centers[:, None], .86 * xyz[faces[rows]] + .14 * centers[:, None]), axis=1)
    flat = samples.reshape(-1, 3)
    any_hit = np.zeros(len(rows), dtype=bool)
    strong_views = np.zeros(len(rows), dtype=int)
    scene = ray_scene(mesh)
    for view in views:
        uv, depth = pinhole_projection(flat, view)
        valid = (depth > 0) & ((uv >= 1) & (uv < [view.camera.width - 1, view.camera.height - 1])).all(axis=1)
        hit = np.zeros(len(flat), dtype=bool)
        hit[valid] = visible_points(scene, flat[valid], view.pose.center)
        hit = hit.reshape(-1, 4)
        any_hit |= hit.any(axis=1)
        strong_views += hit.all(axis=1)
    return any_hit, strong_views


def supported_faces(xyz, candidate, retained, tolerance):
    if not len(candidate) or not len(retained):
        return np.zeros(len(candidate), dtype=bool)
    scene = ray_scene(triangle_patch(xyz, retained))
    corners = xyz[candidate]
    centers, cross, area = face_geometry(xyz, candidate)
    midpoints = (corners + np.roll(corners, 1, axis=1)) / 2
    samples = np.concatenate((corners, midpoints, centers[:, None]), axis=1)
    flat = samples.reshape(-1, 3)
    match = scene.compute_closest_points(o3d.core.Tensor(flat.astype(np.float32)), nthreads=2)
    delta = np.linalg.norm(flat - match["points"].numpy(), axis=1).reshape(-1, 7)
    normal = cross / np.maximum(2 * area[:, None], 1e-30)
    cosine = np.abs(np.sum(match["primitive_normals"].numpy().reshape(-1, 7, 3) * normal[:, None], axis=2))
    return (delta <= tolerance).all(axis=1) & (cosine >= .96).all(axis=1)


def collisions(xyz, faces, rows):
    return np.asarray(triangle_patch(xyz, faces[rows]).get_self_intersecting_triangles(), dtype=int).reshape(-1, 2)


def experiment(surface_dir):
    data = json.loads((surface_dir / "metrics.json").read_text(encoding="utf-8"))
    mesh = o3d.io.read_triangle_mesh(str(surface_dir / "mesh.ply"))
    xyz, faces = mesh_arrays(mesh)
    neighborhoods = list(local_neighborhoods(xyz, faces))
    probe = [(cell, rows, collisions(xyz, faces, rows)) for cell, rows in neighborhoods]
    intersecting = np.unique(np.concatenate([rows[pairs].ravel() for _, rows, pairs in probe]))
    rows = np.unique(np.concatenate([rows for _, rows in neighborhoods]))
    any_hit, strong_views = face_visibility(mesh, xyz, faces, rows, surface_views(data["sparse_dir"]))
    candidate = np.intersect1d(intersecting, rows[~any_hit])
    retained = rows[strong_views >= 2]
    tolerance = data["voxel_size"] / 2
    removable = candidate[supported_faces(xyz, faces[candidate], faces[retained], tolerance)]
    remove = np.zeros(len(faces), dtype=bool)
    remove[removable] = True
    report = {"intersecting_faces_probed": len(intersecting), "unseen_intersecting_faces": len(candidate),
              "observed_support_faces": len(retained), "removable_faces": len(removable), "world_distance_limit": tolerance,
              "scope": "Eight compact neighborhoods. Retain every face hit at any of four probes by any registered view."}
    report["intersections_before"] = sum(len(pairs) for _, _, pairs in probe)
    if len(removable):
        candidate_mesh = o3d.geometry.TriangleMesh(mesh)
        candidate_mesh.remove_triangles_by_mask(remove)
        candidate_mesh.remove_unreferenced_vertices()
        candidate_mesh.compute_vertex_normals()
        report["intersections_after"] = sum(len(collisions(xyz, faces, rows[~remove[rows]])) for _, rows in neighborhoods)
        report["topology"] = topology(candidate_mesh)[1]
        report["before_to_after"] = sampled_deviation(mesh, candidate_mesh)
    else:
        report["intersections_after"] = report["intersections_before"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = experiment(args.surface_dir)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
