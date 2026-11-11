"""Trace raw-mesh intersections to the image grids that generated the faces."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from audit_surface_topology import local_neighborhoods
from experiment_surface_layer_pruning import collisions
from vision3d.reconstruction.surface_validation import face_geometry, mesh_arrays


def face_keys(faces):
    return np.ascontiguousarray(np.sort(faces, axis=1), dtype=np.int64).view(np.dtype("V24")).ravel()


def trace(dense):
    mesh = o3d.io.read_triangle_mesh(str(dense / "mesh.ply"))
    xyz, faces = mesh_arrays(mesh)
    neighborhoods = list(local_neighborhoods(xyz, faces))
    rows = np.unique(np.concatenate([indices for _, indices in neighborhoods]))
    entries = [indices[collisions(xyz, faces, indices)] for _, indices in neighborhoods]
    pairs = np.unique(np.sort(np.concatenate(entries), axis=1), axis=0)
    _, _, area = face_geometry(xyz, faces[rows])
    if not len(pairs):
        return {"intersection_pairs": 0, "evaluated_faces": len(rows), "evaluated_area_sfm_squared": float(area.sum())}

    paired_faces = np.unique(pairs)
    vertices, inverse = np.unique(faces[paired_faces], return_inverse=True)
    cloud = o3d.io.read_point_cloud(str(dense / "dense.ply"))
    distance, nearest = cKDTree(np.asarray(cloud.points)).query(xyz[vertices], workers=1)
    # Raw mesh creation normally preserves dense vertices. Refuse to assign a
    # source after smoothing or simplification moved any triangle corner.
    unchanged = (distance[inverse].reshape(-1, 3) <= 1e-10).all(axis=1)
    candidates = face_keys(nearest[inverse].reshape(-1, 3))
    sources = [set() for _ in paired_faces]
    with np.load(dense / "depth-observations/grid-faces.npz") as data:
        grid = data["triangles"]
        offset = 0
        keys, inverse = np.unique(candidates[unchanged], return_inverse=True)
        owners = [set() for _ in keys]
        for reference, count in zip(data["references"], data["counts"], strict=True):
            match = np.intersect1d(keys, face_keys(grid[offset:offset + count]), return_indices=True)[1]
            for position in match:
                owners[position].add(int(reference))
            offset += count
        for i, owner in zip(np.flatnonzero(unchanged), inverse, strict=True):
            sources[i] = owners[owner]

    indexed = np.searchsorted(paired_faces, pairs)
    traced = np.array([bool(sources[i] and sources[j]) for i, j in indexed])
    common = np.array([bool(sources[i] & sources[j]) for i, j in indexed])
    _, normal, _ = face_geometry(xyz, faces[paired_faces])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    cosine = abs(np.einsum("ij,ij->i", normal[indexed[:, 0]], normal[indexed[:, 1]]))
    angle = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
    classes = {}
    for label, selection in (("shared_reference", traced & common), ("different_references", traced & ~common),
                             ("untraced", ~traced)):
        classes[label] = {"pairs": int(selection.sum()), "near_parallel_under_15_degrees": int((selection & (angle < 15)).sum()),
                          "crossing_over_45_degrees": int((selection & (angle > 45)).sum())}
    return {"intersection_pairs": len(pairs), "evaluated_faces": len(rows), "evaluated_area_sfm_squared": float(area.sum()),
            "pairs_per_evaluated_area": float(len(pairs) / area.sum()), "classes": classes,
            "scope": "Eight compact neighborhoods ranked within this mesh. Source membership is exact for unchanged "
                     "fused triangle corners. It identifies where the triangles were created."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(trace(args.dense), handle, indent=2)
        handle.write("\n")
