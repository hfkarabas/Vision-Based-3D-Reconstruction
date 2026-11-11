"""Reject cross-view grid faces whose three corners contradict two independent normals."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from audit_grid_intersections import face_keys
from audit_surface_topology import local_neighborhoods
from experiment_surface_layer_pruning import collisions
from vision3d.reconstruction.surface import write_mesh_ply
from vision3d.reconstruction.surface_validation import face_geometry, mesh_arrays, sampled_deviation, topology


def vertex_normals(points, faces):
    _, cross, area = face_geometry(points, faces)
    ids, inverse = np.unique(faces, return_inverse=True)
    keys = inverse.ravel()
    vectors = np.column_stack([np.bincount(keys, weights=np.repeat(axis, 3), minlength=len(ids)) for axis in cross.T])
    mass = np.bincount(keys, weights=np.repeat(2 * area, 3), minlength=len(ids))
    lengths = np.linalg.norm(vectors, axis=1)
    coherent = (mass > 0) & (lengths >= .8 * mass)
    return ids[coherent], (vectors[coherent] / lengths[coherent, None]).astype(np.float32)


def supported_grid_masks(points, grids, edges):
    normals = [vertex_normals(points, grid) for grid in grids]
    conflicts = [np.zeros(len(ids), bool) for ids, _ in normals]
    agree, disagree = np.cos(np.deg2rad(25)), np.cos(np.deg2rad(45))
    for i, (a, u) in enumerate(normals):
        for j in range(i + 1, len(normals)):
            if edges[i] & edges[j]:
                continue
            b, v = normals[j]
            common, ia, ib = np.intersect1d(a, b, assume_unique=True, return_indices=True)
            coherent = abs(np.einsum("ij,ij->i", u[ia], v[ib])) >= agree
            common, ua, vb = common[coherent], u[ia[coherent]], v[ib[coherent]]
            if not len(common):
                continue
            for k, (ids, candidate) in enumerate(normals):
                if k in (i, j):
                    continue
                _, pair_idx, row = np.intersect1d(common, ids, assume_unique=True, return_indices=True)
                opposed = ((abs(np.einsum("ij,ij->i", ua[pair_idx], candidate[row])) <= disagree) &
                           (abs(np.einsum("ij,ij->i", vb[pair_idx], candidate[row])) <= disagree))
                conflicts[k][row[opposed]] = True
    masks = []
    for grid, (ids, _), conflict in zip(grids, normals, conflicts, strict=True):
        masks.append(~np.isin(grid, ids[conflict]).all(axis=1))
    return masks


def experiment(dense, output):
    start = perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    root = dense / "depth-observations"
    source = {x["reference"]: x["source"] for x in json.loads((root / "manifest.json").read_text())["pairs"]}
    with np.load(root / "grid-faces.npz") as data:
        grids = np.split(data["triangles"], np.cumsum(data["counts"])[:-1])
        refs = data["references"]
    cloud = o3d.io.read_point_cloud(str(dense / "dense.ply"))
    points = np.asarray(cloud.points)
    masks = supported_grid_masks(points, grids, [{int(i), source[int(i)]} for i in refs])
    removed = {int(i): int((~mask).sum()) for i, mask in zip(refs, masks, strict=True)}
    report = {"removed_grid_faces_by_reference": removed, "removed_grid_faces": sum(removed.values())}
    if not report["removed_grid_faces"]:
        report["status"] = "no_corroborated_normal_conflicts"
    else:
        before = o3d.io.read_triangle_mesh(str(dense / "mesh.ply"))
        before_xyz, before_faces = mesh_arrays(before)
        distance, index = cKDTree(points).query(before_xyz, workers=1)
        if (distance > 1e-10).any():
            raise ValueError("Only unmodified raw mesh corners support this experiment")
        supported = face_keys(np.concatenate([grid[mask] for grid, mask in zip(grids, masks, strict=True)]))
        keep = np.isin(face_keys(index[before_faces]), supported)
        after = o3d.geometry.TriangleMesh(before)
        after.remove_triangles_by_mask(~keep)
        after.remove_unreferenced_vertices()
        after.compute_vertex_normals()
        report["removed_output_faces"] = int((~keep).sum())
        probes = []
        for cell, rows in local_neighborhoods(before_xyz, before_faces):
            kept = rows[keep[rows]]
            _, _, area = face_geometry(before_xyz, before_faces[rows])
            probes.append({"cell": int(cell), "before": len(collisions(before_xyz, before_faces, rows)),
                           "after": len(collisions(before_xyz, before_faces, kept)) if len(kept) else 0,
                           "baseline_area": float(area.sum())})
        old, new = topology(before)[1], topology(after)[1]
        report.update(status="experimental", baseline=old, candidate=new, frozen_intersection_probes=probes,
                      area_retention=new["area"] / old["area"],
                      baseline_to_candidate=sampled_deviation(before, after))
        write_mesh_ply(output / "candidate.ply", after)
    report["runtime_seconds"] = perf_counter() - start
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment(args.dense, args.output)
