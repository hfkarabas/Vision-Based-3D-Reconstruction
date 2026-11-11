"""Compare grid triangulation with ball pivoting on frozen raw-mesh patches."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import open3d as o3d

from audit_surface_topology import local_neighborhoods, triangle_patch
from vision3d.reconstruction.surface_validation import mesh_arrays, sampled_deviation, topology


def pivot(patch, voxel, camera_center):
    cloud = o3d.geometry.PointCloud(patch.vertices)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=20))
    cloud.orient_normals_towards_camera_location(camera_center)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cloud, o3d.utility.DoubleVector([1.25 * voxel, 2.25 * voxel]))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def measure(mesh):
    report = topology(mesh)[1]
    report["intersection_pairs"] = len(mesh.get_self_intersecting_triangles())
    report["intersections_per_unit_area"] = report["intersection_pairs"] / report["area"]
    return report


def experiment(dense, output):
    start = perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    baseline = o3d.io.read_triangle_mesh(str(dense / "mesh.ply"))
    xyz, faces = mesh_arrays(baseline)
    voxel = json.loads((dense / "metrics.json").read_text(encoding="utf-8"))["voxel_size"]
    poses = json.loads((dense / "poses.json").read_text(encoding="utf-8"))
    cameras = []
    for record in poses:
        if record["status"] == "registered":
            transform = np.asarray(record["world_to_camera"])
            cameras.append(-transform[:3, :3].T @ transform[:3, 3])
    comparisons = []
    for cell, rows in local_neighborhoods(xyz, faces):
        patch = triangle_patch(xyz, faces[rows])
        center = np.asarray(patch.vertices).mean(axis=0)
        camera = min(cameras, key=lambda camera: np.linalg.norm(camera - center))
        candidate = pivot(patch, voxel, camera)
        result = {"cell": int(cell), "baseline": measure(patch)}
        if candidate.has_triangles():
            result["candidate"] = measure(candidate)
            result["baseline_to_candidate"] = sampled_deviation(patch, candidate, samples=6000)
            result["candidate_to_baseline"] = sampled_deviation(candidate, patch, samples=6000)
            o3d.io.write_triangle_mesh(str(output / f"cell-{cell}-pivot.ply"), candidate)
        else:
            result["reason"] = "Ball pivoting found no surface"
        o3d.io.write_triangle_mesh(str(output / f"cell-{cell}-grid.ply"), patch)
        comparisons.append(result)
    report = {"voxel_size": voxel, "radii_in_voxels": [1.25, 2.25], "comparisons": comparisons,
              "elapsed_seconds": perf_counter() - start,
              "scope": "Fixed compact patches and identical input vertices. Candidate triangles lack per-grid "
                       "observation provenance. These are local diagnostics, not whole-mesh intersection counts."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment(args.dense, args.output)
