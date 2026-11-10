"""Audit component extent and bounded local intersection checks on a PLY surface."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from vision3d.reconstruction.surface_validation import mesh_arrays, topology


def local_neighborhoods(xyz, faces):
    center = (xyz[faces[:, 0]] + xyz[faces[:, 1]] + xyz[faces[:, 2]]) / 3
    low, high = np.percentile(center, [1, 99], axis=0)
    cells = np.floor((center - low) / np.maximum(high - low, 1e-12) * 8).clip(0, 7).astype(int)
    ids = cells[:, 0] * 64 + cells[:, 1] * 8 + cells[:, 2]
    count = np.bincount(ids, minlength=512)
    for cell in np.argsort(-count, kind="stable")[:8]:
        rows = np.flatnonzero(ids == cell)
        if not len(rows):
            continue
        if len(rows) > 12000:
            # Keep one compact neighborhood so potential colliding pairs remain together.
            distance = np.linalg.norm(center[rows] - np.median(center[rows], axis=0), axis=1)
            rows = rows[np.argsort(distance, kind="stable")[:12000]]
        yield cell, rows


def triangle_patch(xyz, faces):
    ids, indices = np.unique(faces.ravel(), return_inverse=True)
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz[ids]),
                                    o3d.utility.Vector3iVector(indices.reshape(-1, 3)))


def audit(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    xyz, faces = mesh_arrays(mesh)
    labels, summary = topology(mesh)
    sizes = np.bincount(labels)
    components = []
    for component in np.argsort(-sizes, kind="stable")[:8]:
        points = xyz[np.unique(faces[labels == component])]
        components.append({"triangles": int(sizes[component]), "extent": np.ptp(points, axis=0).tolist()})
    local = []
    for cell, rows in local_neighborhoods(xyz, faces):
        patch = triangle_patch(xyz, faces[rows])
        local.append({"cell": int(cell), "triangles": len(rows), "intersection_detected": patch.is_self_intersecting()})
    return {"summary": summary, "largest_components": components, "bounded_intersection_probes": local,
            "scope": "At most eight 12000-triangle neighborhoods. A negative probe does not certify the mesh."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.mesh)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
