"""Check saved UAV production geometry, preview and optional Blender imports."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.sparse.csgraph import connected_components

from analyze_airzoo_geometry import read
from audit_surface_topology import audit
from vision3d.demo.viewer import load_textured_meshes
from vision3d.reconstruction.surface_validation import surface_views


def verify(run, blender_reports):
    workflow = read(run / "workflow.json")
    if workflow["status"] != "success" or any(row["status"] != "success" for row in workflow["stages"].values()):
        raise ValueError("Production workflow did not complete all stages")
    assets = {key: Path(path) for key, path in workflow["assets"].items()}
    if any(not p.is_file() or not p.stat().st_size for p in assets.values()):
        raise ValueError("A recorded artifact is missing or empty")
    sparse = Path(workflow["stages"]["sparse"]["directory"])
    views = surface_views(sparse)
    index = {view.id: i for i, view in enumerate(views)}
    if len(index) != workflow["registered_images"] or len(index) < 3:
        raise ValueError("Registered camera count differs from workflow")
    graph = np.zeros((len(views), len(views)), bool)
    with np.load(sparse / "verified_pairs.npz") as pairs:
        for key in pairs:
            a, b = (int(i) for i in key.split("_"))
            if a in index and b in index and len(pairs[key]):
                graph[index[a], index[b]] = graph[index[b], index[a]] = True
    components = connected_components(graph, directed=False, return_labels=False)
    if components != 1:
        raise ValueError(f"Registered cameras form {components} verified components")
    sparse_metrics = read(sparse / "metrics.json")
    reprojection = sparse_metrics["after_final_ba"]
    if not np.isfinite([reprojection["median_error_px"], reprojection["rms_error_px"]]).all():
        raise ValueError("Sparse projection metrics must be finite")
    cloud = o3d.io.read_point_cloud(str(assets["dense_ply"]))
    points = np.asarray(cloud.points)
    dense = read(run / "dense/metrics.json")
    if (not len(points) or len(points) != workflow["dense_points"] or not np.isfinite(points).all()
            or not cloud.has_colors() or dense["checked_samples"] < len(points)):
        raise ValueError("Invalid dense points or depth support")
    surface = audit(assets["surface_ply"])
    topology = surface["summary"]
    if any(topology[key] for key in ("nonmanifold_edges", "winding_conflicts", "degenerate_triangles", "invalid_vertex_normals")):
        raise ValueError("Optimized mesh failed a topology gate")
    texture = read(run / "texture/metrics.json")
    photo = texture["texture"]
    meshes = load_textured_meshes(assets["textured_glb"])
    if sum(len(mesh.triangles) for mesh in meshes) != photo["textured_faces"] + photo["fallback_faces"]:
        raise ValueError("Desktop preview did not retain the exported faces")
    for mesh in meshes:
        if mesh.has_triangle_uvs():
            uv = np.asarray(mesh.triangle_uvs)
            if not np.isfinite(uv).all() or (uv < -1e-6).any() or (uv > 1 + 1e-6).any():
                raise ValueError("Texture coordinates outside the atlas")
    independent = {}
    for path in blender_reports:
        report = read(path)
        asset = Path(report["asset"]).resolve()
        if report["status"] != "passed" or asset not in {p.resolve() for p in assets.values()}:
            raise ValueError("Blender report must pass and refer to this workflow's asset")
        independent[asset.suffix] = str(path.resolve())
    return {"status": "passed", "registered_images": len(views), "registered_graph_components": int(components),
            "projection": reprojection, "dense_points": len(points), "supported_samples": dense["checked_samples"],
            "dense_bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()], "surface": surface,
            "texture_area_fraction": photo["textured_area_fraction"], "desktop_preview_mesh_groups": len(meshes),
            "blender_reports": independent, "bytes": {key: p.stat().st_size for key, p in assets.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--blender-reports", nargs="+", default=[], type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run, args.blender_reports)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
