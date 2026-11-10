"""Build and validate an optimized surface asset from a completed dense run."""
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import open3d as o3d

from .asset_export import export_glb
from .pair import write_json
from .surface import write_mesh_ply
from .surface_validation import optimize_surface, surface_views


def build_surface_asset(dense_dir, output, target_faces=450000, max_error=None, sparse_dir=None,
                        stage=lambda message: None):
    start = perf_counter()
    dense_dir, output = Path(dense_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new surface output directory: {output}")
    dense = json.loads((dense_dir / "metrics.json").read_text(encoding="utf-8"))
    if dense.get("mesh", {}).get("status") != "success":
        raise ValueError("Surface assets require a successful dense mesh")
    sparse_dir = Path(sparse_dir or dense["source_sparse"])
    for name in ("intrinsics.json", "poses.json"):
        saved = json.loads((dense_dir / name).read_text(encoding="utf-8"))
        actual = json.loads((sparse_dir / name).read_text(encoding="utf-8"))
        if saved != actual:
            raise ValueError(f"Sparse input differs from the dense model's {name}")
    views = surface_views(sparse_dir)
    if max_error is None:
        max_error = 2 * dense["voxel_size"]
    stage("Auditing, simplifying and checking the surface against the input mesh")
    original = o3d.io.read_triangle_mesh(str(dense_dir / "mesh.ply"))
    if not original.has_vertex_colors():
        raise ValueError("Surface assets require a colored source mesh")
    mesh, report = optimize_surface(original, views, target_faces, max_error)
    del original
    stage("Exporting and reloading the optimized PLY and GLB")
    output.mkdir(parents=True, exist_ok=False)
    write_mesh_ply(output / "mesh.ply", mesh)
    export_glb(mesh, output / "mesh.glb")
    for name in ("intrinsics.json", "poses.json"):
        shutil.copyfile(dense_dir / name, output / name)
    metrics = {"status": "success", "dense_dir": str(dense_dir.resolve()),
               "sparse_dir": str(sparse_dir.resolve()), "target_faces": target_faces,
               "geometry": report, "registered_views": len(views), "voxel_size": float(dense["voxel_size"]),
               "glb_bytes": (output / "mesh.glb").stat().st_size,
               "runtime_seconds": perf_counter() - start, "peak_memory_bytes": None}
    try:
        import psutil
        metrics["peak_memory_bytes"] = getattr(psutil.Process().memory_info(), "peak_wset", None)
    except (ImportError, OSError):
        pass
    write_json(output / "metrics.json", metrics)
    return metrics
