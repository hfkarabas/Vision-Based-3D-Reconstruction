"""Create a restartable photographic asset from a validated surface directory."""
import json
from pathlib import Path
from time import perf_counter

import open3d as o3d

from .pair import write_json
from .surface_texture import bake_texture
from .surface_validation import surface_views
from .textured_export import export_textured_glb, export_textured_obj


def build_texture_asset(surface_dir, output, atlas_size=4096, holdout_stride=0, units_per_world=1.,
                        stage=lambda message: None, front_faces_only=False, obj=False):
    start = perf_counter()
    surface_dir, output = Path(surface_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new texture output directory: {output}")
    result = json.loads((surface_dir / "metrics.json").read_text(encoding="utf-8"))
    if result.get("status") != "success" or "geometry" not in result:
        raise ValueError("Expected a successful surface-asset run")
    views = surface_views(result["sparse_dir"])
    mesh = o3d.io.read_triangle_mesh(str(surface_dir / "mesh.ply"))
    atlas = bake_texture(mesh, views, atlas_size, holdout_stride, stage=stage, front_faces_only=front_faces_only)
    stage("Exporting and independently reloading the photographic GLB")
    output.mkdir(parents=True, exist_ok=False)
    encoded = export_textured_glb(mesh, atlas, output / "textured.glb", units_per_world=units_per_world)
    (output / "atlas.jpg").write_bytes(encoded)
    if obj:
        stage("Exporting and independently reloading the photographic OBJ")
        export_textured_obj(mesh, atlas, output / "textured.obj", units_per_world=units_per_world)
    report = {"status": "success", "surface_dir": str(surface_dir.resolve()),
              "export_units_per_sfm_unit": units_per_world, "metric_scale_verified": False,
              "texture": atlas.metrics, "atlas_bytes": len(encoded),
              "obj_bytes": (output / "textured.obj").stat().st_size if obj else None,
              "glb_bytes": (output / "textured.glb").stat().st_size, "runtime_seconds": perf_counter() - start,
              "peak_memory_bytes": None}
    try:
        import psutil
        report["peak_memory_bytes"] = getattr(psutil.Process().memory_info(), "peak_wset", None)
    except (ImportError, OSError):
        pass
    write_json(output / "metrics.json", report)
    return report
