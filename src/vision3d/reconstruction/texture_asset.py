"""Create a restartable photographic asset from a validated surface directory."""
import json
from pathlib import Path
from time import perf_counter

import open3d as o3d

from .pair import write_json
from .surface_subdivision import subdivide_visible_candidates
from .surface_texture import bake_texture
from .surface_validation import surface_views
from .textured_export import export_textured_glb, export_textured_obj, prepare_texture_mesh


def build_texture_asset(surface_dir, output, atlas_size=4096, holdout_stride=0, units_per_world=1.,
                        stage=lambda message: None, front_faces_only=False, obj=False, refine_visible_faces=0):
    start = perf_counter()
    surface_dir, output = Path(surface_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new texture output directory: {output}")
    if not 0 <= refine_visible_faces <= 100000:
        raise ValueError("Visible face refinement budget must be between 0 and 100000")
    result = json.loads((surface_dir / "metrics.json").read_text(encoding="utf-8"))
    if result.get("status") != "success" or "geometry" not in result:
        raise ValueError("Expected a successful surface-asset run")
    views = surface_views(result["sparse_dir"])
    mesh = o3d.io.read_triangle_mesh(str(surface_dir / "mesh.ply"))
    mesh, export_geometry = prepare_texture_mesh(mesh, units_per_world)
    atlas = bake_texture(mesh, views, atlas_size, holdout_stride, stage=stage, front_faces_only=front_faces_only,
                         track_partial=bool(refine_visible_faces))
    refinement = None
    if refine_visible_faces and atlas.partially_visible.any():
        stage("Subdividing partially visible faces along their existing surface")
        refined, split = subdivide_visible_candidates(mesh, atlas.partially_visible, refine_visible_faces)
        refined, cleanup = prepare_texture_mesh(refined, units_per_world)
        if any(cleanup[name] for name in ("duplicate_faces_removed", "rounded_zero_area_faces_removed",
                                          "nonmanifold_faces_removed", "winding_faces_removed")):
            raise ValueError("Refined surface is not representable without face changes at export precision")
        projected = bake_texture(refined, views, atlas_size, holdout_stride, stage=stage, front_faces_only=front_faces_only)
        previous = atlas.metrics["textured_area_fraction"]
        current = projected.metrics["textured_area_fraction"]
        refinement = {"parent_faces_split": split, "input_triangles": len(mesh.triangles),
                      "refined_triangles": len(refined.triangles), "previous_area_fraction": previous,
                      "refined_area_fraction": current, "accepted": current > previous}
        if refinement["accepted"]:
            mesh, atlas = refined, projected
    stage("Exporting and independently reloading the photographic GLB")
    output.mkdir(parents=True, exist_ok=False)
    encoded = export_textured_glb(mesh, atlas, output / "textured.glb", units_per_world=units_per_world)
    (output / "atlas.jpg").write_bytes(encoded)
    if obj:
        stage("Exporting and independently reloading the photographic OBJ")
        export_textured_obj(mesh, atlas, output / "textured.obj", units_per_world=units_per_world)
    report = {"status": "success", "surface_dir": str(surface_dir.resolve()),
              "export_units_per_sfm_unit": units_per_world, "metric_scale_verified": False,
              "export_geometry": export_geometry,
              "visibility_refinement": refinement,
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
