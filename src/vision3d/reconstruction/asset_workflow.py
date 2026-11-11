"""Run the production sparse-to-textured asset stages under one output directory."""
import json
from pathlib import Path
from time import perf_counter, sleep

from .multiview import reconstruct_multiview
from .sequence import reconstruct_sequence
from .surface_asset import build_surface_asset
from .texture_asset import build_texture_asset


def _save_manifest(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for delay in (.02, .05, .1, .2):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            sleep(delay)
    temporary.replace(path)


def reconstruct_asset(output, input_path=None, sparse_dir=None, camera=None, profile="uav", max_images=40,
                      max_dimension=1600, position_hints=None, feature_cache=None, pair_cache=None,
                      target_faces=450000, atlas_size=4096, obj=False, stage=lambda _: None):
    if (input_path is None) == (sparse_dir is None):
        raise ValueError("Select photographs or a saved sparse sequence")
    if sparse_dir is not None and any((camera, position_hints, feature_cache, pair_cache)):
        raise ValueError("Camera and feature options apply to photographs, not saved sparse poses")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    saved = Path(sparse_dir).resolve() if sparse_dir is not None else output / "sparse"
    manifest = output / "workflow.json"
    record = {"status": "running", "profile": profile, "output": str(output), "stages": {},
              "assets": {}, "target_faces": target_faces, "atlas_size": atlas_size}
    active = "sparse"

    def save():
        _save_manifest(manifest, record)

    def run(name, path, operation):
        nonlocal active
        active = name
        stage(f"Starting {name}")
        record["stages"][name] = {"status": "running", "directory": str(path)}
        save()
        start = perf_counter()
        result = operation()
        record["stages"][name].update(status="success", elapsed_seconds=perf_counter() - start)
        save()
        return result

    started = perf_counter()
    try:
        if sparse_dir is None:
            sparse = run("sparse", saved, lambda: reconstruct_sequence(
                input_path=input_path, output=saved, camera=camera, auto_intrinsics=camera is None,
                max_dimension=max_dimension, max_images=max_images, min_registered=3, window=4, long_gap=8,
                recovery_pairs=30, verifier="magsac", ratio=.8, probe_seeds=True, local_ba=True,
                refine_points=True, map_recovery_pairs=30, seed_search=7, seed_search_views=6,
                feature_cache=feature_cache, pair_cache=pair_cache, position_hints=position_hints,
                bridge_pairs=24 if position_hints is not None else 0))
        else:
            sparse = run("sparse", saved, lambda: json.loads((saved / "metrics.json").read_text(encoding="utf-8")))
        if sparse.get("status") != "success" or sparse.get("registered_images", 0) < 3:
            raise ValueError("Sparse reconstruction did not register enough cameras")
        record["registered_images"] = sparse["registered_images"]
        if "input_images" in sparse:
            record["input_images"] = sparse["input_images"]
        dense, surface, texture = (output / name for name in ("dense", "surface", "texture"))
        depth = run("dense", dense, lambda: reconstruct_multiview(saved, dense, profile=profile, mesh=True, stage=stage))
        if (depth.get("mesh") or {}).get("status") != "success":
            raise ValueError(f"Dense fusion completed but mesh generation failed: {depth.get('mesh')}")
        record["dense_points"] = depth["dense_points"]
        run("surface", surface, lambda: build_surface_asset(dense, surface, target_faces=target_faces,
                                                            geometry_weighted_winding=True, stage=stage))
        appearance = run("texture", texture, lambda: build_texture_asset(
            surface, texture, atlas_size=atlas_size, obj=obj, stage=stage))
        record["assets"] = {"dense_ply": str(dense / "dense.ply"), "surface_ply": str(surface / "mesh.ply"),
                            "textured_glb": str(texture / "textured.glb")}
        if obj:
            record["assets"].update(textured_obj=str(texture / "textured.obj"),
                                    material=str(texture / "textured.mtl"), atlas=str(texture / "atlas.jpg"))
        for path in record["assets"].values():
            if not Path(path).is_file() or not Path(path).stat().st_size:
                raise ValueError(f"Missing or empty expected artifact: {path}")
        record["photographic_area_fraction"] = appearance["texture"]["textured_area_fraction"]
        record["status"] = "success"
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                      failed_stage=active, error=str(exc))
        record["stages"].setdefault(active, {})["status"] = record["status"]
        raise
    finally:
        record["elapsed_seconds"] = perf_counter() - started
        save()
    return record
