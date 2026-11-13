"""Measure evaluated terrain support using the existing Blender BVH ray operation."""

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from omega_building_blender import meshes, terrain_query


def main(args):
    if args.scene.suffix.lower() == ".glb":
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(args.scene.resolve()))
    else:
        bpy.ops.wm.open_mainfile(filepath=str(args.scene.resolve()))
    terrain = [obj for obj in meshes() if not obj.get("building_id")]
    if not terrain:
        raise ValueError("There is no terrain to measure.")
    query = terrain_query(terrain)
    with np.load(args.samples) as sample:
        points = sample["points"]
    values = [query(float(east), float(north)) for east, north in points]
    result = np.array([float("nan") if value is None else value for value in values], dtype=np.float32)
    np.save(args.output, result)
    if args.preview:
        scene = bpy.context.scene
        scene.render.filepath = str(args.preview.resolve())
        bpy.ops.render.render(write_still=True)
    if args.foundation_preview:
        buildings = [obj for obj in meshes() if obj.get("building_id")]
        if not buildings:
            raise ValueError("There is no verified building for a foundation close-up.")
        scene = bpy.context.scene
        building = sorted(buildings, key=lambda obj: obj["building_id"])[0]
        corners = np.array([list(building.matrix_world @ Vector(corner)) for corner in building.bound_box])
        center = (corners.min(axis=0) + corners.max(axis=0)) / 2
        center[2] = corners[:, 2].min() + 1
        extent = max(np.ptp(corners, axis=0))
        camera = scene.camera
        camera.location = Vector(center) + Vector((1, -1, 0.55)).normalized() * (4 * extent)
        camera.rotation_euler = (Vector(center) - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = 1.8 * extent
        scene.render.filepath = str(args.foundation_preview.resolve())
        bpy.ops.render.render(write_still=True)
    print(json.dumps(dict(scene=str(args.scene), points=len(points), hits=int(np.isfinite(result).sum()))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "samples", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--foundation-preview", type=Path)
    main(parser.parse_args(sys.argv[sys.argv.index("--") + 1:]))
