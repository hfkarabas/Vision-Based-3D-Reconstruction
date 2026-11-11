"""Independently render a photographic GLB with the frozen audit cameras."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_inspect_asset import inspect_materials


def render(asset, frame_path, size, output):
    frame = json.loads(frame_path.read_text(encoding="utf-8"))
    basis, center, span = np.asarray(frame["basis"]), Vector(frame["center"]), frame["span"]
    output.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(asset.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    to_sfm = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    for obj in meshes:
        obj.matrix_world = to_sfm @ obj.matrix_world
    atlas = hashlib.sha256((asset.parent / "atlas.jpg").read_bytes()).hexdigest()
    for material, (_, base, surface) in inspect_materials(meshes, atlas).items():
        tree = material.node_tree
        emit = tree.nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value = base.default_value
        if base.is_linked:
            tree.links.new(base.links[0].from_socket, emit.inputs["Color"])
        tree.links.new(emit.outputs[0], surface)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device, scene.cycles.samples = "CPU", 8
    scene.render.threads_mode, scene.render.threads = "FIXED", 3
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    camera = bpy.data.objects.new("Fixed audit camera", bpy.data.cameras.new("Orthographic"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type, camera.data.ortho_scale = "ORTHO", span
    camera.data.clip_start, camera.data.clip_end = span / 10000, span * 10
    for name, direction in (("top", [0, 0, 1]), ("oblique", [0, -1, .82]), ("side", [0, -1, .07])):
        front = Vector(basis @ direction).normalized()
        right = Vector(basis[:, 1] if direction[1] == 0 else basis[:, 2]).cross(front).normalized()
        up = front.cross(right)
        camera.matrix_world = Matrix((right, up, front)).transposed().to_4x4()
        camera.location = center + 2 * span * front
        scene.render.filepath = str((output / f"{name}.png").resolve())
        bpy.ops.render.render(write_still=True)
    (output / "report.json").write_text(json.dumps({"asset": str(asset.resolve()), "frame": frame,
        "blender": bpy.app.version_string, "faces": sum(len(obj.data.polygons) for obj in meshes)}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    render(args.asset, args.frame, args.size, args.output)
