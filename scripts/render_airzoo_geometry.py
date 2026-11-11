"""Render imported UAV surfaces from a shared gravity-aligned frame in Blender."""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector


def render(paths, frame, output, resolution):
    basis = np.asarray(frame["basis"])
    origin = np.asarray(frame["origin"])
    output.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    paths = [Path(path).resolve() for path in paths]

    def import_mesh(path):
        if path.suffix.lower() != ".ply":
            raise ValueError("Use the original-coordinate PLY for a controlled geometry comparison")
        bpy.ops.wm.ply_import(filepath=str(path), forward_axis="Y", up_axis="Z")
        objects = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
        if not objects:
            raise ValueError(f"Blender did not import any meshes from {path}")
        for obj in objects:
            world = obj.matrix_world.copy()
            xyz = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
            obj.data.vertices.foreach_get("co", xyz)
            xyz = xyz.reshape(-1, 3) @ np.asarray(world.to_3x3()).T + np.asarray(world.translation)
            obj.matrix_world = Matrix.Identity(4)
            obj.data.vertices.foreach_set("co", np.asarray((xyz - origin) @ basis, dtype=np.float32).ravel())
            obj.data.update()
        return objects

    scenes = [import_mesh(path) for path in paths]
    baseline = []
    for obj in scenes[0]:
        vertices = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", vertices)
        baseline.append(vertices.reshape(-1, 3))
    low, high = np.percentile(np.concatenate(baseline), [.2, 99.8], axis=0)
    target = (high + low) / 2
    scale = np.linalg.norm(high - low) * 1.12
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 6
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 3
    scene.render.resolution_x = resolution
    scene.render.resolution_y = int(resolution * .75)
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.28, .28, .28, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = .65
    scene.view_settings.view_transform = "Standard"
    camera = bpy.data.objects.new("Comparison camera", bpy.data.cameras.new("Orthographic"))
    scene.collection.objects.link(camera)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = scale
    camera.data.clip_start = scale / 1000
    camera.data.clip_end = scale * 20
    scene.camera = camera
    sun = bpy.data.objects.new("Directional light", bpy.data.lights.new("Directional light", "SUN"))
    sun.data.energy = 1.1
    sun.rotation_euler = (.45, -.4, 0)
    scene.collection.objects.link(sun)

    metrics = []
    for index, (path, objects) in enumerate(zip(paths, scenes, strict=True)):
        for group in scenes:
            for obj in group:
                obj.hide_render = group is not objects
        for obj in objects:
            material = bpy.data.materials.new("PLY vertex color")
            material.use_nodes = True
            color = obj.data.color_attributes.active_color
            if color is not None:
                node = material.node_tree.nodes.new("ShaderNodeVertexColor")
                node.layer_name = color.name
                material.node_tree.links.new(node.outputs["Color"], material.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
            obj.data.materials.clear()
            obj.data.materials.append(material)
        names = []
        for label, offset, up in (("top", (0, 0, 1), "Y"), ("oblique", (0, -1, .64), "Y"),
                                  ("side", (0, -1, .025), "Y")):
            camera.location = Vector(target) + Vector(offset).normalized() * scale * 2
            camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", up).to_euler()
            scene.render.filepath = str((output / f"{index}-{label}.png").resolve())
            bpy.ops.render.render(write_still=True)
            names.append(Path(scene.render.filepath).name)
        metrics.append({"input": str(path), "objects": len(objects), "renders": names,
                        "vertices": sum(len(obj.data.vertices) for obj in objects),
                        "faces": sum(len(obj.data.polygons) for obj in objects)})
    (output / "render_report.json").write_text(json.dumps({
        "assets": metrics, "blender": bpy.app.version_string, "target": target.tolist(),
        "ortho_scale": scale, "basis": basis.tolist(), "origin": origin.tolist(),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, action="append")
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=800)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    render(args.mesh, json.loads(args.frame.read_text(encoding="utf-8")), args.output, args.resolution)
