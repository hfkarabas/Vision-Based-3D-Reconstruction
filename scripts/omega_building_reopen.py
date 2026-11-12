"""Reopen the published native scene and render its saved review camera."""

import argparse
import json
from pathlib import Path
import sys

import bpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    expected = json.loads(args.validation.read_text(encoding="utf-8"))
    bpy.ops.wm.open_mainfile(filepath=str(args.scene.resolve()))
    scene = bpy.context.scene
    buildings = [obj for obj in scene.objects if obj.type == "MESH" and obj.get("building_id")]
    terrain = [obj for obj in scene.objects if obj.type == "MESH" and not obj.get("building_id")]
    images = {
        node.image for obj in terrain for material in obj.data.materials
        for node in material.node_tree.nodes if node.type == "TEX_IMAGE" and node.image
    }
    if (
        len(buildings) != expected["building_objects"]
        or len(terrain) != expected["terrain_objects"]
        or not images
        or any(not image.packed_file for image in images)
        or any(not obj.data.uv_layers for obj in terrain)
        or scene.camera is None
    ):
        raise ValueError("The saved BLEND scene failed its object, texture, UV or camera checks.")
    scene.render.filepath = str(args.render.resolve())
    bpy.ops.render.render(write_still=True)
    report = dict(
        blender=bpy.app.version_string,
        saved_scene=str(args.scene.resolve()),
        buildings=len(buildings),
        terrain=len(terrain),
        texture_sizes=[list(image.size) for image in images],
        terrain_textures_packed=True,
        render=str(args.render.resolve()),
    )
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
