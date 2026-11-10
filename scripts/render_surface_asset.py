"""Render a saved vertex-color baseline or photographic GLB using the desktop loader."""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from vision3d.demo.view_geometry import camera_view, load_context
from vision3d.demo.viewer import load_mesh, load_textured_meshes


def render(model, context, output):
    textured = model.suffix.lower() == ".glb"
    meshes = load_textured_meshes(model) if textured else [load_mesh(model)]
    basis = np.array([1, -1, -1]) if textured else np.ones(3)
    points = np.concatenate([np.asarray(mesh.vertices) for mesh in meshes])
    view = camera_view(points * basis, [p["world_to_camera"] for p in load_context(context)])
    viewer = o3d.visualization.Visualizer()
    if not viewer.create_window(width=1100, height=800, visible=False):
        raise RuntimeError("Open3D cannot initialize a rendering context")
    try:
        if not all(viewer.add_geometry(mesh) for mesh in meshes):
            raise RuntimeError("Open3D cannot upload the mesh groups")
        options = viewer.get_render_option()
        options.background_color = np.ones(3)
        options.mesh_color_option = o3d.visualization.MeshColorOption.Color
        options.mesh_show_back_face = True
        options.light_on = False
        control = viewer.get_view_control()
        control.set_lookat(view["lookat"] * basis)
        control.set_front(view["front"] * basis)
        control.set_up(view["up"] * basis)
        control.set_zoom(.55)
        viewer.poll_events()
        buffer = viewer.capture_screen_float_buffer(do_render=True)
        rgb = np.rint(np.asarray(buffer).clip(0, 1) * 255).astype(np.uint8)
        if (np.ptp(rgb.astype(np.int16), axis=2) > 3).mean() < .002:
            raise RuntimeError("Render did not contain usable model colors")
        Image.fromarray(rgb).save(output)
    finally:
        viewer.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path, help="Original-world PLY beside the saved poses")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.model, args.context, args.output)
