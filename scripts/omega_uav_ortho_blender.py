"""Convert COLMAP UV PLY and render a registered camera-image RGB/height pair in Blender."""

import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def read_ply(path):
    # COLMAP's binary WriteTexturedPly layout includes six per-face UV scalars.
    with path.open("rb") as source:
        lines = []
        for _ in range(128):
            line = source.readline().decode("ascii").strip()
            if not line:
                raise ValueError("Incomplete COLMAP PLY header.")
            lines.append(line)
            if line == "end_header":
                break
        properties = [line for line in lines if line.startswith("property ")]
        if "format binary_little_endian 1.0" not in lines or properties != [
            "property float x", "property float y", "property float z",
            "property list uchar int vertex_indices", "property list uchar float texcoord",
        ]:
            raise ValueError("Expected COLMAP's binary textured triangle PLY.")
        vertices = int(next(line for line in lines if line.startswith("element vertex ")).split()[-1])
        faces = int(next(line for line in lines if line.startswith("element face ")).split()[-1])
        texture = next(line for line in lines if line.startswith("comment TextureFile ")).split(" ", 2)[2]
        if Path(texture).name != texture or not 0 < faces <= 1_800_000:
            raise ValueError("The texture path or triangle count is outside the bounded conversion contract.")
        xyz = np.fromfile(source, dtype="<f4", count=3 * vertices).reshape(-1, 3)
        dtype = np.dtype([("count", "u1"), ("vertices", "<i4", 3), ("uvcount", "u1"), ("uv", "<f4", (3, 2))])
        rows = np.fromfile(source, dtype=dtype, count=faces)
        if (
            len(xyz) != vertices or len(rows) != faces or source.read(1)
            or not np.all(rows["count"] == 3) or not np.all(rows["uvcount"] == 6)
            or rows["vertices"].min() < 0 or rows["vertices"].max() >= vertices
            or not np.isfinite(xyz).all() or not np.isfinite(rows["uv"]).all()
        ):
            raise ValueError("Malformed COLMAP triangle or texture records.")
    uv = rows["uv"]
    first, second = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    assigned = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) > 1e-14
    if not assigned.any():
        raise ValueError("COLMAP did not assign usable camera UVs to the mesh.")
    return xyz, rows["vertices"][assigned], uv[assigned], path.parent / texture, faces


def add_surface(xyz, faces, uv):
    mesh = bpy.data.meshes.new("COLMAP camera-textured surface")
    mesh.vertices.add(len(xyz))
    mesh.vertices.foreach_set("co", xyz.astype(np.float32).ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(0, faces.size, 3, dtype=np.int32))
    mesh.update(calc_edges=True)
    if mesh.validate(verbose=False):
        raise ValueError("Blender could not import the original COLMAP topology unchanged.")
    mesh.uv_layers.new(name="COLMAP camera atlas").uv.foreach_set("vector", uv.astype(np.float32).ravel())
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def photo_shader(path):
    mat = bpy.data.materials.new("Original camera imagery")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    image = nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(path))
    emit = nodes.new("ShaderNodeEmission")
    mat.node_tree.links.new(image.outputs["Color"], emit.inputs["Color"])
    mat.node_tree.links.new(emit.outputs[0], nodes.get("Material Output").inputs["Surface"])
    return mat


def elevation_shader(shift):
    mat = bpy.data.materials.new("World elevation float pass")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    geom = nodes.new("ShaderNodeNewGeometry")
    xyz = nodes.new("ShaderNodeSeparateXYZ")
    add = nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    add.inputs[1].default_value = shift
    emit = nodes.new("ShaderNodeEmission")
    links.new(geom.outputs["Position"], xyz.inputs[0])
    links.new(xyz.outputs["Z"], add.inputs[0])
    links.new(add.outputs[0], emit.inputs["Color"])
    links.new(emit.outputs[0], nodes.get("Material Output").inputs["Surface"])
    return mat


def render(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    xyz, faces, uv, texture, total = read_ply(args.mesh)
    frame = json.loads(args.frame.read_text(encoding="utf-8"))
    transform = np.loadtxt(frame["transform"])
    xyz = xyz.astype(np.float64) @ transform[:, :3].T + transform[:, 3]
    origin = np.asarray(frame["origin_projected_m"])
    obj = add_surface(xyz, faces, uv)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = True
    scene.render.resolution_x, scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    west, south, east, north = args.extent
    camera = bpy.data.objects.new("Projected nadir camera", bpy.data.cameras.new("Projected nadir"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.location = ((west + east) / 2 - origin[0], (south + north) / 2 - origin[1], xyz[:, 2].max() + 10)
    camera.data.clip_end = float(np.ptp(xyz[:, 2])) + 30
    camera.data.ortho_scale = max(east - west, north - south)
    frame_xy = np.array([list(v)[:2] for v in camera.data.view_frame(scene=scene)])
    camera.data.ortho_scale *= (east - west) / np.ptp(frame_xy[:, 0])
    frame_xy = np.array([list(v)[:2] for v in camera.data.view_frame(scene=scene)])
    if not np.allclose(np.ptp(frame_xy, axis=0), [east - west, north - south], atol=0.002, rtol=0):
        raise ValueError("Blender's camera does not match the declared projected extent.")
    obj.data.materials.append(photo_shader(texture))
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = str(args.output / "orthophoto.png")
    bpy.ops.render.render(write_still=True)
    shift = 1 - float(xyz[:, 2].min())
    obj.data.materials[0] = elevation_shader(shift)
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = str(args.output / "surface.exr")
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath)
    pixels = np.empty(args.size[0] * args.size[1] * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    rgba = pixels.reshape(args.size[1], args.size[0], 4)[::-1]
    np.savez_compressed(args.output / "surface.npz", elevation=rgba[:, :, 0] - shift + origin[2],
                        alpha=rgba[:, :, 3])
    report = dict(blender=bpy.app.version_string, input_triangles=total, camera_uv_triangles=len(faces),
                  bounds=list(args.extent), resolution=list(args.size), crs=frame["crs"],
                  texture=str(texture), source_mesh=str(args.mesh),
                  method="Native orthographic RGB and world-elevation passes on COLMAP camera-textured mesh")
    (args.output / "renderer.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mesh", "frame", "output"):
        parser.add_argument(f"--{name}", type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument("--extent", nargs=4, type=float, required=True)
    parser.add_argument("--size", nargs=2, type=int, required=True)
    render(parser.parse_args(sys.argv[sys.argv.index("--") + 1:]))
