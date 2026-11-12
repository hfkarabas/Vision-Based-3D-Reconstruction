"""Reuse native Blender export, validate foundations and render matched views."""

import argparse
import math
import os
from pathlib import Path
import sys

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omega_building_blender as shared


def conform_foundation(obj, feature, origin, terrain, original_embed):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    edges = [edge for edge in bm.edges if all(abs(v.co.z) < 1e-5 for v in edge.verts)]
    if not edges:
        bm.free()
        return {"foundation_mode": "existing_flat", "reason": "no bottom edges"}
    cuts = min(8, max(1, math.ceil(max(edge.calc_length() for edge in edges) / 2)))
    bmesh.ops.subdivide_edges(bm, edges=edges, cuts=cuts, use_grid_fill=True)
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    floor = {v for v in bm.verts if abs(v.co.z) < 1e-5}
    z_offset = obj.location.z
    values = {v: terrain(v.co.x, v.co.y) for v in floor}
    if any(value is None for value in values.values()):
        bm.free()
        return {"foundation_mode": "existing_flat", "reason": "missing terrain vertex hit"}
    for v, z in values.items():
        v.co.z = z - 0.15 - z_offset
    floor_faces = [face for face in bm.faces if set(face.verts) <= floor]
    bm.verts.index_update()
    points = [(v.co.x, v.co.y, v.co.z + z_offset) for v in bm.verts]
    if not floor_faces:
        bm.free()
        return {"foundation_mode": "existing_flat", "reason": "no native floor triangles"}
    tree = BVHTree.FromPolygons(points, [tuple(v.index for v in f.verts) for f in floor_faces], all_triangles=True)
    rings = shared.rings_for(feature, origin)
    xy = list(shared.ring_sample_points(rings))
    xy.extend(tuple(f.calc_center_median()[:2]) for f in floor_faces)
    ground_deltas = []
    ray_start = min(p[2] for p in points) - 1.0
    for x, y in xy:
        base, _, _, _ = tree.ray_cast((x, y, ray_start), (0, 0, 1))
        ground = terrain(x, y)
        if base is not None and ground is not None:
            ground_deltas.append(ground - base.z)
    if len(ground_deltas) < 0.9 * len(xy):
        bm.free()
        return {"foundation_mode": "existing_flat", "reason": "dense foundation ray coverage insufficient"}
    shift = max(0.0, 0.05 - min(ground_deltas))
    burial = max(ground_deltas) + shift
    if burial >= original_embed - 0.02:
        bm.free()
        return {"foundation_mode": "existing_flat", "reason": "terrain-following trial did not reduce burial"}
    for vert in floor:
        vert.co.z -= shift
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    if bm.calc_volume(signed=True) < 0:
        bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
    if any(not edge.is_manifold for edge in bm.edges) or any(face.calc_area() < 1e-8 for face in bm.faces):
        bm.free()
        return {"foundation_mode": "existing_flat", "reason": "terrain-following trial failed solid checks"}
    new_base = min(v.co.z for v in floor) + z_offset
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    obj["foundation_z_m"] = new_base
    obj["mesh_height_m"] = obj["roof_z_m"] - new_base
    obj["foundation_mode"] = "terrain_following"
    obj.data.calc_loop_triangles()
    return dict(foundation_mode="terrain_following", foundation_z_m=new_base,
                mesh_height_m=obj["mesh_height_m"], vertices=len(obj.data.vertices),
                triangles=len(obj.data.loop_triangles), max_foundation_burial_m=burial,
                min_foundation_burial_m=min(ground_deltas) + shift,
                foundation_ray_count=len(ground_deltas), **shared.geometry_stats(obj))


def urban():
    from omega_urban_44 import build as existing

    existing.original_apply_photo = shared.apply_photo
    existing.original_camera_setup = shared.camera_setup
    shared.apply_photo = existing.apply_reference_photo
    shared.camera_setup = existing.frame_review

    def extrude(feature, origin, terrain, material, collection, max_burial):
        result = existing.extrude(feature, origin, terrain, material, collection, max_burial)
        if result["status"] != "built":
            return result
        obj = collection.objects[result["building_id"]]
        result["max_foundation_burial_m"] = result["max_boundary_burial_m"]
        result.update(conform_foundation(obj, feature, origin, terrain, result["max_boundary_burial_m"]))
        return result

    shared.extrude = extrude
    shared.main()


def import_mesh(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path.resolve()))
    return shared.meshes()


def terrain_asset(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    obj = shared.terrain_from_grid(args.height_grid)
    grid = shared.read(args.grid)
    shared.apply_photo([obj], grid, args.photo.resolve())
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1
    scene["projected_crs"] = grid["crs"]
    scene["origin_east_north_up_m"] = grid["origin_enu"]
    path = args.output.resolve()
    path.parent.mkdir(exist_ok=True, parents=True)
    triangles = len(obj.data.polygons)
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB")
    shared.write(path.with_suffix(".json"), shared.inspect_portable(path, {}, triangles))


def lighting():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 980
    scene.render.resolution_y = 760
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world = bpy.data.worlds.new("Comparison background")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.18, 0.20, 0.23, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.65
    scene.view_settings.view_transform = "AgX"
    bpy.ops.object.light_add(type="SUN")
    bpy.context.object.data.energy = 2.3
    bpy.context.object.rotation_euler = (0.47, -0.43, -0.25)
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.data.type = "ORTHO"
    scene.camera = cam
    return scene, cam


def comparison(args):
    before = import_mesh(args.before)
    before_points = np.concatenate([shared.world_vertices(obj) for obj in before])
    low, high = before_points.min(axis=0), before_points.max(axis=0)
    terrain_before = [obj for obj in before if not obj.get("building_id")]
    query = shared.terrain_query(terrain_before)
    views = [] if args.local_only else [
        ("top", (0, 0, 1)), ("oblique", (1, -1, 0.95)), ("low_angle", (0.9, -1, 0.28))
    ]
    frames = {}
    for name, direction in views:
        frames[name] = (low, high, direction)
    if args.views:
        for name, item in shared.read(args.views).items():
            east, north = item["center_local_enu"]
            z = query(east, north)
            z = float(np.median(before_points[:, 2])) if z is None else z
            half = item["width_m"] / 2
            frames[name] = (
                np.array([east-half, north-half, z-item.get("z_below_m", 4)]),
                np.array([east+half, north+half, z+item.get("z_above_m", 10)]),
                item.get("direction", [-1, -1, 0.9]),
            )
    if not frames:
        raise ValueError("At least one comparison view is required.")
    out = args.output.resolve()
    out.mkdir(exist_ok=False, parents=True)
    stats = {}
    cameras = {}
    for label, path in (("before", args.before), ("after", args.after)):
        objects = import_mesh(path)
        terrain = [obj for obj in objects if not obj.get("building_id")]
        terrain_triangles = 0
        degenerate = 0
        for obj in terrain:
            obj.data.calc_loop_triangles()
            xyz = shared.world_vertices(obj)
            idx = np.array([t.vertices for t in obj.data.loop_triangles])
            normal = np.cross(xyz[idx[:, 1]]-xyz[idx[:, 0]], xyz[idx[:, 2]]-xyz[idx[:, 0]])
            degenerate += int(np.count_nonzero(~np.isfinite(normal).all(axis=1) | (np.linalg.norm(normal, axis=1) < 1e-8)))
            terrain_triangles += len(idx)
        stats[label] = dict(terrain_meshes=len(terrain), terrain_triangles=terrain_triangles,
                            terrain_degenerate_triangles=degenerate,
                            building_meshes=len(objects)-len(terrain))
        scene, cam = lighting()
        for name, (a, b, direction) in frames.items():
            midpoint = Vector((a+b)/2)
            diagonal = float(np.linalg.norm(b-a))
            cam.location = midpoint + Vector(direction).normalized()*max(100, diagonal*2.5)
            cam.rotation_euler = (midpoint-cam.location).to_track_quat("-Z", "Y").to_euler()
            cam.data.clip_end = max(2500, diagonal*8)
            corners = np.array([[x,y,z] for x in [a[0],b[0]] for y in [a[1],b[1]] for z in [a[2],b[2]]])
            view = (corners-np.array(midpoint)) @ np.array(cam.rotation_euler.to_matrix())
            cam.data.ortho_scale = max(np.ptp(view[:,0]), np.ptp(view[:,1])*980/760)*1.10
            cameras[name] = dict(position=list(cam.location), euler=list(cam.rotation_euler), ortho_scale=cam.data.ortho_scale)
            scene.render.filepath = str(out / f"{name}_{label}.png")
            bpy.ops.render.render(write_still=True)
    shared.write(out / "validation.json", dict(geometry=stats, cameras=cameras, fixed_comparison=True))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(required=True)
    asset = commands.add_parser("terrain-only")
    for name in ("height-grid", "grid", "photo", "output"):
        asset.add_argument(f"--{name}", type=Path, required=True)
    asset.set_defaults(run=terrain_asset)
    compare = commands.add_parser("compare")
    for name in ("before", "after", "output"):
        compare.add_argument(f"--{name}", type=Path, required=True)
    compare.add_argument("--views", type=Path)
    compare.add_argument("--local-only", action="store_true")
    compare.set_defaults(run=comparison)
    args = cli.parse_args(sys.argv[sys.argv.index("--")+1:])
    args.run(args)


if __name__ == "__main__":
    if os.environ.get("OMEGA_TERRAIN_URBAN") == "1":
        urban()
    else:
        main()
