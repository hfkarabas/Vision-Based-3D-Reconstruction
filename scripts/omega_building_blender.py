"""Assemble the detected roof polygons and previously reconstructed terrain in Blender."""

import argparse
import json
import math
from pathlib import Path
import sys

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def meshes():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def world_vertices(obj):
    local = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", local)
    matrix = np.array(obj.matrix_world)
    return local.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]


def terrain_from_grid(path):
    with np.load(path) as grid:
        x, y, z = grid["x"], grid["y"], grid["z"]
    height, width = z.shape
    if width * height > 500_000 or (len(x), len(y)) != (width, height):
        raise ValueError("Expected a bounded height grid with valid axes.")
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=width - 1, y_segments=height - 1, size=1)
    missing = []
    for vert in bm.verts:
        col = round((vert.co.x + 1) * (width - 1) / 2)
        row = round((1 - vert.co.y) * (height - 1) / 2)
        if np.isfinite(z[row, col]):
            vert.co = (x[col], y[row], z[row, col])
        else:
            missing.append(vert)
    bmesh.ops.delete(bm, geom=missing, context="VERTS")
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    mesh = bpy.data.meshes.new("PDAL-classified and GDAL-interpolated terrain")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("Survey-assisted height terrain", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = 1.0
    return mat


def apply_photo(terrain, grid, photo_path):
    mat = material(grid.get("texture_label", "Airborne orthophoto"), (1, 1, 1, 1))
    texture = mat.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(str(photo_path))
    texture.image.pack()
    texture.extension = "CLIP"
    shader = mat.node_tree.nodes.get("Principled BSDF")
    mat.node_tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
    # Keep relief legible without casting full-strength roof shadows onto the orthophoto.
    mat.node_tree.links.new(texture.outputs["Color"], shader.inputs["Emission Color"])
    shader.inputs["Emission Strength"].default_value = 0.25
    west, south, east, north = grid["bounds"]
    origin = np.array(grid["origin_enu"])
    for obj in terrain:
        world = world_vertices(obj)
        mesh = obj.data
        indices = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", indices)
        uv = mesh.uv_layers.active or mesh.uv_layers.new(name="Projected RGB grid")
        coords = (world[indices, :2] + origin[:2] - [west, south]) / [east - west, north - south]
        if coords.min() < -0.002 or coords.max() > 1.002:
            raise ValueError("Terrain coordinates do not match the orthophoto and ENU origin.")
        uv.uv.foreach_set("vector", coords.astype(np.float32).ravel())
        mesh.materials.clear()
        mesh.materials.append(mat)
        mesh.polygons.foreach_set("material_index", np.zeros(len(mesh.polygons), dtype=np.int32))
        obj.name = "Height-derived UAV terrain"


def terrain_query(terrain):
    trees = []
    for obj in terrain:
        mesh = obj.data
        mesh.calc_loop_triangles()
        trees.append(BVHTree.FromPolygons(
            world_vertices(obj).tolist(), [tuple(t.vertices) for t in mesh.loop_triangles], all_triangles=True
        ))
    ceiling = max(world_vertices(obj)[:, 2].max() for obj in terrain) + 10

    def height(x, y):
        hits = [tree.ray_cast((x, y, ceiling), (0, 0, -1))[0] for tree in trees]
        return max((hit.z for hit in hits if hit is not None), default=None)

    return height


def rings_for(feature, origin):
    rings = []
    for ring in feature["geometry"]["coordinates"]:
        rings.append([(x - origin[0], y - origin[1]) for x, y in ring[:-1]])
    return rings


def ring_sample_points(rings):
    for ring in rings:
        for start, end in zip(ring, ring[1:] + ring[:1]):
            steps = max(1, math.ceil(math.dist(start, end) / 1.5))
            for i in range(steps):
                t = i / steps
                yield ((1 - t) * start[0] + t * end[0], (1 - t) * start[1] + t * end[1])


def geometry_stats(obj, merge=False):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if merge:
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=0.00001)
    info = dict(
        nonmanifold_edges=sum(not edge.is_manifold for edge in bm.edges),
        degenerate_faces=sum(face.calc_area() < 1e-8 for face in bm.faces),
        signed_volume_m3=bm.calc_volume(signed=True),
    )
    bm.free()
    return info


def extrude(feature, origin, query, mat, collection, max_burial):
    props = feature["properties"]
    name = props["building_id"]
    rings = rings_for(feature, origin)
    ground = [query(*xy) for xy in ring_sample_points(rings)]
    if not ground or None in ground:
        return dict(building_id=name, status="terrain_mesh_has_foundation_gap")
    base = min(ground) - 0.10
    top = props["roof_in_dtm_frame_m"] - origin[2]
    if max(ground) - base > max_burial or top - max(ground) < 1.5:
        return dict(
            building_id=name, status="foundation_relief_or_roof_clearance_failed",
            boundary_min_m=min(ground), boundary_max_m=max(ground), roof_z_m=top,
        )

    # GEOS triangulates the existing polygon, including interior holes.
    # Blender extrudes the connected face region and creates the wall topology.
    bm = bmesh.new()
    vertices = {}
    for triangle in props["foundation_triangles_xy"]:
        face = []
        for east, north in triangle:
            key = (east, north)
            if key not in vertices:
                vertices[key] = bm.verts.new((east - origin[0], north - origin[1], 0))
            face.append(vertices[key])
        bm.faces.new(face)
    raised = bmesh.ops.extrude_face_region(bm, geom=list(bm.faces) + list(bm.edges))
    bmesh.ops.translate(
        bm,
        verts=[vert for vert in raised["geom"] if isinstance(vert, bmesh.types.BMVert)],
        vec=(0, 0, top - base),
    )
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    if bm.calc_volume(signed=True) < 0:
        bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.location.z = base
    bpy.context.view_layer.update()
    obj.data.materials.append(mat)
    stats = geometry_stats(obj)
    area_delta = abs(stats["signed_volume_m3"] / (top - base) - props["area_m2"]) / props["area_m2"]
    if stats["nonmanifold_edges"] or stats["degenerate_faces"] or area_delta > 0.005:
        bpy.data.objects.remove(obj, do_unlink=True)
        return dict(building_id=name, status="native_extrusion_geometry_failed", **stats, area_relative_error=area_delta)
    # Check actual terrain through the interior of the filled roof, too.
    obj.data.calc_loop_triangles()
    interior = []
    for tri in obj.data.loop_triangles:
        coords = [obj.matrix_world @ obj.data.vertices[index].co for index in tri.vertices]
        if all(abs(v.z - top) < 0.01 for v in coords):
            center = sum(coords, Vector()) / 3
            interior.append(query(center.x, center.y))
    if not interior or any(value is None or not base <= value < top - 1.5 for value in interior):
        bpy.data.objects.remove(obj, do_unlink=True)
        return dict(building_id=name, status="terrain_mesh_interior_gap_or_invalid_clearance")
    for key in ("building_id", "lod1_height_m", "height_provenance", "footprint_provenance"):
        obj[key] = props[key]
    obj["foundation_z_m"] = base
    obj["roof_z_m"] = top
    return dict(
        building_id=name,
        status="built",
        vertices=len(obj.data.vertices),
        triangles=len(obj.data.loop_triangles),
        height_m=props["lod1_height_m"],
        foundation_z_m=base,
        roof_z_m=top,
        boundary_ray_count=len(ground),
        interior_ray_count=len(interior),
        min_boundary_burial_m=min(ground) - base,
        max_boundary_burial_m=max(ground) - base,
        minimum_roof_clearance_m=top - max(ground + interior),
        area_relative_error=area_delta,
        **stats,
    )


def camera_setup(terrain, buildings, output):
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1400
    scene.render.resolution_y = 1120
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world = bpy.data.worlds.new("Neutral background")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.18, 0.20, 0.23, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.65
    scene.view_settings.view_transform = "AgX"
    bpy.ops.object.light_add(type="SUN")
    bpy.context.object.data.energy = 2.3
    bpy.context.object.rotation_euler = (0.47, -0.43, -0.25)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    scene.camera = camera
    output.mkdir(exist_ok=True)

    def frame(objects, direction, label):
        points = np.concatenate([world_vertices(obj) for obj in objects])
        low, high = points.min(0), points.max(0)
        center = Vector((low + high) / 2)
        radius = float(np.linalg.norm(high - low))
        camera.location = center + Vector(direction).normalized() * max(50, radius * 2.5)
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera.data.clip_end = max(500, radius * 8)
        rotation = np.array(camera.rotation_euler.to_matrix())
        views = (points - np.array(center)) @ rotation
        aspect = scene.render.resolution_x / scene.render.resolution_y
        camera.data.ortho_scale = max(np.ptp(views[:, 0]), np.ptp(views[:, 1]) * aspect) * 1.13
        scene.render.filepath = str(output / f"{label}.png")
        bpy.ops.render.render(write_still=True)
        return center, radius

    full = terrain + buildings
    frame(full, (0, 0, 1), "top")
    frame(full, (0.9, -1, 0.28), "low_angle")
    if buildings:
        ordered = sorted(buildings, key=lambda obj: obj["lod1_height_m"])
        frame([ordered[len(ordered) // 2]], (-0.7, -1, 0.65), "close_up")
        if len(ordered) > 1:
            frame([ordered[-1]], (-0.7, -1, 0.65), "close_up_highrise")
    center, radius = frame(full, (1, -1, 0.95), "oblique")
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                view = area.spaces.active
                view.shading.type = "MATERIAL"
                view.clip_end = max(1500, radius * 8)
                view.region_3d.view_location = center
                view.region_3d.view_distance = radius * 1.2
                view.region_3d.view_rotation = camera.rotation_euler.to_quaternion()


def inspect_portable(path, expected, terrain_reference):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = {obj.get("building_id"): obj for obj in meshes() if obj.get("building_id")}
    terrain = [obj for obj in meshes() if not obj.get("building_id")]
    if set(imported) != set(expected) or not terrain:
        raise ValueError("GLB reimport lost building identities or the terrain.")
    packed_images = [
        dict(name=img.name, resolution=list(img.size))
        for img in bpy.data.images if img.packed_file and min(img.size) > 1
    ]
    if not packed_images or any(not obj.data.uv_layers for obj in terrain):
        raise ValueError("GLB reimport lost the embedded photographic texture or UVs.")
    if sum(len(obj.data.polygons) for obj in terrain) != terrain_reference:
        raise ValueError("Terrain triangle count changed in the GLB round-trip.")
    for name, obj in imported.items():
        shape = geometry_stats(obj, merge=True)
        if shape["nonmanifold_edges"] or shape["degenerate_faces"] or shape["signed_volume_m3"] <= 0:
            raise ValueError(f"Building solid did not survive GLB reimport: {name}")
        xyz = world_vertices(obj)
        if not np.allclose([xyz[:, 2].min(), xyz[:, 2].max()], expected[name], atol=0.002):
            raise ValueError(f"Building elevation changed during GLB reimport: {name}")
        shader = obj.data.materials[0].node_tree.nodes.get("Principled BSDF")
        if not np.allclose(shader.inputs["Base Color"].default_value, [1, 1, 1, 1]):
            raise ValueError(f"White material lost during GLB reimport: {name}")
        if abs(shader.inputs["Roughness"].default_value - 1.0) > 0.001:
            raise ValueError(f"Matte material lost during GLB reimport: {name}")
    points = np.concatenate([world_vertices(obj) for obj in meshes()])
    return dict(
        building_objects=len(imported),
        terrain_objects=len(terrain),
        terrain_triangles=terrain_reference,
        bounds_local_enu_m=[points.min(0).tolist(), points.max(0).tolist()],
        dimensions_m=np.ptp(points, axis=0).tolist(),
        packed_textures=packed_images,
        blender_version=bpy.app.version_string,
        all_building_solids_closed=True,
        material_roundtrip="plain white, roughness 1",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    terrain_input = parser.add_mutually_exclusive_group(required=True)
    terrain_input.add_argument("--terrain", type=Path)
    terrain_input.add_argument("--height-grid", type=Path)
    parser.add_argument("--terrain-validation", type=Path)
    parser.add_argument("--name", default="terrain_white_buildings")
    parser.add_argument("--max-burial", type=float, default=4)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--visual-review", type=Path)
    parser.add_argument("--allow-terrain-only", action="store_true",
                        help="Save verified terrain when no candidate supports a building volume.")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    out = args.output.resolve()
    report_dir = out / "reports"
    report_dir.mkdir(exist_ok=True)
    grid = read(out / "grid.json")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if args.terrain:
        if not args.terrain_validation:
            raise ValueError("The terrain GLB requires its prior validation report.")
        prior = read(args.terrain_validation)
        if prior["crs"] != grid["crs"] or not np.allclose(prior["geospatial_origin"], grid["origin_enu"], rtol=0, atol=0.01):
            raise ValueError("The existing terrain uses a different CRS or local origin.")
        bpy.ops.import_scene.gltf(filepath=str(args.terrain.resolve()))
    else:
        terrain_from_grid(args.height_grid)
    terrain = meshes()
    if not terrain:
        raise ValueError("Terrain GLB contains no mesh.")
    apply_photo(terrain, grid, out / "terrain_photo.png")
    terrain_triangles = 0
    for obj in terrain:
        obj.data.calc_loop_triangles()
        terrain_triangles += len(obj.data.loop_triangles)
    query = terrain_query(terrain)
    buildings = bpy.data.collections.new("Buildings - detected roof LOD1")
    bpy.context.scene.collection.children.link(buildings)
    white = material("Matte white LOD1", (1, 1, 1, 1))
    features = read(out / "building_footprints_projected.json")["features"]
    excluded = read(args.visual_review).get("exclude_from_asset", {}) if args.visual_review else {}
    eligible = {
        feature["properties"]["building_id"]: feature
        for feature in features if feature["properties"]["status"] == "eligible"
    }
    if unknown := excluded.keys() - eligible.keys():
        raise ValueError(f"Visual review refers to ineligible or missing candidates: {sorted(unknown)}")
    results = []
    for name, feature in eligible.items():
        if name in excluded:
            results.append(dict(building_id=name, status="visual_review_rejected", reason=excluded[name]))
        else:
            results.append(extrude(feature, grid["origin_enu"], query, white, buildings, args.max_burial))
    write(report_dir / "blender_buildings.json", results)
    built = [record for record in results if record["status"] == "built"]
    if not built and not args.allow_terrain_only:
        raise ValueError("No detected polygon passed the terrain and mesh geometry checks.")
    if args.validate_only:
        print(f"{len(built)} buildings passed native Blender and foundation checks.")
        return
    scene = bpy.context.scene
    scene["projected_crs"] = grid["crs"]
    scene["origin_east_north_up_m"] = grid["origin_enu"]
    camera_setup(terrain, list(buildings.objects), out / "preview")
    bpy.ops.wm.save_as_mainfile(filepath=str(out / f"{args.name}.blend"))
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes():
        obj.select_set(True)
    glb = out / f"{args.name}.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb), export_format="GLB", use_selection=True, export_extras=True
    )
    expected = {row["building_id"]: [row["foundation_z_m"], row["roof_z_m"]] for row in built}
    write(report_dir / "blender_glb_reimport.json", inspect_portable(glb, expected, terrain_triangles))


if __name__ == "__main__":
    main()
