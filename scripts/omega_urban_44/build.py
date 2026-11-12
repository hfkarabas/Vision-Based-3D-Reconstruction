"""Reuse Mission Omega's Blender export and validate expanded footprint placement."""
import os
from pathlib import Path
import sys

import bmesh
import bpy

repo = Path(os.environ["OMEGA_PROJECT_ROOT"])
sys.path.insert(0, str(repo / "scripts"))
import omega_building_blender as original
from omega_building_blender import geometry_stats, ring_sample_points, rings_for

def extrude(feature, origin, query, mat, collection, max_burial):
    props = feature["properties"]
    name = props["building_id"]
    rings = rings_for(feature, origin)
    ground = [query(*xy) for xy in ring_sample_points(rings)]
    triangles = props["foundation_triangles_xy"]
    interior = [
        query(sum(point[0] for point in tri) / 3 - origin[0],
              sum(point[1] for point in tri) / 3 - origin[1])
        for tri in triangles
    ]
    samples = ground + interior
    if not ground or not interior or None in samples:
        return dict(building_id=name, status="terrain_mesh_has_foundation_gap")
    base = min(samples) - 0.10
    high_ground = max(samples)
    provisional = props["height_support"] == "unknown"
    top = high_ground + props["display_height_m"] if provisional else props["reference_dsm_roof_m"] - origin[2]
    if top - high_ground < 0.75 and props["allow_display_proxy"]:
        provisional = True
        top = high_ground + 3.0
    if top - high_ground < 0.75 or high_ground - base > max_burial:
        return dict(
            building_id=name, status="foundation_relief_or_roof_clearance_failed",
            boundary_min_m=min(ground), boundary_max_m=max(ground), roof_z_m=top,
        )

    # GEOS triangulates the existing polygon, including interior holes.
    # Blender extrudes the connected face region and creates the wall topology.
    bm = bmesh.new()
    vertices = {}
    for triangle in triangles:
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
    obj.data.calc_loop_triangles()
    for key in ("building_id", "height_support", "roof_support", "height_provenance", "footprint_provenance", "footprint_tier"):
        obj[key] = props[key]
    if provisional:
        obj["height_support"] = "unknown"
        obj["height_provenance"] = "Unknown height; 3 m display proxy above highest sampled terrain"
    else:
        obj["lod1_height_m"] = props["lod1_height_m"]
    obj["mesh_height_m"] = top - base
    obj["foundation_z_m"] = base
    obj["roof_z_m"] = top
    return dict(
        building_id=name,
        status="built",
        vertices=len(obj.data.vertices),
        triangles=len(obj.data.loop_triangles),
        height_m=None if provisional else props["lod1_height_m"],
        mesh_height_m=top - base,
        height_support=obj["height_support"],
        height_provenance=obj["height_provenance"],
        footprint_tier=props["footprint_tier"],
        foundation_z_m=base,
        roof_z_m=top,
        boundary_ray_count=len(ground),
        interior_ray_count=len(interior),
        min_boundary_burial_m=min(ground) - base,
        max_boundary_burial_m=max(ground) - base,
        minimum_roof_clearance_m=top - high_ground,
        area_relative_error=area_delta,
        **stats,
    )


def apply_reference_photo(terrain, grid, photo_path):
    original_apply_photo(terrain, grid, photo_path)
    for obj in terrain:
        obj.name = "Reference DSM-derived terrain"


def frame_review(terrain, buildings, output):
    unknown = [obj for obj in buildings if obj["height_support"] == "unknown"]
    for obj in unknown:
        obj["lod1_height_m"] = obj["mesh_height_m"]
    try:
        original_camera_setup(terrain, buildings, output)
    finally:
        for obj in unknown:
            del obj["lod1_height_m"]


if __name__ == "__main__":
    original_apply_photo = original.apply_photo
    original_camera_setup = original.camera_setup
    original.apply_photo = apply_reference_photo
    original.camera_setup = frame_review
    original.extrude = extrude
    original.main()
