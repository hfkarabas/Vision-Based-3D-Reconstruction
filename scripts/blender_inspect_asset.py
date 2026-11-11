"""Import, inspect and render one photogrammetry asset inside Blender."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np


def read_array(collection, attribute, dtype, width=1):
    array = np.empty(len(collection) * width, dtype=dtype)
    collection.foreach_get(attribute, array)
    return array.reshape(-1, width) if width > 1 else array


def sha(data):
    return hashlib.sha256(data).hexdigest()


def image_digest(image):
    if image.packed_file:
        return sha(image.packed_file.data)
    return sha(Path(bpy.path.abspath(image.filepath, library=image.library)).read_bytes())


def inspect_materials(objects, atlas):
    materials = {slot.material for obj in objects for slot in obj.material_slots}
    found = {}
    for material in materials:
        if material is None or material.node_tree is None:
            continue
        nodes = material.node_tree.nodes
        output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output), None)
        surface = output.inputs["Surface"] if output else None
        shader = surface.links[0].from_node if surface and surface.is_linked else None
        if shader is None or shader.type != "BSDF_PRINCIPLED":
            raise ValueError(f"Expected an imported Principled material for {material.name}")
        base = shader.inputs["Base Color"]
        source = base.links[0].from_node if base.is_linked else None
        record = {"source": source.type if source else "constant"}
        if source and source.type == "TEX_IMAGE":
            image = source.image
            if image is None or min(image.size) < 2 or not image.has_data:
                raise ValueError(f"Could not decode image for {material.name}")
            digest = image_digest(image)
            if digest != atlas:
                raise ValueError(f"Material {material.name} is bound to different image bytes")
            record.update(image=image.name, size=list(image.size), sha256=digest, packed=bool(image.packed_file))
        found[material] = (record, base, surface)
    return found


def mesh_data(objects, materials):
    positions, faces, photographic = [], [], []
    counts = {}
    offset = 0
    invalid_normals = 0
    for obj in objects:
        mesh = obj.data
        mesh.calc_loop_triangles()
        if len(mesh.polygons) != len(mesh.loop_triangles) or not len(mesh.polygons):
            raise ValueError("Expected imported triangle polygons")
        xyz = read_array(mesh.vertices, "co", np.float32, 3)
        transform = np.array(obj.matrix_world)
        xyz = (xyz @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
        if not np.isfinite(xyz).all():
            raise ValueError("Mesh positions contain non-finite coordinates")
        normals = read_array(mesh.corner_normals, "vector", np.float32, 3)
        lengths = np.linalg.norm(normals, axis=1)
        invalid_normals += int((~np.isfinite(lengths) | (lengths < .99) | (lengths > 1.01)).sum())
        triangles = read_array(mesh.loop_triangles, "vertices", np.int32, 3)
        polygons = read_array(mesh.loop_triangles, "polygon_index", np.int32)
        slots = read_array(mesh.polygons, "material_index", np.int32)[polygons]
        textured = np.zeros(len(triangles), dtype=bool)
        for slot_index in np.unique(slots):
            if slot_index >= len(obj.material_slots) or obj.material_slots[slot_index].material not in materials:
                raise ValueError("Triangle has no supported material")
            material = obj.material_slots[slot_index].material
            record = materials[material][0]
            assigned = slots == slot_index
            counts[material.name] = counts.get(material.name, 0) + int(assigned.sum())
            if record["source"] == "TEX_IMAGE":
                textured |= assigned
        if textured.any():
            layer = mesh.uv_layers.active
            if layer is None:
                raise ValueError("A photographic mesh is missing an active UV layer")
            coordinates = read_array(layer.uv, "vector", np.float32, 2)
            loops = read_array(mesh.loop_triangles, "loops", np.int32, 3)
            uv = coordinates[loops[textured]].astype(np.float64)
            a, b = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
            signed = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            if not np.isfinite(uv).all() or uv.min() < 0 or uv.max() > 1 or np.any(abs(signed) <= 1e-12):
                raise ValueError("Photographic UVs must be finite, inside the atlas and nondegenerate")
        positions.append(xyz)
        faces.append(triangles + offset)
        photographic.append(textured)
        offset += len(xyz)
    return np.concatenate(positions), np.concatenate(faces), np.concatenate(photographic), counts, invalid_normals


def inspect_topology(points, faces):
    """Join exact coordinates for analysis because export charts split vertex IDs."""
    xyz, inverse = np.unique(points, axis=0, return_inverse=True)
    faces = inverse[faces].astype(np.int32)
    begin, end = faces.ravel(), faces[:, [1, 2, 0]].ravel()
    key = np.minimum(begin, end).astype(np.int64) * len(xyz) + np.maximum(begin, end)
    order = np.argsort(key)
    _, start, count = np.unique(key[order], return_index=True, return_counts=True)
    shared = start[count == 2]
    left, right = order[shared], order[shared + 1]
    roots = np.arange(len(faces), dtype=np.int32)
    a, b = left // 3, right // 3
    while True:
        previous = roots.copy()
        np.minimum.at(roots, np.maximum(roots[a], roots[b]), np.minimum(roots[a], roots[b]))
        while not np.array_equal(roots, roots[roots]):
            roots = roots[roots]
        if np.array_equal(roots, previous):
            break
    sizes = np.bincount(roots)
    mesh = {"import_vertices": len(points), "exact_position_vertices": len(xyz),
            "edge_components_across_two_face_edges": int(np.count_nonzero(sizes)),
            "largest_component_faces": int(sizes.max()), "boundary_edges": int((count == 1).sum()),
            "edges_with_more_than_two_faces": int((count > 2).sum()),
            "shared_edge_winding_conflicts": int(((begin[left] < end[left]) == (begin[right] < end[right])).sum())}
    return mesh, xyz, faces


def intersection_probes(xyz, faces):
    centers = (xyz[faces[:, 0]] + xyz[faces[:, 1]] + xyz[faces[:, 2]]) / 3
    low, high = np.percentile(centers, [1, 99], axis=0)
    cells = np.floor((centers - low) / np.maximum(high - low, 1e-12) * 8).clip(0, 7).astype(int)
    bins = cells @ [64, 8, 1]
    counts = np.bincount(bins, minlength=512)
    report = []
    for cell in np.argsort(-counts, kind="stable")[:8]:
        rows = np.flatnonzero(bins == cell)
        if not len(rows):
            continue
        distance = np.linalg.norm(centers[rows] - np.median(centers[rows], axis=0), axis=1)
        rows = rows[np.argsort(distance, kind="stable")[:12000]]
        ids, indices = np.unique(faces[rows].ravel(), return_inverse=True)
        triangles = indices.reshape(-1, 3)
        tree = BVHTree.FromPolygons(xyz[ids].tolist(), triangles.tolist(), all_triangles=True)
        pairs = np.asarray(tree.overlap(tree), dtype=np.int32).reshape(-1, 2)
        a, b = pairs[pairs[:, 0] < pairs[:, 1]].T
        nonadjacent = ~(triangles[a, :, None] == triangles[b, None, :]).any(axis=(1, 2))
        report.append({"cell": int(cell), "triangles": len(rows), "nonadjacent_bvh_overlaps": int(nonadjacent.sum())})
    return report


def render_views(materials, xyz, output, framing, resolution):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 4
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    for material, (_, base, surface) in materials.items():
        tree = material.node_tree
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = base.default_value
        if base.is_linked:
            tree.links.new(base.links[0].from_socket, emission.inputs["Color"])
        tree.links.new(emission.outputs[0], surface)
    camera = bpy.data.objects.new("Regression camera", bpy.data.cameras.new("Regression camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    if framing is None:
        low, high = xyz.min(axis=0), xyz.max(axis=0)
        framing = {"center": ((low + high) / 2).tolist(), "span": float(np.linalg.norm(high - low))}
    center, span = Vector(framing["center"]), framing["span"]
    camera.data.clip_start = span / 10000
    camera.data.clip_end = 10 * span
    camera.data.ortho_scale = span * 1.06
    for name, direction in (("top", (0, 0, 1)), ("oblique", (1, -2, 2))):
        view = Vector(direction).normalized()
        camera.location = center + 2 * span * view
        camera.rotation_euler = (-view).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(output / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    return framing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--framing", type=Path)
    parser.add_argument("--resolution", type=int, default=640)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    started = perf_counter()
    asset, output = args.asset.resolve(), args.output.resolve()
    expected = json.loads((asset.parent / "metrics.json").read_text(encoding="utf-8"))["texture"]
    atlas = sha((asset.parent / "atlas.jpg").read_bytes())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if asset.suffix.lower() == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(asset))
    elif asset.suffix.lower() == ".obj":
        bpy.ops.wm.obj_import(filepath=str(asset), forward_axis="NEGATIVE_Z", up_axis="Y")
    else:
        raise ValueError("Expected GLB or OBJ")
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise ValueError("Importer produced no meshes")
    materials = inspect_materials(objects, atlas)
    points, faces, photo, counts, invalid_normals = mesh_data(objects, materials)
    area = np.linalg.norm(np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]]), axis=1) / 2
    coverage = float(area[photo].sum(dtype=np.float64) / area.sum(dtype=np.float64))
    issues = []
    photo_delta = int(photo.sum()) - expected["textured_faces"]
    fallback_delta = int((~photo).sum()) - expected["fallback_faces"]
    if photo_delta or fallback_delta:
        issues.append("Importer changed the triangle count")
    if abs(coverage - expected["textured_area_fraction"]) > 1e-5 or (area <= 0).any():
        issues.append("Imported mesh area or photographic coverage changed")
    if invalid_normals:
        issues.append("Imported mesh has invalid corner normals")
    used = {material.name: record for material, (record, _, _) in materials.items() if material.name in counts}
    if any(data["size"] != [expected["texture_size"]] * 2 for data in used.values() if data["source"] == "TEX_IMAGE"):
        raise ValueError("Imported atlas size differs from export metadata")
    if asset.suffix.lower() == ".glb" and (~photo).any():
        if not any(data["source"] in {"VERTEX_COLOR", "ATTRIBUTE"} for data in used.values()):
            raise ValueError("GLB fallback lost its vertex color binding")
    topology, unique, welded = inspect_topology(points, faces)
    if topology["edges_with_more_than_two_faces"] or topology["shared_edge_winding_conflicts"]:
        issues.append("Exact-position topology has nonmanifold edges or inconsistent winding")
    probes = intersection_probes(unique, welded)
    framing = json.loads(args.framing.read_text(encoding="utf-8"))["framing"] if args.framing else None
    output.mkdir(parents=True, exist_ok=False)
    frame = render_views(materials, points, output, framing, args.resolution)
    report = {"blender": bpy.app.version_string, "asset": str(asset), "triangles": len(faces),
              "textured_faces": int(photo.sum()), "textured_area_fraction": coverage, "materials": used,
              "material_faces": counts, "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
              "topology": topology, "bounded_intersection_probes": probes, "framing": frame,
              "triangle_count_delta": {"photographic": photo_delta, "fallback": fallback_delta},
              "zero_area_faces": int((area <= 0).sum()), "invalid_corner_normals": invalid_normals,
              "status": "passed" if not issues else "failed", "issues": issues,
              "resolution": args.resolution, "runtime_seconds": perf_counter() - started}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if issues:
        raise ValueError(". ".join(issues))


if __name__ == "__main__":
    main()
