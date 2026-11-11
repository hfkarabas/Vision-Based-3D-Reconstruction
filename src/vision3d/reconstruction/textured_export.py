"""Textured glTF export with separate photographic and vertex-color materials."""
import io
import json
from pathlib import Path
import struct
import tempfile

import numpy as np
import open3d as o3d
from PIL import Image

from .surface_validation import face_geometry, mesh_arrays, repair_winding, topology, update_surface_normals


def prepare_texture_mesh(mesh, units_per_world):
    """Resolve duplicate positions and faces at the precision used by both exporters."""
    if not np.isfinite(units_per_world) or units_per_world <= 0:
        raise ValueError("Export units per SfM unit must be positive and finite")
    xyz, faces = mesh_arrays(mesh)
    colors = np.asarray(mesh.vertex_colors)
    if colors.shape != xyz.shape:
        raise ValueError("Photographic assets need fallback vertex colors")
    scaled = np.asarray(xyz * units_per_world, dtype=np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("Scaled positions exceed the export coordinate range")
    _, first, index = np.unique(scaled, axis=0, return_index=True, return_inverse=True)
    indexed = index[faces]
    _, _, original_area = face_geometry(xyz, faces)
    _, _, rounded_area = face_geometry(scaled.astype(np.float64), faces)
    finite = np.flatnonzero(rounded_area > 0)
    _, unique_faces = np.unique(np.sort(indexed[finite], axis=1), axis=0, return_index=True)
    keep = np.sort(finite[unique_faces])
    result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz[first]),
                                     o3d.utility.Vector3iVector(indexed[keep]))
    result.vertex_colors = o3d.utility.Vector3dVector(colors[first])
    result.remove_non_manifold_edges()
    result.remove_unreferenced_vertices()
    _, stats = topology(result)
    winding_cuts = 0
    if stats["winding_conflicts"]:
        winding_cuts = repair_winding(result, geometry_weighted=True)["contradictory_faces_cut"]
    cancelled = update_surface_normals(result)
    _, stats = topology(result)
    _, _, result_area = face_geometry(*mesh_arrays(result))
    retained_unique_area = original_area[keep].sum()
    area_loss = (original_area[rounded_area == 0].sum()
                 + max(0, retained_unique_area - result_area.sum())) / retained_unique_area
    if (stats["winding_conflicts"] or stats["nonmanifold_edges"] or stats["degenerate_triangles"]
            or stats["invalid_vertex_normals"] or area_loss > .0001):
        raise ValueError(f"Export precision cleanup exceeds its topology or area bound: {stats}, {area_loss:g}")
    return result, {"coincident_vertices": len(xyz) - len(first), "duplicate_faces_removed": len(finite) - len(keep),
                    "rounded_zero_area_faces_removed": len(faces) - len(finite),
                    "nonmanifold_faces_removed": len(keep) - len(result.triangles) - winding_cuts,
                    "winding_faces_removed": winding_cuts,
                    "relative_area_loss": float(area_loss), "cancelled_vertex_normals_recovered": cancelled}


def textured_primitives(mesh, atlas, units_per_world):
    if not np.isfinite(units_per_world) or units_per_world <= 0:
        raise ValueError("Export units per SfM unit must be positive and finite")
    xyz, faces = mesh_arrays(mesh)
    if (atlas.chart.shape != (len(faces),) or atlas.uv.shape != (len(faces), 3, 2)
            or not np.any(atlas.chart >= 0)):
        raise ValueError("Texture atlas does not match the surface triangles")
    normals = np.asarray(mesh.vertex_normals)
    if normals.shape != xyz.shape or not np.isfinite(normals).all() or np.any(np.linalg.norm(normals, axis=1) < 1e-7):
        raise ValueError("Textured export needs finite, nonzero vertex normals")
    basis = np.array([1, -1, -1])
    xyz = np.asarray(xyz * basis * units_per_world, dtype="<f4")
    if not np.isfinite(xyz).all():
        raise ValueError("Scaled positions exceed the export coordinate range")
    normals = np.asarray(normals * basis / np.linalg.norm(normals, axis=1, keepdims=True), dtype="<f4")
    _, cross, areas = face_geometry(xyz.astype(np.float64), faces)
    face_normals = cross / np.maximum(2 * areas[:, None], 1e-30)
    alignment = np.sum(normals[faces] * face_normals[:, None, :], axis=2)
    sharp_vertices = np.zeros(len(xyz), dtype=bool)
    sharp_vertices[faces[alignment <= .2]] = True

    def indices_for(selected, chart):
        local = faces[selected].ravel()
        sharp = sharp_vertices[local]
        shading = np.where(sharp, np.repeat(np.flatnonzero(selected) + 1, 3), 0)
        keys = np.column_stack((np.repeat(chart, 3).astype(np.int64) * len(xyz) + local, shading))
        keys, first, indices = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        rows = keys[:, 0] % len(xyz)
        values = normals[rows].copy()
        split = keys[:, 1] > 0
        values[split] = face_normals[keys[split, 1] - 1]
        return rows, first, indices, values

    primitives = []
    selected = atlas.chart >= 0
    uv = atlas.uv[selected]
    p, q = (uv[:, 1] - uv[:, 0]).astype(float), (uv[:, 2] - uv[:, 0]).astype(float)
    if (not np.isfinite(uv).all() or np.any((uv < 0) | (uv > 1))
            or np.any(np.abs(p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]) <= 1e-12)):
        raise ValueError("Atlas includes degenerate or out-of-bounds UV coordinates")
    rows, first, indices, values = indices_for(selected, atlas.chart[selected])
    primitives.append(({"POSITION": xyz[rows], "NORMAL": values, "TEXCOORD_0": uv.reshape(-1, 2)[first]}, indices))
    if (~selected).any():
        colors = np.asarray(mesh.vertex_colors)
        if colors.shape != xyz.shape or not np.isfinite(colors).all() or np.any((colors < 0) | (colors > 1)):
            raise ValueError("Unsupported faces need finite fallback vertex colors")
        rows, _, indices, values = indices_for(~selected, np.zeros(np.count_nonzero(~selected), np.int32))
        rgb = colors[rows]
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        primitives.append(({"POSITION": xyz[rows], "NORMAL": values, "COLOR_0": linear}, indices))
    return primitives


def verify_textured_model(path, primitives, texture_size, *, fallback_colors):
    model = o3d.io.read_triangle_model(str(path))
    expected = {i: (attributes, indices) for i, (attributes, indices) in enumerate(primitives)}
    if len(model.meshes) != len(expected):
        raise OSError("Independent model reload lost a material group")
    verified = set()
    for group in model.meshes:
        mesh = group.mesh
        material = model.materials[group.material_idx]
        image = np.asarray(material.albedo_img)
        textured = image.ndim == 3 and image.size > 0
        index = 0 if textured else 1
        if index in verified or index not in expected:
            raise OSError("Model reload altered material bindings")
        verified.add(index)
        attributes, indices = expected[index]
        actual = np.asarray(mesh.triangles).ravel()
        if len(actual) != len(indices) or not mesh.has_vertex_normals():
            raise OSError("Model reload altered triangle count or normals")
        for name, data in (("POSITION", mesh.vertices), ("NORMAL", mesh.vertex_normals)):
            values = np.asarray(data)
            for start in range(0, len(actual), 150000):
                span = slice(start, start + 150000)
                if not np.allclose(values[actual[span]], attributes[name][indices[span]], atol=1e-5):
                    raise OSError(f"Model reload altered triangle {name}")
        if textured:
            uv = np.asarray(mesh.triangle_uvs)
            if image.shape[:2] != texture_size or uv.shape != (len(indices), 2):
                raise OSError("Model reload lost texture dimensions or UVs")
            original = attributes["TEXCOORD_0"][indices]
            # Legacy Open3D texture coordinates use a bottom-left image origin.
            if not np.allclose(uv, original * [1, -1] + [0, 1], atol=1e-6):
                raise OSError("Model reload altered texture coordinates")
        elif fallback_colors and (not mesh.has_vertex_colors() or not np.allclose(
                np.asarray(mesh.vertex_colors)[actual], attributes["COLOR_0"][indices], atol=1e-5)):
            raise OSError("Model reload lost fallback colors")


def export_textured_glb(mesh, atlas, path, units_per_world=1.):
    path = Path(path)
    primitives = textured_primitives(mesh, atlas, units_per_world)
    jpeg = io.BytesIO()
    Image.fromarray(atlas.rgb).save(jpeg, format="JPEG", quality=92, subsampling=0)
    encoded = jpeg.getvalue()
    views, accessors, parts = [], [], []
    offset = 0

    def append_view(blob, target=None):
        nonlocal offset
        view = {"buffer": 0, "byteOffset": offset, "byteLength": blob.nbytes}
        if target is not None:
            view["target"] = target
        views.append(view)
        pad = (-blob.nbytes) % 4
        parts.extend((blob.cast("B"), b"\0" * pad))
        offset += blob.nbytes + pad
        return len(views) - 1

    def accessor(array, name):
        is_index = name == "indices"
        array = np.ascontiguousarray(array, dtype="<u4" if is_index else "<f4")
        view = append_view(memoryview(array), 34963 if is_index else 34962)
        item = {"bufferView": view, "componentType": 5125 if is_index else 5126, "count": len(array),
                "type": "SCALAR" if is_index else f"VEC{array.shape[1]}"}
        if name == "POSITION":
            item.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        accessors.append(item)
        return len(accessors) - 1

    encoded_primitives, materials = [], []
    for i, (attributes, indices) in enumerate(primitives):
        encoded_primitives.append({"attributes": {name: accessor(data, name) for name, data in attributes.items()},
                                   "indices": accessor(indices, "indices"), "material": i, "mode": 4})
        pbr = {"metallicFactor": 0, "roughnessFactor": 1}
        if i == 0:
            pbr["baseColorTexture"] = {"index": 0}
        materials.append({"name": "Photographic atlas" if i == 0 else "Vertex-color fallback",
                          "pbrMetallicRoughness": pbr, "doubleSided": True})
    image_view = append_view(memoryview(encoded))
    document = {"asset": {"version": "2.0", "generator": "vision3d"},
                "buffers": [{"byteLength": offset}], "bufferViews": views, "accessors": accessors,
                "images": [{"bufferView": image_view, "mimeType": "image/jpeg"}],
                "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
                "textures": [{"source": 0, "sampler": 0}], "materials": materials,
                "meshes": [{"primitives": encoded_primitives}],
                "nodes": [{"mesh": 0, "extras": {"exportUnitsPerSfmUnit": units_per_world,
                                                 "metricScaleVerified": False}}],
                "scenes": [{"nodes": [0]}], "scene": 0}
    metadata = json.dumps(document, separators=(",", ":")).encode()
    metadata += b" " * (-len(metadata) % 4)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".glb", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(struct.pack("<4sII", b"glTF", 2, 28 + len(metadata) + offset))
            stream.write(struct.pack("<I4s", len(metadata), b"JSON"))
            stream.write(metadata)
            stream.write(struct.pack("<I4s", offset, b"BIN\0"))
            for part in parts:
                stream.write(part)
        verify_textured_model(temporary, primitives, atlas.rgb.shape[:2], fallback_colors=True)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return encoded


def export_textured_obj(mesh, atlas, path, units_per_world=1.):
    """Write OBJ and MTL referencing the sibling photographic atlas."""
    path = Path(path)
    if not (path.parent / "atlas.jpg").is_file():
        raise ValueError("Textured OBJ requires a sibling atlas.jpg")
    primitives = textured_primitives(mesh, atlas, units_per_world)
    material_path = path.with_suffix(".mtl")
    material_path.write_text("newmtl photographic\nKd 1 1 1\nmap_Kd atlas.jpg\n\n"
                             "newmtl fallback\nKd 0.65 0.65 0.65\n", encoding="ascii")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".obj", delete=False, mode="w", encoding="ascii") as stream:
            temporary = Path(stream.name)
            stream.write(f"mtllib {material_path.name}\n")
            first_vertex = 1
            for attributes, indices in primitives:
                uv = attributes.get("TEXCOORD_0")
                stream.write(f"usemtl {'photographic' if uv is not None else 'fallback'}\n")
                np.savetxt(stream, attributes["POSITION"], fmt="v %.9g %.9g %.9g")
                np.savetxt(stream, attributes["NORMAL"], fmt="vn %.9g %.9g %.9g")
                if uv is not None:
                    np.savetxt(stream, uv * [1, -1] + [0, 1], fmt="vt %.9g %.9g")
                for start in range(0, len(indices), 90000):
                    rows = indices[start:start + 90000].astype(np.int64)
                    geometry = rows + first_vertex
                    if uv is not None:
                        vertex_triplets = np.column_stack((geometry, rows + 1, geometry))
                        np.savetxt(stream, vertex_triplets.reshape(-1, 9), fmt="f %d/%d/%d %d/%d/%d %d/%d/%d")
                    else:
                        vertex_pairs = np.column_stack((geometry, geometry))
                        np.savetxt(stream, vertex_pairs.reshape(-1, 6), fmt="f %d//%d %d//%d %d//%d")
                first_vertex += len(attributes["POSITION"])
        verify_textured_model(temporary, primitives, atlas.rgb.shape[:2], fallback_colors=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
