"""glTF 2.0 binary mesh export with validated colors and coordinate transform."""
import json
from pathlib import Path
import struct
import tempfile

import numpy as np


def export_obj(mesh, path):
    """Write interoperable OBJ geometry and normals. Use PLY or GLB for colors."""
    import open3d as o3d

    path = Path(path)
    xyz, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)
    # A sibling file also permits Open3D's native writer to reopen it on Windows.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".obj", delete=False) as stream:
            temporary = Path(stream.name)
        if not o3d.io.write_triangle_mesh(str(temporary), mesh, write_vertex_colors=False, write_triangle_uvs=False):
            raise OSError("OBJ export failed")
        loaded = o3d.io.read_triangle_mesh(str(temporary))
        actual = np.asarray(loaded.triangles)
        if not loaded.has_vertex_normals() or actual.shape != faces.shape:
            raise OSError("OBJ reload lost normals or triangles")
        # Readers may weld identical vertices. Verify each indexed triangle corner
        # against its source position and normal instead of assuming identical IDs.
        for expected_data, actual_data in ((xyz, np.asarray(loaded.vertices)),
                                           (normals, np.asarray(loaded.vertex_normals))):
            for start in range(0, len(faces), 100000):
                span = slice(start, start + 100000)
                if not np.allclose(expected_data[faces[span]], actual_data[actual[span]], atol=1e-6):
                    raise OSError("OBJ reload altered mesh positions or normals")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_glb(mesh, path):
    import open3d as o3d

    path = Path(path)
    # OpenCV world axes are right/down/forward, GLB uses right/up/back.
    basis = np.array([1, -1, -1])
    xyz = np.asarray(np.asarray(mesh.vertices) * basis, dtype="<f4")
    normals = np.asarray(np.asarray(mesh.vertex_normals) * basis, dtype="<f4")
    rgb = np.asarray(mesh.vertex_colors, dtype=np.float32)
    linear = np.asarray(np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4), dtype="<f4")
    faces = np.asarray(mesh.triangles, dtype="<u4").reshape(-1)
    if (not len(xyz) or not len(faces) or len(faces) % 3 or faces.max() >= len(xyz)
            or normals.shape != xyz.shape or rgb.shape != xyz.shape
            or any(not np.isfinite(a).all() for a in (xyz, normals, rgb)) or np.any((rgb < 0) | (rgb > 1))):
        raise ValueError("GLB needs a finite indexed triangle mesh with normals and vertex colors")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-7):
        raise ValueError("GLB needs nonzero vertex normals")
    normals /= lengths[:, None]

    offsets, accessors = [], []
    offset = 0
    arrays = (xyz, normals, linear, faces)
    for i, array in enumerate(arrays):
        index = i == 3
        offsets.append({"buffer": 0, "byteOffset": offset, "byteLength": array.nbytes,
                        "target": 34963 if index else 34962})
        accessors.append({"bufferView": i, "componentType": 5125 if index else 5126,
                          "count": len(array), "type": "SCALAR" if index else "VEC3"})
        offset += array.nbytes
    accessors[0].update(min=xyz.min(axis=0).tolist(), max=xyz.max(axis=0).tolist())
    document = {"asset": {"version": "2.0", "generator": "vision3d"},
                "buffers": [{"byteLength": offset}], "bufferViews": offsets, "accessors": accessors,
                "materials": [{"doubleSided": True, "pbrMetallicRoughness": {"metallicFactor": 0}}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
                                             "indices": 3, "material": 0, "mode": 4}]}],
                "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0}
    metadata = json.dumps(document, separators=(",", ":")).encode()
    metadata += b" " * (-len(metadata) % 4)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".glb", delete=False) as stream:
            temp = Path(stream.name)
            stream.write(struct.pack("<4sII", b"glTF", 2, 28 + len(metadata) + offset))
            stream.write(struct.pack("<I4s", len(metadata), b"JSON"))
            stream.write(metadata)
            stream.write(struct.pack("<I4s", offset, b"BIN\0"))
            for array in arrays:
                stream.write(memoryview(np.ascontiguousarray(array)).cast("B"))
        check = o3d.io.read_triangle_mesh(str(temp))
        loaded_faces = np.asarray(check.triangles).ravel()
        if not check.has_vertex_normals() or not check.has_vertex_colors() or len(loaded_faces) != len(faces):
            raise OSError("GLB reload lost normals, colors, or triangles")
        for actual, expected in zip((check.vertices, check.vertex_normals, check.vertex_colors), arrays):
            actual = np.asarray(actual)
            for start in range(0, len(faces), 100000):
                span = slice(start, start + 100000)
                if not np.allclose(actual[loaded_faces[span]], expected[faces[span]], atol=1e-5):
                    raise OSError("GLB reload altered triangle positions, normals, or colors")
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
