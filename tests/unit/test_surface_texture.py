import numpy as np
import open3d as o3d
from PIL import Image
import pytest

from vision3d.geometry.camera import Camera
from vision3d.demo.viewer import load_textured_meshes
from vision3d.reconstruction.pair import source_hash
from vision3d.reconstruction.state import Pose
from vision3d.reconstruction.surface_texture import TextureAtlas, bake_texture
from vision3d.reconstruction.surface_subdivision import subdivide_visible_candidates
from vision3d.reconstruction.surface_validation import SurfaceView, topology
from vision3d.reconstruction.textured_export import export_textured_glb, export_textured_obj, prepare_texture_mesh, textured_primitives


def test_photographic_atlas_rejects_hidden_faces_and_roundtrips_fallback(tmp_path):
    photo = np.zeros((64, 64, 3), dtype=np.uint8)
    photo[:, :, 0] = np.arange(64, dtype=np.uint8)[None, :] * 4
    photo[:, :, 2] = 60
    path = tmp_path / "photo.png"
    Image.fromarray(photo).save(path)
    view = SurfaceView(0, Pose(np.eye(3), [0, 0, 0]), Camera(64, 64, 40, 40, 31.5, 31.5), path, source_hash(path))
    near = np.array([[-.7, -.7, 2], [.7, -.7, 2], [-.7, .7, 2], [.7, .7, 2]])
    xyz = np.concatenate((near, near + [0, 0, 1]))
    faces = [[0, 2, 1], [1, 2, 3], [4, 6, 5], [5, 6, 7]]
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz), o3d.utility.Vector3iVector(faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.full_like(xyz, .4))
    mesh.compute_vertex_normals()
    atlas = bake_texture(mesh, [view], atlas_size=512)
    assert atlas.chart.tolist() == [0, 0, -1, -1]
    assert atlas.metrics["textured_area_fraction"] == pytest.approx(.5)
    assert atlas.metrics["uv_center_projection_error_px_p50_p95"] == pytest.approx([0, 0], abs=1e-6)
    expected = np.rint(atlas.uv[0].mean(axis=0) * 512 - .5).astype(int)
    assert atlas.rgb[expected[1], expected[0], 2] == 60
    output = tmp_path / "model.glb"
    encoded = export_textured_glb(mesh, atlas, output, units_per_world=2.)
    assert encoded[:2] == b"\xff\xd8"
    assert output.read_bytes()[:4] == b"glTF"
    preview = load_textured_meshes(output)
    assert sum(len(group.triangles) for group in preview) == len(faces)
    assert sum(group.has_textures() for group in preview) == 1
    (tmp_path / "atlas.jpg").write_bytes(encoded)
    obj = tmp_path / "model.obj"
    export_textured_obj(mesh, atlas, obj, units_per_world=2.)
    assert "usemtl photographic" in obj.read_text(encoding="ascii")
    atlas.uv[0, 2] = atlas.uv[0, 1]
    with pytest.raises(ValueError, match="degenerate"):
        textured_primitives(mesh, atlas, 1.)


def test_precision_cleanup_drops_only_redundant_face():
    xyz = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz), o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5]]))
    mesh.paint_uniform_color([.1, .2, .3])
    result, stats = prepare_texture_mesh(mesh, 1.)
    assert len(result.triangles) == 1
    assert stats["duplicate_faces_removed"] == 1 and stats["relative_area_loss"] == 0


@pytest.mark.parametrize(("opposite", "maximum_normal_dot"), [([0, 1, .1], -.99), ([0, .5, np.sqrt(.75)], -.49)])
def test_export_splits_shading_at_a_fold_without_moving_vertices(opposite, maximum_normal_dot):
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], opposite])
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz), o3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()
    uv = np.array([[[.1, .1], [.9, .1], [.1, .9]], [[.9, .1], [.1, .1], [.1, .9]]], dtype=np.float32)
    atlas = TextureAtlas(np.zeros((4, 4, 3), np.uint8), np.zeros(2, np.int32), uv, {})
    [(attributes, indices)] = textured_primitives(mesh, atlas, 1.)
    corner_normals = attributes["NORMAL"][indices].reshape(2, 3, 3)
    assert len(attributes["POSITION"]) == 6
    assert corner_normals[0, 0] @ corner_normals[1, 1] < maximum_normal_dot
    assert np.allclose(attributes["POSITION"][indices].reshape(2, 3, 3), xyz[faces] * [1, -1, -1])


def test_visibility_subdivision_keeps_neighbor_conforming_and_area_exact():
    xyz = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz), o3d.utility.Vector3iVector([[0, 1, 2], [1, 3, 2]]))
    mesh.paint_uniform_color([.1, .2, .3])
    result, count = subdivide_visible_candidates(mesh, np.array([True, False]), 1)
    _, audit = topology(result)
    assert count == 1 and len(result.triangles) == 6
    assert audit["components"] == 1 and audit["nonmanifold_edges"] == 0 and audit["winding_conflicts"] == 0
    assert audit["area"] == pytest.approx(1.)
