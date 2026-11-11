from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

from vision3d.geometry.camera import Camera
from vision3d.reconstruction.surface import footprint_edge_gate


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))


def test_material_render_retains_white_faces_and_reports_front_hits(tmp_path, scripts):
    from audit_reconstruction_visuals import View, render_mesh

    xyz = np.array([[-.6, -.5, 0], [.6, -.5, 0], [0, .6, 0]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz), o3d.utility.Vector3iVector([[0, 1, 2]]))
    mesh.paint_uniform_color([1, 1, 1])
    view = View({"basis": np.eye(3), "center": [0, 0, 0], "span": 2}, np.array([0, 0, 1]), 64)
    stats, masks = render_mesh([mesh], {"top": view}, tmp_path, "glb", material_modes=True)
    top = stats["views"]["top"]
    assert top["front_face_pixels"] == top["hit_pixels"] == masks["top"].sum() > 0
    assert top["near_white_fallback_pixels"] == top["hit_pixels"]
    assert top["photographic_first_hits"] == 0


def test_common_pose_alignment_recovers_scale_even_for_collinear_centers(scripts):
    from render_dense_ablation import common_pose_similarity

    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    shift = np.array([1, 2, 3])

    def pose(matrix, center):
        result = np.eye(4)
        result[:3, :3] = matrix
        result[:3, 3] = -matrix @ center
        return result

    centers = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    source = {i: pose(np.eye(3), center) for i, center in enumerate(centers)}
    target = {i: pose(rotation.T, 2 * rotation @ center + shift) for i, center in enumerate(centers)}
    scale, fitted, offset, info = common_pose_similarity(source, target)
    assert scale == pytest.approx(2)
    assert np.allclose(fitted, rotation) and np.allclose(offset, shift)
    assert info["center_residual_sfm_p50_p95"] == [0., 0.]


def test_external_depth_camera_checks_half_pixel_resize(scripts):
    from replay_patchmatch_depth import camera_intrinsic

    camera = Camera(1200, 800, 800, 800, 599.5, 399.5)
    external = SimpleNamespace(width=600, height=400, model_name="PINHOLE", params=[400, 400, 300, 200])
    assert np.array_equal(camera_intrinsic(external, camera), camera.resized(600, 400).K)
    external.params[2] -= .5
    with pytest.raises(ValueError, match="differs"):
        camera_intrinsic(external, camera)


def test_footprint_edge_gate_requires_shared_witnesses_and_facing_surface():
    xyz = np.array([[0, 0, 10], [.8, 0, 10], [0, .4, 10], [0, 0, 10.4]])
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 2]])
    keep = footprint_edge_gate(xyz, faces, np.zeros(3), 30, .1, np.array([True, True, False]))
    assert keep.tolist() == [True, False, False]
