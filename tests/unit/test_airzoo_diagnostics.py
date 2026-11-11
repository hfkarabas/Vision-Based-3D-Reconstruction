import importlib
from pathlib import Path

import numpy as np
import open3d as o3d


def script(monkeypatch, name):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module(name)


def test_depth_evidence_ignores_holes_depth_edges_and_points_behind_surface(monkeypatch):
    module = script(monkeypatch, "audit_intersecting_depth_layers")
    depth = np.full((6, 9), 10.)
    depth[1, 5] = 0
    depth[1, 7] = 20
    uv = np.array([[1., 1.], [2., 1.], [3., 1.], [5., 1.], [7., 1.], [8.6, 1.], [2., 2.]])
    z = np.array([10.1, 9.5, 11., 8., 8., 8., -1.])
    valid, agree, free = module.depth_evidence(uv, z, depth, np.full(len(z), .2))
    assert valid.tolist() == [True, True, True, False, False, False, False]
    assert agree.tolist() == [True, False, False, False, False, False, False]
    assert free.tolist() == [False, True, False, False, False, False, False]


def test_removal_needs_separated_votes_and_a_supported_competing_face(monkeypatch):
    module = script(monkeypatch, "audit_intersecting_depth_layers")
    centers = np.array([[0., 0., 0.], [.01, 0., 0.], [1., 0., 0.]])
    points = np.array([[0., 0., 10.], [0., 0., 10.]])
    votes = np.array([[True, True], [True, False], [False, True]])
    assert module.separated_votes(votes, centers, points, 2).tolist() == [False, True]
    pairs = np.array([[0, 1], [1, 2], [3, 4], [4, 5]])
    supported = np.array([True, False, False, False, False, True])
    contradicted = np.array([False, True, True, True, True, False])
    partial = np.array([True, False, False, False, True, True])
    assert module.removal_candidates(pairs, supported, contradicted, partial).tolist() == [
        False, True, False, False, False, False]


def test_first_hit_requires_the_target_face_and_its_depth(monkeypatch):
    module = script(monkeypatch, "audit_intersecting_depth_layers")
    vertices = np.array([[-1, -1, 2], [1, -1, 2], [0, 1, 2], [-1, -1, 3], [1, -1, 3], [0, 1, 3]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5]]))
    rays = np.array([[0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 2]], np.float32)
    assert module.first_hit_faces(module.ray_scene(mesh), rays, np.array([0, 1, 1])).tolist() == [True, False, False]


def test_dsm_exact_membership_reports_removed_and_added_residuals(monkeypatch):
    module = script(monkeypatch, "compare_airzoo_dsm")
    points = np.array([[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    result = module.exact_point_comparison((points[:2], np.array([.5, -2.])),
                                          (points[1:], np.array([-2., 1.])))
    assert result["identical_points"] == result["only_baseline_points"] == result["only_variant_points"] == 1
    assert result["identical_absolute_residual_m"] == [2.] * 5
    assert result["only_baseline_absolute_residual_m"] == [.5] * 5
    assert result["only_variant_absolute_residual_m"] == [1.] * 5


def test_grid_normal_filter_requires_two_disjoint_supporting_image_pairs(monkeypatch):
    module = script(monkeypatch, "experiment_grid_normal_consensus")
    base = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
    points = np.concatenate((base, base + [0, 0, .1], base + [0, .1, 0]))
    main = np.array([[0, 1, 2]])
    witnesses = np.array([[0, 3, 6], [1, 4, 7], [2, 5, 8]])
    grids = [main, witnesses, witnesses]
    independent = module.supported_grid_masks(points, grids, [{0, 1}, {2, 3}, {4, 5}])
    correlated = module.supported_grid_masks(points, grids, [{0, 1}, {2, 3}, {3, 4}])
    assert independent[0].tolist() == [False]
    assert correlated[0].tolist() == [True]
    assert independent[1].all() and independent[2].all()
