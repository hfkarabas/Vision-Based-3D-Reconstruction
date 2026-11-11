import numpy as np
import open3d as o3d
import pytest

from vision3d.reconstruction.surface_validation import repair_winding, sampled_deviation, topology
from scripts.experiment_surface_seams import weld_boundary_seams


def plane(z, inconsistent=False):
    xyz = np.array([[0., 0., z], [1., 0., z], [0., 1., z], [1., 1., z]])
    faces = [[0, 1, 2], [1, 2, 3] if inconsistent else [1, 3, 2]]
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz), o3d.utility.Vector3iVector(faces))


def test_intersection_probe_distinguishes_crossings_from_shared_edges_and_depth_separation():
    mesh = plane(0)
    assert not len(mesh.get_self_intersecting_triangles())
    upright = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector([[.2, .3, -1.], [.8, .3, -1.], [.3, .3, 1.]]),
        o3d.utility.Vector3iVector([[0, 1, 2]]))
    crossing = mesh + upright
    assert (0, 2) in {tuple(sorted(pair)) for pair in crossing.get_self_intersecting_triangles()}
    assert not len((mesh + upright.translate([0., 0., 2.])).get_self_intersecting_triangles())


def test_winding_repair_resolves_shared_edge_without_discarding_valid_faces():
    mesh = plane(0, inconsistent=True)
    assert topology(mesh)[1]["winding_conflicts"] == 1
    result = repair_winding(mesh)
    assert result["winding_faces_flipped"] == 1
    assert result["contradictory_faces_cut"] == 0
    report = topology(mesh)[1]
    assert report["winding_conflicts"] == 0
    assert report["boundary_edges"] == 4


def test_weighted_winding_preserves_planar_edge_under_opposing_input_order():
    mesh = plane(0, inconsistent=True)
    result = repair_winding(mesh, geometry_weighted=True)
    assert result["contradictory_edges"] == 0
    assert topology(mesh)[1]["winding_conflicts"] == 0
    assert len(mesh.triangles) == 2


def test_surface_deviation_detects_parallel_offset():
    report = sampled_deviation(plane(0), plane(.02), samples=1000)
    assert report["distance_p50_p95_p99"] == pytest.approx([.02] * 3, abs=1e-7)
    assert report["unsigned_normal_angle_p50_p95_deg"] == pytest.approx([0.] * 2, abs=.03)


def test_seam_join_keeps_all_faces_and_refuses_offset_surfaces():
    def surface(z):
        mesh = plane(0) + plane(z).translate([1.01, 0, 0])
        mesh.paint_uniform_color([.2, .3, .4])
        return mesh

    candidate, report = weld_boundary_seams(surface(0), .02)
    assert report == {"joined_edge_pairs": 1, "merged_vertices": 2}
    assert topology(candidate)[1]["components"] == 1
    assert len(candidate.triangles) == 4
    separated, report = weld_boundary_seams(surface(.01), .02)
    assert report["joined_edge_pairs"] == 0
    assert topology(separated)[1]["components"] == 2
