import numpy as np
import open3d as o3d
import pytest

from vision3d.reconstruction.surface_validation import repair_winding, sampled_deviation, topology


def plane(z, inconsistent=False):
    xyz = np.array([[0., 0., z], [1., 0., z], [0., 1., z], [1., 1., z]])
    faces = [[0, 1, 2], [1, 2, 3] if inconsistent else [1, 3, 2]]
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz), o3d.utility.Vector3iVector(faces))


def test_winding_repair_resolves_shared_edge_without_discarding_valid_faces():
    mesh = plane(0, inconsistent=True)
    assert topology(mesh)[1]["winding_conflicts"] == 1
    result = repair_winding(mesh)
    assert result["winding_faces_flipped"] == 1
    assert result["contradictory_faces_cut"] == 0
    report = topology(mesh)[1]
    assert report["winding_conflicts"] == 0
    assert report["boundary_edges"] == 4


def test_surface_deviation_detects_parallel_offset():
    report = sampled_deviation(plane(0), plane(.02), samples=1000)
    assert report["distance_p50_p95_p99"] == pytest.approx([.02] * 3, abs=1e-7)
    assert report["unsigned_normal_angle_p50_p95_deg"] == pytest.approx([0.] * 2, abs=.03)
