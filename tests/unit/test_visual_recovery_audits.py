import numpy as np

from vision3d.reconstruction.surface import footprint_edge_gate


def test_footprint_edge_gate_requires_shared_witnesses_and_facing_surface():
    xyz = np.array([[0, 0, 10], [.8, 0, 10], [0, .4, 10], [0, 0, 10.4]])
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 2]])
    keep = footprint_edge_gate(xyz, faces, np.zeros(3), 30, .1, np.array([True, True, False]))
    assert keep.tolist() == [True, False, False]
