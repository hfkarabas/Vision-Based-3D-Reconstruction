import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vision3d.geometry.camera import Camera
from vision3d.geometry.depth import project_depth, unproject_depth
from vision3d.reconstruction.multiview import plan_depth
from vision3d.reconstruction.state import Pose


@pytest.mark.parametrize("vertical", [False, True])
def test_world_pixel_roundtrip_with_rotation_translation_and_transposed_camera(vertical):
    rotation = Rotation.from_euler("xyz", [14, -22, 36], degrees=True).as_matrix()
    translation = np.array([2., -1., 4.])
    intrinsic = np.array([[690., 0, 294.2], [0, 710., 235.4], [0, 0, 1.]])
    if vertical:
        intrinsic = intrinsic[[1, 0, 2]]
    points = np.array([[1., 2., 8.], [-2., 3., 5.], [3., -2., 10.]])
    pixels, z = project_depth(points, rotation, translation, intrinsic)
    assert (z > 0).all()
    assert np.allclose(unproject_depth(pixels, z, rotation, translation, intrinsic), points, atol=1e-12, rtol=0)


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("recenter", [False, True])
def test_rectified_q_matches_world_for_rotated_camera_pair(vertical, recenter):
    rotation = Rotation.from_euler("xyz", [14, -22, 36], degrees=True).as_matrix()
    first_center = np.array([2., -1., 4.])
    second_center = first_center + rotation.T @ ([0., .7, .02] if vertical else [.7, 0., .02])
    other_rotation = Rotation.from_euler("xyz", [1, -2, .5], degrees=True).as_matrix() @ rotation
    first = Pose(rotation, -rotation @ first_center)
    second = Pose(other_rotation, -other_rotation @ second_center)
    a = Camera(960, 720, 830., 825., 485., 350.)
    b = Camera(960, 720, 839., 837., 462., 370.)
    rng = np.random.default_rng(2)
    # Two tilted surfaces at different depths.
    xy = rng.uniform([-1., -.8], [1.6, .8], (200, 2))
    z = np.repeat([6., 11.], 100) + .3 * xy[:, 0] - .2 * xy[:, 1]
    points = np.column_stack((xy, z)) @ rotation + first_center
    plan = plan_depth(first, second, a, b, points, recenter=recenter)
    assert plan.vertical == vertical
    ra, rb = plan.rotations
    pa, pb = plan.projections
    xa = (points @ first.rotation.T + first.translation) @ ra.T
    xb = (points @ second.rotation.T + second.translation) @ rb.T
    qa, qb = xa @ pa[:, :3].T, xb @ pb[:, :3].T
    uv_a, uv_b = qa[:, :2] / qa[:, 2:3], qb[:, :2] / qb[:, 2:3]
    axis = int(vertical)
    assert np.allclose(uv_a[:, 1-axis], uv_b[:, 1-axis], atol=1e-10, rtol=0)
    disparity = uv_a[:, axis] - uv_b[:, axis]
    homogeneous = np.column_stack((uv_a, disparity, np.ones(len(xa)))) @ plan.disparity_to_depth.T
    rectified = homogeneous[:, :3] / homogeneous[:, 3:]
    assert np.allclose(rectified, xa, atol=1e-10, rtol=0)
    recovered = (rectified @ ra - first.translation) @ first.rotation
    assert np.allclose(recovered, points, atol=1e-10, rtol=0)
