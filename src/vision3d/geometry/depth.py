"""Projection and back-projection for pinhole or transposed rectified canvases."""
import numpy as np


def project_depth(points, rotation, translation, intrinsic):
    local = points @ rotation.T + translation
    homogeneous = local @ intrinsic.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return homogeneous[:, :2] / homogeneous[:, 2:3], local[:, 2]


def unproject_depth(pixels, depth, rotation, translation, intrinsic):
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(intrinsic).T
    return (rays * depth[:, None] - translation) @ rotation
