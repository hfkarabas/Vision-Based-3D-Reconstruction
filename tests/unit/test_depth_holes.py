import numpy as np

from vision3d.reconstruction.depth_holes import supported_holes
from vision3d.reconstruction.multiview import DepthMap


def test_empty_primary_depth_requires_two_disjoint_coherent_witnesses():
    k = np.array([[50, 0, 12], [0, 50, 8], [0, 0, 1]])
    image = np.full((20, 26, 3), 128, np.uint8)
    centers = {i: np.array([i / 4, 0., 0.]) for i in range(7)}

    def depth(reference, source, pixels=((8, 12), (9, 12))):
        xyz = np.array([[0., 0., 10.], [0., .2, 10.]])
        return DepthMap(reference, source, np.eye(3), -centers[reference], k,
                        np.full(image.shape[:2], 10., np.float32), image, np.array(pixels), xyz,
                        np.full((2, 3), 128 / 255), focal_baseline=500.)

    primary, alternate = depth(0, 1), depth(0, 2)
    primary.depth[:] = 0
    first, second = depth(3, 4), depth(5, 6)

    def recovered(witnesses):
        return supported_holes(primary, alternate, witnesses, centers, 28, .02, 1.5)

    assert recovered([first, second]).tolist() == [[0, 8, 12, 0, 1], [1, 9, 12, 0, 1]]
    assert not len(recovered([first, depth(5, 4)]))
    assert not len(recovered([first, depth(5, 2)]))
    second.depth[:] = 11
    assert not len(recovered([first, second]))
    second.depth[:] = 10
    primary.depth[7, 12] = 10  # Conservative guard beside a depth boundary.
    assert recovered([first, second]).tolist() == [[1, 9, 12, 0, 1]]
    alternate.focal_baseline = 200
    assert not len(recovered([first, second]))
