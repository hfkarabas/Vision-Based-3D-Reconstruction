import json

import numpy as np
from PIL import Image
import pytest

from vision3d.geometry.camera import Camera
from vision3d.reconstruction.multiview import make_depth, reconstruct_multiview, relative_pose, voxel_fuse
from vision3d.reconstruction.pair import source_hash
from vision3d.reconstruction.state import Pose


def test_world_to_relative_camera_transform():
    a = Pose([[0, -1, 0], [1, 0, 0], [0, 0, 1]], [1, 2, 3])
    b = Pose([[1, 0, 0], [0, 0, -1], [0, 1, 0]], [-1, 0, 2])
    r, t = relative_pose(a, b)
    world = np.array([.5, 1, 9])
    assert np.allclose(r @ (a.rotation @ world + a.translation) + t, b.rotation @ world + b.translation)


def test_vertical_stereo_keeps_world_scale():
    left = np.random.default_rng(4).integers(0, 256, (192, 256, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:-10] = left[10:]
    camera = Camera(256, 192, 120, 120, 127.5, 95.5)
    a = Pose(np.eye(3), [0, 0, 0])
    b = Pose(np.eye(3), [0, -1, 0])
    sparse = np.tile([0, 0, 12], (40, 1))
    result = make_depth(0, 1, a, b, camera, camera, left, right, sparse)
    assert result.transposed
    assert len(result.xyz) > 10000
    assert np.median(result.xyz[:, 2]) == pytest.approx(12, abs=.2)


def test_voxel_averages_both_geometry_and_color():
    points = np.array([[0, 0, 0], [.01, 0, 0], [.5, 0, 0]])
    xyz, rgb, ids = voxel_fuse(points, np.eye(3), .1)
    assert ids.tolist() == [0, 0, 1]
    assert np.allclose(xyz[0], [.005, 0, 0])
    assert np.allclose(rgb[0], [.5, .5, 0])


@pytest.mark.parametrize(("profile", "common_witnesses"), [("fast", False), ("uav", False), ("uav", True)])
def test_three_view_dense_pipeline_with_mesh_and_glb(tmp_path, profile, common_witnesses):
    camera = Camera(256, 192, 120, 120, 127.5, 95.5)
    base = np.random.default_rng(17).integers(0, 256, (192, 256, 3), dtype=np.uint8)
    source = tmp_path / "sparse"
    source.mkdir()
    poses, inputs, intrinsics = [], [], []
    for i in range(3):
        shift = i * 10
        image = np.roll(base, -shift, axis=1)
        path = tmp_path / f"{i}.png"
        Image.fromarray(image).save(path)
        pose = Pose(np.eye(3), [-i, 0, 0])
        poses.append({"image_id": i, "status": "registered", "world_to_camera": pose.matrix.tolist()})
        inputs.append({"image_id": i, "source": str(path), "sha256": source_hash(path)})
        intrinsics.append({"image_id": i, "original": camera.to_dict(), "processed": camera.to_dict()})
    landmarks = [{"xyz": [float(x), float(y), 12], "observations": {"0": 0, "1": 0, "2": 0}}
                 for x in np.linspace(-3, 3, 10) for y in np.linspace(-3, 3, 10)]
    files = {"poses.json": poses, "intrinsics.json": intrinsics,
             "manifest.json": {"inputs": inputs}, "landmarks.json": landmarks}
    for name, data in files.items():
        (source / name).write_text(json.dumps(data))
    out = tmp_path / "out"
    result = reconstruct_multiview(source, out, profile=profile, mesh=True, glb=True, obj=True,
                                   mesh_common_witnesses=common_witnesses)
    assert result["depth_maps"] == 3 and len(result["fused_references"]) == 3
    assert result["checked_samples"] > result["dense_points"] > 1000
    assert result["mesh"]["multi_image_color_vertices"] > 100
    assert result["mesh"]["glb"]["status"] == "success"
    assert result["mesh"]["obj"]["status"] == "success"
    if common_witnesses:
        assert sum(row["faces_kept"] for row in result["mesh_common_witnesses"]) > 100
    if profile == "uav":
        assert result["mesh"]["open_manifold_edges"]
        assert all(row["checked_samples"] > 100 for row in result["depth_support"])
    assert (out / "mesh.glb").stat().st_size
    with pytest.raises(FileExistsError):
        reconstruct_multiview(source, out)
    (tmp_path / "1.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        reconstruct_multiview(source, tmp_path / "invalid")
    poses.pop()
    (source / "poses.json").write_text(json.dumps(poses))
    with pytest.raises(ValueError, match="three registered"):
        reconstruct_multiview(source, tmp_path / "missing")



def test_object_polygon_bounds_and_fill():
    from vision3d.reconstruction.multiview import rasterize_polygon

    mask = rasterize_polygon([[.25, .25], [.75, .25], [.75, .75], [.25, .75]], (200, 100))
    assert mask.shape == (100, 200) and mask[50, 100] and not mask[0, 0]
    with pytest.raises(ValueError, match="normalized"):
        rasterize_polygon([[-1, 0], [1, 0], [1, 1]], (200, 100))
