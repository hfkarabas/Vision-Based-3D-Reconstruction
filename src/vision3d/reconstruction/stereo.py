"""Experimental bounded dense stereo from a completed calibrated pair run."""
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from vision3d.geometry.camera import parse_camera
from vision3d.io import load_image
from vision3d.preprocessing import grayscale
from .state import Pose
from .pair import export_cloud, source_hash, write_json


def disparity_range(disparities, width):
    """Infer a bounded search interval with padding from robust sparse support."""
    values = np.asarray(disparities)
    if len(values) < 20 or not np.isfinite(values).all():
        raise ValueError("Dense stereo requires at least 20 finite sparse disparities")
    low, high = np.percentile(values, [2, 98])
    padding = max(8, (high - low) * 0.25)
    start = int(np.floor(low - padding))
    count = int(np.ceil((high + padding - start + 1) / 16)) * 16
    if count > min(256, width // 2) or abs(start) + count >= width:
        raise ValueError("Sparse disparity range exceeds the bounded dense search")
    return start, count


def stereo_points(left, right, valid_left, valid_right, minimum, count, q, rotation, depth_bounds):
    """Check bidirectional matches before Q reprojection and return world XYZ/RGB."""
    def match(a, b, start):
        solver = cv2.StereoSGBM_create(minDisparity=start, numDisparities=count, blockSize=5,
                                      P1=200, P2=800, uniquenessRatio=10, speckleWindowSize=80,
                                      speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        return solver.compute(a, b).astype(np.float32) / 16

    luma_left, luma_right = grayscale(left), grayscale(right)
    dl = match(luma_left, luma_right, minimum)
    reverse_minimum = -(minimum + count - 1)
    dr = match(luma_right, luma_left, reverse_minimum)
    y, x = np.nonzero((dl >= minimum) & valid_left)
    disparity = dl[y, x]
    xr = np.rint(x - disparity).astype(int)
    in_bounds = (xr >= 0) & (xr < dl.shape[1])
    y, x, xr, disparity = y[in_bounds], x[in_bounds], xr[in_bounds], disparity[in_bounds]
    reverse = dr[y, xr]
    consistent = valid_right[y, xr] & (reverse >= reverse_minimum) & (np.abs(disparity + reverse) <= 1)
    y, x, disparity = y[consistent], x[consistent], disparity[consistent]
    homogeneous = np.column_stack([x, y, disparity, np.ones(len(x))]) @ q.T
    nonzero = np.abs(homogeneous[:, 3]) > 1e-12
    xyz = homogeneous[nonzero, :3] / homogeneous[nonzero, 3:]
    good = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] >= depth_bounds[0]) & (xyz[:, 2] <= depth_bounds[1])
    y, x = y[nonzero][good], x[nonzero][good]
    return xyz[good] @ rotation, left[y, x, ::-1] / 255.0, np.column_stack([y, x])


def surface_triangles(pixels, size, depth, max_relative_jump=0.02):
    """Connect adjacent valid samples only, leaving holes and depth boundaries open."""
    grid = np.full(size[::-1], -1, dtype=np.int32)
    grid[pixels[:, 0], pixels[:, 1]] = np.arange(len(pixels))
    a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]
    halves = []
    for corners in ((a, c, b), (b, c, d)):
        tris = np.stack(corners, axis=-1).reshape(-1, 3)
        tris = tris[(tris >= 0).all(axis=1)]
        z = depth[tris]
        halves.append(tris[np.ptp(z, axis=1) <= max_relative_jump * z.min(axis=1)])
    return np.concatenate(halves)


def export_stereo_mesh(path, xyz, rgb, triangles):
    import open3d as o3d

    if not len(triangles):
        raise ValueError("No continuous disparity regions support a mesh")
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz), o3d.utility.Vector3iVector(triangles))
    mesh.vertex_colors = o3d.utility.Vector3dVector(rgb)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(path), mesh):
        raise OSError(f"Cannot export stereo mesh: {path}")
    loaded = o3d.io.read_triangle_mesh(str(path))
    if (not loaded.has_vertex_colors() or not loaded.has_vertex_normals() or
            not np.array_equal(np.asarray(loaded.triangles), np.asarray(mesh.triangles)) or
            not np.allclose(np.asarray(loaded.vertices), np.asarray(mesh.vertices), rtol=1e-6, atol=1e-8) or
            not np.allclose(np.asarray(loaded.vertex_colors), np.asarray(mesh.vertex_colors), atol=1/255 + 1e-9)):
        raise OSError("Stereo mesh reload validation failed")
    return len(mesh.vertices), len(mesh.triangles)


def reconstruct_dense_pair(pair_dir, output, mesh=False):
    """Run opt-in CPU stereo on two equal-size views with explicit intrinsics."""
    started = perf_counter()
    pair_dir, output = Path(pair_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new dense output directory: {output}")

    def read(name):
        return json.loads((pair_dir / name).read_text(encoding="utf-8"))

    manifest, intrinsics, poses = read("manifest.json"), read("intrinsics.json"), read("poses.json")
    if (not isinstance(intrinsics, dict) or not isinstance(poses, dict) or
            read("metrics.json").get("status") != "success" or len(manifest["inputs"]) != 2):
        raise ValueError("Dense stereo needs a successful reconstruct-pair result")
    cameras = [parse_camera(entry) for entry in intrinsics["processed"]]
    if len(cameras) != 2 or any(c.approximate for c in cameras):
        raise ValueError("Dense stereo currently requires two explicitly calibrated cameras")
    a, b = cameras
    size = a.width, a.height
    if size != (b.width, b.height) or min(size) < 64 or max(size) > 1600:
        raise ValueError("Dense stereo requires equal processed sizes between 64 and 1600 pixels per side")
    pose = np.asarray(poses["world_to_camera2"], dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or np.linalg.norm(pose[:3, 3]) < 1e-9:
        raise ValueError("Invalid relative pose for stereo")
    Pose(pose[:3, :3], pose[:3, 3])
    images = []
    for entry in manifest["inputs"]:
        source = Path(entry["source"])
        if source_hash(source) != entry["sha256"]:
            raise ValueError(f"Image changed since sparse reconstruction: {source}")
        image = load_image(source).image
        if image.shape[:2] != size[::-1]:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        images.append(image)

    cv2.setNumThreads(1)
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(a.K, a.D, b.K, b.D, size, pose[:3, :3], pose[:3, 3],
                                               flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    if abs(p2[0, 3]) <= abs(p2[1, 3]) or not np.isfinite(q).all():
        raise ValueError("This dense stereo mode requires horizontal rectification")
    with np.load(pair_dir / "correspondences.npz", allow_pickle=False) as points:
        xyz = points["xyz_pose_points"][points["valid_pose_points"]] @ r1.T
    depths = xyz[np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0), 2]
    minimum, count = disparity_range(-p2[0, 3] / depths, size[0])
    near, far = np.percentile(depths, [2, 98])
    rectified, masks = [], []
    for image, camera, r, p in zip(images, cameras, (r1, r2), (p1, p2)):
        mx, my = cv2.initUndistortRectifyMap(camera.K, camera.D, r, p[:, :3], size, cv2.CV_32FC1)
        masks.append((mx >= 2) & (my >= 2) & (mx < size[0] - 3) & (my < size[1] - 3))
        rectified.append(cv2.remap(image, mx, my, cv2.INTER_LINEAR))
    xyz, colors, pixels = stereo_points(*rectified, *masks, minimum, count, q, r1, (near * 0.5, far * 2))
    if len(xyz) < 100:
        raise ValueError("Too little consistent dense disparity; use a better overlapping calibrated pair")
    triangles = surface_triangles(pixels, size, xyz @ r1[2]) if mesh else None
    if mesh and not len(triangles):
        raise ValueError("No continuous disparity regions support a mesh")
    output.mkdir(parents=True, exist_ok=False)
    export_cloud(output / "dense.ply", xyz, colors)
    write_json(output / "poses.json", poses)
    write_json(output / "intrinsics.json", intrinsics)
    metrics = {"status": "success", "registered_images": 2, "representation": "dense_two_view_stereo", "source_pair": str(pair_dir.resolve()),
               "dense_points": len(xyz), "disparity_min": minimum, "disparity_count": count,
               "resolution": list(size), "runtime_seconds": perf_counter() - started,
               "coordinate_convention": "sparse pair world frame (first camera), arbitrary baseline units",
               "validation": "left-right disparity consistency within 1 px; sparse-derived depth bounds"}
    if mesh:
        vertices, faces = export_stereo_mesh(output / "mesh.ply", xyz, colors, triangles)
        metrics["mesh"] = {"representation": "vertex_colored_stereo_surface", "vertices": vertices,
                           "triangles": faces, "relative_depth_jump_limit": 0.02}
    metrics["runtime_seconds"] = perf_counter() - started
    write_json(output / "metrics.json", metrics)
    return metrics
