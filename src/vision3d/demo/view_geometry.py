"""Viewer-only camera context. Never transforms, interpolates or filters cloud points."""
import json
from pathlib import Path
import numpy as np
from vision3d.reconstruction.state import Pose


def camera_view(points, matrices):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Expected nonempty finite Nx3 points")
    matrices = [np.asarray(m, dtype=float) for m in matrices]
    if any(m.shape != (4, 4) or not np.isfinite(m).all() for m in matrices):
        raise ValueError("Expected finite 4x4 world-to-camera matrices")
    poses = [Pose(m[:3, :3], m[:3, 3]) for m in matrices]
    if not poses:
        raise ValueError("At least one camera pose is required")
    target = np.median(points, axis=0)
    front = np.mean([p.center for p in poses], axis=0) - target
    if np.linalg.norm(front) < 1e-9:
        front = -poses[0].rotation.T[:, 2]
    front /= np.linalg.norm(front)
    up = -np.mean([p.rotation.T[:, 1] for p in poses], axis=0)
    up -= (up @ front) * front
    if np.linalg.norm(up) < 1e-9:
        axis = np.eye(3)[np.argmin(np.abs(front))]
        up = axis - (axis @ front) * front
    up /= np.linalg.norm(up)
    return {"lookat": target, "front": front, "up": up}


def camera_frustum(matrix, intrinsic, width, height, size):
    """Actual pinhole image boundary at a display-only camera-space depth."""
    matrix, intrinsic = np.asarray(matrix, dtype=float), np.asarray(intrinsic, dtype=float)
    if (matrix.shape != (4, 4) or intrinsic.shape != (3, 3) or min(width, height, size) <= 0
            or not np.isfinite(intrinsic).all() or not np.isfinite(size)):
        raise ValueError("Invalid camera frustum input")
    pose = Pose(matrix[:3, :3], matrix[:3, 3])
    pixels = np.array([[0, 0, 1], [width-1, 0, 1], [width-1, height-1, 1], [0, height-1, 1]], float)
    rays = pixels @ np.linalg.inv(intrinsic).T
    corners = (rays * (size / rays[:, 2:3])) @ pose.rotation + pose.center
    points = np.vstack([pose.center, corners])
    lines = np.array([[0, k] for k in range(1, 5)] + [[1, 2], [2, 3], [3, 4], [4, 1]], int)
    return points, lines


def load_context(ply):
    folder = Path(ply).parent
    poses = json.loads((folder / 'poses.json').read_text(encoding='utf-8'))
    if isinstance(poses, dict):
        return [{"image_id": i, "world_to_camera": poses[f"world_to_camera{i + 1}"]} for i in range(2)]
    return [p for p in poses if p.get('status') == 'registered']


def load_intrinsics(ply):
    data = json.loads((Path(ply).parent / 'intrinsics.json').read_text(encoding='utf-8'))
    if isinstance(data, dict):
        return dict(enumerate(data["processed"]))
    return {p["image_id"]: p["processed"] for p in data}
