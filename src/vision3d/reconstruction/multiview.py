"""Bounded CPU multi-view stereo with independent third-view depth checks."""
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from vision3d.geometry.camera import parse_camera
from vision3d.io import load_image
from .pair import export_cloud, source_hash, write_json
from .asset_export import export_glb
from .state import Pose
from .surface import build_surface, write_mesh_ply
from .stereo import disparity_range, stereo_points, surface_triangles


@dataclass(frozen=True)
class DenseProfile:
    dimension: int
    references: int
    attempts: int
    relative_depth: float


PROFILES = {
    "fast": DenseProfile(640, 8, 2, 0.025),
    "balanced": DenseProfile(900, 12, 3, 0.02),
    "high": DenseProfile(1200, 16, 4, 0.015),
}


@dataclass
class DepthMap:
    reference: int
    source: int
    rotation: np.ndarray
    translation: np.ndarray
    intrinsic: np.ndarray
    depth: np.ndarray
    image: np.ndarray
    pixels: np.ndarray
    xyz: np.ndarray
    colors: np.ndarray
    transposed: bool = False


def relative_pose(a, b):
    """Transform a point in camera a coordinates into camera b coordinates."""
    rotation = b.rotation @ a.rotation.T
    return rotation, b.translation - rotation @ a.translation


def make_depth(reference, source, a, b, camera_a, camera_b, left, right, sparse, object_masks=(None, None)):
    size = (camera_a.width, camera_a.height)
    object_a, object_b = object_masks
    # stereoRectify uses one canvas size. Preserve each source's pixel geometry
    # when mapping a differently sized source onto that canvas.
    if (camera_b.width, camera_b.height) != size:
        right = cv2.resize(right, size, interpolation=cv2.INTER_LINEAR)
        camera_b = camera_b.resized(*size)
        if object_b is not None:
            object_b = cv2.resize(object_b, size, interpolation=cv2.INTER_NEAREST)
    r, t = relative_pose(a, b)
    if np.linalg.norm(t) < 1e-8:
        raise ValueError("negligible baseline")
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        camera_a.K, camera_a.D, camera_b.K, camera_b.D, size, r, t,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    if not np.isfinite(q).all():
        raise ValueError("invalid rectification")
    vertical = abs(p2[1, 3]) > abs(p2[0, 3])
    axis = int(vertical)
    rectified_rotation = r1 @ a.rotation
    rectified_translation = r1 @ a.translation
    z = (sparse @ rectified_rotation.T + rectified_translation)[:, 2]
    z = z[np.isfinite(z) & (z > 0)]
    minimum, count = disparity_range(-p2[axis, 3] / z, size[axis])
    near, far = np.percentile(z, [2, 98])
    images, masks = [], []
    for image, camera, rect, projection, object_mask in ((left, camera_a, r1, p1, object_a),
                                                        (right, camera_b, r2, p2, object_b)):
        mx, my = cv2.initUndistortRectifyMap(camera.K, camera.D, rect, projection[:, :3], size, cv2.CV_32FC1)
        masks.append((mx >= 2) & (my >= 2) & (mx < size[0] - 3) & (my < size[1] - 3))
        if object_mask is not None:
            masks[-1] &= cv2.remap(object_mask, mx, my, cv2.INTER_NEAREST).astype(bool)
        images.append(cv2.remap(image, mx, my, cv2.INTER_LINEAR))
    projection = p1[:, :3]
    if vertical:
        images = [cv2.transpose(image) for image in images]
        masks = [mask.T for mask in masks]
        q = q[:, [1, 0, 2, 3]]
        projection = projection[[1, 0, 2], :]
    local, colors, pixels = stereo_points(*images, *masks, minimum, count, q, r1, (near * .65, far * 1.55))
    if len(local) < 100:
        raise ValueError("insufficient bidirectional disparity")
    depth = np.zeros(images[0].shape[:2], np.float32)
    depth[pixels[:, 0], pixels[:, 1]] = local @ r1[2]
    xyz = (local - a.translation) @ a.rotation
    return DepthMap(reference, source, rectified_rotation, rectified_translation, projection,
                    depth, images[0], pixels, xyz, colors, vertical)


def corroborate(xyz, rgb, witness, tolerance):
    """Test projected depth and color against another left-right checked depth map."""
    camera_xyz = xyz @ witness.rotation.T + witness.translation
    z = camera_xyz[:, 2]
    finite = np.isfinite(camera_xyz).all(axis=1) & (z > 0)
    rows = np.flatnonzero(finite)
    projected = camera_xyz[rows] @ witness.intrinsic.T
    uv = projected[:, :2] / projected[:, 2:3]
    height, width = witness.depth.shape
    inside = ((uv >= 0) & (uv < [width - .5, height - .5])).all(axis=1)
    rows = rows[inside]
    x, y = np.rint(uv[inside]).astype(int).T
    depth = witness.depth[y, x]
    color = witness.image[y, x, ::-1] / 255.0
    compatible = (depth > 0) & (np.abs(z[rows] - depth) <= tolerance * depth)
    compatible &= np.mean(np.abs(rgb[rows] - color), axis=1) <= .25
    return rows[compatible], color[compatible]


def checked_samples(maps, tolerance):
    """Require a distinct reference and a third camera across the two stereo edges."""
    accepted = []
    for sample in maps:
        edges = {sample.reference, sample.source}
        counts = np.ones(len(sample.xyz), np.int32)
        colors = sample.colors.copy()
        for witness in maps:
            if witness.reference == sample.reference or {witness.reference, witness.source} == edges:
                continue
            indices, rgb = corroborate(sample.xyz, sample.colors, witness, tolerance)
            counts[indices] += 1
            colors[indices] += rgb
        mask = counts >= 2
        if mask.any():
            accepted.append((sample, mask, colors[mask] / counts[mask, None]))
    return accepted


def voxel_fuse(points, colors, size):
    """Reduce duplicate samples in world voxels and return sample-to-vertex IDs."""
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Invalid fusion voxel size")
    cells = np.floor((points - points.min(axis=0)) / size).astype(np.int64)
    _, inverse, count = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    fused = [np.column_stack([np.bincount(inverse, weights=column) / count for column in a.T])
             for a in (points, colors)]
    return *fused, inverse


def overlapping_pairs(landmarks, poses):
    pairs = defaultdict(list)
    for landmark in landmarks:
        xyz = np.asarray(landmark["xyz"], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            continue
        visible = sorted(set(map(int, landmark["observations"])) & poses.keys())
        for pair in combinations(visible, 2):
            pairs[pair].append(xyz)
    neighbors = defaultdict(list)
    for (a, b), xyz in pairs.items():
        if len(xyz) < 30:
            continue
        xyz = np.asarray(xyz)
        ray_a, ray_b = xyz - poses[a].center, xyz - poses[b].center
        norms = np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)
        cosine = np.sum(ray_a * ray_b, axis=1) / np.maximum(norms, 1e-12)
        angle = np.degrees(np.arccos(np.clip(np.median(cosine), -1, 1)))
        if 1 <= angle <= 35:
            score = len(xyz) * min(angle / 4, 1) / (1 + angle / 12)
            neighbors[a].append((score, b, xyz))
            neighbors[b].append((score, a, xyz))
    return {i: sorted(values, key=lambda entry: (-entry[0], entry[1])) for i, values in neighbors.items()}


def rasterize_polygon(vertices, size):
    vertices = np.asarray(vertices, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3
            or not np.isfinite(vertices).all() or np.any((vertices < 0) | (vertices > 1))):
        raise ValueError("Object polygons require at least three normalized [x, y] vertices")
    polygon = np.rint(vertices * (np.array(size) - 1)).astype(np.int32)
    if cv2.contourArea(polygon) < 1:
        raise ValueError("Object polygon has no area")
    mask = np.zeros(size[::-1], np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def reconstruct_multiview(sparse_dir, output, profile="balanced", mesh=False, target_faces=None,
                          glb=False, stage=lambda name: None, object_masks=None):
    """Densify one sequence result without editing any sparse artifacts."""
    start = perf_counter()
    sparse_dir, output = Path(sparse_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new dense output directory: {output}")
    if profile not in PROFILES:
        raise ValueError(f"Unknown dense profile: {profile}")
    if target_faces is not None and (not mesh or target_faces < 100):
        raise ValueError("Target faces requires a mesh and must be at least 100")
    if glb and not mesh:
        raise ValueError("GLB export requires --mesh")
    settings = PROFILES[profile]

    def read(name):
        return json.loads((sparse_dir / name).read_text(encoding="utf-8"))

    stage("Selecting overlapping registered views")
    manifest, intrinsics, pose_records = read("manifest.json"), read("intrinsics.json"), read("poses.json")
    if not isinstance(pose_records, list) or not isinstance(intrinsics, list):
        raise ValueError("Multi-view dense reconstruction needs a sparse sequence result")
    poses = {}
    for entry in pose_records:
        if entry["status"] != "registered":
            continue
        matrix = np.asarray(entry["world_to_camera"], dtype=float)
        if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("Invalid world-to-camera matrix")
        poses[entry["image_id"]] = Pose(matrix[:3, :3], matrix[:3, 3])
    if len(poses) < 3:
        raise ValueError("At least three registered views are required for dense fusion")
    neighbors = overlapping_pairs(read("landmarks.json"), poses)
    selected = sorted(neighbors, key=lambda i: (-neighbors[i][0][0], i))[:settings.references]
    if len(selected) < 3:
        raise ValueError("Insufficient shared landmarks and parallax across three registered views")

    sources = {entry["image_id"]: entry for entry in manifest["inputs"]}
    original = {entry["image_id"]: parse_camera(entry["original"]) for entry in intrinsics}
    inputs = {}
    polygons = None
    if object_masks is not None:
        polygons = json.loads(Path(object_masks).read_text(encoding="utf-8"))
        if not isinstance(polygons, dict) or any(sources[i]["name"] not in polygons for i in poses):
            raise ValueError("Object masks must map every registered image filename to a polygon")

    def image(i):
        if i not in inputs:
            source = sources[i]
            if source_hash(source["source"]) != source["sha256"]:
                raise ValueError(f"Source photograph changed: {source['source']}")
            frame = load_image(source["source"]).image
            height, width = frame.shape[:2]
            camera = original[i]
            if (width, height) != (camera.width, camera.height):
                raise ValueError("Source image dimensions disagree with sparse intrinsics")
            scale = min(1, settings.dimension / max(width, height))
            size = (round(width * scale), round(height * scale))
            if min(size) < 64:
                raise ValueError("Dense processing needs at least 64 pixels on each side")
            mask = rasterize_polygon(polygons[source["name"]], size) if polygons is not None else None
            inputs[i] = cv2.resize(frame, size, interpolation=cv2.INTER_AREA), camera.resized(*size), mask
        return inputs[i]

    cv2.setNumThreads(1)
    depth_start = perf_counter()
    maps, attempts = [], []
    used_pixels = 0
    for index, reference in enumerate(selected, 1):
        stage(f"Estimating depth for reference {index}/{len(selected)}")
        left, a, object_a = image(reference)
        if used_pixels + a.width * a.height > 8_000_000:
            attempts.append({"reference": reference, "reason": "total depth-map pixel budget reached"})
            continue
        for _, source, xyz in neighbors[reference][:settings.attempts]:
            right, b, object_b = image(source)
            try:
                depth = make_depth(reference, source, poses[reference], poses[source], a, b, left, right, xyz, (object_a, object_b))
            except (ValueError, cv2.error) as exc:
                attempts.append({"reference": reference, "source": source, "reason": str(exc)})
                stage(f"Depth pair {reference}/{source} skipped: {exc}")
                continue
            maps.append(depth)
            used_pixels += a.width * a.height
            attempts.append({"reference": reference, "source": source, "depth_pixels": len(depth.xyz)})
            break
    depth_seconds = perf_counter() - depth_start
    stage("Checking third-view depth and fusing world voxels")
    fusion_start = perf_counter()
    accepted = checked_samples(maps, settings.relative_depth)
    sample_count = sum(int(mask.sum()) for _, mask, _ in accepted)
    if len(accepted) < 2 or sample_count < 100:
        reason = f"Too little independent third-view support ({len(maps)} depth maps, {sample_count} samples)"
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "metrics.json", {"status": "rejected", "reason": reason, "pair_attempts": attempts})
        raise ValueError(reason)
    points = np.concatenate([depth.xyz[mask] for depth, mask, _ in accepted])
    colors = np.concatenate([rgb for _, _, rgb in accepted])
    voxel = float(np.median([np.median(d.depth[d.depth > 0]) / np.linalg.norm(d.intrinsic[0, :2])
                            for d, _, _ in accepted]))
    xyz, colors, inverse = voxel_fuse(points, colors, voxel)
    fusion_seconds = perf_counter() - fusion_start
    surface, mesh_info = None, None
    mesh_start = perf_counter()
    if mesh:
        stage("Building open surface from corroborated depth")
        offset = 0
        triangles = []
        for depth, mask, _ in accepted:
            pixels = depth.pixels[mask]
            heights = depth.depth[pixels[:, 0], pixels[:, 1]]
            local = surface_triangles(pixels, depth.depth.shape[::-1], heights, max_relative_jump=.025)
            if depth.transposed:
                local = local[:, [0, 2, 1]]
            triangles.append(inverse[offset + local])
            offset += len(pixels)
        try:
            surface, mesh_info = build_surface(xyz, colors, np.concatenate(triangles), target_faces)
        except ValueError as exc:
            mesh_info = {"status": "rejected", "reason": str(exc)}
    mesh_seconds = perf_counter() - mesh_start
    appearance_start = perf_counter()
    if surface is not None:
        stage("Reprojecting mesh vertices into depth-checked photographs")
        vertices = np.asarray(surface.vertices)
        prior = np.asarray(surface.vertex_colors)
        sampled = np.zeros_like(prior)
        counts = np.zeros(len(vertices), np.int32)
        for depth in maps:
            indices, rgb = corroborate(vertices, prior, depth, settings.relative_depth)
            sampled[indices] += rgb
            counts[indices] += 1
        hit = counts > 0
        prior[hit] = sampled[hit] / counts[hit, None]
        mesh_info["recolored_vertices"] = int(hit.sum())
        mesh_info["multi_image_color_vertices"] = int((counts >= 2).sum())
        mesh_info["fallback_fused_color_vertices"] = int((~hit).sum())
    appearance_seconds = perf_counter() - appearance_start
    stage("Validating and exporting colored PLY")
    export_start = perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    export_cloud(output / "dense.ply", xyz, colors)
    if surface is not None:
        write_mesh_ply(output / "mesh.ply", surface)
        mesh_info.update(status="success", representation="vertex_colored_open_surface")
        if glb:
            stage("Writing and independently reloading GLB")
            try:
                export_glb(surface, output / "mesh.glb")
            except (ValueError, OSError) as exc:
                mesh_info["glb"] = {"status": "failed", "reason": str(exc)}
            else:
                mesh_info["glb"] = {"status": "success", "axis_transform": [1, -1, -1], "color": "linear COLOR_0"}
    if polygons is not None:
        write_json(output / "object_masks.json", polygons)
    write_json(output / "poses.json", pose_records)
    write_json(output / "intrinsics.json", intrinsics)
    metrics = {"status": "success", "representation": "dense_multiview", "source_sparse": str(sparse_dir.resolve()),
               "profile": profile, "settings": asdict(settings), "object_masks": bool(polygons), "registered_images": len(poses), "dense_points": len(xyz),
               "depth_maps": len(maps), "fused_references": [d.reference for d, _, _ in accepted],
               "raw_depth_samples": sum(len(d.xyz) for d in maps), "checked_samples": sample_count,
               "depth_canvas_pixels": used_pixels,
               "voxel_size": voxel, "pair_attempts": attempts, "mesh_requested": mesh,
               "approximate_intrinsics": any(original[i].approximate for d in maps for i in (d.reference, d.source)),
               "mesh": mesh_info,
               "stage_seconds": {"depth": depth_seconds, "fusion": fusion_seconds, "mesh": mesh_seconds,
                                 "appearance": appearance_seconds, "export": perf_counter()-export_start},
               "runtime_seconds": perf_counter() - start}
    try:
        import psutil
        metrics["peak_memory_bytes"] = getattr(psutil.Process().memory_info(), "peak_wset", None)
    except (ImportError, OSError):
        metrics["peak_memory_bytes"] = None
    write_json(output / "metrics.json", metrics)
    return metrics
