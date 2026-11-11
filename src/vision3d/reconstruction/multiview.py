"""Bounded CPU multi-view stereo with independent third-view depth checks."""
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from vision3d.geometry.camera import parse_camera
from vision3d.io import load_image
from .pair import export_cloud, source_hash, write_json
from .asset_export import export_glb, export_obj
from .dense_quality import landmark_visibility, shared_witness_faces, spread_references, supported_samples
from .state import Pose
from .surface import build_surface, write_mesh_ply
from .stereo import DisparitySearchLimit, disparity_range, stereo_points, surface_triangles


@dataclass(frozen=True)
class DenseProfile:
    dimension: int
    references: int
    attempts: int
    relative_depth: float
    canvas_budget: int = 8_000_000
    spread_views: bool = False
    max_sparse_error_px: float | None = None
    max_return_px: float | None = None
    clean_surface: bool = False
    preflight_pairs: bool = False
    recenter_pairs: bool = False
    recentered_witnesses: int = 1
    min_effective_disparity: float | None = None
    prefer_conditioned_pairs: bool = False
    recovery_pairs: int = 0


PROFILES = {
    "fast": DenseProfile(640, 8, 2, 0.025),
    "balanced": DenseProfile(900, 12, 3, 0.02),
    "high": DenseProfile(1200, 16, 4, 0.015),
    "uav": DenseProfile(900, 18, 4, 0.02, canvas_budget=11_000_000, spread_views=True,
                        max_sparse_error_px=2.5, max_return_px=1.5, clean_surface=True,
                        preflight_pairs=True, recenter_pairs=True, recentered_witnesses=2),
}
PROFILES["uav-conditioned"] = replace(PROFILES["uav"], min_effective_disparity=28, prefer_conditioned_pairs=True)
PROFILES["uav-recovery"] = replace(PROFILES["uav-conditioned"], recovery_pairs=16)


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
    recentered: bool = False
    witness_bits: np.ndarray | None = None
    overlap_pixels: int = 0
    focal_baseline: float = 0


@dataclass(frozen=True)
class StereoPlan:
    rotations: tuple
    projections: tuple
    disparity_to_depth: np.ndarray
    vertical: bool
    minimum: int
    count: int
    depth_bounds: tuple
    recentered: bool
    median_effective_disparity: float


def relative_pose(a, b):
    """Transform a point in camera a coordinates into camera b coordinates."""
    rotation = b.rotation @ a.rotation.T
    return rotation, b.translation - rotation @ a.translation


def recenter_rectification(local_a, local_b, p1, p2, size, axis):
    """Center a virtual stereo canvas on shared geometry, keeping epipolar lines aligned."""
    valid = np.isfinite(local_a).all(axis=1) & np.isfinite(local_b).all(axis=1)
    valid &= (local_a[:, 2] > 0) & (local_b[:, 2] > 0)
    if valid.sum() < 30:
        raise ValueError("Too few shared positive-depth points to recenter rectification")
    ua = local_a[valid, :2] / local_a[valid, 2:3]
    ub = local_b[valid, :2] / local_b[valid, 2:3]
    focal = p1[0, 0]
    span = np.max([np.ptp(np.percentile(uv, [5, 95], axis=0), axis=0) for uv in (ua, ub)], axis=0)
    disparity_span = np.ptp(np.percentile((ua - ub)[:, axis], [2, 98]))
    disparity_budget = min(256, size[axis] // 2)
    target_focal = min(focal, np.min(.85 * np.array(size) / np.maximum(span, 1e-9)),
                       .6 * disparity_budget / max(disparity_span, 1e-9))
    if target_focal < focal / 3:
        raise ValueError("Shared rectification would require excessive downscaling")
    center = (np.array(size) - 1) / 2
    pa = center - target_focal * np.median(ua, axis=0)
    pb = pa.copy()
    pb[axis] = center[axis] - target_focal * np.median(ub[:, axis])
    uv_a, uv_b = target_focal * ua + pa, target_focal * ub + pb
    common = ((uv_a >= 0) & (uv_a < size) & (uv_b >= 0) & (uv_b < size)).all(axis=1)
    if common.sum() < max(30, .6 * len(common)):
        raise ValueError("Recentering does not preserve sufficient shared canvas coverage")
    a, b = np.zeros((3, 4)), np.zeros((3, 4))
    for projection, principal in ((a, pa), (b, pb)):
        projection[:2, :2] = target_focal * np.eye(2)
        projection[:2, 2] = principal
        projection[2, 2] = 1
    baseline = p2[axis, 3] / focal
    b[axis, 3] = target_focal * baseline
    offset = pa[axis] - pb[axis]
    q = np.array([[1, 0, 0, -pa[0]], [0, 1, 0, -pa[1]], [0, 0, 0, target_focal],
                  [0, 0, -1 / baseline, offset / baseline]])
    return a, b, q, offset


def plan_depth(a, b, camera_a, camera_b, sparse, *, recenter=False):
    """Preflight rectification and the bounded disparity search without decoding images."""
    size = (camera_a.width, camera_a.height)
    if (camera_b.width, camera_b.height) != size:
        camera_b = camera_b.resized(*size)
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
    local_a = sparse @ rectified_rotation.T + rectified_translation
    offset = 0
    if recenter:
        local_b = (sparse @ b.rotation.T + b.translation) @ r2.T
        p1, p2, q, offset = recenter_rectification(local_a, local_b, p1, p2, size, axis)
    z = local_a[:, 2]
    z = z[np.isfinite(z) & (z > 0)]
    minimum, count = disparity_range(offset - p2[axis, 3] / z, size[axis])
    near, far = np.percentile(z, [2, 98])
    effective_disparity = abs(p2[axis, 3]) / np.median(z)
    return StereoPlan((r1, r2), (p1, p2), q, bool(vertical), minimum, count, (near * .65, far * 1.55),
                      recenter, float(effective_disparity))


def make_depth(reference, source, a, b, camera_a, camera_b, left, right, sparse, object_masks=(None, None), *, plan=None):
    size = (camera_a.width, camera_a.height)
    object_a, object_b = object_masks
    if plan is None:
        plan = plan_depth(a, b, camera_a, camera_b, sparse)
    # Stereo rectification uses one canvas, so scale the source model with its pixels.
    if (camera_b.width, camera_b.height) != size:
        right = cv2.resize(right, size, interpolation=cv2.INTER_LINEAR)
        camera_b = camera_b.resized(*size)
        if object_b is not None:
            object_b = cv2.resize(object_b, size, interpolation=cv2.INTER_NEAREST)
    r1, r2 = plan.rotations
    p1, p2 = plan.projections
    q = plan.disparity_to_depth
    vertical = plan.vertical
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
    local, colors, pixels = stereo_points(*images, *masks, plan.minimum, plan.count, q, r1, plan.depth_bounds)
    if len(local) < 100:
        raise ValueError("insufficient bidirectional disparity")
    depth = np.zeros(images[0].shape[:2], np.float32)
    depth[pixels[:, 0], pixels[:, 1]] = local @ r1[2]
    xyz = (local - a.translation) @ a.rotation
    return DepthMap(reference, source, r1 @ a.rotation, r1 @ a.translation, projection,
                    depth, images[0], pixels, xyz, colors, vertical, plan.recentered,
                    overlap_pixels=int(np.count_nonzero(masks[0] & masks[1])),
                    focal_baseline=abs(float(p2[int(vertical), 3])))


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


def voxel_fuse(points, colors, size, *, origin=None):
    """Reduce duplicate samples in world voxels and return sample-to-vertex IDs."""
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Invalid fusion voxel size")
    anchor = points.min(axis=0) if origin is None else np.asarray(origin, dtype=float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("Invalid fusion voxel origin")
    scaled = (points - anchor) / size
    if scaled.min() < -(2. ** 63) or not scaled.max() < 2. ** 63:
        raise ValueError("Fusion voxel indices exceed the supported integer range")
    np.floor(scaled, out=scaled)
    cells = scaled.astype(np.int64)
    del scaled
    _, inverse, count = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    fused = [np.column_stack([np.bincount(inverse, weights=column) / count for column in a.T])
             for a in (points, colors)]
    return *fused, inverse


def overlapping_pairs(landmarks, poses, visibility=None):
    pairs = defaultdict(list)
    for index, landmark in enumerate(landmarks):
        xyz = np.asarray(landmark["xyz"], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            continue
        visible = sorted(set(map(int, landmark["observations"])) & poses.keys())
        if visibility is not None:
            visible = [i for i in visible if index in visibility[i]]
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
                          glb=False, stage=lambda name: None, object_masks=None, obj=False, mesh_common_witnesses=False,
                          depth_diagnostics=False, min_effective_disparity=None, prefer_conditioned_pairs=False,
                          fusion_voxel_size=None, fusion_voxel_origin=None, recovery_pairs=None):
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
    if obj and not mesh:
        raise ValueError("OBJ export requires --mesh")
    settings = PROFILES[profile]
    if min_effective_disparity is None:
        min_effective_disparity = settings.min_effective_disparity
    prefer_conditioned_pairs = prefer_conditioned_pairs or settings.prefer_conditioned_pairs
    if recovery_pairs is None:
        recovery_pairs = settings.recovery_pairs
    if mesh_common_witnesses and (not mesh or settings.max_return_px is None):
        raise ValueError("Common mesh witnesses require the UAV profile and --mesh")
    if min_effective_disparity is not None and (
            settings.max_return_px is None or not np.isfinite(min_effective_disparity)
            or min_effective_disparity <= 0):
        raise ValueError("Minimum effective disparity must be positive, finite and used with the UAV profile")
    if prefer_conditioned_pairs and min_effective_disparity is None:
        raise ValueError("Conditioned pair selection requires a minimum effective disparity")
    if (not isinstance(recovery_pairs, int) or not 0 <= recovery_pairs <= 32 or
            (recovery_pairs and (min_effective_disparity is None or mesh_common_witnesses))):
        raise ValueError("Stereo recovery requires conditioning, a budget of 0–32 pairs and no common-witness mesh gate")
    if fusion_voxel_size is not None and (not np.isfinite(fusion_voxel_size) or fusion_voxel_size <= 0):
        raise ValueError("Fusion voxel size must be positive and finite")
    if fusion_voxel_origin is not None and (fusion_voxel_size is None or
            np.shape(fusion_voxel_origin) != (3,) or not np.isfinite(fusion_voxel_origin).all()):
        raise ValueError("Fusion voxel origin requires a fixed voxel size and three finite coordinates")

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
    landmarks = read("landmarks.json")
    visibility = (landmark_visibility(landmarks, poses, settings.max_sparse_error_px)
                  if settings.max_sparse_error_px is not None else None)
    neighbors = overlapping_pairs(landmarks, poses, visibility)
    if settings.spread_views:
        selected = spread_references(neighbors, visibility, settings.references)
    else:
        selected = sorted(neighbors, key=lambda i: (-neighbors[i][0][0], i))[:settings.references]
    if len(selected) < 3:
        raise ValueError("Insufficient shared landmarks and parallax across three registered views")

    sources = {entry["image_id"]: entry for entry in manifest["inputs"]}
    original = {entry["image_id"]: parse_camera(entry["original"]) for entry in intrinsics}
    cameras = {}
    for i in neighbors:
        camera = original[i]
        scale = min(1, settings.dimension / max(camera.width, camera.height))
        size = (round(camera.width * scale), round(camera.height * scale))
        cameras[i] = camera.resized(*size)
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
            scaled = cameras[i]
            size = (scaled.width, scaled.height)
            if min(size) < 64:
                raise ValueError("Dense processing needs at least 64 pixels on each side")
            mask = rasterize_polygon(polygons[source["name"]], size) if polygons is not None else None
            inputs[i] = cv2.resize(frame, size, interpolation=cv2.INTER_AREA), mask
        return inputs[i]

    cv2.setNumThreads(1)
    depth_start = perf_counter()
    maps, attempts = [], []
    plans = {}
    used_pixels = 0
    for index, reference in enumerate(selected, 1):
        stage(f"Estimating depth for reference {index}/{len(selected)}")
        a = cameras[reference]
        if used_pixels + a.width * a.height > settings.canvas_budget:
            attempts.append({"reference": reference, "reason": "total depth-map pixel budget reached"})
            continue
        depth_attempts = 0
        candidates = [(source, xyz, False) for _, source, xyz in neighbors[reference]]
        for source, xyz, recenter in candidates:
            if depth_attempts == settings.attempts:
                break
            b, plan = cameras[source], None
            if settings.preflight_pairs or depth_diagnostics:
                try:
                    plan = plan_depth(poses[reference], poses[source], a, b, xyz, recenter=recenter)
                except (ValueError, cv2.error) as exc:
                    attempts.append({"reference": reference, "source": source, "reason": str(exc),
                                     "preflight": True, "recentered": recenter})
                    if settings.recenter_pairs and not recenter and isinstance(exc, DisparitySearchLimit):
                        candidates.append((source, xyz, True))
                    continue
                if prefer_conditioned_pairs and plan.median_effective_disparity < min_effective_disparity:
                    attempts.append({"reference": reference, "source": source, "preflight": True,
                                     "recentered": recenter, "reason": "weak median effective disparity",
                                     "median_effective_disparity_px": plan.median_effective_disparity})
                    continue
            depth_attempts += 1
            left, object_a = image(reference)
            right, object_b = image(source)
            try:
                depth = make_depth(reference, source, poses[reference], poses[source], a, b, left, right, xyz,
                                   (object_a, object_b), plan=plan)
            except (ValueError, cv2.error) as exc:
                attempts.append({"reference": reference, "source": source, "reason": str(exc), "recentered": recenter})
                stage(f"Depth pair {reference}/{source} skipped: {exc}")
                continue
            maps.append(depth)
            if depth_diagnostics:
                plans[reference] = plan
            used_pixels += a.width * a.height
            attempts.append({"reference": reference, "source": source, "depth_pixels": len(depth.xyz),
                             "recentered": recenter, "median_effective_disparity_px":
                             plan.median_effective_disparity if plan is not None else None})
            break
    depth_seconds = perf_counter() - depth_start
    stage("Checking third-view depth and fusing world voxels")
    fusion_start = perf_counter()
    if settings.max_return_px is None:
        accepted, support = checked_samples(maps, settings.relative_depth), None
    else:
        accepted, support = supported_samples(maps, settings.relative_depth, settings.max_return_px,
                                              settings.recentered_witnesses,
                                              keep_witnesses=mesh_common_witnesses or depth_diagnostics or bool(recovery_pairs),
                                              min_effective_disparity=min_effective_disparity)
    recovery, recovery_provenance = None, {}
    if recovery_pairs:
        from .depth_recovery import recover_depth

        def build_alternate(reference, source, xyz):
            a, b = cameras[reference], cameras[source]
            try:
                try:
                    plan = plan_depth(poses[reference], poses[source], a, b, xyz)
                except DisparitySearchLimit:
                    if not settings.recenter_pairs:
                        raise
                    plan = plan_depth(poses[reference], poses[source], a, b, xyz, recenter=True)
                if plan.median_effective_disparity < min_effective_disparity:
                    return None, "weak median effective disparity"
                left, object_a = image(reference)
                right, object_b = image(source)
                return make_depth(reference, source, poses[reference], poses[source], a, b, left, right, xyz,
                                  (object_a, object_b), plan=plan), None
            except (ValueError, cv2.error) as exc:
                return None, str(exc)

        recovery_start = perf_counter()
        accepted, recovery, recovery_provenance = recover_depth(
            maps, accepted, {i: p.center for i, p in poses.items()}, neighbors, build_alternate, recovery_pairs,
            settings.relative_depth, settings.max_return_px, min_effective_disparity, settings.recentered_witnesses, stage)
        recovery["runtime_seconds"] = perf_counter() - recovery_start
    sample_count = sum(int(mask.sum()) for _, mask, _ in accepted)
    if len(accepted) < 2 or sample_count < 100:
        reason = f"Too little independent third-view support ({len(maps)} depth maps, {sample_count} samples)"
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "metrics.json", {"status": "rejected", "reason": reason, "pair_attempts": attempts,
                                             "depth_support": support, "settings": asdict(settings),
                                             "min_effective_disparity_px": min_effective_disparity,
                                             "prefer_conditioned_pairs": prefer_conditioned_pairs})
        raise ValueError(reason)
    points = np.concatenate([depth.xyz[mask] for depth, mask, _ in accepted])
    colors = np.concatenate([rgb for _, _, rgb in accepted])
    voxel = fusion_voxel_size
    if voxel is None:
        voxel = float(np.median([np.median(d.depth[d.depth > 0]) / np.linalg.norm(d.intrinsic[0, :2])
                                for d, _, _ in accepted]))
    origin = points.min(axis=0) if fusion_voxel_origin is None else np.asarray(fusion_voxel_origin, dtype=float)
    xyz, colors, inverse = voxel_fuse(points, colors, voxel, origin=origin)
    fusion_seconds = perf_counter() - fusion_start
    surface, mesh_info = None, None
    face_evidence = []
    grid_provenance = None
    mesh_start = perf_counter()
    if mesh:
        stage("Building open surface from corroborated depth")
        offset = 0
        triangles = []
        grid_references = []
        for depth, mask, _ in accepted:
            pixels = depth.pixels[mask]
            heights = depth.depth[pixels[:, 0], pixels[:, 1]]
            local = surface_triangles(pixels, depth.depth.shape[::-1], heights, max_relative_jump=.025)
            if mesh_common_witnesses:
                minimum = settings.recentered_witnesses if depth.recentered else 1
                valid = shared_witness_faces(depth.witness_bits[mask], local, minimum)
                face_evidence.append({"reference": depth.reference, "minimum_common_witnesses": minimum,
                                      "faces_tested": len(local), "faces_kept": int(valid.sum())})
                local = local[valid]
            if depth.transposed:
                local = local[:, [0, 2, 1]]
            triangles.append(inverse[offset + local])
            grid_references.append(depth.reference)
            offset += len(pixels)
        try:
            grid_faces = np.concatenate(triangles)
            if depth_diagnostics:
                grid_provenance = (grid_faces, grid_references, [len(face_set) for face_set in triangles])
            surface, mesh_info = build_surface(xyz, colors, grid_faces, target_faces,
                                              max_edge=voxel * 6 if settings.clean_surface else None)
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
    if depth_diagnostics:
        from .depth_diagnostics import save_depth_diagnostics
        save_depth_diagnostics(output, maps, accepted, inverse, plans, poses, recovery=recovery_provenance)
        if grid_provenance is not None:
            triangles, references, counts = grid_provenance
            np.savez_compressed(output / "depth-observations" / "grid-faces.npz",
                                triangles=triangles, references=references, counts=counts)
    if support is not None:
        coverage = output / "depth-coverage"
        coverage.mkdir()
        checked = {depth.reference: mask for depth, mask, _ in accepted}
        for depth, record in zip(maps, support, strict=True):
            mask = np.zeros(depth.depth.shape, np.uint8)
            y, x = depth.pixels.T
            mask[y, x] = 96
            if depth.reference in checked:
                rows = checked[depth.reference]
                mask[y[rows], x[rows]] = 255
            if depth.transposed:
                mask = mask.T
            ok, encoded = cv2.imencode(".png", mask)
            if ok:
                name = f"depth-coverage/ref-{depth.reference:04d}.png"
                (output / name).write_bytes(encoded.tobytes())
                record["coverage_mask"] = name
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
        if obj:
            stage("Writing and independently reloading OBJ")
            try:
                export_obj(surface, output / "mesh.obj")
            except (ValueError, OSError) as exc:
                mesh_info["obj"] = {"status": "failed", "reason": str(exc)}
            else:
                mesh_info["obj"] = {"status": "success", "color": None, "normals": True}
    if polygons is not None:
        write_json(output / "object_masks.json", polygons)
    write_json(output / "poses.json", pose_records)
    write_json(output / "intrinsics.json", intrinsics)
    metrics = {"status": "success", "representation": "dense_multiview", "source_sparse": str(sparse_dir.resolve()),
               "profile": profile, "settings": asdict(settings), "object_masks": bool(polygons), "registered_images": len(poses), "dense_points": len(xyz),
               "depth_maps": len(maps), "fused_references": [d.reference for d, _, _ in accepted],
               "min_effective_disparity_px": min_effective_disparity,
               "prefer_conditioned_pairs": prefer_conditioned_pairs,
               "mesh_common_witnesses": face_evidence if mesh_common_witnesses else None,
               "eligible_references": sorted(neighbors), "planned_references": selected, "depth_support": support,
               "raw_depth_samples": sum(len(d.xyz) for d in maps), "checked_samples": sample_count,
               "depth_canvas_pixels": used_pixels,
               "coordinate_frame": "input SfM world, right/down/forward of the anchor camera",
               "units": "arbitrary monocular scale",
               "bounds": {"min": xyz.min(axis=0).tolist(), "max": xyz.max(axis=0).tolist()},
               "voxel_size": voxel, "pair_attempts": attempts, "mesh_requested": mesh,
               "voxel_origin": origin.tolist(),
               "depth_recovery": recovery,
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
