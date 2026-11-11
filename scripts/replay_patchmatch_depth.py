"""Test external PatchMatch hypotheses with project poses, fusion and surface gates."""
import argparse
import json
from pathlib import Path
import shutil
from time import perf_counter

import cv2
import numpy as np

from compare_airzoo_depth_backends import read_depth
from analyze_airzoo_geometry import read
from vision3d.geometry.depth import unproject_depth
from vision3d.io import load_image
from vision3d.reconstruction.dense_quality import round_trip_depth, shared_witness_faces
from vision3d.reconstruction.multiview import DepthMap, overlapping_pairs, voxel_fuse
from vision3d.reconstruction.pair import export_cloud, source_hash
from vision3d.reconstruction.stereo import surface_triangles
from vision3d.reconstruction.surface import build_surface, footprint_edge_gate, write_mesh_ply
from vision3d.reconstruction.surface_validation import surface_views


def camera_intrinsic(external, original):
    if external.model_name != "PINHOLE" or np.any(original.D):
        raise ValueError("This replay expects undistorted PINHOLE photographs")
    fx, fy, cx, cy = external.params
    intrinsic = np.array([[fx, 0., cx - .5], [0., fy, cy - .5], [0., 0., 1.]])
    if not np.allclose(intrinsic, original.resized(external.width, external.height).K, atol=.02, rtol=0):
        raise ValueError("PatchMatch camera differs from the resized project intrinsics")
    return intrinsic


def read_maps(sparse, workspace, views):
    import pycolmap

    model = pycolmap.Reconstruction(workspace / "sparse")
    sources = {row["image_id"]: row for row in read(sparse / "manifest.json")["inputs"]}
    maps = {}
    for image in model.images.values():
        if not image.has_pose:
            continue
        key = int(Path(image.name).stem)
        view = views[key]
        if key in maps or image.name != f"{key:04d}.png":
            raise ValueError("Unexpected or duplicate workspace image identity")
        if not np.allclose(image.cam_from_world().matrix(), view.pose.matrix[:3], atol=1e-8, rtol=0):
            raise ValueError(f"PatchMatch pose changed for view {key}")
        intrinsic = camera_intrinsic(image.camera, view.camera)
        depth = read_depth(workspace / "stereo/depth_maps" / f"{image.name}.geometric.bin")
        shape = (image.camera.height, image.camera.width)
        if depth.shape != shape or depth.size > 1_000_000:
            raise ValueError("Unexpected PatchMatch depth dimensions")
        depth[~np.isfinite(depth) | (depth <= 0)] = 0
        if source_hash(view.source) != sources[key]["sha256"]:
            raise ValueError(f"Original photograph changed: {view.source}")
        rgb = cv2.resize(load_image(view.source).image, shape[::-1], interpolation=cv2.INTER_AREA)
        pixels = np.argwhere(depth > 0)
        world = unproject_depth(pixels[:, ::-1], depth[tuple(pixels.T)], view.pose.rotation, view.pose.translation, intrinsic)
        maps[key] = DepthMap(key, -1, view.pose.rotation, view.pose.translation, intrinsic,
                             depth, rgb, pixels, world, rgb[tuple(pixels.T)][:, ::-1] / 255.)
    if set(maps) != set(views):
        raise ValueError("Registered workspace and sparse camera sets differ")
    return maps


def verified_maps(maps, views, neighbors):
    if len(maps) > 64:
        raise ValueError("This replay supports at most 64 witness cameras")
    bits = {key: np.uint64(1) << np.uint64(i) for i, key in enumerate(maps)}
    for key, sample in maps.items():
        votes = np.zeros(len(sample.xyz), np.uint8)
        support = np.zeros(len(sample.xyz), np.uint64)
        conditioned = np.zeros(len(sample.xyz), bool)
        for _, other, _ in neighbors.get(key, [])[:12]:
            witness = maps[other]
            rows, _, _ = round_trip_depth(sample, witness, .02, 1.5)
            votes[rows] += 1
            support[rows] |= bits[other]
            rays = sample.xyz[rows] - views[key].pose.center
            other_rays = sample.xyz[rows] - views[other].pose.center
            norm = np.linalg.norm(rays, axis=1) * np.linalg.norm(other_rays, axis=1)
            sine = np.linalg.norm(np.cross(rays, other_rays), axis=1) / np.maximum(norm, 1e-12)
            focal = min(sample.intrinsic[0, 0], witness.intrinsic[0, 0])
            conditioned[rows] |= focal * sine >= 28
        yield key, (votes >= 2) & conditioned, votes, support


def replay(sparse, workspace, grid, output, footprint_edges=False):
    start = perf_counter()
    views = {view.id: view for view in surface_views(sparse)}
    maps = read_maps(sparse, workspace, views)
    settings = read(grid / "metrics.json")
    if Path(settings["source_sparse"]).resolve() != sparse.resolve():
        raise ValueError("Comparison voxel grid must use the same sparse gauge")
    voxel, origin = settings["voxel_size"], settings["voxel_origin"]
    neighbors = overlapping_pairs(read(sparse / "landmarks.json"), {i: v.pose for i, v in views.items()})
    points, colors, faces, stats, shared = [], [], [], [], []
    offset = 0
    for key, keep, votes, support in verified_maps(maps, views, neighbors):
        sample = maps[key]
        pixels = sample.pixels[keep]
        points.append(sample.xyz[keep])
        colors.append(sample.colors[keep])
        local = surface_triangles(pixels, sample.depth.shape[::-1], sample.depth[tuple(pixels.T)], .025)
        faces.append(local + offset)
        if footprint_edges:
            shared.append(shared_witness_faces(support[keep], local, 2))
        offset += len(pixels)
        stats.append({"reference": key, "raw": len(keep), "accepted": int(keep.sum()),
                      "at_least_two_witnesses": int((votes >= 2).sum())})
    if offset < 100:
        raise ValueError("Insufficient corroborated PatchMatch samples")
    xyz, rgb, inverse = voxel_fuse(np.concatenate(points), np.concatenate(colors), voxel, origin=origin)
    if footprint_edges:
        selected = []
        for grid_faces, agreement, record in zip(faces, shared, stats, strict=True):
            key = record["reference"]
            triangles = inverse[grid_faces]
            gate = footprint_edge_gate(xyz, triangles, views[key].pose.center, maps[key].intrinsic[0, 0], voxel, agreement)
            selected.append(triangles[gate])
            record["footprint_faces"] = int(gate.sum())
        grid_faces = np.concatenate(selected)
    else:
        grid_faces = inverse[np.concatenate(faces)]
    surface, mesh = build_surface(xyz, rgb, grid_faces, max_edge=voxel * (12 if footprint_edges else 6))
    output.mkdir(parents=True, exist_ok=False)
    export_cloud(output / "dense.ply", xyz, rgb)
    write_mesh_ply(output / "mesh.ply", surface)
    for name in ("poses.json", "intrinsics.json"):
        shutil.copyfile(sparse / name, output / name)
    record = {"status": "success", "profile": "experimental-patchmatch-replay", "source_sparse": str(sparse.resolve()),
              "workspace": str(workspace.resolve()), "source_grid": str(grid.resolve()),
              "registered_images": len(views), "depth_maps": len(maps), "dense_points": len(xyz), "checked_samples": offset,
              "voxel_size": voxel, "voxel_origin": origin, "mesh": {"status": "success", **mesh}, "per_map": stats,
              "footprint_edges": footprint_edges,
              "criteria": {"relative_depth": .02, "round_trip_px": 1.5, "minimum_witnesses": 2,
                           "minimum_focal_times_sine_parallax_px": 28, "max_witnesses_tested": 12},
              "runtime_seconds": perf_counter() - start}
    (output / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sparse", "workspace", "grid", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--footprint-edges", action="store_true", help="Test longer same-grid faces with two shared witnesses")
    args = parser.parse_args()
    replay(args.sparse, args.workspace, args.grid, args.output, args.footprint_edges)
