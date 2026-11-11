"""Probe intersecting raw-mesh faces against saved same-pose COLMAP geometric depth."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from analyze_airzoo_geometry import read
from audit_surface_topology import local_neighborhoods
from compare_airzoo_depth_backends import read_depth
from experiment_surface_layer_pruning import collisions
from vision3d.reconstruction.surface_validation import face_geometry, mesh_arrays, ray_scene


def depth_evidence(uv, z, depth, tolerance):
    """Use coherent positive footprints. Return agreement and positive free-space evidence."""
    agree = np.zeros(len(z), bool)
    free = np.zeros(len(z), bool)
    valid = np.zeros(len(z), bool)
    pixels = uv - .5  # COLMAP's upper-left pixel center is (0.5, 0.5).
    height, width = depth.shape
    ids = np.flatnonzero((z > 0) & ((pixels >= 0) & (pixels < [width - 1, height - 1])).all(axis=1))
    base = np.floor(pixels[ids]).astype(int)
    x, y = base.T
    footprint = depth[y[:, None] + [0, 0, 1, 1], x[:, None] + [0, 1, 0, 1]]
    lo, hi = footprint.min(axis=1), footprint.max(axis=1)
    tol = tolerance[ids]
    coherent = np.isfinite(footprint).all(axis=1) & (lo > 0) & (hi - lo <= 2 * tol)
    ids, footprint, tol = ids[coherent], footprint[coherent], tol[coherent]
    u, v = (pixels[ids] - base[coherent]).T
    weights = np.column_stack(((1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v))
    expected = 1 / (weights / footprint).sum(axis=1)
    valid[ids] = True
    agree[ids] = np.abs(z[ids] - expected) <= tol
    free[ids] = z[ids] + tol < lo[coherent]
    return valid, agree, free


def separated_votes(votes, centers, points, minimum_degrees):
    """Require two camera rays with distinct viewing directions for the same face."""
    rays = centers[:, None] - points
    rays /= np.maximum(np.linalg.norm(rays, axis=2, keepdims=True), 1e-15)
    threshold = np.cos(np.radians(minimum_degrees))
    result = np.zeros(len(points), bool)
    for i, votes_i in enumerate(votes):
        candidates = np.flatnonzero(votes_i & ~result)
        if len(candidates):
            cosine = np.einsum("ijk,jk->ij", rays[i + 1:, candidates], rays[i, candidates])
            result[candidates] = ((cosine <= threshold) & votes[i + 1:, candidates]).any(axis=0)
    return result


def removal_candidates(pairs, supported, contradicted, partial_agreement):
    """A candidate requires positive free space and a supported intersecting alternative."""
    remove = np.zeros(len(supported), bool)
    for first, second in (pairs.T, pairs[:, ::-1].T):
        selected = contradicted[first] & ~partial_agreement[first] & supported[second]
        remove[first[selected]] = True
    return remove


def first_hit_faces(scene, rays, target_faces):
    hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=2)
    return ((abs(hit["t_hit"].numpy() - 1) < 2e-4) &
            (hit["primitive_ids"].numpy() == target_faces))


def audit(dense, workspace, output, tolerance, view_angle):
    import pycolmap

    metrics = read(dense / "metrics.json")
    mesh = o3d.io.read_triangle_mesh(str(dense / "mesh.ply"))
    xyz, faces = mesh_arrays(mesh)
    probes, controls = [], []
    for cell, indices in local_neighborhoods(xyz, faces):
        pairs = indices[collisions(xyz, faces, indices)]
        if not len(pairs):
            continue
        probes.append((int(cell), pairs))
        intersecting = np.unique(pairs)
        remaining = np.setdiff1d(indices, intersecting)
        if len(remaining):
            reference = xyz[faces[remaining]].mean(axis=1)
            positions = xyz[faces[intersecting]].mean(axis=1)
            _, nearest = cKDTree(reference).query(positions, workers=1)
            controls.append(remaining[np.unique(nearest)])
    if not probes:
        raise ValueError("No intersections in the bounded neighborhoods")
    pairs = np.unique(np.sort(np.concatenate([pairs for _, pairs in probes]), axis=1), axis=0)
    intersecting = np.unique(pairs)
    control_faces = np.setdiff1d(np.concatenate(controls), intersecting) if controls else np.empty(0, int)
    rows = np.union1d(intersecting, control_faces)
    mapped = np.searchsorted(rows, pairs)
    is_intersecting = np.isin(rows, intersecting)
    centers, _, area = face_geometry(xyz, faces[rows])
    samples = np.concatenate((centers[:, None], .86 * xyz[faces[rows]] + .14 * centers[:, None]), axis=1)
    points = samples.reshape(-1, 3)
    rays = np.empty((len(points), 6), np.float32)
    target_faces = np.repeat(rows, 4)
    scene = ray_scene(mesh)
    model = pycolmap.Reconstruction(workspace / "sparse")
    expected_poses = {entry["image_id"]: np.array(entry["world_to_camera"])[:3]
                      for entry in read(dense / "poses.json") if entry["status"] == "registered"}
    visible_votes, depth_votes, contradictions, camera_centers = [], [], [], []
    partial = np.zeros(len(rows), bool)
    has_coherent_depth = np.zeros(len(rows), bool)
    observations = []
    for view in sorted(model.images.values(), key=lambda item: item.name):
        if not view.has_pose:
            continue
        camera_id = int(Path(view.name).stem)
        pose = view.cam_from_world().matrix()
        if camera_id not in expected_poses or not np.allclose(pose, expected_poses[camera_id], atol=1e-8, rtol=0):
            raise ValueError(f"COLMAP pose for image {camera_id} is outside the project gauge")
        path = workspace / "stereo/depth_maps" / (view.name + ".geometric.bin")
        if not path.is_file():
            continue
        camera, depth = view.camera, read_depth(path)
        if camera.model_name != "PINHOLE" or depth.shape != (camera.height, camera.width):
            raise ValueError(f"Expected undistorted COLMAP pinhole depth for image {camera_id}")
        fx, fy, cx, cy = camera.params
        local = points @ pose[:, :3].T + pose[:, 3]
        z = local[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = local[:, :2] / z[:, None] * [fx, fy] + [cx, cy]
        margin = np.maximum(metrics["voxel_size"], tolerance * z)
        valid, agree, free = depth_evidence(uv, z, depth, margin)
        camera_center = view.projection_center()
        rays[:, :3], rays[:, 3:] = camera_center, points - camera_center
        first = first_hit_faces(scene, rays, target_faces)
        agreement = agree.reshape(-1, 4).all(axis=1)
        observed = (agree & first).reshape(-1, 4).all(axis=1)
        contradiction = (free & first).reshape(-1, 4).all(axis=1)
        partial |= agree.reshape(-1, 4).any(axis=1)
        has_coherent_depth |= valid.reshape(-1, 4).any(axis=1)
        camera_centers.append(camera_center)
        depth_votes.append(agreement)
        visible_votes.append(observed)
        contradictions.append(contradiction)
        observations.append({"image": camera_id, "valid_depth_samples": int(valid.sum()),
                             "depth_supported_faces": int(agreement.sum()), "visible_supported_faces": int(observed.sum()),
                             "free_space_contradicted_faces": int(contradiction.sum())})
    if len(camera_centers) < 2:
        raise ValueError("Need at least two same-pose depth maps")
    camera_centers = np.stack(camera_centers)
    supported = separated_votes(np.stack(depth_votes), camera_centers, centers, view_angle)
    visible = separated_votes(np.stack(visible_votes), camera_centers, centers, view_angle)
    contradicted = separated_votes(np.stack(contradictions), camera_centers, centers, view_angle)
    remove = removal_candidates(mapped, supported, contradicted, partial)
    categories = {
        "both_depth_supported": int(supported[mapped].all(axis=1).sum()),
        "one_depth_supported": int((supported[mapped].sum(axis=1) == 1).sum()),
        "neither_depth_supported": int((~supported[mapped].any(axis=1)).sum()),
        "both_visible_depth_supported": int(visible[mapped].all(axis=1).sum()),
        "contradicted_faces": int((is_intersecting & contradicted).sum()),
        "contradicted_without_any_support": int((is_intersecting & contradicted & ~partial).sum()),
    }
    coverage = {}
    for name, selection in (("intersecting", is_intersecting), ("nearest_controls", ~is_intersecting)):
        coverage[name] = {"faces": int(selection.sum()),
                          "has_any_coherent_depth": int((selection & has_coherent_depth).sum()),
                          "has_any_agreeing_sample": int((selection & partial).sum()),
                          "has_two_separated_depth_votes": int((selection & supported).sum()),
                          "has_two_separated_visible_depth_votes": int((selection & visible).sum())}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "intersection_evidence.npz", face_ids=rows, pairs=pairs, is_intersecting=is_intersecting,
                        views=np.array([row["image"] for row in observations]), depth_support=np.stack(depth_votes),
                        visible_support=np.stack(visible_votes), free_space=np.stack(contradictions),
                        remove_candidates=rows[remove])
    report = {"probed_pairs": len(pairs), "probed_faces": len(intersecting), "neighborhoods_with_intersections": len(probes),
              "relative_depth_tolerance": tolerance, "voxel_depth_tolerance_floor": metrics["voxel_size"],
              "minimum_view_separation_deg": view_angle, "categories": categories, "coverage": coverage,
              "candidate_faces": int(remove.sum()), "candidate_area": float(area[remove].sum()),
              "probed_pairs_incident_to_candidates": int(remove[mapped].any(axis=1).sum()), "observations": observations,
              "scope": "Exact Open3D intersections in up to eight compact neighborhoods. Depth-only support can belong to "
                       "an occluded competitor. Free-space contradiction requires the probed face to be the first mesh hit. "
                       "Controls are nearest centroids within the same sampled neighborhoods with no detected intersection "
                       "in the probe. Missing or incoherent depths are unknown. Same-source images can correlate both backends."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--relative-depth", type=float, default=.005)
    parser.add_argument("--view-angle", type=float, default=2.)
    args = parser.parse_args()
    if not (0 < args.relative_depth < .1 and 0 < args.view_angle < 90):
        parser.error("Expected relative depth in (0, 0.1) and view angle in (0, 90)")
    audit(args.dense, args.workspace, args.output, args.relative_depth, args.view_angle)
