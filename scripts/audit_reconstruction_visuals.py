"""Render saved AirZoo reconstruction stages on the same orthographic rays."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from analyze_airzoo_geometry import read, sparse_report, station_metadata
from vision3d.demo.view_geometry import camera_frustum
from vision3d.demo.viewer import load_textured_meshes


VIEWS = {"top": [0, 0, 1], "oblique": [0, -1, .82], "side": [0, -1, .07]}
BACKGROUND = (31, 35, 42)


def normalized(vector):
    return vector / np.linalg.norm(vector)


class View:
    def __init__(self, frame, direction, size):
        basis = np.asarray(frame["basis"])
        self.center = np.asarray(frame["center"])
        self.front = normalized(basis @ direction)
        self.right = normalized(np.cross(basis[:, 1] if direction[1] == 0 else basis[:, 2], self.front))
        self.up = np.cross(self.front, self.right)
        self.span, self.size = frame["span"], size

    def pixels(self, xyz):
        offset = xyz - self.center
        xy = np.column_stack((offset @ self.right, -offset @ self.up))
        return (xy / self.span + .5) * self.size, offset @ self.front

    def rays(self):
        grid = ((np.arange(self.size) + .5) / self.size - .5) * self.span
        origins = (self.center + self.front * 2 * self.span + grid[None, :, None] * self.right
                   - grid[:, None, None] * self.up)
        return o3d.core.Tensor(np.concatenate((origins, np.broadcast_to(-self.front, origins.shape)), axis=2),
                               dtype=o3d.core.Dtype.Float32)


def paint(mask, rgb, background=BACKGROUND):
    image = np.empty((*mask.shape, 3), dtype=np.uint8)
    image[:] = background
    image[mask] = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
    return image


def cloud_image(points, colors, view):
    xy, depth = view.pixels(points)
    xy = np.floor(xy).astype(int)
    inside = ((xy >= 0) & (xy < view.size)).all(axis=1) & (depth < 2 * view.span)
    rows = np.flatnonzero(inside)
    keys = xy[rows, 1] * view.size + xy[rows, 0]
    order = np.lexsort((-depth[rows], keys))
    _, first = np.unique(keys[order], return_index=True)
    keep = rows[order[first]]
    mask = np.zeros((view.size, view.size), bool)
    mask[xy[keep, 1], xy[keep, 0]] = True
    return paint(mask, colors[keep[np.argsort(keys[order[first]])]] * 255), mask


def texture_color(mesh, triangles, weights):
    if mesh.has_triangle_uvs() and mesh.has_textures():
        texture = np.asarray(mesh.textures[0])
        uv = np.einsum("ij,ijk->ik", weights, np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)[triangles])
        h, w = texture.shape[:2]
        xy = np.clip(uv * [w, -h] + [-.5, h - .5], [0, 0], [w - 1, h - 1])
        x, y = xy.astype(int).T
        dx, dy = (xy - np.column_stack((x, y))).T[:, :, None]
        a, b = np.minimum(x + 1, w - 1), np.minimum(y + 1, h - 1)
        return (1 - dy) * ((1 - dx) * texture[y, x, :3] + dx * texture[y, a, :3]) + dy * (
            (1 - dx) * texture[b, x, :3] + dx * texture[b, a, :3]), True
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)[np.asarray(mesh.triangles)[triangles]]
        return 255 * np.einsum("ij,ijk->ik", weights, colors), False
    return np.full((len(triangles), 3), 155.), False


def render_mesh(meshes, views, output, stage, *, material_modes=False):
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    geometry = {}
    area = 0.
    triangle_count = 0
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
    for mesh in meshes:
        xyz, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        identifier = scene.add_triangles(o3d.core.Tensor(xyz, dtype=o3d.core.Dtype.Float32),
                                         o3d.core.Tensor(faces, dtype=o3d.core.Dtype.UInt32))
        geometry[identifier] = mesh
        minimum = np.minimum(minimum, xyz.min(axis=0))
        maximum = np.maximum(maximum, xyz.max(axis=0))
        for start in range(0, len(faces), 250000):
            a, b, c = xyz[faces[start:start + 250000]].transpose(1, 0, 2)
            area += float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2)
        triangle_count += len(faces)
    statistics, masks = {}, {}
    for name, view in views.items():
        rays = view.rays()
        result = scene.cast_rays(rays, nthreads=2)
        mask = np.isfinite(result["t_hit"].numpy())
        normals = result["primitive_normals"].numpy()[mask]
        # Two-sided neutral shading retains every actual first hit.
        shade = 92 + 102 * abs(normals @ normalized(view.front + .6 * view.up - .4 * view.right))
        neutral = np.repeat(shade[:, None], 3, axis=1)
        Image.fromarray(paint(mask, neutral)).save(output / f"{name}-{stage}-neutral.png")
        masks[name] = mask
        statistics[name] = {"hit_pixels": int(mask.sum()), "screen_fraction": float(mask.mean()),
                            "back_facing_first_hits": int((normals @ view.front <= 0).sum())}
        if not material_modes:
            continue
        identifiers = result["geometry_ids"].numpy()[mask]
        triangles = result["primitive_ids"].numpy()[mask]
        uv = result["primitive_uvs"].numpy()[mask]
        weights = np.column_stack((1 - uv.sum(axis=1), uv))
        colors, photographic, edge = np.zeros_like(neutral), np.zeros(len(normals), bool), np.zeros(len(normals), bool)
        for identifier, mesh in geometry.items():
            rows = np.flatnonzero(identifiers == identifier)
            if not len(rows):
                continue
            colors[rows], photographic[rows] = texture_color(mesh, triangles[rows], weights[rows])
            corners = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)[triangles[rows]]]
            projected, _ = view.pixels(corners.reshape(-1, 3))
            xy = projected.reshape(-1, 3, 2)
            a, b = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
            twice_area = abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
            lengths = np.linalg.norm(np.roll(xy, -1, axis=1) - np.roll(xy, 1, axis=1), axis=2)
            edge[rows] = np.min(weights[rows] * twice_area[:, None] / np.maximum(lengths, 1e-8), axis=1) < .35
        Image.fromarray(paint(mask, colors, (255, 255, 255))).save(output / f"{name}-photo-white.png")
        Image.fromarray(paint(mask, colors)).save(output / f"{name}-photo-dark.png")
        neutral[edge] *= .46
        Image.fromarray(paint(mask, neutral)).save(output / f"{name}-wire.png")
        ids = triangles.astype(np.uint64) * 2654435761 + identifiers * 2246822519
        face_ids = 60 + np.column_stack([(ids >> i) & 159 for i in (0, 8, 16)])
        Image.fromarray(paint(mask, face_ids)).save(output / f"{name}-face-id.png")
        facing = np.where((normals @ view.front > 0)[:, None], [140, 202, 230], [238, 140, 102])
        Image.fromarray(paint(mask, facing)).save(output / f"{name}-winding.png")
        all_hits = scene.list_intersections(rays, nthreads=2)
        front_pixels = np.zeros(mask.size, bool)
        for identifier, mesh in geometry.items():
            group = all_hits["geometry_ids"].numpy() == identifier
            hit_faces = all_hits["primitive_ids"].numpy()[group]
            a, b, c = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)[hit_faces]].transpose(1, 0, 2)
            front = np.cross(b - a, c - a) @ view.front > 0
            front_pixels[all_hits["ray_ids"].numpy()[group][front]] = True
        statistics[name].update(
            front_face_pixels=int(front_pixels.sum()), photographic_first_hits=int(photographic.sum()),
            near_white_photo_pixels=int(((255 - colors[photographic]).max(axis=1) <= 16).sum()),
            near_white_fallback_pixels=int(((255 - colors[~photographic]).max(axis=1) <= 16).sum()),
            near_dark_pixels=int((abs(colors - BACKGROUND).max(axis=1) <= 16).sum()))
    return {"faces": triangle_count, "vertices": sum(len(m.vertices) for m in meshes),
            "area_sfm_squared": area, "bounds": [minimum.tolist(), maximum.tolist()], "views": statistics}, masks


def montage(output, rows, columns, name):
    size = 220
    sheet = Image.new("RGB", (len(columns) * size, len(rows) * (size + 22)), "#e7eaf0")
    draw = ImageDraw.Draw(sheet)
    for j, row in enumerate(rows):
        for i, label in enumerate(columns):
            path = output / f"{row}-{label}.png"
            if path.is_file():
                with Image.open(path) as tile:
                    sheet.paste(tile.resize((size, size)), (i * size, j * (size + 22) + 22))
                draw.text((i * size + 6, j * (size + 22) + 5), f"{row} · {label}", fill="#1c2731")
    sheet.save(output / name)


def audit(sparse, dense, raw, surface, glb, output, size, frozen=None):
    output.mkdir(parents=True, exist_ok=False)
    inputs, cameras, stations, positions, attitudes, basis, deviations = station_metadata(sparse)
    sparse_stats, graph, landmarks = sparse_report(sparse, cameras, stations, inputs, attitudes, deviations)
    cloud = o3d.io.read_point_cloud(str(dense / "dense.ply"))
    xyz = np.asarray(cloud.points)
    bounds = xyz @ basis
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    frame = ({"basis": basis.tolist(), "center": ((low + high) / 2 @ basis.T).tolist(),
              "span": float(np.linalg.norm(high - low) * 1.05)} if frozen is None else read(frozen))
    (output / "frame.json").write_text(json.dumps(frame, indent=2), encoding="utf-8")
    views = {name: View(frame, np.asarray(direction), size) for name, direction in VIEWS.items()}
    metrics = {"sources": {"sparse": str(sparse.resolve()), "dense": str(dense.resolve()), "glb": str(glb.resolve())},
               "sparse": sparse_stats, "graph": graph, "dense_points": len(xyz), "resolution": size, "views": {}}
    occupancy = {}
    for name, view in views.items():
        rgb, pixels = cloud_image(xyz, np.asarray(cloud.colors), view)
        Image.fromarray(rgb).save(output / f"{name}-dense.png")
        occupancy["dense", name] = pixels
        rgb, _ = cloud_image(landmarks, np.full_like(landmarks, .85), view)
        plot = Image.fromarray(rgb)
        draw = ImageDraw.Draw(plot)
        for i, camera in cameras.items():
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = camera.pose.rotation, camera.pose.translation
            xyz_frustum, edges = camera_frustum(matrix, camera.camera.K, camera.camera.width, camera.camera.height,
                                                frame["span"] * .035)
            xy, _ = view.pixels(xyz_frustum)
            color = ["#f19c59", "#63c5e8", "#e9c36b", "#a6d085"][stations[i] % 4]
            for a, b in edges:
                draw.line([tuple(xy[a]), tuple(xy[b])], fill=color, width=1)
        plot.save(output / f"{name}-sparse.png")
    del cloud, xyz, bounds
    for stage, path in (("raw", raw), ("optimized", surface), ("glb", glb)):
        if stage == "glb":
            meshes = load_textured_meshes(path)
            for mesh in meshes:
                # GLB stores the proper SfM-to-Y-up rotation (x, -y, -z).
                np.asarray(mesh.vertices)[:] *= [1, -1, -1]
        else:
            meshes = [o3d.io.read_triangle_mesh(str(path))]
        metrics[stage], masks = render_mesh(meshes, views, output, stage, material_modes=stage == "glb")
        occupancy.update({(stage, view): mask for view, mask in masks.items()})
        del meshes
    for name in views:
        dense_mask, raw_mask, opt_mask, glb_mask = (occupancy[stage, name]
                                                   for stage in ("dense", "raw", "optimized", "glb"))
        metrics["views"][name] = {"dense_pixels": int(dense_mask.sum()),
                                   "dense_pixels_without_raw_triangle": int((dense_mask & ~raw_mask).sum()),
                                   "raw_pixels_lost_in_optimized": int((raw_mask & ~opt_mask).sum()),
                                   "optimized_pixels_lost_in_glb": int((opt_mask & ~glb_mask).sum()),
                                   "optimized_glb_iou": float((opt_mask & glb_mask).sum() / (opt_mask | glb_mask).sum())}
    np.savez_compressed(output / "occupancy.npz", **{f"{stage}_{view}": mask for (stage, view), mask in occupancy.items()})
    montage(output, VIEWS, ("photo-white", "photo-dark", "glb-neutral", "wire", "winding", "face-id"), "materials.png")
    montage(output, VIEWS, ("sparse", "dense", "raw-neutral", "optimized-neutral", "photo-dark"), "stages.png")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("sparse", "dense", "raw", "surface", "glb", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=range(128, 801), default=400, metavar="128–800")
    parser.add_argument("--frame", type=Path, help="Reuse a comparison camera frame")
    args = parser.parse_args()
    audit(args.sparse, args.dense, args.raw, args.surface, args.glb, args.output, args.size, args.frame)
