"""Bake per-view UV charts from registered photographs with mesh visibility checks."""
from dataclasses import dataclass, replace
from math import ceil, sqrt

import cv2
import numpy as np

from vision3d.io import load_image
from .pair import source_hash
from .surface_validation import face_geometry, mesh_arrays, pinhole_projection, ray_scene, visible_points


@dataclass
class TextureAtlas:
    rgb: np.ndarray
    chart: np.ndarray
    uv: np.ndarray
    metrics: dict
    partially_visible: np.ndarray | None = None


def read_photo(view, dimension):
    if source_hash(view.source) != view.sha256:
        raise ValueError(f"Source photograph changed: {view.source}")
    image = load_image(view.source).image
    camera = view.camera
    if image.shape[:2] != (camera.height, camera.width):
        raise ValueError(f"Photograph dimensions disagree with camera model: {view.source}")
    scale = min(1, dimension / max(camera.width, camera.height))
    size = (max(2, round(camera.width * scale)), max(2, round(camera.height * scale)))
    camera = camera.resized(*size)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    valid = np.ones(size[::-1], dtype=np.uint8)
    if camera.D is not None:
        mx, my = cv2.initUndistortRectifyMap(camera.K, camera.D, None, camera.K, size, cv2.CV_32FC1)
        image = cv2.remap(image, mx, my, cv2.INTER_LINEAR)
        valid = ((mx >= 1) & (my >= 1) & (mx < size[0] - 2) & (my < size[1] - 2)).astype(np.uint8)
    return replace(view, camera=camera), cv2.cvtColor(image, cv2.COLOR_BGR2RGB), valid


def project_faces(view, xyz, faces):
    uv, z = pinhole_projection(xyz, view)
    pixels = uv[faces]
    valid = (z[faces] > 0).all(axis=1) & ((pixels >= 1) & (
        pixels <= [view.camera.width - 2, view.camera.height - 2])).all(axis=(1, 2))
    a, b = pixels[:, 1] - pixels[:, 0], pixels[:, 2] - pixels[:, 0]
    pixel_area = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    return pixels, pixel_area, valid


def reduce_seams(faces, labels, alternate, best, second, vertex_count):
    """Switch isolated labels only to an already validated near-tie source view."""
    a, b = faces.ravel(), faces[:, [1, 2, 0]].ravel()
    keys = np.minimum(a, b).astype(np.int64) * vertex_count + np.maximum(a, b)
    order = np.argsort(keys)
    edges = keys[order]
    _, start, count = np.unique(edges, return_index=True, return_counts=True)
    start = start[count == 2]
    left, right = order[start] // 3, order[start + 1] // 3
    existing = np.zeros(len(faces), dtype=np.int8)
    alternative = np.zeros(len(faces), dtype=np.int8)
    for current, neighbor in ((left, right), (right, left)):
        np.add.at(existing, current, labels[current] == labels[neighbor])
        np.add.at(alternative, current, alternate[current] == labels[neighbor])
    swap = (alternate >= 0) & (second >= .90 * best) & (alternative >= 2) & (existing == 0)
    labels[swap] = alternate[swap]
    return int(swap.sum())


def sample_rgb(image, pixels):
    rows = np.rint(pixels).astype(int)
    return image[rows[:, 1], rows[:, 0]].astype(np.float32)


def bake_texture(mesh, views, atlas_size=4096, holdout_stride=0, stage=lambda message: None, front_faces_only=False,
                 track_partial=False):
    if atlas_size not in (512, 1024, 2048, 4096, 8192):
        raise ValueError("Atlas size must be 512, 1024, 2048, 4096 or 8192")
    if holdout_stride != 0 and holdout_stride < 3:
        raise ValueError("Holdout stride must be zero or at least three")
    heldout = views[holdout_stride - 1::holdout_stride] if holdout_stride else []
    heldout_ids = {v.id for v in heldout}
    active = [v for v in views if v.id not in heldout_ids]
    if not active:
        raise ValueError("No source views are available for texture mapping")
    grid = ceil(sqrt(len(active)))
    cell, pad = atlas_size // grid, 8
    if cell - 2 * pad < 48:
        raise ValueError("Atlas too small for the registered image count; increase texture size")
    xyz, faces = mesh_arrays(mesh)
    centers, cross, area = face_geometry(xyz, faces)
    normals = cross / np.maximum(2 * area[:, None], 1e-30)
    scene = ray_scene(mesh)
    labels = np.full(len(faces), -1, np.int32)
    alternate = labels.copy()
    best, second = np.zeros(len(faces)), np.zeros(len(faces))
    center_visible = np.zeros(len(faces), dtype=bool) if track_partial else None
    views_and_images = []
    for idx, view in enumerate(active):
        stage(f"Checking texture visibility in photo {idx + 1}/{len(active)}")
        view, photo, validity = read_photo(view, cell - 2 * pad)
        views_and_images.append((view, photo))
        ray = view.pose.center - centers
        cosine = np.sum(ray * normals, axis=1) / np.maximum(np.linalg.norm(ray, axis=1), 1e-30)
        pixels, pixel_area, valid = project_faces(view, xyz, faces)
        incidence = cosine if front_faces_only else np.abs(cosine)
        rows = np.flatnonzero(valid & (incidence >= .2) & (pixel_area > .005))
        corners = np.rint(pixels[rows]).astype(int)
        rows = rows[(validity[corners[:, :, 1], corners[:, :, 0]] > 0).all(axis=1)]
        rows = rows[visible_points(scene, centers[rows], view.pose.center)]
        if center_visible is not None:
            center_visible[rows] = True
        # Test interior points near each corner, avoiding numerically ambiguous mesh edges.
        for corner in range(3):
            probe = .86 * xyz[faces[rows, corner]] + .14 * centers[rows]
            rows = rows[visible_points(scene, probe, view.pose.center)]
        center_uv, _ = pinhole_projection(centers[rows], view)
        xy = np.rint(center_uv).astype(int)
        gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
        contrast = cv2.blur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (7, 7))
        local_sharpness = np.clip(contrast[xy[:, 1], xy[:, 0]] / max(np.median(contrast), 1.), .75, 1.25)
        score = incidence[rows] ** 2 * np.sqrt(pixel_area[rows] / area[rows]) * local_sharpness
        score *= np.where(cosine[rows] > 0, 1., .8)
        new_best = score > best[rows]
        new_second = ~new_best & (score > second[rows])
        replace_best, replace_second = rows[new_best], rows[new_second]
        alternate[replace_best], second[replace_best] = labels[replace_best], best[replace_best]
        labels[replace_best], best[replace_best] = idx, score[new_best]
        alternate[replace_second], second[replace_second] = idx, score[new_second]

    reassigned = reduce_seams(faces, labels, alternate, best, second, len(xyz))
    covered = labels >= 0
    if not covered.any():
        raise ValueError("No triangles have a geometrically visible source photo")
    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
    uv = np.zeros((len(faces), 3, 2), dtype=np.float32)
    chart = np.full(len(faces), -1, np.int32)
    image_texels = 0
    source_faces = {}
    backside_faces = 0
    uv_center_error = np.zeros(len(faces), np.float32)
    for slot, idx in enumerate(np.unique(labels[covered])):
        view, photo = views_and_images[idx]
        rows = np.flatnonzero(labels == idx)
        h, w = photo.shape[:2]
        y, x = (slot // grid) * cell, (slot % grid) * cell
        atlas[y:y + h + 2 * pad, x:x + w + 2 * pad] = cv2.copyMakeBorder(photo, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        projected, _ = pinhole_projection(xyz, view)
        uv[rows] = (projected[faces[rows]] + [x + pad + .5, y + pad + .5]) / atlas_size
        center_projection, _ = pinhole_projection(centers[rows], view)
        uv_center_error[rows] = np.linalg.norm(projected[faces[rows]].mean(axis=1) - center_projection, axis=1)
        backside_faces += int((np.sum((view.pose.center - centers[rows]) * normals[rows], axis=1) < 0).sum())
        chart[rows] = slot
        image_texels += h * w
        source_faces[str(view.id)] = len(rows)

    holdout = []
    for original in heldout:
        view, photo, _ = read_photo(original, cell - 2 * pad)
        rows = np.flatnonzero(covered)[::4]
        pixels, z = pinhole_projection(centers[rows], view)
        inside = (z > 0) & ((pixels >= 1) & (pixels < [view.camera.width - 1, view.camera.height - 1])).all(axis=1)
        rows, pixels = rows[inside], pixels[inside]
        ray = view.pose.center - centers[rows]
        cosine = np.sum(ray * normals[rows], axis=1) / np.maximum(np.linalg.norm(ray, axis=1), 1e-30)
        facing = (cosine if front_faces_only else np.abs(cosine)) >= .2
        visible = facing & visible_points(scene, centers[rows], view.pose.center)
        expected = sample_rgb(photo, pixels[visible])
        sample_ids = rows[visible]
        actual = sample_rgb(atlas, uv[sample_ids].mean(axis=1) * atlas_size - .5)
        errors = np.abs(actual - expected).mean(axis=1)
        holdout.append({"camera": view.id, "samples": len(errors),
                        "rgb_mae_p50_p90": np.percentile(errors, [50, 90]).tolist() if len(errors) else None})
    metrics = {"texture_size": atlas_size, "atlas_photo_texel_fraction": image_texels / (atlas_size ** 2),
               "textured_faces": int(covered.sum()), "fallback_faces": int((~covered).sum()),
               "textured_area_fraction": float(area[covered].sum() / area.sum()), "source_faces": source_faces,
               "seam_label_changes": reassigned, "holdout_photometry": holdout, "front_faces_only": front_faces_only,
               "textured_faces_observed_from_back": backside_faces,
               "uv_center_projection_error_px_p50_p95": np.percentile(uv_center_error[covered], [50, 95]).tolist()}
    return TextureAtlas(atlas, chart, uv, metrics, center_visible & ~covered if track_partial else None)
