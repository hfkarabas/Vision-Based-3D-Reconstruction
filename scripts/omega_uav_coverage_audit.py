"""Audit matched coverage on completed UAV pilots using evaluation-only polygons."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.warp import reproject


COLORS = {
    "road_or_path": (243, 189, 41), "open_ground": (100, 237, 149),
    "roof": (121, 194, 255), "raised_surface": (237, 128, 239),
}


def measure(region, dense, before, final):
    values = {
        "pixels": region.sum(),
        "dense_supported": (region & dense).sum(),
        "render_before_mask": (region & before).sum(),
        "final_rgb_dsm": (region & final).sum(),
        "dense_present_render_missing": (region & dense & ~before).sum(),
        "render_pixels_removed_by_mask": (region & before & ~dense).sum(),
        "neither_dense_nor_render": (region & ~dense & ~before).sum(),
        "no_dense_support": (region & ~dense).sum(),
    }
    return {key: int(value) for key, value in values.items()}


def labels(regions, shape):
    mapped = {}
    used = np.zeros(shape, dtype=bool)
    for item in regions:
        vertices = item["vertices"]
        if not all(0 <= x <= shape[1] and 0 <= y <= shape[0] for x, y in vertices):
            raise ValueError(f"Review polygon outside raster: {item['id']}")
        mask = geometry_mask(
            [{"type": "Polygon", "coordinates": [vertices + [vertices[0]]]}],
            out_shape=shape, transform=Affine.identity(), invert=True,
        )
        if not mask.any() or (used & mask).any() or item["id"] in mapped:
            raise ValueError(f"Empty, duplicate or overlapping review polygon: {item['id']}")
        mapped[item["id"]] = mask
        used |= mask
    return mapped


def scene_data(path):
    with rasterio.open(path / "orthophoto.tif") as rgb, rasterio.open(path / "dsm.tif") as dsm:
        if (rgb.shape, rgb.crs, rgb.transform) != (dsm.shape, dsm.crs, dsm.transform):
            raise ValueError("Photographic and DSM output grids differ.")
        final_rgb = (rgb.read_masks() > 0).all(axis=0)
        final = final_rgb & np.isfinite(dsm.read(1, masked=True).filled(np.nan))
        counts = np.full(rgb.shape, np.nan, dtype=np.float32)
        with rasterio.open(path / "dense_count.tif") as source:
            reproject(rasterio.band(source, 1), counts, dst_transform=rgb.transform,
                      dst_crs=rgb.crs, dst_nodata=np.nan, resampling=Resampling.nearest)
        pixel_area = abs(rgb.transform.a * rgb.transform.e - rgb.transform.b * rgb.transform.d)
    rgba = np.asarray(Image.open(path / "orthophoto.png").convert("RGBA"))
    with np.load(path / "surface.npz") as surface:
        before = (rgba[:, :, 3] >= 252) & (surface["alpha"] >= 0.985) & np.isfinite(surface["elevation"])
    dense = counts > 0
    if not np.array_equal(final, dense & before):
        raise ValueError("Saved output masks differ from the recorded dense/render gate.")
    return rgba, dense, before, final, pixel_area


def panel(title, image, regions, mark=False):
    picture = Image.fromarray(image.astype("uint8")).convert("RGB")
    canvas = Image.new("RGB", (picture.width, picture.height + 26), (25, 29, 34))
    canvas.paste(picture, (0, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill="white")
    for index, item in enumerate(regions, start=1):
        coords = [(x, y + 26) for x, y in item["vertices"]]
        draw.line(coords + [coords[0]], fill=COLORS[item["class"]], width=2)
        if mark:
            draw.text(coords[0], str(index), fill=COLORS[item["class"]], stroke_width=1, stroke_fill="black")
    return canvas


def figure(dataset, rgba, dense, before, final, regions, output):
    rgb = rgba[:, :, :3]
    diagnostic = np.zeros_like(rgb)
    diagnostic[~dense & ~before] = (36, 36, 42)
    diagnostic[before & ~dense] = (167, 58, 91)
    diagnostic[dense & ~before] = (231, 137, 45)
    diagnostic[final] = (35, 139, 125)
    tiles = [
        panel("Camera RGB before dense-support mask", rgb, regions, mark=True),
        panel("Dense support: >=1 point in 0.50 m cell", np.repeat(dense[:, :, None], 3, axis=2) * 235, regions),
        panel("Published RGB + DSM support", np.where(final[:, :, None], rgb, 0), regions),
        panel("Stage attribution", diagnostic, regions),
    ]
    width, height = tiles[0].size
    montage = Image.new("RGB", (width * 2, height * 2 + 70), (25, 29, 34))
    for index, tile in enumerate(tiles):
        montage.paste(tile, (index % 2 * width, index // 2 * height))
    draw = ImageDraw.Draw(montage)
    draw.text((9, height * 2 + 7), f"{dataset} | outlined review polygons are evaluation only", fill="white")
    draw.text((9, height * 2 + 26), "Attribution: teal = retained | rose = render without dense point support", fill=(238, 208, 217))
    draw.text((9, height * 2 + 45), "Orange = dense support without valid render | gray = neither", fill=(237, 205, 155))
    montage.save(output)


def run(root, review, output):
    datasets = json.loads(review.read_text(encoding="utf-8"))["datasets"]
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, metadata in datasets.items():
        path = root / f"{dataset}-pilot" / "orthophoto"
        rgba, dense, before, final, area = scene_data(path)
        regions = metadata["regions"]
        masks = labels(regions, dense.shape)
        classes = {}
        for item in regions:
            mask = masks[item["id"]]
            category = item["class"]
            classes.setdefault(category, np.zeros(dense.shape, dtype=bool))
            classes[category] |= mask
            rows.append(dict(dataset=dataset, scope="region", name=item["id"], category=category,
                             **measure(mask, dense, before, final)))
        visible = classes["road_or_path"] | classes["open_ground"]
        scopes = {"entire_pilot": np.ones_like(dense), "visible_ground": visible, **classes}
        for name, mask in scopes.items():
            rows.append(dict(dataset=dataset, scope="union", name=name, category=name,
                             **measure(mask, dense, before, final)))
        figure(dataset, rgba, dense, before, final, regions, output / f"{dataset}-coverage-comparison.png")
        for row in rows:
            if row["dataset"] != dataset:
                continue
            row["area_m2"] = round(row["pixels"] * area, 4)
            for field in ("dense_supported", "render_before_mask", "final_rgb_dsm",
                          "dense_present_render_missing", "render_pixels_removed_by_mask",
                          "neither_dense_nor_render", "no_dense_support"):
                row[f"{field}_pct"] = round(100 * row[field] / row["pixels"], 4)
    with (output / "coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "coverage.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "review", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.review, args.output)
