"""Compare DSM, DTM and terrain ray coverage on the same evaluation pixels."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import rasterio as rio
from rasterio.transform import xy
from scipy.ndimage import label

from omega_uav_coverage_audit import labels
from omega_uav_products import as_grid


def fraction(mask, region):
    return float((mask & region).sum() / region.sum())


def largest(mask, region):
    components, _ = label(mask & region)
    return float(np.bincount(components.ravel())[1:].max(initial=0) / region.sum())


def run(args):
    with rio.open(args.pilot / "orthophoto/dsm.tif") as dsm:
        grid = dict(width=dsm.width, height=dsm.height, transform=dsm.transform, crs=dsm.crs)
        surface = np.isfinite(dsm.read(1))
    ground = args.ground or args.pilot / "ground"
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        origin = json.loads((ground / "complete.json").read_text(encoding="utf-8"))["origin"]
        rows, cols = np.indices(surface.shape)
        east, north = xy(grid["transform"], rows.ravel(), cols.ravel())
        np.savez_compressed(args.output / "rays.npz", points=np.column_stack([east, north]) - origin[:2])
        return
    terrain = np.isfinite(as_grid(ground / "dtm.tif", grid))
    observed = as_grid(ground / "observed.tif", grid) > 0
    hits = np.load(args.mesh).reshape(surface.shape)
    mesh = np.isfinite(hits)
    if args.glb:
        exported = np.load(args.glb).reshape(surface.shape)
        if not np.array_equal(mesh, np.isfinite(exported)) or not np.allclose(
            hits, exported, rtol=0, atol=0.001, equal_nan=True
        ):
            raise ValueError("The GLB terrain ray geometry disagrees with the native scene.")
    regions = json.loads(args.review.read_text(encoding="utf-8"))["datasets"][args.dataset]["regions"]
    polygons = labels(regions, surface.shape)
    rows = []
    groups = {}
    for item in regions:
        groups.setdefault(item["class"], np.zeros_like(surface))
        groups[item["class"]] |= polygons[item["id"]]
    for name, region in [
        ("whole_block", np.ones_like(surface)),
        ("visible_ground", groups["open_ground"] | groups["road_or_path"]),
        *groups.items(), *polygons.items(),
    ]:
        rows.append(dict(name=name, pixels=int(region.sum()), dsm=fraction(surface, region),
                         observed=fraction(observed & terrain, region),
                         inferred=fraction(terrain & ~observed, region), dtm=fraction(terrain, region),
                         mesh=fraction(mesh, region), largest_mesh_component=largest(mesh, region),
                         largest_dtm_component=largest(terrain, region),
                         dsm_without_dtm=fraction(surface & ~terrain, region),
                         dtm_without_mesh=fraction(terrain & ~mesh, region)))
    with (args.output / "terrain_coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    canvas = Image.new("RGB", (surface.shape[1] * 2, (surface.shape[0] + 25) * 2))
    draw = ImageDraw.Draw(canvas)
    for i, (name, mask) in enumerate((
        ("Camera DSM", surface), ("Observed DTM", observed & terrain),
        ("Final DTM", terrain), ("Blender vertical ray hits", mesh),
    )):
        values = np.where(mask[:, :, None], (75, 174, 140), (49, 35, 53)).astype("uint8")
        x, y = i % 2 * surface.shape[1], i // 2 * (surface.shape[0] + 25)
        canvas.paste(Image.fromarray(values), (x, y + 25))
        draw.text((x + 7, y + 6), name, fill="white")
        for item in regions:
            vertices = [(x + a, y + 25 + b) for a, b in item["vertices"]]
            draw.line(vertices + [vertices[0]], fill=(254, 214, 94), width=2)
    canvas.save(args.output / "terrain_coverage.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "review", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("ground", "mesh", "glb"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prepare", action="store_true")
    run(parser.parse_args())
