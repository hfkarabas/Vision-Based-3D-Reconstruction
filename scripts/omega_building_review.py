"""Create registered visual evidence for the GeoAI building and terrain outputs."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import Polygon


def alternate(args):
    import geoai

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    with args.checkpoint.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if sha != "3aea5d0da7803be31ff4e2e1e5ac747f15cab2a13d6bb1a9319b6a6ba8bc7bca":
        raise ValueError("The alternate checkpoint does not match the pinned public RGB weights.")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    extractor = geoai.BuildingFootprintExtractor(model_path=str(args.checkpoint.resolve()), device="cuda")
    buildings = extractor.process_raster(
        str(args.rgb.resolve()), batch_size=2, confidence_threshold=0.5,
        mask_threshold=0.5, chip_size=(512, 512), overlap=0.5, band_indexes=[1, 2, 3],
        filter_edges=False,
    )
    if buildings is not None:
        buildings = buildings.explode(index_parts=False, ignore_index=True)
        buildings = buildings.loc[buildings.geometry.is_valid & (buildings.geometry.area >= 20)]
        if not buildings.empty:
            buildings[["geometry"]].to_file(args.output / "geoai_polygons.gpkg", driver="GPKG", index=False)
    (args.output / "inference.json").write_text(json.dumps(dict(
        seconds=time.perf_counter() - started,
        polygons=len(buildings) if buildings is not None else 0, checkpoint_sha256=sha,
        method="GeoAI BuildingFootprintExtractor, RGB Mask R-CNN ResNet50",
        revision="aa2b25d27b2c78f0e92c7b6a4813a571f9bc2288",
        settings=dict(tile=512, overlap_fraction=0.5, confidence=0.5, mask=0.5, minimum_area_m2=20),
    ), indent=2), encoding="utf-8")


def overlay(args):
    with rasterio.open(args.rgb) as src:
        factor = min(1, 1800 / max(src.width, src.height))
        width, height = round(src.width * factor), round(src.height * factor)
        pixels = src.read([1, 2, 3], out_shape=(3, height, width), resampling=Resampling.average)
        image = Image.fromarray(np.moveaxis(pixels, 0, -1))
        transform = ~(src.transform * src.transform.scale(src.width / width, src.height / height))
        crs = src.crs
    canvas = ImageDraw.Draw(image)
    options = [
        (args.primary, (25, 226, 240), "WHU candidates"),
        (args.alternate, (255, 186, 34), "Alternate Mask R-CNN candidates"),
        (args.volumes, (245, 245, 245), "Existing 3D volume footprints"),
    ]
    labels = []
    for path, color, label in options:
        if not path:
            continue
        layer = gpd.read_file(path).to_crs(crs).explode(index_parts=False)
        if label == "Existing 3D volume footprints" and "status" in layer.columns:
            layer = layer.loc[layer["status"] == "built"]
        for shape in layer.geometry:
            if isinstance(shape, Polygon):
                canvas.line([transform * point[:2] for point in shape.exterior.coords], fill=color, width=2)
        labels.append((color, label))
    canvas.rectangle((8, 8, 290, 16 + 17 * len(labels)), fill=(24, 24, 24))
    for index, (color, label) in enumerate(labels):
        canvas.text((16, 13 + 17 * index), label, fill=color)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    if args.projected_assessment:
        data = json.loads(args.projected_assessment.read_text(encoding="utf-8"))
        features = gpd.GeoDataFrame.from_features(data, crs=crs)
        features = features.drop(columns=["foundation_triangles_xy"], errors="ignore")
        features.to_crs("EPSG:4326").to_file(args.output.with_name("building_footprints.geojson"),
                                             driver="GeoJSON", index=False)


def foundations(args):
    with rasterio.open(args.rgb) as src:
        rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        inverse, crs = ~src.transform, src.crs
    records = json.loads(args.validation.read_text(encoding="utf-8"))
    results = {row["building_id"]: row["status"] for row in records if row.get("building_id")}
    source = json.loads(args.projected_assessment.read_text(encoding="utf-8"))
    features = gpd.GeoDataFrame.from_features(source, crs=crs)
    features["blender_status"] = features["building_id"].map(results).fillna("not_eligible")
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for feature in features.itertuples():
        poly = feature.geometry
        color = "#7aff38" if feature.blender_status == "built" else "#ffab47"
        draw.line([inverse * xy for xy in poly.exterior.coords], fill=color, width=2)
        number = feature.building_id.removeprefix("Building_")
        draw.text(inverse * poly.representative_point().coords[0], number, fill=color,
                  stroke_width=2, stroke_fill="#202020")
    draw.rectangle((10, 10, 360, 54), fill="#202020")
    draw.text((18, 17), "Green: Blender foundation and roof checks passed", fill="#7aff38")
    draw.text((18, 35), "Orange: detected candidate rejected by a geometry check", fill="#ffab47")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    features.drop(columns=["foundation_triangles_xy"], errors="ignore").to_crs(4326).to_file(
        args.output.with_name("building_footprints.geojson"), driver="GeoJSON", index=False,
    )


def cross_section(args):
    start, end = np.array(args.endpoints, dtype=float).reshape(2, 2)
    positions = np.linspace(start, end, 101)
    distances = np.linalg.norm(positions - start, axis=1)
    values = {}
    for name, filename in zip(args.names, args.rasters, strict=True):
        with rasterio.open(filename) as src:
            if src.crs is None or src.crs.to_string() != args.crs:
                raise ValueError(f"Expected {args.crs} in aligned raster {filename}.")
            values[name] = np.ma.array(list(src.sample(positions, indexes=1, masked=True))).astype(float).filled(np.nan).ravel()
    if args.mesh_section:
        mesh = np.loadtxt(args.mesh_section, delimiter=",", skiprows=1)
        if mesh.shape != (len(positions), 5) or not np.allclose(mesh[:, 1:3], positions, rtol=0, atol=1e-5):
            raise ValueError("The Blender mesh section does not use the same transect.")
        if np.allclose(mesh[:, 3], mesh[:, 4], rtol=0, atol=1e-5, equal_nan=True):
            values["Blender mesh (both)"] = mesh[:, 4]
        else:
            values["Baseline Blender mesh"], values["Candidate Blender mesh"] = mesh[:, 3:5].T
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output.with_suffix(".csv"), np.column_stack([distances, positions, *values.values()]),
               delimiter=",", header="distance_m,easting_m,northing_m," + ",".join(values), comments="")
    fig, ax = plt.subplots(figsize=(7.5, 3.2), layout="constrained")
    for name, z in values.items():
        ax.plot(distances, z, label=name)
    ax.set(xlabel="Distance along cross-section (m)", ylabel="Elevation in source reference (m)")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    detect = commands.add_parser("alternate")
    for name in ("rgb", "checkpoint", "output"):
        detect.add_argument(f"--{name}", type=Path, required=True)
    detect.set_defaults(run=alternate)
    render = commands.add_parser("overlay")
    for name in ("rgb", "primary", "output"):
        render.add_argument(f"--{name}", type=Path, required=True)
    render.add_argument("--alternate", type=Path)
    render.add_argument("--volumes", type=Path)
    render.add_argument("--projected-assessment", type=Path)
    render.set_defaults(run=overlay)
    verified = commands.add_parser("foundations")
    for name in ("rgb", "projected-assessment", "validation", "output"):
        verified.add_argument(f"--{name}", type=Path, required=True)
    verified.set_defaults(run=foundations)
    profile = commands.add_parser("cross-section")
    profile.add_argument("--rasters", nargs="+", type=Path, required=True)
    profile.add_argument("--names", nargs="+", required=True)
    profile.add_argument("--crs", required=True)
    profile.add_argument("--endpoints", nargs=4, type=float, required=True)
    profile.add_argument("--mesh-section", type=Path)
    profile.add_argument("--output", type=Path, required=True)
    profile.set_defaults(run=cross_section)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
