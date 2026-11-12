"""Export standard GeoJSON, image-overlay evidence and validation counts."""

import argparse
from collections import Counter
from math import ceil
import json
from pathlib import Path

import geopandas as gpd
from PIL import Image, ImageDraw, ImageFont
import rasterio
from shapely.geometry import shape


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def overlay(out, features):
    with rasterio.open(out / "inference_rgb.tif") as raster:
        inverse = ~raster.transform
        gsd = raster.res[0]
    photo = Image.open(out / "terrain_photo.png").convert("RGB")
    factor = min(1.0, 1300 / photo.height)
    photo = photo.resize((round(photo.width * factor), round(photo.height * factor)), Image.Resampling.LANCZOS)
    width, height = photo.size
    marks = Image.new("RGBA", photo.size)
    draw = ImageDraw.Draw(marks)
    for feature in features:
        rings = [
            [tuple(coordinate * factor for coordinate in inverse * tuple(point)) for point in ring]
            for ring in feature["geometry"]["coordinates"]
        ]
        placed = feature["properties"]["status"] == "built"
        stroke, fill = ((35, 246, 248, 255), (22, 231, 234, 43)) if placed else (
            (255, 157, 62, 255), (247, 111, 29, 33)
        )
        draw.polygon(rings[0], fill=fill)
        for hole in rings[1:]:
            draw.polygon(hole, fill=(0, 0, 0, 0))
        for ring in rings:
            draw.line(ring, fill=stroke, width=2, joint="curve")
    combined = Image.alpha_composite(photo.convert("RGBA"), marks).convert("RGB")
    margin, heading, footer = 18, 62, 55
    canvas = Image.new("RGB", (2 * width + 3 * margin, height + heading + footer), "#f8f8f8")
    label = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    canvas.paste(photo, (margin, heading))
    canvas.paste(combined, (width + 2 * margin, heading))
    label.text((margin, 17), "Source UAV orthophoto", fill="black", font=font)
    label.text((width + 2 * margin, 17), "GeoAI roof polygon overlay", fill="black", font=font)
    y = height + heading + 18
    scale = 50 * factor / gsd
    label.line((margin, y + 10, margin + scale, y + 10), fill="black", width=4)
    label.text((margin + scale + 10, y), "50 m  |  North up", fill="black", font=font)
    right = width + 2 * margin
    label.rectangle((right, y + 3, right + 22, y + 18), fill=(0, 160, 165))
    label.text((right + 32, y), "LOD1 volume", fill="black", font=font)
    label.rectangle((right + 220, y + 3, right + 242, y + 18), fill=(225, 117, 20))
    label.text((right + 252, y), "Detected; no validated volume", fill="black", font=font)
    canvas.save(out / "building_detection_overlay.png")


def contact_sheets(out, features, all_detections):
    chosen = features if all_detections else [
        f for f in features if f["properties"]["status"] in {"built", "eligible"}
    ]
    prefix = "candidate_review" if all_detections else "roof_review"
    with rasterio.open(out / "inference_rgb.tif") as src:
        inverse = ~src.transform
    photo = Image.open(out / "terrain_photo.png").convert("RGB")
    edge, columns, rows = 220, 5, 4
    per_page = columns * rows
    font = ImageFont.load_default(size=15)
    for page, offset in enumerate(range(0, len(chosen), per_page), 1):
        visible = chosen[offset : offset + per_page]
        sheet = Image.new("RGB", (min(columns, len(visible)) * edge, ceil(len(visible) / columns) * (edge + 24)), "white")
        caption = ImageDraw.Draw(sheet)
        for index, feature in enumerate(visible):
            geom = shape(feature["geometry"])
            x0, y0, x1, y1 = geom.bounds
            left, top = inverse * (x0, y1)
            right, bottom = inverse * (x1, y0)
            side = int(max(right - left, bottom - top) * 1.45) + 12
            u = int((left + right - side) / 2)
            v = int((top + bottom - side) / 2)
            chip = photo.crop((u, v, u + side, v + side)).resize((edge, edge), Image.Resampling.LANCZOS)
            pen = ImageDraw.Draw(chip)
            for ring in feature["geometry"]["coordinates"]:
                points = [inverse * tuple(point) for point in ring]
                pen.line([((x - u) * edge / side, (y - v) * edge / side) for x, y in points], fill="#05FFFF", width=2)
            x = (index % columns) * edge
            y = (index // columns) * (edge + 24)
            sheet.paste(chip, (x, y))
            props = feature["properties"]
            height = props.get("lod1_height_m")
            caption.text(
                (x + 3, y + edge + 3),
                props["building_id"] + (f"   {height:.1f} m" if height is not None else ""),
                fill="black", font=font,
            )
        sheet.save(out / "reports" / f"{prefix}_{page:02}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-all", action="store_true")
    parser.add_argument("--visual-review", type=Path)
    args = parser.parse_args()
    out = args.output
    reports = out / "reports"
    reports.mkdir(exist_ok=True)
    grid = read(out / "grid.json")
    inference = read(out / "inference.json")
    features = read(out / "building_footprints_projected.json")["features"]
    blender = reports / "blender_buildings.json"
    mesh = {row["building_id"]: row for row in read(blender)} if blender.exists() else {}
    for feature in features:
        props = feature["properties"]
        props.pop("foundation_triangles_xy", None)
        props["raster_elevation_status"] = props["status"]
        props.update(mesh.get(props["building_id"], {}))
    # Standard GeoJSON uses WGS84. Geometry for construction stays in metric ENU.
    if features:
        source = gpd.GeoDataFrame.from_features(features, crs=grid["crs"])
        source.to_crs("EPSG:4326").to_file(out / "building_footprints.geojson", driver="GeoJSON", index=False)
    else:
        write(out / "building_footprints.geojson", dict(type="FeatureCollection", features=[]))
    overlay(out, features)
    contact_sheets(out, features, args.review_all)
    placed = [row for row in mesh.values() if row["status"] == "built"]
    checks = reports / "blender_glb_reimport.json"
    review = read(args.visual_review) if args.visual_review else None
    if review:
        write(reports / "visual_review.json", review)
    write(
        reports / "building_validation.json",
        dict(
            detected_polygons=len(features),
            built_volumes=len(placed),
            final_status_counts=dict(Counter(f["properties"]["status"] for f in features)),
            visual_review=review,
            lidar_or_survey_ground_truth_validated=False,
            building_heights_m=[row["height_m"] for row in placed],
            boundary_ray_count=sum(row["boundary_ray_count"] for row in placed),
            interior_ray_count=sum(row["interior_ray_count"] for row in placed),
            maximum_foundation_embed_m=max((row["max_boundary_burial_m"] for row in placed), default=None),
            minimum_roof_clearance_m=min((row["minimum_roof_clearance_m"] for row in placed), default=None),
            max_extruded_area_relative_error=max((row["area_relative_error"] for row in placed), default=None),
            blender_roundtrip=read(checks) if checks.exists() else None,
        ),
    )
    write(
        reports / "model_provenance.json",
        dict(
            **inference,
            checkpoint_license="Apache-2.0 (published Hugging Face model card)",
            checkpoint_url=f"https://huggingface.co/{inference['model']}/tree/{inference['model_revision']}",
            geoai_source="https://github.com/opengeos/geoai",
            geoai_license="MIT",
            geoai_api=["timm_segmentation_from_hub", "orthogonalize"],
            geometry_stack=["GDAL/Rasterio", "Shapely", "GeoPandas/PyProj", "Blender"],
            coordinates=grid,
            scientific_baseline_eligible=all(grid.get(f"{key}_role") == "reconstruction" for key in ("rgb", "dsm", "dtm")),
            source_roles={key: grid.get(f"{key}_role", "undeclared") for key in ("rgb", "dsm", "dtm")},
            roof_geometry="GeoAI roof polygons; approximation to ground footprint, no façade texture",
            roof_elevation="DSM-derived estimate with a local ground-based offset against the provided DTM",
            external_footprints_used=False,
        ),
    )


if __name__ == "__main__":
    main()
