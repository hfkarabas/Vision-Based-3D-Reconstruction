"""Publish geospatial layers and verify the expanded Blender delivery."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
from pathlib import Path
import sys

import geopandas as gpd
from PIL import Image, ImageChops
from shapely.geometry import shape

from recover import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "scripts"))
    from omega_building_report import contact_sheets, overlay

    out = args.output
    reports = out / "reports"
    summary = read(reports / "recovery.json")
    baseline = Path(summary["baseline"])
    source = read(baseline / "delivery/building_footprints_projected.json")["features"]
    features = read(out / "building_footprints_projected.json")["features"]
    meshes = {row["building_id"]: row for row in read(reports / "blender_buildings.json")}
    imported = read(reports / "blender_glb_reimport.json")
    reopened = read(reports / "blender_reopen.json")
    built = [row for row in meshes.values() if row["status"] == "built"]
    if len(built) != imported["building_objects"] or len(built) != reopened["buildings"]:
        raise ValueError("Mesh counts disagree across export and reopen")
    with Image.open(out / "preview/oblique.png") as a, Image.open(out / "preview/oblique_reopened.png") as b:
        if ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox():
            raise ValueError("Saved scene did not reproduce the oblique preview")

    for feature in features:
        props = feature["properties"]
        result = meshes.get(props["building_id"], {"status": "footprint_only"})
        props["status"] = result["status"]
        if result["status"] == "built":
            props["height_support"] = result["height_support"]
            props["height_provenance"] = result["height_provenance"]
            props["lod1_height_m"] = result["height_m"]
            props["mesh_height_m"] = result["mesh_height_m"]
        props.pop("foundation_triangles_xy")

    audit_path = reports / "candidate_audit.csv"
    with audit_path.open(newline="", encoding="utf-8") as stream:
        audit = list(csv.DictReader(stream))
    targets = [(f["properties"], shape(f["geometry"])) for f in features]
    for original, row in zip(source, audit, strict=True):
        if row["building_id"] != original["properties"]["building_id"]:
            raise ValueError("Candidate audit order differs from the source")
        geom = shape(original["geometry"])
        overlaps = [p["building_id"] for p, target in targets
                    if geom.intersection(target).area >= 0.30 * geom.area]
        row["overlapping_recovery_ids"] = " ".join(overlaps)
        row["overlapping_exported_ids"] = " ".join(
            name for name in overlaps if meshes.get(name, {}).get("status") == "built")
        original["properties"].update(row)
        original["properties"].pop("foundation_triangles_xy", None)
    with audit_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)

    grid = read(out / "grid.json")
    for layer, items in (("reviewed_footprints", features), ("original_candidates", source)):
        frame = gpd.GeoDataFrame.from_features(items, crs=grid["crs"])
        frame.to_file(out / "urban_building_review.gpkg", layer=layer)
        if len(gpd.read_file(out / "urban_building_review.gpkg", layer=layer)) != len(items):
            raise ValueError(f"GeoPackage count changed for {layer}")
        if layer == "reviewed_footprints":
            frame.to_crs("EPSG:4326").to_file(out / "reviewed_footprints.geojson", driver="GeoJSON")

    overlay(out, features)
    contact_sheets(out, source, True)
    contact_sheets(out, features, False)
    images = out / "building_detection_overlay.png"
    # Generic overlay labels the input as UAV imagery and all outlines as model output.
    # Relabel the headers to match this reference-assisted delivery.
    from PIL import ImageDraw, ImageFont
    with Image.open(images) as src:
        canvas = src.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    draw.rectangle((0, 0, canvas.width, 60), fill="#f8f8f8")
    draw.text((18, 17), "AirZoo reference DOM", fill="black", font=font)
    draw.text((canvas.width // 2 + 9, 17), "Reviewed footprints and white LOD1 placement", fill="black", font=font)
    canvas.save(images)

    summary.update(
        delivered_volumes=len(built),
        delivered_height_support=dict(Counter(row["height_support"] for row in built)),
        delivered_footprint_tiers=dict(Counter(row["footprint_tier"] for row in built)),
        footprint_only=[f["properties"]["building_id"] for f in features if f["properties"]["status"] != "built"],
        maximum_boundary_burial_m=max(row["max_boundary_burial_m"] for row in built),
        minimum_sampled_roof_clearance_m=min(row["minimum_roof_clearance_m"] for row in built),
        original_model=read(out / "inference.json"),
        source_sha256={
            key: sha256((baseline / key).read_bytes()).hexdigest()
            for key in ("delivery/inference_rgb.tif", "delivery/survey_dsm.tif",
                        "survey-ground/blender_height_grid.npz", "survey-ground/observed_ground.tif",
                        "delivery/building_footprints_projected.json", "delivery/urban_terrain_white_buildings.glb")
        },
        saved_oblique_decoded_rgb_matches=True,
    )
    write(reports / "recovery.json", summary)


if __name__ == "__main__":
    main()
