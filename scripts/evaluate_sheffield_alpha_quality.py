"""Evaluate two Sheffield alpha runs without supplying any labels to production."""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio as rio
from rasterio.features import geometry_mask
from rasterio.transform import xy
from shapely.geometry import box, shape
from shapely.ops import unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def grid(path):
    with rio.open(path) as source:
        data = source.read(1, masked=True).astype("float32").filled(np.nan)
        return data, source.transform


def mask(geometry, array, affine):
    return geometry_mask([geometry], array.shape, affine, invert=True)


def results(run):
    features = read(run / "scene/building_footprints_projected.json")["features"]
    records = read(run / "scene/reports/blender_buildings.json")
    built = {record["building_id"] for record in records if record["status"] == "built"}
    return features, built


def raster_changes(original, surface, observed, features, built, affine):
    approved = unary_union([shape(f["geometry"]).buffer(1.5) for f in features
                            if f["properties"]["building_id"] in built])
    rejected = unary_union([shape(f["geometry"]).buffer(1.5) for f in features
                            if f["properties"]["building_id"] not in built])
    approved_mask = mask(approved, original, affine) if not approved.is_empty else np.zeros_like(original, dtype=bool)
    rejected_mask = mask(rejected, original, affine) if not rejected.is_empty else np.zeros_like(original, dtype=bool)
    changed = ~np.isclose(original, surface, rtol=0, atol=0, equal_nan=True)
    return dict(changed=int(changed.sum()), changed_observed=int((changed & observed).sum()),
                changed_rejected_only=int((changed & rejected_mask & ~approved_mask).sum()))


def coverage(geometry, footprints):
    return float(geometry.intersection(unary_union(footprints)).area / geometry.area)


def review_regions(config, broad_controls, after, old, new, original, old_dtm, published, affine, csf):
    ortho = after / "products/orthophoto"
    dsm, dsm_affine = grid(ortho / "dsm.tif")
    smrf, _ = grid(after / "observations/classified_ground.tif")
    csf, csf_affine = grid(csf)
    if csf.shape != original.shape or csf_affine != affine:
        raise ValueError("CSF must be evaluated on the same grid as SMRF.")
    models = [
        gpd.read_file(after / "detections/geoai_polygons.gpkg").geometry,
        gpd.read_file(after / "detections/whu/geoai_polygons.gpkg").geometry,
    ]
    roads = {name: shape(broad_controls["regions"][name]) for name in config["control_regions_from_full_site"]}
    regions = {name: box(*bounds) for kind in ("roofs", "controls") for name, bounds in config[kind].items()}
    regions.update(roads)

    def review_label(fraction):
        return "full" if fraction >= 0.8 else "partial" if fraction > 0.05 else "none"

    def median(values):
        return float(np.nanmedian(values)) if np.isfinite(values).any() else None

    def labels(geometry, features, built):
        found = []
        for feature in features:
            polygon = shape(feature["geometry"])
            overlap = geometry.intersection(polygon).area
            if overlap <= 0.05 * min(geometry.area, polygon.area):
                continue
            p = feature["properties"]
            found.append(dict(id=p["building_id"], status="built" if p["building_id"] in built else p["status"],
                              detector=p.get("detector", "mask-rcnn"),
                              lod1_height_m=p["lod1_height_m"], roof_support=p["roof_core_fraction"],
                              foundation_support=p["foundation_fraction"],
                              observed_ground_cells=p["surrounding_observed_ground_cells"]))
        return found

    records = {}
    for name, geometry in regions.items():
        roi = mask(geometry, original, affine)
        raw = [coverage(geometry, model.tolist()) for model in models]
        before = coverage(geometry, [shape(f["geometry"]) for f in old[0]])
        after = coverage(geometry, [shape(f["geometry"]) for f in new[0]])
        records[name] = dict(
            kind="roof" if name in config["roofs"] else "control",
            primary_coverage=raw[0], whu_coverage=raw[1],
            primary_label=review_label(raw[0]), whu_label=review_label(raw[1]),
            final_detection_coverage=after, final_detection_label=review_label(after),
            old_detection_coverage=before, old_detection_label=review_label(before),
            dsm_median_m=median(dsm[mask(geometry, dsm, dsm_affine)]),
            original_dtm_median_m=median(original[roi]),
            smrf_observed_cells=int(np.isfinite(smrf[roi]).sum()),
            csf_observed_cells=int(np.isfinite(csf[roi]).sum()), roi_cells=int(roi.sum()),
            original_dtm_cells=int(np.isfinite(original[roi]).sum()),
            published_dtm_cells=int(np.isfinite(published[roi]).sum()),
            old_changed_cells=int((roi & ~np.isclose(original, old_dtm, rtol=0, atol=0, equal_nan=True)).sum()),
            new_changed_cells=int((roi & ~np.isclose(original, published, rtol=0, atol=0, equal_nan=True)).sum()),
            before=labels(geometry, *old), after=labels(geometry, *new),
        )
    return records


def sample_meshes(args, points, origin):
    output = args.output
    np.savez(output / "terrain_samples_xy.npz", points=points - origin[:2])
    surfaces = []
    for name, run in (("before", args.before), ("after", args.after)):
        source = run / "scene/terrain_white_buildings.blend"
        target = output / f"terrain_samples_{name}.npy"
        script = Path(__file__).with_name("omega_uav_ground_mesh_audit.py")
        with (output / f"terrain_samples_{name}.log").open("w") as log:
            subprocess.run([args.blender, "-b", "-t", "4", "--python-exit-code", "1", "--python", script,
                            "--", "--scene", source, "--samples", output / "terrain_samples_xy.npz", "--output", target],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        surfaces.append(np.load(target) + origin[2])
    return surfaces


def road_section(output, points, terrain, original, old_ground, new_ground, old_mask, new_mask):

    def sample(path):
        with rio.open(path) as source:
            return np.ma.array(list(source.sample(points, masked=True))).astype(float).filled(np.nan).ravel()

    values = [sample(p) for p in (original, old_ground, new_ground, old_mask, new_mask)] + terrain
    rows = np.column_stack([points, *values])
    with (output / "east_road_section.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["easting_m", "northing_m", "base_dtm_m", "old_dtm_m", "new_dtm_m", "old_mask", "new_mask",
                         "old_blender_m", "new_blender_m"])
        writer.writerows(rows)
    fig, axis = plt.subplots(figsize=(10, 4.5), layout="constrained")
    axis.plot(points[:-2, 0], values[0][:-2], color="black", label="Pre-foundation DTM", linewidth=1)
    axis.plot(points[:-2, 0], values[1][:-2], color="#a02b18", label="Old DTM", alpha=0.55)
    axis.plot(points[:-2, 0], values[5][:-2], color="#a02b18", linestyle="--", label="Old Blender terrain")
    axis.plot(points[:-2, 0], values[6][:-2], color="#15784b", linestyle=":", label="New Blender terrain")
    axis.axvline(points[-2, 0], color="grey", alpha=0.4)
    axis.set(xlabel="Easting (m), EPSG:32617", ylabel="Reconstruction Z (m)", title="Sheffield east road cross-section")
    axis.ticklabel_format(useOffset=False)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left")
    fig.savefig(output / "east_road_section.png", dpi=140)
    plt.close(fig)
    return [[float(v) if np.isfinite(v) else None for v in row] for row in rows[-2:]]


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    config = read(args.regions)
    broad_controls = read(args.regions.with_name("sheffield_full_regions.json"))
    old, new = results(args.before), results(args.after)
    base_path = args.before / "products/ground/dtm.tif"
    old_path = args.before / "foundation/ground/dtm.tif"
    new_path = Path(read(args.after / "run.json")["published_dtm"])
    old_mask = old_path.with_name("inferred_foundation.tif")
    new_mask = new_path.with_name("inferred_foundation.tif")
    base, affine = grid(base_path)
    old_dtm, old_affine = grid(old_path)
    new_dtm, new_affine = grid(new_path)
    current_base, current_affine = grid(args.after / "products/ground/dtm.tif")
    if not old_affine == new_affine == current_affine == affine or not np.array_equal(base, current_base, equal_nan=True):
        raise ValueError("The runs must begin from the same DTM and grid.")
    observed = grid(base_path.with_name("observed.tif"))[0] > 0
    case = next(f for f in old[0] if f["properties"]["building_id"] == config["east_road"]["old_candidate"])
    area = mask(shape(case["geometry"]).buffer(1.5), base, affine)
    paired = area & np.isfinite(old_dtm) & np.isfinite(base)
    change = old_dtm - base
    max_cell = np.unravel_index(np.nanargmax(np.where(paired, change, np.nan)), base.shape)
    match_error = np.where(paired, abs(base + 66.65) + abs(old_dtm + 62.33), np.nan)
    match_cell = np.unravel_index(np.nanargmin(match_error), base.shape)
    origin = np.array(read(args.before / "scene/grid.json")["origin_enu"])
    views = {}
    for name, data in config["views"].items():
        item = dict(data)
        item["center_local_enu"] = (item.pop("center") - origin[:2]).tolist()
        views[name] = item
    write(args.output / "views.json", views)
    spike_xy, match_xy = xy(affine, *max_cell), xy(affine, *match_cell)
    transect = np.vstack([np.linspace(*config["east_road"]["section"], 121), spike_xy, match_xy])
    controls = {name: box(*bounds) for name, bounds in config["controls"].items()}
    controls.update((name, shape(broad_controls["regions"][name])) for name in config["control_regions_from_full_site"])
    samples = {name: np.column_stack(xy(affine, *np.nonzero(mask(poly, base, affine))))
               for name, poly in controls.items()}
    surfaces = sample_meshes(args, np.vstack([transect, *samples.values()]), origin)
    section = road_section(args.output, transect, [z[:len(transect)] for z in surfaces],
                           base_path, old_path, new_path, old_mask, new_mask)
    records = review_regions(config, broad_controls, args.after, old, new, base, old_dtm, new_dtm, affine, args.csf)
    offset = len(transect)
    for name, points in samples.items():
        first, second = [surface[offset:offset + len(points)] for surface in surfaces]
        paired = np.isfinite(first) & np.isfinite(second)
        difference = abs(first[paired] - second[paired])
        records[name].update(
            mesh_hits_before=int(np.isfinite(first).sum()), mesh_hits_after=int(np.isfinite(second).sum()),
            mesh_paired_median_abs_change_m=float(np.median(difference)) if difference.size else None,
        )
        offset += len(points)
    write(args.output / "evaluation.json", dict(
        before=raster_changes(base, old_dtm, observed, *old, affine),
        after=raster_changes(base, new_dtm, observed, *new, affine),
        old_spike_candidate=case["properties"],
        old_spike_buffer_changed=int((area & ~np.isclose(base, old_dtm, rtol=0, atol=0, equal_nan=True)).sum()),
        new_spike_buffer_changed=int((area & ~np.isclose(base, new_dtm, rtol=0, atol=0, equal_nan=True)).sum()),
        max_change_and_requested_match=section, reviewed_regions=records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "output", "regions", "blender", "csf"):
        parser.add_argument(f"--{name}", required=True, type=lambda path: Path(path).resolve())
    main(parser.parse_args())
