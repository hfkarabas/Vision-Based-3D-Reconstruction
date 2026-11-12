from pathlib import Path
import argparse
import csv
import json
import shutil
from collections import Counter

import numpy as np
import rasterio as rio
from rasterio.features import geometry_mask, shapes
from rasterio.fill import fillnodata
from scipy import ndimage as ndi
from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon, mapping, shape

# Scene-specific visual labels from the numbered DOM contact sheets.
ACCEPT = {1,4,14,15,19,26,32,37,41,57,60,62,65,67,68,70,72,74,76,79,80,84,87,88,90,91}
FRAGMENT = {2,3,8,9,10,11,12,13,16,17,18,20,21,22,24,29,31,33,35,36,38,39,40,42,43,44,45,46,47,48,49,50,54,55,58,59,63,69,71,77,78,81,82,89,92,93}
FALSE = {5,6,7,23,25,27,28,30,34,51,53,56,61,64,66,73,83,85,86}
# Seed points and local windows on the saved 1620 x 1917 reference DOM.
# Roof level components are extracted with SciPy/Rasterio and then image-reviewed.
SEEDS = {
    "northwest_1": (193,278,130,225,355,332),
    "northeast_1": (512,271,442,204,755,303),
    "west_2": (205,471,155,398,377,515),
    "east_2": (540,436,462,392,719,483),
    "west_3": (303,665,164,604,404,724),
    "east_3": (548,648,481,597,731,698),
    "west_4": (398,837,314,784,508,890),
    "east_4": (675,822,535,774,739,877),
    "central_cross_w": (261,1068,199,1004,349,1139),
    "central_cross_mid": (466,1075,406,982,550,1131),
    "south_block_w": (390,1502,337,1446,437,1554),
    "south_block_e": (486,1513,433,1445,556,1552),
}

# Approximate outer roof lines traced on the same reference DOM. Connected
# wings stay one footprint. DSM steps within a roof remain an LOD1 limitation.
TRACES = {
    "central_cross_e": [(618,1074),(647,1044),(630,1027),(650,1000),(678,1024),
        (709,994),(734,1020),(703,1050),(741,1083),(721,1106),(692,1084),(665,1112)],
    "west_3": [(174,645),(207,641),(207,634),(270,632),(269,618),(326,615),
        (326,623),(363,622),(362,654),(346,657),(345,678),(284,680),
        (282,669),(236,670),(233,707),(183,711),(176,692)],
    "south_cross_n": [(630,1509),(648,1487),(640,1473),(659,1445),(687,1469),
        (716,1441),(740,1463),(711,1496),(749,1534),(726,1553),(696,1523),(669,1548)],
    "south_cross_e": [(689,1596),(718,1565),(710,1555),(735,1531),(763,1559),
        (787,1543),(819,1578),(785,1611),(759,1595),(731,1630)],
    "south_cross_sw": [(283,1773),(311,1740),(326,1753),(343,1747),(367,1773),
        (394,1765),(424,1795),(401,1823),(379,1810),(351,1833)],
    "south_cross_s": [(365,1838),(394,1809),(416,1826),(452,1805),(478,1829),
        (452,1854),(480,1881),(453,1908),(426,1882),(396,1894),(368,1866)],
    "south_cross_se": [(683,1779),(707,1754),(724,1767),(758,1726),(790,1757),
        (765,1780),(811,1813),(787,1843),(754,1815),(726,1843)],
}

# Roof surfaces partially traced by the old model or skipped by the DSM seeds.
# These low roofs use image-derived footprint proxies and the same height audit.
LOW_TRACES = {
    "west_service_roof": [(174,771),(290,767),(290,794),(175,799)],
    "school_west_south": [(243,1222),(309,1220),(310,1285),(245,1288)],
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def raster(path):
    with rio.open(path) as src:
        return src.read(1, masked=True).astype("float32").filled(np.nan)


def component(dsm, transform, spec):
    x, y, left, top, right, bottom = spec
    area = dsm[top:bottom, left:right]
    level = np.nanmedian(dsm[y-2:y+3, x-2:x+3])
    mask = ndi.binary_closing(area > level - 5, iterations=2)
    labels, _ = ndi.label(mask)
    label = labels[y-top, x-left]
    if not label or np.count_nonzero(labels == label) < 180:
        return None
    selected = ndi.binary_fill_holes(labels == label)
    local = transform * rio.Affine.translation(left, top)
    return max(
        (shape(geom).simplify(0.7, preserve_topology=True)
         for geom, _ in shapes(selected.astype("uint8"), mask=selected, transform=local)),
        key=lambda geom: geom.area,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    base, out = args.baseline.resolve(), args.output.resolve()
    if base == out or base in out.parents or out.exists():
        raise ValueError("Use a fresh output directory outside the baseline.")
    (out / "reports").mkdir(parents=True)
    for name in ("inference_rgb.tif", "terrain_photo.png", "grid.json", "inference.json"):
        shutil.copy2(base / "delivery" / name, out / name)
    baseline = read(base / "delivery/building_footprints_projected.json")["features"]
    previous = {row["building_id"]: row for row in read(base / "reports/blender_buildings.json")}
    with rio.open(out / "inference_rgb.tif") as src:
        transform, grid_shape = src.transform, src.shape
    dsm = raster(base / "delivery/survey_dsm.tif")
    grid = read(out / "grid.json")
    origin = grid["origin_enu"]
    with np.load(base / "survey-ground/blender_height_grid.npz") as src:
        axes = {axis: src[axis] for axis in ("x", "y", "z")}
    terrain = axes["z"]
    terrain_transform = rio.Affine(1, 0, axes["x"][0]+origin[0]-0.5, 0, -1, axes["y"][0]+origin[1]+0.5)

    def mask(geom, terrain_grid=False):
        affine, size = (terrain_transform, terrain.shape) if terrain_grid else (transform, grid_shape)
        return geometry_mask([mapping(geom)], out_shape=size, transform=affine, invert=True)

    reviewed = []
    accepted = []
    for i, feature in enumerate(baseline, 1):
        props = feature["properties"]
        geom = shape(feature["geometry"])
        decision = ("roof" if i in ACCEPT else "roof_fragment" if i in FRAGMENT
                    else "false_positive" if i in FALSE else "uncertain")
        name = props["building_id"]
        reviewed.append(dict(building_id=name, image_review=decision,
                             raster_gate=props["status"],
                             blender_gate=previous.get(name, {}).get("status", "")))
        if i in ACCEPT:
            accepted.append((name, geom, "reviewed_model_outline", props))
    for label, spec in SEEDS.items():
        if label in TRACES:
            continue
        geom = component(dsm, transform, spec)
        if geom is None or not geom.is_valid or geom.area > 1800:
            continue
        if any(geom.intersection(other).area > 0.20 * min(geom.area, other.area)
               for _, other, _, _ in accepted):
            continue
        accepted.append((label, geom, "reviewed_dsm_roof_component", {}))
    for label, points in (TRACES | LOW_TRACES).items():
        geom = Polygon([transform * point for point in points])
        if not geom.is_valid or any(geom.intersection(other).area > 0.20 * min(geom.area, other.area)
                                    for _, other, _, _ in accepted):
            raise ValueError(f"Invalid or overlapping reviewed trace: {label}")
        accepted.append((label, geom, "provisional_dom_trace", {}))

    # Remove classified roof remnants and interpolate only in the selected
    # roof neighborhoods. Preserve the source terrain everywhere else.
    correction = np.zeros(terrain.shape, dtype=bool)
    terrain_source = terrain.copy()
    observed_ground = raster(base / "survey-ground/observed_ground.tif") > 0
    for _, geom, _, _ in accepted:
        correction |= mask(geom.buffer(2.5), True)
    # Remove local outlier remnants near a reviewed outline only. Retain
    # classified near-ground cells as anchors and leave unrelated cells intact.
    for _, geom, _, _ in accepted:
        nearby = mask(geom.buffer(9), True)
        samples = terrain[nearby & observed_ground & ~correction]
        if samples.size >= 15:
            correction |= nearby & (terrain > np.percentile(samples, 35) + 3)
    known = np.isfinite(terrain) & ~correction
    fitted = fillnodata(np.where(known, terrain, np.nan).astype("float32"),
                        mask=known.astype("uint8"), max_search_distance=36)
    filled = correction & np.isfinite(fitted)
    terrain[filled] = fitted[filled]
    np.savez_compressed(out / "terrain.npz", **axes)
    features = []
    for name, geom, tier, original in accepted:
        inner = geom.buffer(-0.7)
        roof_pixels = dsm[mask(inner if not inner.is_empty else geom)]
        roof_pixels = roof_pixels[np.isfinite(roof_pixels)]
        ring = mask(geom.buffer(12).difference(geom.buffer(2)), True) & known
        observed = terrain_source[ring & observed_ground]
        ground = observed if observed.size >= 12 else terrain_source[ring]
        roof = float(np.median(roof_pixels)) if roof_pixels.size >= 15 else None
        base_z = float(np.median(ground)) if ground.size >= 12 else None
        height = roof - base_z if roof is not None and base_z is not None else None
        placement_height = height if height is not None and 1.5 <= height <= 120 else None
        support = "inferred" if placement_height is not None else "unknown"
        # A small proxy is used only for image-reviewed low roof candidates.
        # Unmeasured tall components remain in the GIS layer if support fails.
        allow_proxy = tier == "reviewed_model_outline" or name in LOW_TRACES
        if support == "unknown" and allow_proxy:
            placement_height = 3.0
        tri = constrained_delaunay_triangles(geom)
        if not np.isclose(sum(t.area for t in tri.geoms), geom.area, rtol=1e-6):
            raise ValueError(f"Triangulation changed the area of {name}")
        props = dict(
            building_id=name, status="eligible" if placement_height is not None else "footprint_only",
            footprint_tier=tier, footprint_provenance=f"{tier}; reference DOM review; roof projection",
            height_support=support, height_provenance=(
                "Reference DSM roof samples and inferred local ground"
                if support == "inferred" else "Unknown height; any extrusion is a 3 m display proxy"
            ),
            roof_support="reference_dsm_sampled" if roof is not None else "unknown",
            reference_dsm_roof_m=roof, local_ground_estimate_m=base_z,
            roof_samples=int(roof_pixels.size), ground_samples=int(ground.size),
            classified_ground_samples=int(observed.size),
            lod1_height_m=placement_height if support == "inferred" else None,
            display_height_m=placement_height, area_m2=geom.area,
            allow_display_proxy=allow_proxy,
            foundation_triangles_xy=[list(t.exterior.coords)[:-1] for t in tri.geoms],
        )
        features.append(dict(type="Feature", properties=props, geometry=mapping(geom)))
    write(out / "building_footprints_projected.json", dict(type="FeatureCollection", features=features))
    with (out / "reports/candidate_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reviewed[0]))
        writer.writeheader()
        writer.writerows(reviewed)
    write(out / "reports/recovery.json", dict(
        baseline=str(base), original_candidates=len(reviewed), image_review=dict(Counter(r["image_review"] for r in reviewed)),
        original_first_raster_gate=dict(Counter(r["raster_gate"] for r in reviewed)),
        original_blender_disposition=dict(Counter(r["blender_gate"] for r in reviewed if r["blender_gate"])),
        new_footprints=dict(Counter(f["properties"]["footprint_tier"] for f in features)),
        height_support=dict(Counter(f["properties"]["height_support"] for f in features)),
        local_terrain_cells_interpolated=int(filled.sum()),
        local_terrain_cells_without_interpolation=int((correction & ~filled).sum()),
        reference_assisted=True, independent_truth_inventory=False,
    ))


if __name__ == "__main__":
    main()
