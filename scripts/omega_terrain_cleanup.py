"""Run bounded PDAL classification and repair selected height-grid defects."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import psutil
import rasterio
from rasterio.features import geometry_mask
from rasterio.fill import fillnodata
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from scipy.ndimage import binary_dilation, distance_transform_edt, maximum_filter, minimum_filter
from shapely.geometry import box, mapping, shape
from shapely import constrained_delaunay_triangles
from sklearn.linear_model import RANSACRegressor


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def raster(path):
    with rasterio.open(path) as src:
        return src.read(1, masked=True).astype("float32").filled(np.nan), src.profile


def save_raster(path, z, profile, mask=False):
    dtype = "uint8" if mask else "float32"
    nodata = None if mask else -9999
    with rasterio.open(path, "w", **dict(profile, count=1, dtype=dtype, nodata=nodata, compress="deflate")) as dst:
        dst.write(z.astype(dtype) if mask else np.where(np.isfinite(z), z, nodata).astype(dtype), 1)


def aligned(profile, other):
    return all(profile[key] == other[key] for key in ("crs", "transform", "width", "height"))


def classify(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    z, p = raster(args.dsm)
    h, w = z.shape
    a = p["transform"]
    if h * w > args.max_cells or a.b or a.d or a.a <= 0 or not np.isclose(a.a, -a.e):
        raise ValueError("Classification requires a bounded, square-cell north-up grid.")
    if not p["crs"] or not p["crs"].is_projected or p["crs"].linear_units_factor[1] != 1:
        raise ValueError("A metre-based projected CRS is required.")
    results = {}
    for name, stage in (
        ("smrf", dict(type="filters.smrf", cell=a.a, slope=0.2, scalar=1.2, threshold=0.45, window=18)),
        ("csf", dict(type="filters.csf", resolution=a.a, rigidness=3, iterations=500, threshold=0.5)),
    ):
        target = out / f"{name}.tif"
        pipeline = [
            dict(type="readers.gdal", filename=str(args.dsm.resolve()), header="Z"),
            dict(type="filters.expression", expression="Z > -9998 && Z < 12000"),
            stage,
            dict(type="filters.expression", expression="Classification == 2"),
            dict(type="writers.gdal", filename=str(target), resolution=a.a, radius=a.a * 0.51,
                 origin_x=a.c, origin_y=a.f + a.e * h, width=w, height=h,
                 output_type="min", data_type="float32", nodata=-9999, override_srs=p["crs"].to_string()),
        ]
        recipe = out / f"{name}.json"
        dump(recipe, pipeline)
        start = time.perf_counter()
        peak = 0
        with (out / f"{name}.log").open("wb") as log:
            with subprocess.Popen([str(args.pdal.resolve()), "pipeline", str(recipe)], stdout=log, stderr=subprocess.STDOUT) as proc:
                observer = psutil.Process(proc.pid)
                while proc.poll() is None:
                    if time.perf_counter() - start > args.timeout:
                        proc.kill()
                        raise TimeoutError(f"PDAL {name} exceeded its time limit")
                    try:
                        peak = max(peak, observer.memory_info().rss)
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(0.15)
                if proc.wait() != 0:
                    raise RuntimeError(f"PDAL {name} failed, see {out / (name + '.log')}")
        result, rp = raster(target)
        if not aligned(p, rp):
            raise ValueError(f"PDAL {name} changed the grid")
        results[name] = dict(seconds=time.perf_counter() - start, peak_rss_bytes=peak,
                             ground_cells=int(np.isfinite(result).sum()))
    dump(out / "classification.json", dict(surface=str(args.dsm.resolve()), valid_surface_cells=int(np.isfinite(z).sum()),
                                           total_cells=z.size, filters=results))


def rasterize(poly, profile):
    return geometry_mask([mapping(poly)], out_shape=(profile["height"], profile["width"]),
                         transform=profile["transform"], invert=True)


def interpolate(z, anchors, max_distance, cell):
    if not anchors.any():
        return np.full_like(z, np.nan), np.full_like(z, np.inf)
    dist = distance_transform_edt(~anchors) * cell
    values = fillnodata(np.where(anchors, z, np.nan).astype("float32"), mask=anchors.astype("uint8"),
                        max_search_distance=max_distance / cell, smoothing_iterations=0)
    return np.where(dist <= max_distance, values, np.nan), dist


def court_plane(z, core, cell, target=None, baseline=None):
    r, c = np.where(core & np.isfinite(z))
    if r.size < 50:
        raise ValueError("Insufficient court samples")
    xy = np.column_stack((c * cell, r * cell))
    observed = z[r, c]
    held = (r + 2 * c) % 5 == 0
    fit = RANSACRegressor(min_samples=3, residual_threshold=0.35, random_state=7)
    fit.fit(xy[~held], observed[~held])
    err = np.abs(observed[held] - fit.predict(xy[held]))
    slope = float(np.linalg.norm(fit.estimator_.coef_))
    support = float(fit.inlier_mask_.mean())
    if slope > 0.15 or support < 0.80 or np.percentile(err, 95) > 0.50:
        raise ValueError("Court did not have a stable plane supported by held-out source cells")
    rr, cc = np.where(core if target is None else target)
    plane = np.full_like(z, np.nan)
    plane[rr, cc] = fit.predict(np.column_stack((cc * cell, rr * cell)))
    evidence = dict(observed_cells=r.size, heldout_cells=int(held.sum()), fitted_slope_percent=slope * 100,
                    fit_inlier_fraction=support, heldout_abs_p95_m=float(np.percentile(err, 95)))
    if baseline is not None:
        before = np.abs(observed[held] - baseline[r[held], c[held]])
        evidence["baseline_against_same_heldout_dsm_abs_p95_m"] = float(np.nanpercentile(before, 95))
    return plane, evidence


def stats(z, roi, reference, cell):
    valid = np.isfinite(z) & roi
    v = z[valid]
    error = np.abs(z - reference)[valid & np.isfinite(reference)]
    gradients = []
    for axis in (0, 1):
        r = np.moveaxis(valid, axis, 0)
        d = np.abs(np.diff(np.moveaxis(z, axis, 0), axis=0))[r[1:] & r[:-1]] / cell
        gradients.extend(d)
    return dict(
        valid_cells=int(valid.sum()), median_z_m=float(np.median(v)) if v.size else None,
        p95_reference_abs_residual_m=float(np.percentile(error, 95)) if error.size else None,
        median_reference_abs_residual_m=float(np.median(error)) if error.size else None,
        p95_adjacent_gradient_m_per_m=float(np.percentile(gradients, 95)) if gradients else None,
        max_adjacent_gradient_m_per_m=float(max(gradients)) if gradients else None,
        jumps_gt_1m_per_m=int(np.count_nonzero(np.asarray(gradients) > 1)),
    )


def clean(args):
    start = time.perf_counter()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    masks = out / "masks"
    masks.mkdir()
    z, profile = raster(args.dsm)
    smrf, smrf_profile = raster(args.filters / "smrf.tif")
    csf, csf_profile = raster(args.filters / "csf.tif")
    if not aligned(profile, smrf_profile) or not aligned(profile, csf_profile):
        raise ValueError("Ground classification grids differ from the input surface")
    aff = profile["transform"]
    cell = aff.a
    r, c = np.indices(z.shape)
    if args.base.suffix == ".npz":
        with np.load(args.base) as grid:
            base, x, y = (grid[k].copy() for k in ("z", "x", "y"))
        origin = [aff.c + cell / 2 - x[0], aff.f - cell / 2 - y[0], 0.0]
        if base.shape != z.shape or not np.allclose(np.diff(x), cell) or not np.allclose(np.diff(y), -cell):
            raise ValueError("Local NPZ axes disagree with the surface grid")
    else:
        base, bp = raster(args.base)
        if not aligned(profile, bp):
            raise ValueError("Baseline terrain and source surface use different grids")
        origin = [aff.c + cell * z.shape[1] / 2, aff.f - cell * z.shape[0] / 2, 0.0]
        x, y = aff.c + (c[0] + 0.5) * cell - origin[0], aff.f - (r[:, 0] + 0.5) * cell - origin[1]
    regions = load(args.regions) if args.regions else {}
    if regions and rasterio.crs.CRS.from_string(regions["projected_crs"]) != profile["crs"]:
        raise ValueError("Region CRS differs from the terrain CRS")
    specs = regions.get("features", [])
    rois = {f["properties"]["name"]: (f["properties"]["mode"], rasterize(shape(f["geometry"]), profile))
            for f in specs}
    fit_rois = {f["properties"]["name"]: rasterize(shape(f["properties"]["fit_geometry"]), profile)
                for f in specs if "fit_geometry" in f["properties"]}
    widths = {f["properties"]["name"]: f["properties"]["transition_width_m"]
              for f in specs if "transition_width_m" in f["properties"]}
    classified = np.isfinite(z) & np.isfinite(smrf) & np.isfinite(csf)
    accepted = classified.copy()
    if args.footprints:
        footprint_ground = np.zeros_like(accepted)
        for f in load(args.footprints)["features"]:
            geom = shape(f["geometry"])
            accepted &= ~rasterize(geom.buffer(1.0), profile)
            footprint_ground |= rasterize(geom, profile)
        rois["building_footprints"] = ("structures", footprint_ground)
        widths["building_footprints"] = args.footprint_transition_width
    for mode, roi in rois.values():
        if mode == "non_ground":
            accepted &= ~roi
    strict = np.where(accepted, z, np.nan)
    candidate, dist = interpolate(z, accepted, args.radius, cell)
    clean = base.copy()
    original_support = np.isfinite(base)
    hi = maximum_filter(np.where(original_support, base, -np.inf), size=3)
    lo = minimum_filter(np.where(original_support, base, np.inf), size=3)
    steep = (hi - lo) > 1.5 * cell
    if not rois:
        rois = {"domain": ("spikes", original_support)}
    reference_by_roi = {}
    protected = np.zeros_like(accepted)
    for mode, roi in rois.values():
        if mode in ("control", "boundary"):
            protected |= roi
    court_results = {}
    rejected = {}
    for name, (mode, roi) in rois.items():
        target = candidate
        if mode == "court":
            target, evidence = court_plane(z, fit_rois.get(name, roi), cell, target=roi, baseline=base)
            court_results[name] = evidence
            take = roi & np.isfinite(target) & original_support
            clean[take] = target[take]
        elif mode in ("spikes", "non_ground", "structures"):
            residual = base - target
            if mode == "spikes":
                flagged = roi & steep & (np.abs(residual) > args.trigger)
            else:
                flagged = roi & (residual > args.trigger)
            width = widths.get(name, args.transition_width)
            if not np.isfinite(width) or width < cell:
                raise ValueError(f"Transition width for {name} must be at least one grid cell")
            steps = int(np.ceil(width / cell))
            supported = roi & np.isfinite(target) & original_support & ~protected
            take = binary_dilation(flagged & supported, iterations=steps) & supported
            if take.any():
                # Blend only the mask fringe. All target elevations come from GDAL.
                weight = np.minimum(distance_transform_edt(take) * cell / width, 1)
                proposal = clean.copy()
                proposal[take] = base[take] - weight[take] * residual[take]
                surrounding = binary_dilation(roi)
                before = stats(clean, surrounding, target, cell)
                after = stats(proposal, surrounding, target, cell)
                if (
                    after["max_adjacent_gradient_m_per_m"] <= before["max_adjacent_gradient_m_per_m"] + 0.1
                    and after["jumps_gt_1m_per_m"] <= before["jumps_gt_1m_per_m"]
                ):
                    clean = proposal
                else:
                    rejected[name] = dict(before=before, trial=after)
        save_raster(masks / f"{name}.tif", roi, profile, mask=True)
        reference_by_roi[name] = target
    changed = original_support & (clean != base)
    rows = []
    for name, (_, roi) in rois.items():
        for variant, surface in (("baseline", base), ("cleaned", clean), ("SMRF", smrf), ("CSF", csf)):
            rows.append(dict(region=name, variant=variant, roi_cells=int(roi.sum()),
                             changed_cells=int((changed & roi).sum()) if variant == "cleaned" else 0,
                             **stats(surface, roi, reference_by_roi[name], cell)))
    interpolation = changed & ((dist > 0) | ~accepted)
    for mode, roi in rois.values():
        if mode == "court":
            interpolation |= changed & roi
    preserve = ~(changed | ~original_support)
    boundary = binary_dilation(changed) & ~changed & original_support
    save_raster(out / "terrain_cleaned_production.tif", clean, profile)
    save_raster(out / "terrain_difference.tif", clean - base, profile)
    strict_name = "terrain_strict_uav_only.tif" if args.role == "uav-only" else "terrain_strict_reference_assisted.tif"
    save_raster(out / strict_name, strict, profile)
    save_raster(out / "interpolated_or_uncertain_mask.tif", interpolation | (original_support & ~accepted), profile, mask=True)
    save_raster(masks / "changed.tif", changed, profile, mask=True)
    save_raster(masks / "interpolated_changed.tif", interpolation, profile, mask=True)
    save_raster(masks / "ground_anchor.tif", accepted, profile, mask=True)
    np.savez_compressed(out / "terrain.npz", x=x, y=y, z=clean)
    np.savez_compressed(out / "baseline.npz", x=x, y=y, z=base)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump(out / "preservation.json", dict(
        role=args.role, crs=profile["crs"].to_string(), origin=origin, z_scale=1.0, pixel_m=cell,
        baseline=str(args.base.resolve()), source_surface=str(args.dsm.resolve()), max_fill_distance_m=args.radius,
        residual_trigger_m=args.trigger, valid_before=int(original_support.sum()), valid_after=int(np.isfinite(clean).sum()),
        changed_cells=int(changed.sum()), unchanged_valid_cells=int(preserve.sum()),
        rejected_transition_trials=rejected, transition_width_m=args.transition_width, region_transition_widths_m=widths,
        strict_support_cells=int(accepted.sum()), interpolated_changed_cells=int(interpolation.sum()),
        original_nodata_preserved=bool(np.array_equal(np.isfinite(clean), original_support)),
        boundary=dict(before=stats(base, boundary | changed, candidate, cell),
                      after=stats(clean, boundary | changed, candidate, cell)),
        courts=court_results, runtime_s=time.perf_counter()-start, final_rss_bytes=psutil.Process().memory_info().rss,
        uncertainty="Classification agreement is not ground truth; no outside-mask edits or new elevation observations.",
    ))


def prepare_asset(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for name in ("terrain_photo.png", "grid.json"):
        shutil.copy2(args.baseline / name, out / name)
    payload = load(args.baseline / "building_footprints_projected.json")
    previous = {r["building_id"]: r for r in load(args.baseline / "reports/blender_buildings.json")}
    ground, profile = raster(args.terrain / "terrain_cleaned_production.tif")
    anchor, _ = raster(args.terrain / "masks/ground_anchor.tif")
    dsm, dp = raster(args.dsm)
    if not aligned(profile, dp):
        raise ValueError("The roof DSM must share the ground grid")
    geometries = [shape(f["geometry"]) for f in payload["features"]]
    for feature, geom in zip(payload["features"], geometries, strict=True):
        p = feature["properties"]
        old = previous.get(p["building_id"], {})
        if old.get("status") != "built":
            p["status"] = "footprint_only"
            continue
        p["height_support"] = old["height_support"]
        if p["height_support"] == "unknown":
            p["display_height_m"] = 3.0
            p["lod1_height_m"] = None
            continue
        p["allow_display_proxy"] = False
        # Preserve the validated top. Report a new ground-relative estimate
        # only where nearby retained ground is adequately sampled.
        p["reference_dsm_roof_m"] = old["roof_z_m"]
        ring = rasterize(geom.buffer(12).difference(geom.buffer(2)), profile)
        samples = ground[ring & (anchor == 1) & np.isfinite(ground)]
        if samples.size >= 15:
            local = float(np.median(samples))
            estimate = old["roof_z_m"] - local
            if 1.5 <= estimate <= 120:
                p["local_ground_estimate_m"] = local
                p["lod1_height_m"] = estimate
                p["height_provenance"] = "Preserved roof elevation minus median nearby retained classified ground"
    for review in load(args.regions).get("reviewed_roofs", []):
        geom = shape(review["geometry"])
        if not geom.is_valid or any(geom.intersection(other).area > 0.05 for other in geometries):
            raise ValueError(f"Invalid or overlapping reviewed roof: {review['id']}")
        core = geom.buffer(-1)
        values = dsm[rasterize(core, profile) & np.isfinite(dsm)]
        ring = rasterize(geom.buffer(7).difference(geom.buffer(1.5)), profile)
        observed = ground[ring & (anchor == 1) & np.isfinite(ground)]
        roof = float(np.median(values)) if values.size >= 15 else None
        local = float(np.median(observed)) if observed.size >= 15 else None
        height = roof-local if roof is not None and local is not None else None
        eligible = height is not None and 1.5 <= height <= 10 and np.percentile(values, 90)-np.percentile(values, 10) < 1.5
        triangles = constrained_delaunay_triangles(geom)
        p = dict(building_id=review["id"], status="eligible" if eligible else "footprint_only",
                 footprint_tier="reviewed_dom_roof", footprint_provenance=review["provenance"],
                 height_support="inferred" if eligible else "unknown", roof_support="reference_dsm_sampled",
                 height_provenance="Reference DSM roof minus classified nearby ground" if eligible else "Uncertain",
                 reference_dsm_roof_m=roof, local_ground_estimate_m=local, lod1_height_m=height if eligible else None,
                 display_height_m=height if eligible else None, allow_display_proxy=False,
                 roof_samples=int(values.size), ground_samples=int(observed.size), area_m2=geom.area,
                 foundation_triangles_xy=[list(t.exterior.coords)[:-1] for t in triangles.geoms])
        payload["features"].append(dict(type="Feature", properties=p, geometry=mapping(geom)))
    dump(out / "building_footprints_projected.json", payload)


def boundary_grid(args):
    from omega_building_rasters import on_grid

    grid = load(args.grid)
    bounds = [value + sign * args.halo for value, sign in zip(grid["bounds"], [-1, -1, 1, 1])]
    w = int(np.ceil((bounds[2]-bounds[0]) / 0.3))
    h = int(np.ceil((bounds[3]-bounds[1]) / 0.3))
    if w * h > 5_000_000 or args.halo <= 0:
        raise ValueError("Boundary inference crop exceeds the bounded budget")
    profile = dict(driver="GTiff", crs=grid["crs"], width=w, height=h,
                   transform=from_bounds(*bounds, w, h), compress="deflate")
    rgb = on_grid(Path(grid["rgb"]), profile, grid["rgb_assigned_crs"], [1, 2, 3], Resampling.average, 0)
    args.output.mkdir(parents=True, exist_ok=False)
    with rasterio.open(args.output / "inference_rgb.tif", "w", count=3, dtype="uint8", **profile) as dst:
        dst.write(rgb)
    dump(args.output / "boundary_grid.json", dict(bounds=bounds, halo_m=args.halo, source=grid["rgb"], crs=grid["crs"]))


def boundary_matches(args):
    import geopandas as gpd

    grid = load(args.grid)
    edge = box(*grid["bounds"])
    polygons = list(gpd.read_file(args.output / "geoai_polygons.gpkg").to_crs(grid["crs"]).geometry)
    with rasterio.open(args.output / "inference_rgb.tif") as src:
        inset = box(*src.bounds).buffer(-max(src.res))
    rows = []
    unique = {}
    for f in load(args.candidates)["features"]:
        prop = f["properties"]
        if prop["status"] != "footprint_clipped_by_inference_extent":
            continue
        original = shape(f["geometry"])
        ranked = sorted(((original.intersection(g).area, i, g) for i, g in enumerate(polygons)), reverse=True)
        overlap, index, match = ranked[0] if ranked else (0, -1, None)
        matched = match is not None and overlap > 0.45 * original.area
        recovered = matched and inset.covers(match) and not edge.covers(match) and match.area > 1.05 * original.area
        props = dict(building_id=prop["building_id"], original_area_m2=original.area,
                     match_index=index if matched else None, overlap_of_original=overlap/original.area,
                     match_area_m2=match.area if matched else None,
                     status="extended_candidate_outside_fixed_terrain" if recovered else "unresolved")
        rows.append(props)
        if recovered:
            if index not in unique:
                unique[index] = dict(type="Feature", geometry=mapping(match),
                                     properties=dict(match_index=index, area_m2=match.area, matched_original_ids=[],
                                                     status="extended_candidate_outside_fixed_terrain"))
            unique[index]["properties"]["matched_original_ids"].append(prop["building_id"])
    features = list(unique.values())
    if features:
        data = gpd.GeoDataFrame.from_features(features, crs=grid["crs"])
        data.to_crs(4326).to_file(args.output / "extended_footprint_candidates.geojson", index=False)
    dump(args.output / "boundary_recovery.json", dict(
        fixed_terrain_bounds=grid["bounds"], original_clipped=len(rows), distinct_extended_candidates=len(features),
        original_fragments_matched=sum(r["status"] == "extended_candidate_outside_fixed_terrain" for r in rows),
        matches=rows, warning="Candidate evidence only. Volumes outside the protected terrain footprint were not extruded.",
    ))


def prepare_survey(args):
    """Align a public orthophoto and DSM on a bounded metre-based grid."""
    from PIL import Image

    with rasterio.open(args.dsm) as dsm, rasterio.open(args.rgb) as rgb:
        crs = dsm.crs
        if not crs or crs != rgb.crs or not crs.is_projected or crs.linear_units_factor[1] != 1:
            raise ValueError("Survey inputs must share a projected metre-based CRS")
        bounds = [
            int(np.ceil(max(dsm.bounds.left, rgb.bounds.left))),
            int(np.ceil(max(dsm.bounds.bottom, rgb.bounds.bottom))),
            int(np.floor(min(dsm.bounds.right, rgb.bounds.right))),
            int(np.floor(min(dsm.bounds.top, rgb.bounds.top))),
        ]
        w, h = bounds[2]-bounds[0], bounds[3]-bounds[1]
        if min(w, h) < 3 or w*h > 500_000:
            raise ValueError("Overlapping survey region is empty or exceeds the bounded 1 m grid")
        args.output.mkdir(parents=True, exist_ok=False)
        options = dict(crs=crs, width=w, height=h, transform=from_bounds(*bounds, w, h),
                       resampling=Resampling.average)
        with WarpedVRT(dsm, **options, nodata=-9999) as surface:
            save_raster(args.output / "dsm.tif", surface.read(1, masked=True).astype("float32").filled(np.nan),
                        dict(surface.profile, driver="GTiff"))
        rgb_options = dict(options, width=w*3, height=h*3, transform=from_bounds(*bounds, w*3, h*3))
        with WarpedVRT(rgb, **rgb_options) as photo:
            colors = photo.read([1, 2, 3])
        Image.fromarray(np.moveaxis(colors, 0, -1)).save(args.output / "terrain_photo.png")
    dump(args.output / "grid.json", dict(
        crs=crs.to_string(), bounds=bounds, origin_enu=[(bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2, 0],
        dsm=str(args.dsm.resolve()), rgb=str(args.rgb.resolve()),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    fit = commands.add_parser("classify")
    fit.add_argument("--dsm", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--pdal", type=Path, required=True)
    fit.add_argument("--max-cells", type=int, default=1_000_000)
    fit.add_argument("--timeout", type=int, default=240)
    fit.set_defaults(run=classify)
    repair = commands.add_parser("clean")
    for name in ("dsm", "base", "filters", "output"):
        repair.add_argument(f"--{name}", type=Path, required=True)
    repair.add_argument("--regions", type=Path)
    repair.add_argument("--footprints", type=Path)
    repair.add_argument("--role", choices=("reference-assisted", "uav-only"), required=True)
    repair.add_argument("--radius", type=float, default=12.0)
    repair.add_argument("--trigger", type=float, default=1.25)
    repair.add_argument("--transition-width", type=float, default=4.0)
    repair.add_argument("--footprint-transition-width", type=float, default=10.0)
    repair.set_defaults(run=clean)
    stage = commands.add_parser("prepare-asset")
    for name in ("baseline", "terrain", "dsm", "regions", "output"):
        stage.add_argument(f"--{name}", type=Path, required=True)
    stage.set_defaults(run=prepare_asset)
    halo = commands.add_parser("boundary-grid")
    halo.add_argument("--grid", type=Path, required=True)
    halo.add_argument("--output", type=Path, required=True)
    halo.add_argument("--halo", type=float, default=22.0)
    halo.set_defaults(run=boundary_grid)
    matches = commands.add_parser("boundary-matches")
    for name in ("grid", "candidates", "output"):
        matches.add_argument(f"--{name}", type=Path, required=True)
    matches.set_defaults(run=boundary_matches)
    survey = commands.add_parser("prepare-survey")
    for name in ("dsm", "rgb", "output"):
        survey.add_argument(f"--{name}", type=Path, required=True)
    survey.set_defaults(run=prepare_survey)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
