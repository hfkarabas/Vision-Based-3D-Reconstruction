"""Building footprints (roofprints) for an area: OpenStreetMap via the Overpass API.

Only the bounding box of the reconstructed area is sent to the endpoint (configs/defaults.json).
Returned polygons are reprojected to the point-cloud CRS and kept only where the reconstruction
covers them, so Roofer is never asked to model a building we did not observe.
"""
from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.enums
import rasterio.warp
import requests
from rasterio.features import geometry_mask
from shapely.geometry import Polygon
from shapely.ops import unary_union


class FootprintError(RuntimeError):
    pass


def dsm_area(dsm_path) -> tuple[tuple[float, float, float, float], str]:
    """(west, south, east, north) in WGS84 and the DSM CRS as WKT."""
    with rasterio.open(dsm_path) as ds:
        if ds.crs is None:
            raise FootprintError("DSM has no CRS (no georeferencing); OSM footprints cannot be placed.")
        west, south, east, north = rasterio.warp.transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
        return (west, south, east, north), ds.crs.to_wkt()


def _overpass(query: str, urls: list[str], retries: int, timeout_s: int) -> tuple[list, str]:
    """POST to each endpoint in turn, retrying transient failures with backoff."""
    errors = []
    for url in urls:
        for attempt in range(retries):
            try:
                resp = requests.post(url, data={"data": query}, timeout=timeout_s + 30,
                                     headers={"User-Agent": "uav3d-research-pipeline"})
                if resp.status_code in (429, 502, 503, 504):
                    raise requests.HTTPError(f"{resp.status_code} from {url}")
                resp.raise_for_status()
                return resp.json().get("elements", []), url
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"{url} #{attempt + 1}: {exc}")
                time.sleep(min(60, 5 * 2 ** attempt))
    raise FootprintError("All Overpass endpoints failed: " + " | ".join(errors[-3:]))


def fetch_osm(bbox_wgs84, urls: list[str], retries: int = 3, timeout_s: int = 180) -> gpd.GeoDataFrame:
    west, south, east, north = bbox_wgs84
    query = (f"[out:json][timeout:{timeout_s}];"
             f"(way[\"building\"]({south},{west},{north},{east});"
             f"relation[\"building\"][\"type\"=\"multipolygon\"]({south},{west},{north},{east}););"
             "out geom;")
    elements, used = _overpass(query, urls, retries, timeout_s)
    rows = []
    for el in elements:
        if el["type"] == "way":
            rings = [el.get("geometry", [])]
        else:  # multipolygon relation: outer rings only (holes are irrelevant for roofprints)
            rings = [m.get("geometry", []) for m in el.get("members", []) if m.get("role") == "outer"]
        for ring in rings:
            coords = [(p["lon"], p["lat"]) for p in ring]
            if len(coords) >= 4 and coords[0] == coords[-1]:
                poly = Polygon(coords)
                if poly.is_valid and poly.area > 0:
                    rows.append({"osm_id": f"{el['type'][0]}{el['id']}", "geometry": poly})
    if rows:
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    else:  # no OSM buildings in the area: an empty frame, not an error
        gdf = gpd.GeoDataFrame({"osm_id": []}, geometry=[], crs="EPSG:4326")
    gdf.attrs["endpoint"] = used
    return gdf


def osm_cached(bbox, fcfg: dict, cache_dir: Path) -> tuple[gpd.GeoDataFrame, str]:
    """OSM footprints for the bbox, cached per run folder. If every endpoint fails the run goes on
    without OSM (point-cloud and image footprints remain) and the reason is recorded."""
    key = "_".join(f"{v:.6f}" for v in bbox)
    cache = cache_dir / f"osm_{key}.gpkg"
    if cache.is_file():
        return gpd.read_file(cache), f"cache:{cache.name}"
    try:
        osm = fetch_osm(bbox, fcfg["osm_overpass_urls"], fcfg["osm_retries_per_endpoint"],
                        fcfg["osm_timeout_s"])
    except FootprintError as exc:
        return gpd.GeoDataFrame({"osm_id": []}, geometry=[], crs="EPSG:4326"), f"unavailable: {exc}"[:300]
    cache_dir.mkdir(parents=True, exist_ok=True)
    import pyogrio
    pyogrio.write_dataframe(osm, cache, geometry_type="Polygon")  # also remembers "no buildings here"
    return osm, osm.attrs.get("endpoint", "")


def covered(footprints: gpd.GeoDataFrame, dsm_path, min_fraction: float) -> gpd.Series:
    """Fraction of each footprint (already in the DSM CRS) that has valid DSM cells."""
    with rasterio.open(dsm_path) as ds:
        data = ds.read(1, masked=True)
        valid = ~np.ma.getmaskarray(data)
        fractions = []
        for geom in footprints.geometry:
            inside = ~geometry_mask([geom], out_shape=data.shape, transform=ds.transform, all_touched=False)
            n = inside.sum()
            fractions.append(float(valid[inside].sum() / n) if n else 0.0)
    return pd.Series(fractions, index=footprints.index) >= min_fraction


def rectangularity(poly) -> float:
    """Area over the area of the minimum rotated rectangle (1 = rectangle; ragged traced outlines are low)."""
    rect = poly.minimum_rotated_rectangle
    return float(poly.area / rect.area) if rect.area > 0 else 0.0


def regularise(poly: Polygon, rectangularity: float) -> Polygon:
    """Replace a ragged outline by its minimum rotated rectangle when it fills that rectangle
    well enough (most roofprints are rectangular); otherwise keep the simplified outline."""
    rect = poly.minimum_rotated_rectangle
    return rect if rect.area > 0 and poly.area / rect.area >= rectangularity else poly


def dominant_angle(poly: Polygon, tolerance: float = 0.0) -> float:
    """Main orientation of an outline in radians, modulo 90 deg: length-weighted mean of the edge angles
    on the 4-fold circle (an edge and its perpendicular vote for the same direction). The outline is first
    simplified by `tolerance` so that short ragged edges (tracing noise) do not bias the vote."""
    simple = poly.simplify(tolerance) if tolerance > 0 else poly
    if simple.is_empty or simple.geom_type != "Polygon":
        simple = poly
    c = np.asarray(simple.exterior.coords)
    d = np.diff(c, axis=0)
    # squared length: long walls set the direction, short jagged edges hardly vote
    z = (np.hypot(d[:, 0], d[:, 1]) ** 2 * np.exp(4j * np.arctan2(d[:, 1], d[:, 0]))).sum()
    return float(np.angle(z) / 4)


def orthogonalise(poly: Polygon, step_m: float, min_iou: float) -> Polygon:
    """Right-angled outline along the dominant orientation (the usual footprint regularisation of building
    extraction; cf. JOSM "orthogonalize"): rotate to the dominant direction, simplify by step/2, drop edges
    shorter than step/2, label each edge horizontal or vertical, merge consecutive edges of the same label
    into one line (length-weighted position), and take the corners as intersections of consecutive lines.
    Falls back to the input when the result departs from it (IoU < min_iou) or degenerates."""
    from shapely.affinity import rotate

    if poly.is_empty or poly.area <= 0:
        return poly
    theta = dominant_angle(poly, step_m / 2)
    c = poly.centroid
    r = rotate(poly, -theta, origin=c, use_radians=True).simplify(step_m / 2)
    if r.is_empty or r.geom_type != "Polygon":
        return poly
    pts = np.asarray(r.exterior.coords)[:-1]
    edges = []  # [label, length-weighted position, length]
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        d = b - a
        length = float(np.hypot(*d))
        if length == 0:
            continue
        horizontal = abs(d[0]) >= abs(d[1])
        edges.append(["H" if horizontal else "V", (a[1] + b[1]) / 2 * length if horizontal
                      else (a[0] + b[0]) / 2 * length, length])
    long_edges = [e for e in edges if e[2] >= step_m / 2]
    if len(long_edges) >= 4:
        edges = long_edges
    lines: list[list] = []
    for e in edges:
        if lines and lines[-1][0] == e[0]:
            lines[-1][1] += e[1]
            lines[-1][2] += e[2]
        else:
            lines.append(list(e))
    if len(lines) > 1 and lines[0][0] == lines[-1][0]:
        lines[0][1] += lines[-1][1]
        lines[0][2] += lines[-1][2]
        lines.pop()
    if len(lines) < 4:
        return poly
    pos = [(lab, s / length) for lab, s, length in lines]
    corners = []
    for (l1, v1), (_, v2) in zip(pos, pos[1:] + pos[:1]):
        corners.append((v2, v1) if l1 == "H" else (v1, v2))  # H: y = v1 meets V: x = v2, and vice versa
    o = rotate(Polygon(corners), theta, origin=c, use_radians=True)
    if not o.is_valid or o.area <= 0:
        return poly
    iou = o.intersection(poly).area / o.union(poly).area
    return o if iou >= min_iou else poly


def regularise_all(fp: gpd.GeoDataFrame, rcfg: dict) -> gpd.GeoDataFrame:
    """Right-angled outlines for the traced roofprints (OSM outlines are kept as mapped)."""
    traced = fp["source"] != "osm"
    if not traced.any():
        return fp
    fp = fp.copy()
    fp.loc[traced, "geometry"] = [orthogonalise(p, rcfg["step_m"], rcfg["min_iou"]) for p in fp.geometry[traced]]
    return fp


def from_pointcloud(laz_path, crs_wkt: str, gsd_m: float | None, pcfg: dict, building_class: int,
                    min_area: float) -> gpd.GeoDataFrame:
    """Roofprints from the building-classified points: occupancy raster -> morphological
    clean-up -> connected regions -> simplified polygons. Cell size and kernels scale with GSD."""
    import laspy
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    from scipy import ndimage

    las = laspy.read(laz_path)
    keep = np.asarray(las.classification) == building_class
    pts = np.column_stack([las.x, las.y, las.z])[keep]
    n_class = len(pts)
    all_thin = None
    if "planarity" in pcfg and len(pts):
        all_thin, planar = planarity_mask(pts, **{k: v for k, v in pcfg["planarity"].items()
                                                 if k != "min_planar_fraction"})
        pts = all_thin[planar]
    x, y = pts[:, 0], pts[:, 1]
    if len(x) == 0:
        empty = gpd.GeoDataFrame({"source": []}, geometry=[], crs=crs_wkt)
        empty.attrs.update(class_points=n_class, planar_points=0)
        return empty
    cell = float(np.clip((gsd_m or 0.05) * pcfg["cell_gsd_factor"], pcfg["cell_min_m"], pcfg["cell_max_m"]))
    # pad by more than the largest kernel so closing does not erode regions at the raster edge
    pad = (max(pcfg["close_m"], pcfg["open_m"]) + 2 * cell)
    x0, y1 = x.min() - pad, y.max() + pad
    x_max, y_min = x.max() + pad, y.min() - pad
    cols = int((x_max - x0) / cell) + 1
    rows = int((y1 - y_min) / cell) + 1
    grid = np.zeros((rows, cols), dtype=bool)
    grid[((y1 - y) / cell).astype(int), ((x - x0) / cell).astype(int)] = True

    def disk(radius_m: float) -> np.ndarray:
        r = max(1, int(round(radius_m / cell)))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        return xx * xx + yy * yy <= r * r

    grid = ndimage.binary_closing(grid, structure=disk(pcfg["close_m"]))
    grid = ndimage.binary_fill_holes(grid)
    grid = ndimage.binary_opening(grid, structure=disk(pcfg["open_m"]))
    transform = from_origin(x0, y1, cell, cell)
    polys = []
    for geom, value in shapes(grid.astype(np.uint8), mask=grid, transform=transform):
        if value:
            poly = Polygon(geom["coordinates"][0]).simplify(cell * pcfg["simplify_cells"])
            if poly.is_valid and poly.area >= min_area:
                polys.append(regularise(poly, pcfg["rectangularity"]))
    gdf = gpd.GeoDataFrame({"source": ["pointcloud"] * len(polys)}, geometry=polys, crs=crs_wkt)
    if all_thin is not None and len(gdf):
        # Share of the building-class points inside each candidate that are planar: high on roofs,
        # low on tree crowns that only contributed a few locally planar leaf patches.
        import shapely
        fractions = []
        for geom in gdf.geometry:
            inside = shapely.contains_xy(geom, all_thin[:, 0], all_thin[:, 1])
            fractions.append(float(planar[inside].mean()) if inside.any() else 0.0)
        gdf["planar_fraction"] = np.round(fractions, 3)
        gdf = gdf[gdf["planar_fraction"] >= pcfg["planarity"]["min_planar_fraction"]]
    gdf.attrs.update(cell_m=round(cell, 3), class_points=n_class, planar_points=len(x),
                     candidates=len(polys))
    return gdf


def planarity_mask(pts: np.ndarray, voxel_m: float, k: int,
                   max_surface_variation: float) -> tuple[np.ndarray, np.ndarray]:
    """(voxel-thinned points, mask of those on locally planar surfaces); see planar_points."""
    from scipy.spatial import cKDTree

    _, first = np.unique(np.floor(pts / voxel_m).astype(np.int64), axis=0, return_index=True)
    thin = pts[first]
    if len(thin) < k:
        return thin, np.zeros(len(thin), dtype=bool)
    _, nn = cKDTree(thin).query(thin, k=k, workers=-1)
    nb = thin[nn] - thin[nn].mean(axis=1, keepdims=True)
    ev = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", nb, nb) / k)
    return thin, ev[:, 0] / ev.sum(axis=1).clip(1e-12) <= max_surface_variation


def planar_points(pts: np.ndarray, voxel_m: float, k: int, max_surface_variation: float) -> np.ndarray:
    """Keep points that lie on locally planar surfaces (roofs), dropping tree crowns.

    Points are voxel-thinned (uniform density), then each point's surface variation
    l3 / (l1 + l2 + l3) of its k nearest neighbours is computed (Pauly et al. 2002; used for
    roof/vegetation separation in photogrammetric clouds, e.g. Rönnholm et al. 2025). With
    voxel 0.3 m and k = 30 the neighbourhood is ~1 m, large enough to average out
    photogrammetric noise but smaller than a roof face."""
    thin, mask = planarity_mask(pts, voxel_m, k, max_surface_variation)
    return thin[mask]


def heights(fp: gpd.GeoDataFrame, dsm_path, dtm_path, pct: float = 90) -> pd.Series:
    """pct-th percentile of DSM - DTM inside each footprint (same grid in our terrain output)."""
    with rasterio.open(dsm_path) as dsm, rasterio.open(dtm_path) as dtm:
        if dsm.shape != dtm.shape or not dsm.transform.almost_equals(dtm.transform, precision=1e-6):
            raise FootprintError(f"DSM {dsm.shape} and DTM {dtm.shape} grids differ")
        diff = dsm.read(1, masked=True) - dtm.read(1, masked=True)
        out = []
        for geom in fp.to_crs(dsm.crs).geometry:
            v = diff[~geometry_mask([geom], out_shape=diff.shape, transform=dsm.transform)].compressed()
            out.append(float(np.percentile(v, pct)) if len(v) else 0.0)
    return pd.Series(out, index=fp.index)


def image_cover(fp: gpd.GeoDataFrame, score_path, min_score_pct: int, tolerance_m: pd.Series | None = None) -> pd.Series:
    """Share of each footprint's cells where the image model says 'building'.

    With ``tolerance_m`` a cell also counts when the nearest 'building' cell is within that
    distance: in a (non-true) orthophoto a building of height h appears displaced by up to
    h * tan(view angle), so tall buildings get a proportional tolerance and low ones none."""
    from scipy import ndimage

    with rasterio.open(score_path) as ds:
        s = ds.read(1) >= min_score_pct
        dist = ndimage.distance_transform_edt(~s) * ds.res[0] if tolerance_m is not None else None
        values = []
        for idx, geom in zip(fp.index, fp.to_crs(ds.crs).geometry):
            inside = ~geometry_mask([geom], out_shape=s.shape, transform=ds.transform)
            n = inside.sum()
            if not n:
                values.append(0.0)
            elif dist is None:
                values.append(float(s[inside].sum() / n))
            else:
                values.append(float((dist[inside] <= tolerance_m[idx]).sum() / n))
    return pd.Series(values, index=fp.index)


def from_image(score_path, crs_wkt: str, icfg: dict, min_score_pct: int, min_area: float) -> gpd.GeoDataFrame:
    """Footprints straight from the image-model mask (buildings the point-cloud classes missed)."""
    from rasterio.features import shapes
    from scipy import ndimage

    with rasterio.open(score_path) as ds:
        mask = ds.read(1) >= min_score_pct
        res, transform, crs = ds.res[0], ds.transform, ds.crs
    r = max(1, int(round(icfg["open_m"] / res)))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    mask = ndimage.binary_opening(mask, structure=xx * xx + yy * yy <= r * r)
    polys = []
    for geom, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        if value:
            poly = Polygon(geom["coordinates"][0]).simplify(res * 2)
            if poly.is_valid and poly.area >= min_area:
                polys.append(regularise(poly, icfg["rectangularity"]))
    return gpd.GeoDataFrame({"source": ["image"] * len(polys)}, geometry=polys, crs=crs).to_crs(crs_wkt)


def from_elevated(dsm_path, dtm_path, score_path, crs_wkt: str, ecfg: dict, min_score_pct: int,
                  min_area: float) -> gpd.GeoDataFrame:
    """Footprints whose geometry comes from the points and whose 'building' evidence comes from
    the image: connected regions at least min_height_m above the DTM (nDSM), kept when enough
    of them lies inside the image-model mask dilated by support_buffer_m. The orthophoto of
    tall buildings is not a true orthophoto (facades lean), so image masks alone are displaced;
    the nDSM puts the roof where the points are."""
    from rasterio.features import shapes
    from scipy import ndimage

    with rasterio.open(dsm_path) as dsm, rasterio.open(dtm_path) as dtm:
        f = max(1, int(round(ecfg["cell_m"] / dsm.res[0])))
        shape = (dsm.height // f, dsm.width // f)
        a = dsm.read(1, out_shape=shape, resampling=rasterio.enums.Resampling.average, masked=True)
        b = dtm.read(1, out_shape=shape, resampling=rasterio.enums.Resampling.average, masked=True)
        transform = dsm.transform * dsm.transform.scale(dsm.width / shape[1], dsm.height / shape[0])
        crs = dsm.crs
    cell = abs(transform.a)
    elevated = ((a - b).filled(0) >= ecfg["min_height_m"])
    r = max(1, int(round(ecfg["open_m"] / cell)))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    elevated = ndimage.binary_opening(elevated, structure=xx * xx + yy * yy <= r * r)
    support = np.zeros(shape, dtype=bool)
    with rasterio.open(score_path) as ds:
        s = np.zeros(shape, dtype="uint8")
        rasterio.warp.reproject(ds.read(1), s, src_transform=ds.transform, src_crs=ds.crs, dst_transform=transform,
                                dst_crs=crs, resampling=rasterio.enums.Resampling.max)
    support = s >= min_score_pct
    rb = max(1, int(round(ecfg["support_buffer_m"] / cell)))
    yy, xx = np.ogrid[-rb:rb + 1, -rb:rb + 1]
    support = ndimage.binary_dilation(support, structure=xx * xx + yy * yy <= rb * rb)
    labels, n = ndimage.label(elevated)
    if n == 0:
        return gpd.GeoDataFrame({"source": []}, geometry=[], crs=crs_wkt)
    share = ndimage.mean(support, labels, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = share >= ecfg["min_support"]
    mask = keep[labels]
    polys = []
    for geom, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        if value:
            poly = Polygon(geom["coordinates"][0]).simplify(cell * 1.5)
            if poly.is_valid and poly.area >= min_area:
                polys.append(regularise(poly, ecfg["rectangularity"]))
    return gpd.GeoDataFrame({"source": ["elevated"] * len(polys)}, geometry=polys, crs=crs).to_crs(crs_wkt)


def merge_fragments(fp: gpd.GeoDataFrame, mcfg: dict) -> gpd.GeoDataFrame:
    """Merge roofprint fragments of one building (roofs split by lift shafts, parapets, terraces).

    Two fragments join when the gap between them is at most max(base_gap_m, gap_per_height *
    lower height) and their heights agree within max(height_tol_m, height_tol_frac * taller).
    Each group becomes the morphological closing of its parts. Needs the 'height_p90_m' column."""
    if len(fp) < 2 or "height_p90_m" not in fp:
        return fp
    fp = fp.reset_index(drop=True)
    h = fp["height_p90_m"].clip(lower=0).to_numpy()
    reach = np.maximum(mcfg["base_gap_m"], mcfg["gap_per_height"] * h)
    parent = list(range(len(fp)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = gpd.sjoin(fp.assign(geometry=fp.buffer(reach)), fp[["geometry"]], predicate="intersects")
    for i, j in zip(pairs.index, pairs["index_right"]):
        if i >= j:
            continue
        gap = fp.geometry[i].distance(fp.geometry[j])
        lo, hi = min(h[i], h[j]), max(h[i], h[j])
        if gap <= max(mcfg["base_gap_m"], mcfg["gap_per_height"] * lo) and \
                hi - lo <= max(mcfg["height_tol_m"], mcfg["height_tol_frac"] * hi):
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(fp)):
        groups.setdefault(find(i), []).append(i)
    rows = []
    for members in groups.values():
        first = fp.iloc[members[0]].to_dict()
        if len(members) == 1:
            rows.append(first)
            continue
        r = max(reach[members].min() / 2, mcfg["base_gap_m"] / 2)
        merged = unary_union([fp.geometry[m].buffer(r) for m in members]).buffer(-r)
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)
        first.update(geometry=regularise(merged, mcfg["rectangularity"]), merged_parts=len(members),
                     height_p90_m=float(h[members].max()))
        rows.append(first)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=fp.crs)


def _not_overlapping(new: gpd.GeoDataFrame, existing: gpd.GeoDataFrame, max_overlap: float) -> gpd.GeoDataFrame:
    if not len(existing) or not len(new):
        return new
    union = unary_union(list(existing.geometry))
    return new[new.geometry.intersection(union).area / new.geometry.area < max_overlap]


def prepare(dsm_path, laz_path, gsd_m: float | None, cfg: dict, out_gpkg, score_path=None,
            elevation: tuple | None = None) -> dict:
    """Write roofprints for the reconstructed area to ``out_gpkg``; return a summary.

    OSM polygons are used where they exist. Point-cloud roofprints fill the gaps, kept only
    where they do not substantially overlap an OSM polygon and, when an image-model score
    raster is given, only where the image model also sees a building (trees, terraces and
    other planar non-buildings fail this). Image-model footprints then add buildings the
    point-cloud classes missed; their height is checked later from the points (Roofer)."""
    fcfg, pcfg, icfg = cfg["footprints"], cfg["footprints"]["pointcloud"], cfg["footprints"]["image"]
    min_score_pct = int(round(cfg["detection"]["min_score"] * 100))
    bbox, crs_wkt = dsm_area(dsm_path)
    min_area = max((fcfg["min_side_gsd"] * gsd_m) ** 2 if gsd_m else 0.0, fcfg["min_area_m2"])
    osm, osm_note = osm_cached(bbox, fcfg, Path(out_gpkg).parent.parent / "cache")
    summary = {"osm": osm_note, "bbox_wgs84": [round(v, 6) for v in bbox],
               "osm_polygons": len(osm), "min_area_m2": round(min_area, 2)}
    fp = osm.to_crs(crs_wkt) if len(osm) else gpd.GeoDataFrame({"source": []}, geometry=[], crs=crs_wkt)
    fp = fp[~fp.geometry.duplicated()].assign(source="osm")
    fp = fp[fp.geometry.area >= min_area]
    if score_path is not None and elevation is not None and fcfg["elevated"]["enabled"]:
        el = from_elevated(elevation[0], elevation[1], score_path, crs_wkt, fcfg["elevated"], min_score_pct, min_area)
        el = _not_overlapping(el, fp, fcfg["elevated"]["max_overlap"])
        summary["elevated_polygons"] = len(el)
        fp = pd.concat([fp, el], ignore_index=True)
    if pcfg["enabled"] and laz_path:
        pc = from_pointcloud(laz_path, crs_wkt, gsd_m, pcfg, cfg["roofer"]["building_class"], min_area)
        summary["pointcloud_cell_m"] = pc.attrs.get("cell_m")
        summary["pointcloud_polygons"] = len(pc)
        summary["building_class_points"] = pc.attrs.get("class_points")
        summary["planar_points"] = pc.attrs.get("planar_points")
        summary["pointcloud_candidates"] = pc.attrs.get("candidates")
        pc = _not_overlapping(pc, fp, pcfg["max_osm_overlap"])
        if score_path is not None and len(pc):
            tol = None
            if elevation is not None:
                h = heights(pc, elevation[0], elevation[1])
                tol = (h.clip(lower=0) * icfg["lean_factor"]).clip(upper=icfg["max_lean_m"])
                pc = pc.assign(height_p90_m=h.round(1))
            cover = image_cover(pc, score_path, min_score_pct, tol)
            pc = pc.assign(image_cover=cover.round(3))
            summary["pointcloud_rejected_by_image"] = int((cover < icfg["min_cover"]).sum())
            pc = pc[cover >= icfg["min_cover"]]
        if fcfg["merge"]["enabled"] and elevation is not None and len(pc) > 1:
            if "height_p90_m" not in pc:
                pc = pc.assign(height_p90_m=heights(pc, elevation[0], elevation[1]).round(1))
            before = len(pc)
            pc = merge_fragments(pc, fcfg["merge"])
            summary["pointcloud_merged_from"] = before
        fp = pd.concat([fp, pc], ignore_index=True)
    if score_path is not None:
        img = _not_overlapping(from_image(score_path, crs_wkt, icfg, min_score_pct, min_area), fp,
                               icfg["max_overlap"])
        summary["image_polygons_added"] = len(img)
        fp = pd.concat([fp, img], ignore_index=True)
    fp = gpd.GeoDataFrame(fp.reset_index(drop=True), geometry="geometry", crs=crs_wkt)
    if len(fp) and fcfg.get("regularise", {}).get("enabled"):
        fp = regularise_all(fp, fcfg["regularise"])
    if len(fp):
        fp = fp[covered(fp, dsm_path, fcfg["min_coverage"])]
    summary.update(kept=len(fp), kept_by_source=fp["source"].value_counts().to_dict() if len(fp) else {},
                   kept_area_m2=round(float(unary_union(list(fp.geometry)).area), 1) if len(fp) else 0.0)
    if len(fp):
        fp = fp.reset_index(drop=True)
        fp["fp_id"] = [f"fp{i:05d}" for i in range(len(fp))]  # Roofer --id-attribute
        # outline quality, carried by Roofer into the CityJSON attributes (buildings._assess)
        fp["rectangularity"] = [round(rectangularity(p), 3) for p in fp.geometry]
        fp.to_file(out_gpkg, layer="roofprints", driver="GPKG")
    return summary
