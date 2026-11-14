"""Building footprints (roofprints) for an area: OpenStreetMap via the Overpass API.

Only the bounding box of the reconstructed area is sent to the endpoint (configs/defaults.json).
Returned polygons are reprojected to the point-cloud CRS and kept only where the reconstruction
covers them, so Roofer is never asked to model a building we did not observe.
"""
from __future__ import annotations

import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
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


def regularise(poly: Polygon, rectangularity: float) -> Polygon:
    """Replace a ragged outline by its minimum rotated rectangle when it fills that rectangle
    well enough (most roofprints are rectangular); otherwise keep the simplified outline."""
    rect = poly.minimum_rotated_rectangle
    return rect if rect.area > 0 and poly.area / rect.area >= rectangularity else poly


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


def prepare(dsm_path, laz_path, gsd_m: float | None, cfg: dict, out_gpkg) -> dict:
    """Write roofprints for the reconstructed area to ``out_gpkg``; return a summary.

    OSM polygons are used where they exist; point-cloud roofprints fill the gaps, kept only
    where they do not substantially overlap an OSM polygon."""
    fcfg, pcfg = cfg["footprints"], cfg["footprints"]["pointcloud"]
    bbox, crs_wkt = dsm_area(dsm_path)
    min_area = max((fcfg["min_side_gsd"] * gsd_m) ** 2 if gsd_m else 0.0, fcfg["min_area_m2"])
    osm = fetch_osm(bbox, fcfg["osm_overpass_urls"], fcfg["osm_retries_per_endpoint"])
    summary = {"endpoint": osm.attrs["endpoint"], "bbox_wgs84": [round(v, 6) for v in bbox],
               "osm_polygons": len(osm), "min_area_m2": round(min_area, 2)}
    fp = osm.to_crs(crs_wkt) if len(osm) else gpd.GeoDataFrame({"source": []}, geometry=[], crs=crs_wkt)
    fp = fp[~fp.geometry.duplicated()].assign(source="osm")
    fp = fp[fp.geometry.area >= min_area]
    if pcfg["enabled"] and laz_path:
        pc = from_pointcloud(laz_path, crs_wkt, gsd_m, pcfg, cfg["roofer"]["building_class"], min_area)
        summary["pointcloud_cell_m"] = pc.attrs.get("cell_m")
        summary["pointcloud_polygons"] = len(pc)
        summary["building_class_points"] = pc.attrs.get("class_points")
        summary["planar_points"] = pc.attrs.get("planar_points")
        summary["pointcloud_candidates"] = pc.attrs.get("candidates")
        if len(fp) and len(pc):
            osm_union = unary_union(list(fp.geometry))
            overlap = pc.geometry.intersection(osm_union).area / pc.geometry.area
            pc = pc[overlap < pcfg["max_osm_overlap"]]
        fp = pd.concat([fp, pc], ignore_index=True)
    fp = gpd.GeoDataFrame(fp.reset_index(drop=True), geometry="geometry", crs=crs_wkt)
    if len(fp):
        fp = fp[covered(fp, dsm_path, fcfg["min_coverage"])]
    summary.update(kept=len(fp), kept_by_source=fp["source"].value_counts().to_dict() if len(fp) else {},
                   kept_area_m2=round(float(unary_union(list(fp.geometry)).area), 1) if len(fp) else 0.0)
    if len(fp):
        fp = fp.reset_index(drop=True)
        fp["fp_id"] = [f"fp{i:05d}" for i in range(len(fp))]  # Roofer --id-attribute
        fp.to_file(out_gpkg, layer="roofprints", driver="GPKG")
    return summary
