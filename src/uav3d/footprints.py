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
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
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


def prepare(dsm_path, gsd_m: float | None, cfg: dict, out_gpkg) -> dict:
    """Write roofprints for the reconstructed area to ``out_gpkg``; return a summary."""
    fcfg = cfg["footprints"]
    bbox, crs_wkt = dsm_area(dsm_path)
    osm = fetch_osm(bbox, fcfg["osm_overpass_urls"], fcfg["osm_retries_per_endpoint"])
    summary = {"source": "osm", "endpoint": osm.attrs["endpoint"], "bbox_wgs84": [round(v, 6) for v in bbox],
               "osm_polygons": len(osm)}
    if osm.empty:
        summary["kept"] = 0
        return summary
    fp = osm.to_crs(crs_wkt)
    # dissolve exact duplicates / overlaps from way+relation double mapping
    fp = fp[~fp.geometry.duplicated()].reset_index(drop=True)
    min_area = max((fcfg["min_side_gsd"] * gsd_m) ** 2 if gsd_m else 0.0, fcfg["min_area_m2"])
    fp = fp[fp.geometry.area >= min_area]
    keep = covered(fp, dsm_path, fcfg["min_coverage"]) if len(fp) else []
    fp = fp[keep] if len(fp) else fp
    summary.update(min_area_m2=round(min_area, 2), kept=len(fp),
                   kept_area_m2=round(float(unary_union(list(fp.geometry)).area), 1) if len(fp) else 0.0)
    if len(fp):
        fp.to_file(out_gpkg, layer="roofprints", driver="GPKG")
    return summary
