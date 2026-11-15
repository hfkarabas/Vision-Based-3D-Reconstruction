"""Build a building-labelling kit for tune scenes (QGIS).

For each scene: labels/buildings/<scene>.gpkg with
  - layer "area": the region(s) to label completely. Either the whole survey, or N random
    squares (fixed seed, fully inside the survey interior, non-overlapping) so the labelled
    sample is not chosen by the person who designs the method;
  - layer "buildings": empty polygons with an integer field "uncertain" (1 = not sure).
and a background image in labels/buildings/_backgrounds/ (git-ignored): the run's own orthophoto,
JPEG-compressed with overviews.
Usage: python bench/make_label_kit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import box, shape
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "labels" / "buildings"
SEED = 20261114
SCENES = [
    {"name": "sheffield", "run": "outputs/uav3d/sheffield", "area": "full"},
]


def survey_polygon(dsm_path: Path, step: int = 20):
    with rasterio.open(dsm_path) as ds:
        a = ds.read(1, out_shape=(ds.height // step, ds.width // step), resampling=Resampling.nearest)
        tr = ds.transform * ds.transform.scale(ds.width / a.shape[1], ds.height / a.shape[0])
        valid = (a != ds.nodata).astype("uint8")
        crs = ds.crs
    polys = [shape(g) for g, v in shapes(valid, mask=valid.astype(bool), transform=tr) if v]
    return unary_union(polys), crs


def image_footprint(path: Path, crs, step: int = 16):
    """Polygon (in ``crs``) where the background image has data (non-black)."""
    with rasterio.open(path) as ds:
        a = ds.read(out_shape=(ds.count, ds.height // step, ds.width // step), resampling=Resampling.nearest)
        tr = ds.transform * ds.transform.scale(ds.width / a.shape[2], ds.height / a.shape[1])
        valid = (a.max(axis=0) > 0).astype("uint8")
        src_crs = ds.crs
    poly = unary_union([shape(g) for g, v in shapes(valid, mask=valid.astype(bool), transform=tr) if v])
    return gpd.GeoSeries([poly], crs=src_crs).to_crs(crs).iloc[0]


def random_squares(survey, n: int, size: float, rng) -> list:
    interior = survey.buffer(-size / 2 - 10)
    minx, miny, maxx, maxy = interior.bounds
    squares = []
    for _ in range(20000):
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        sq = box(x - size / 2, y - size / 2, x + size / 2, y + size / 2)
        if survey.buffer(-10).contains(sq) and not any(sq.intersects(s) for s in squares):
            squares.append(sq)
            if len(squares) == n:
                return squares
    raise RuntimeError(f"could only place {len(squares)} of {n} squares")


def background_odm(run: Path, out: Path) -> None:
    src = run / "odm" / "project" / "odm_orthophoto" / "odm_orthophoto.tif"
    with rasterio.open(src) as ds:
        f = max(1, int(round(0.05 / ds.res[0])))  # ~5 cm is plenty for roof outlines
        h, w = ds.height // f, ds.width // f
        rgb = ds.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average)
        tr = ds.transform * ds.transform.scale(ds.width / w, ds.height / h)
        _write_rgb(out, rgb, tr, ds.crs)


def _write_rgb(out: Path, rgb, tr, crs) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "height": rgb.shape[1], "width": rgb.shape[2], "count": 3, "dtype": "uint8",
               "crs": crs, "transform": tr, "compress": "JPEG", "photometric": "YCBCR", "tiled": True,
               "blockxsize": 512, "blockysize": 512}
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(rgb)
        dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)


def main() -> None:
    rng = np.random.default_rng(SEED)
    manifest = []
    for sc in SCENES:
        run = REPO / sc["run"]
        survey, crs = survey_polygon(run / "odm" / "project" / "odm_dem" / "dsm.tif")
        bg = OUT / "_backgrounds" / f"{sc['name']}.tif"
        if not bg.exists():
            background_odm(run, bg)
        survey = survey.intersection(image_footprint(bg, crs))  # label only what is both reconstructed and visible
        if sc["area"] == "full":
            areas = [survey.buffer(0)]
        else:
            areas = random_squares(survey, sc["n"], sc["size_m"], rng)
        gpkg = OUT / f"{sc['name']}.gpkg"
        OUT.mkdir(parents=True, exist_ok=True)
        gpkg.unlink(missing_ok=True)
        area_df = gpd.GeoDataFrame({"area_id": list(range(1, len(areas) + 1)), "seed": SEED}, geometry=areas, crs=crs)
        pyogrio.write_dataframe(area_df, gpkg, layer="area")
        empty = gpd.GeoDataFrame({"uncertain": np.array([], dtype="int32")}, geometry=[], crs=crs)
        pyogrio.write_dataframe(empty, gpkg, layer="buildings", geometry_type="Polygon", append=False)
        manifest.append({"scene": sc["name"], "gpkg": str(gpkg.relative_to(REPO)), "background": str(bg.relative_to(REPO)),
                         "crs": crs.to_string(), "areas": len(areas), "area_m2": round(sum(a.area for a in areas))})
        print(manifest[-1])
    (OUT / "kit.json").write_text(json.dumps({"seed": SEED, "scenes": manifest}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
