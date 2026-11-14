import geopandas as gpd
import laspy
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import box

from uav3d import buildings, detect, footprints

CRS = "EPSG:32617"


def test_tiles_cover_image_and_mosaic_keeps_best_score(tmp_path):
    ortho = tmp_path / "ortho.tif"
    rgb = np.full((3, 400, 400), 120, dtype="uint8")
    with rasterio.open(ortho, "w", driver="GTiff", height=400, width=400, count=3, dtype="uint8", crs=CRS,
                       transform=from_origin(500000, 3100040, 0.1, 0.1)) as ds:
        ds.write(rgb)
    grid = detect.tile_orthophoto(str(ortho), tmp_path / "tiles", res=0.2, tile_px=128, overlap=0.25)
    assert grid["shape"] == (200, 200) and len(grid["tiles"]) == 4
    t = grid["tiles"][0]
    assert Image.open(t["png"]).size == (128, 128)
    m = np.zeros((2, 128, 128), bool)
    m[0, 10:20, 10:20] = True
    m[1, 15:25, 15:25] = True
    np.savez(t["npz"], masks=m, scores=np.array([0.9, 0.6]), prompt_index=np.array([0, 0]))
    for other in grid["tiles"][1:]:
        np.savez(other["npz"], masks=np.zeros((0, 128, 128), bool), scores=np.zeros(0), prompt_index=np.zeros(0))
    s = detect.mosaic(grid, min_score=0.5)
    assert s[12, 12] == 90 and s[17, 17] == 90 and s[23, 23] == 60 and s[100, 100] == 0


def score_raster(tmp_path, building_box_px):
    s = np.zeros((200, 200), dtype="uint8")
    r0, r1, c0, c1 = building_box_px
    s[r0:r1, c0:c1] = 80
    path = tmp_path / "score.tif"
    with rasterio.open(path, "w", driver="GTiff", height=200, width=200, count=1, dtype="uint8", crs=CRS,
                       transform=from_origin(500000, 3100040, 0.2, 0.2), nodata=0) as ds:
        ds.write(s, 1)
    return path


def test_image_cover_and_image_footprints(tmp_path):
    path = score_raster(tmp_path, (50, 100, 50, 100))      # a 10 m x 10 m building at x 10-20, y 20-30
    fp = gpd.GeoDataFrame(geometry=[box(500010, 3100020, 500020, 3100030), box(500030, 3100000, 500036, 3100006)], crs=CRS)
    cover = footprints.image_cover(fp, path, 50)
    assert cover.iloc[0] > 0.95 and cover.iloc[1] == 0.0
    icfg = {"open_m": 0.5, "rectangularity": 0.8}
    img = footprints.from_image(path, CRS, icfg, 50, min_area=10)
    assert len(img) == 1 and abs(img.geometry.iloc[0].area - 100) < 10


def test_elevated_footprints_need_image_support(tmp_path):
    tr = from_origin(500000, 3100100, 0.5, 0.5)
    dtm = np.full((200, 200), 10.0, dtype="float32")
    dsm = dtm.copy()
    dsm[20:80, 20:80] = 90.0     # tower roof, 30 m x 30 m
    dsm[120:140, 120:140] = 18.0  # tree crown
    score = np.zeros((200, 200), dtype="uint8")
    score[26:86, 26:86] = 80      # image mask displaced by 3 m (leaning facades)
    paths = {}
    for name, arr in (("dsm", dsm), ("dtm", dtm), ("score", score)):
        paths[name] = tmp_path / f"{name}.tif"
        with rasterio.open(paths[name], "w", driver="GTiff", height=200, width=200, count=1, dtype=arr.dtype,
                           crs=CRS, transform=tr) as ds:
            ds.write(arr, 1)
    ecfg = {"min_height_m": 2.5, "cell_m": 0.5, "open_m": 1.0, "support_buffer_m": 3.0, "min_support": 0.3,
            "rectangularity": 0.8}
    el = footprints.from_elevated(paths["dsm"], paths["dtm"], paths["score"], CRS, ecfg, 50, min_area=10)
    assert len(el) == 1
    assert abs(el.geometry.iloc[0].area - 900) < 60      # the roof where the points are, not the image mask


def test_relabel_marks_elevated_points_inside_roofprints(tmp_path):
    dtm = tmp_path / "dtm.tif"
    with rasterio.open(dtm, "w", driver="GTiff", height=40, width=40, count=1, dtype="float32", crs=CRS,
                       transform=from_origin(500000, 3100040, 1, 1), nodata=-9999) as ds:
        ds.write(np.full((40, 40), 10.0, dtype="float32"), 1)
    gpkg = tmp_path / "roof.gpkg"
    gpd.GeoDataFrame({"fp_id": ["fp00000"]}, geometry=[box(500010, 3100010, 500020, 3100020)], crs=CRS).to_file(gpkg)
    pts = np.array([[500015, 3100015, 18.0, 2],    # tower roof SMRF called ground -> building
                    [500015, 3100015, 10.2, 2],    # ground inside the footprint stays ground
                    [500030, 3100030, 15.0, 6],    # 'building' outside every footprint (tree) -> unclassified
                    [500012, 3100012, 25.0, 7]])   # noise stays noise
    h = laspy.LasHeader(point_format=6, version="1.4")
    h.offsets, h.scales = [500000, 3100000, 0], [0.001] * 3
    las = laspy.LasData(h)
    las.x, las.y, las.z, las.classification = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3].astype(np.uint8)
    las.write(tmp_path / "in.laz")
    r = buildings.relabel(tmp_path / "in.laz", gpkg, dtm, tmp_path / "out.laz", min_above_m=1.0, building_class=6)
    assert list(laspy.read(tmp_path / "out.laz").classification) == [6, 2, 1, 7]
    assert r == {"building_points_before": 1, "building_points_after": 1}
