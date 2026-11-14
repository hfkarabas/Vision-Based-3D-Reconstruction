import numpy as np
import rasterio
from rasterio.transform import from_origin

from uav3d import terrain


def test_building_mask_lookup(tmp_path):
    s = np.zeros((100, 100), dtype="uint8")
    s[20:40, 20:40] = 90      # building
    s[60:70, 60:70] = 30      # low-score detection, below threshold
    path = tmp_path / "score.tif"
    with rasterio.open(path, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8",
                       crs="EPSG:32617", transform=from_origin(0, 100, 1, 1), nodata=0) as ds:
        ds.write(s, 1)
    x = np.array([30.5, 65.5, 5.0, 500.0])
    y = np.array([70.5, 35.5, 5.0, 500.0])   # row = 100 - y
    assert terrain.in_building_mask(x, y, path, 50).tolist() == [True, False, False, False]
