"""GDAL georeferencing of an actual orthographic photo render; no new ortho engine."""
from pathlib import Path
import json,shutil,sys
from osgeo import gdal
import rasterio
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mission-omega'
scene=sys.argv[1];variant=sys.argv[2] if len(sys.argv)>2 else 'production8'
src=OUT/'raw'/scene/variant;dest=OUT/'raw'/scene/(variant+'-ortho');dest.mkdir(exist_ok=True)
asset=OUT/'asset'/f'{scene}-{variant}'
qa=json.loads((src/'raster_qa.json').read_text());xmin,ymin,xmax,ymax=qa['roi_bounds']
for name in ['dtm.tif','geospatial_transform.json','source_manifest.json']:
    shutil.copy2(src/name,dest/name)
gdal.UseExceptions()
ds=gdal.Translate(str(dest/'orthographic_photo_mosaic.tif'),str(asset/'orthographic_photo_mosaic.png'),
    format='GTiff',outputSRS='EPSG:4547',outputBounds=[xmin,ymax,xmax,ymin],creationOptions=['COMPRESS=DEFLATE'])
ds=None
with rasterio.open(dest/'orthographic_photo_mosaic.tif') as ds:
    rgba=ds.read();valid=rgba[3]>0
    qa['orthographic_mosaic']=dict(bounds=list(ds.bounds),transform=list(ds.transform),crs=str(ds.crs),shape=[ds.height,ds.width],
        alpha_supported_pixels=int(valid.sum()),alpha_supported_fraction=float(valid.mean()),
        nominal_pixel_size_m=list(ds.res),source_render=str(asset/'orthographic_photo_mosaic.png'),
        alignment='Same declared ROI; north-up (negative raster row step), X=east Y=north; Blender camera above Z with image up +Y',
        note='Not ODM ortho or DSM building-correct true ortho. Rendered COLMAP-photo-textured DTM; sampled ground and photographic support are different masks.')
qa['texture_status']='Georeferenced orthographic photo mosaic from COLMAP-photo-textured UAV-derived DTM; not building-correct true ortho'
(dest/'raster_qa.json').write_text(json.dumps(qa,indent=2));print(json.dumps(qa['orthographic_mosaic']))
