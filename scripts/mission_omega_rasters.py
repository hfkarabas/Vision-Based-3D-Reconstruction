"""PDAL raster configuration and GDAL/rasterio quality measurement only."""
from pathlib import Path
import json, math, os, subprocess, sys
import numpy as np
from pyproj import Transformer
import rasterio
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/mission-omega'
scene=sys.argv[1]; raw=OUT/'raw'/scene
if len(sys.argv)>2: raw=raw/sys.argv[2]
meta=json.loads((ROOT/'outputs/research-baseline/reports/dataset-audit.json').read_text())[scene]['images']
east,north=Transformer.from_crs(4326,4547,always_xy=True).transform(
    [float(r['xmp']['GpsLongitude']) for r in meta],[float(r['xmp']['GpsLatitude']) for r in meta])
buffer=float(sys.argv[3]) if len(sys.argv)>3 else 50.
xmin=math.floor(min(east)-buffer);xmax=math.ceil(max(east)+buffer)
ymin=math.floor(min(north)-buffer);ymax=math.ceil(max(north)+buffer)
pdal=OUT/'tools/gis/Library/bin/pdal.exe'
report=dict(roi_bounds=[xmin,ymin,xmax,ymax],roi_selection=f'camera position envelope plus fixed {buffer} m; full dense-cloud bounds separately retained',
    crs='EPSG:4547',pixel_size_m=1.,raster_footprint_m2=(xmax-xmin)*(ymax-ymin),products={})
for name,dim,stat,ground in [('dsm','Z','max',False),('dtm','Z','idw',True),('red','Red','mean',False),('green','Green','mean',False),('blue','Blue','mean',False)]:
    job=[dict(type='readers.las',filename=str(raw/'classified.laz')),
         dict(type='filters.crop',bounds=f'([{xmin},{xmax}],[{ymin},{ymax}])')]
    if ground:
        job.append(dict(type='filters.expression',expression='Classification == 2'))
        job.append(dict(type='filters.mad',dimension='Z',k=10.))
    job.append(dict(type='writers.gdal',filename=str(raw/f'{name}.tif'),resolution=1.,radius=1.5 if ground else .71,
        dimension=dim,output_type=stat,window_size=0,nodata=-9999,data_type='float',
        origin_x=xmin,origin_y=ymin,width=xmax-xmin,height=ymax-ymin,gdalopts='COMPRESS=DEFLATE'))
    path=raw/f'{name}_pipeline.json';path.write_text(json.dumps(job,indent=2))
    subprocess.run([str(pdal),'pipeline',str(path)],check=True)
    with rasterio.open(raw/f'{name}.tif') as ds:
        z=ds.read(1,masked=True);valid=z.compressed()
        report['products'][name]=dict(bounds=list(ds.bounds),crs=str(ds.crs),shape=z.shape,
            valid_cells=int(z.count()),valid_fraction=float(z.count()/z.size),
            valid_area_m2=int(z.count()),range=[float(valid.min()),float(valid.max())] if valid.size else None,
            percentiles=np.percentile(valid,[1,5,50,95,99]).tolist() if valid.size else [])
# RGB point projection is explicitly NOT yet an orthophoto: preserve honest naming.
with rasterio.open(raw/'red.tif') as ds:
    profile=ds.profile; bands=[]
    for name in ['red','green','blue']:
        with rasterio.open(raw/f'{name}.tif') as src:bands.append(src.read(1))
    bands=np.array(bands);mask=np.all(bands!=-9999,axis=0)
    profile.update(count=4,dtype='uint8',nodata=None)
    with rasterio.open(raw/'point_color_projection.tif','w',**profile) as dst:
        dst.write(np.vstack([np.clip(bands,0,255).astype('uint8'),(mask*255).astype('uint8')[None]]))
report['texture_status']='Point-color orthographic raster only; photogrammetric orthophoto NOT VERIFIED'
(raw/'raster_qa.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
