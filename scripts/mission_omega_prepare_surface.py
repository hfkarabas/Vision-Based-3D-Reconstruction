"""Declared quality masks + existing GDAL bounded gap filling for production.

Strict science rasters are untouched. No external elevation is consumed.
"""
from pathlib import Path
import json, shutil, sys
import numpy as np
import rasterio
from rasterio.fill import fillnodata
from scipy.ndimage import distance_transform_edt, median_filter, label
from scipy.stats import median_abs_deviation
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mission-omega'
scene=sys.argv[1];source_variant=sys.argv[2] if len(sys.argv)>2 else 'v3-mad'
variant=sys.argv[3] if len(sys.argv)>3 else 'production8';gap=int(sys.argv[4]) if len(sys.argv)>4 else 8
lowtail=len(sys.argv)>5 and sys.argv[5]=='lowtail'
src=OUT/'raw'/scene/source_variant;dst=OUT/'raw'/scene/variant;dst.mkdir(parents=True,exist_ok=True)
with rasterio.open(src/'dtm.tif') as ds:
    z=ds.read(1);profile=ds.profile
    valid=z!=-9999
    _,idx=distance_transform_edt(~valid,return_indices=True)
    # Nearest values are used ONLY for calculating a local quality residual.
    local_median=median_filter(z[tuple(idx)],size=11)
    suspicious=valid & (np.abs(z-local_median)>3.)
    threshold=None
    if lowtail:
        threshold=float(np.median(z[valid])-3*median_abs_deviation(z[valid],scale='normal'))
        suspicious|=valid & (z<threshold)
    clean=valid & ~suspicious
    zmasked=np.where(clean,z,0).astype('float32')
    filled=fillnodata(zmasked,mask=clean.astype('uint8'),max_search_distance=gap,smoothing_iterations=0)
    newvalid=filled!=0
    production=np.where(newvalid,filled,-9999).astype('float32')
    with rasterio.open(dst/'dtm.tif','w',**profile) as out:out.write(production,1)
    for name,mask in [('strict_support',valid),('quality_rejected',suspicious),('interpolated',newvalid & ~clean),('production_support',newvalid)]:
        mp=profile.copy();mp.update(dtype='uint8',nodata=0)
        with rasterio.open(dst/f'{name}.tif','w',**mp) as out:out.write(mask.astype('uint8'),1)
    delta=np.diff(production,axis=0);both=newvalid[1:]&newvalid[:-1]
    qa=json.loads((src/'raster_qa.json').read_text())
    stats=dict(method=f'COLMAP cached 900px dense + PDAL {source_variant} + quality abstention + GDAL FillNodata',
        quality_rule='Reject |DTM - scipy.ndimage median_filter(size=11) local median| > 3m; nearest completion used ONLY for QA metric',
        low_tail_quality_threshold_m=threshold,
        low_tail_rule='Quality abstention below median - 3 * scipy.stats.median_abs_deviation(scale=normal); no upper-tail censoring' if lowtail else None,
        gapfill=f'GDAL FillNodata max_search_distance={gap} pixels (={gap}m); smoothing_iterations=0',
        original_valid_cells=int(valid.sum()),quality_rejected_cells=int(suspicious.sum()),
        interpolated_cells=int((newvalid&~clean).sum()),production_valid_cells=int(newvalid.sum()),
        missing_cells=int((~newvalid).sum()),production_valid_fraction=float(newvalid.mean()),
        production_vertical_range_m=[float(production[newvalid].min()),float(production[newvalid].max())],
        vertical_edge_abs_percentiles_m=np.percentile(np.abs(delta[both]),[50,95,99,100]).tolist(),
        components=int(label(newvalid)[1]),area_m2=int(newvalid.sum()),reference_geometry_used=False)
    qa['production']=stats;qa['products']['dtm'].update(valid_cells=int(newvalid.sum()),valid_fraction=float(newvalid.mean()),
        valid_area_m2=int(newvalid.sum()),range=stats['production_vertical_range_m'])
    (dst/'raster_qa.json').write_text(json.dumps(qa,indent=2))
    fig,ax=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    ax[0].imshow(np.where(valid,z,np.nan),vmin=np.percentile(z[valid],1),vmax=np.percentile(z[valid],99));ax[0].set_title('Strict UAV-only DTM; NoData blank')
    ax[1].imshow(np.where(newvalid,production,np.nan),vmin=np.percentile(z[valid],1),vmax=np.percentile(z[valid],99));ax[1].set_title(f'Production DTM; bounded fill {gap}m')
    ax[2].imshow(valid.astype(int)+2*(newvalid&~clean)+4*suspicious);ax[2].set_title('Support / interpolation / rejection masks')
    fig.suptitle(f'{scene}: full ROI EPSG:4547 {qa["roi_bounds"]}; Z scale 1:1')
    (OUT/'preview').mkdir(exist_ok=True);fig.savefig(OUT/'preview'/f'{scene}_{variant}_terrain_quality_map.png',dpi=150);plt.close(fig)
for name in ['point_color_projection.tif','geospatial_transform.json','source_manifest.json']:
    shutil.copy2(src/name,dst/name)
print(json.dumps(stats))
