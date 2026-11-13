"""Rasterio/GDAL merge and standard raster quality measurements for real trials."""
from pathlib import Path
import csv,json,sys
import numpy as np
import rasterio
from rasterio.merge import merge
from scipy.ndimage import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mission-omega'
scene,mode=sys.argv[1:3];work=OUT/'research'/scene;work.mkdir(parents=True,exist_ok=True)
baseline=OUT/'raw'/scene/'v3-mad/dtm.tif'
def state(path):
    with rasterio.open(path) as ds:
        z=ds.read(1);valid=z!=ds.nodata;tiles=[]
        rowedges=np.linspace(0,z.shape[0],5,dtype=int);coledges=np.linspace(0,z.shape[1],5,dtype=int)
        for r in range(4):
            for c in range(4):
                y0,y1=rowedges[r:r+2];x0,x1=coledges[c:c+2];a=z[y0:y1,x0:x1];v=valid[y0:y1,x0:x1]
                delta=np.abs(np.diff(a,axis=0));good=v[1:]&v[:-1];p95=float(np.percentile(delta[good],95)) if good.any() else 2.
                left,bottom=ds.transform*(int(x0),int(y1));right,top=ds.transform*(int(x1),int(y0))
                tiles.append(dict(id=f'r{r}c{c}',row=r,col=c,bounds=[left,bottom,right,top],
                    valid_fraction=float(v.mean()),nodata_fraction=float(1-v.mean()),edge_p95_m=p95,
                    score=.7*(1-float(v.mean()))+.3*min(p95/2.,1.)))
        edge=np.abs(np.diff(z,axis=0));good=valid[1:]&valid[:-1]
        metrics=dict(valid_cells=int(valid.sum()),total_cells=z.size,valid_fraction=float(valid.mean()),
            valid_area_m2=int(valid.sum()),components=int(label(valid)[1]),
            edge_p95_m=float(np.percentile(edge[good],95)),edge_max_m=float(edge[good].max()),
            jumps_over_3m=int((edge[good]>3).sum()),z_range=[float(z[valid].min()),float(z[valid].max())])
        return tiles,metrics,z,ds.profile
if mode=='prepare':
    tiles,metrics,z,profile=state(baseline)
    (work/'tile_state.json').write_text(json.dumps(tiles,indent=2));(work/'baseline_metrics.json').write_text(json.dumps(metrics,indent=2))
elif mode=='merge':
    arm=sys.argv[3];action=Path(sys.argv[4]);dest=work/arm;dest.mkdir(exist_ok=True)
    current=dest/'merged_dtm.tif';src=current if current.exists() else baseline
    with rasterio.open(src) as a,rasterio.open(action/'dtm.tif') as b:
        values,transform=merge([a,b],bounds=a.bounds,res=a.res,nodata=-9999,method='first')
        profile=a.profile;profile.update(transform=transform,width=values.shape[2],height=values.shape[1])
    tmp=dest/'merge_next.tif'
    with rasterio.open(tmp,'w',**profile) as ds:ds.write(values)
    # Preserve each executed checkpoint; final path points to its actual data copy.
    import shutil
    shutil.copy2(tmp,action/'merged_checkpoint.tif');shutil.copy2(tmp,current)
    tiles,metrics,_,_=state(current)
    (dest/'feedback_tile_state.json').write_text(json.dumps(tiles,indent=2))
    (action/'merge_metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics))
elif mode=='summary':
    rows=[];base=state(baseline)[1];baseline_z=state(baseline)[2]
    rows.append(dict(arm='frozen_900px_start_state',**base,incremental_seconds=0.,peak_rss_bytes=0,reference_depth_maps=0))
    arms=['B0_uniform','B1_random_seed42','B2_feedback']
    if (work/'B2_feedback_revisit/merged_dtm.tif').exists():arms.append('B2_feedback_revisit')
    if (work/'B3_support_proxy/merged_dtm.tif').exists():arms.append('B3_support_proxy')
    fig,axs=plt.subplots(2,len(arms),figsize=(5*len(arms),9),constrained_layout=True)
    for i,arm in enumerate(arms):
        tiles,metrics,z,_=state(work/arm/'merged_dtm.tif')
        stages=[json.loads(p.read_text()) for p in (OUT/'reports/stages').glob(f'{scene}-{arm}-*.json')]
        maps=list((work/arm).glob('action-*/stereo/depth_maps/*.geometric.bin'))
        pixels=0
        for action in (work/arm).glob('action-*'):
            decision=json.loads((action/'decision.json').read_text())
            for name in decision['reference_images']:
                if (action/'stereo/depth_maps'/f'{name}.geometric.bin').exists():
                    with Image.open(action/'images'/name) as im:pixels+=im.width*im.height
        row=dict(arm=arm,**metrics,incremental_seconds=sum(s['seconds'] for s in stages),
            peak_rss_bytes=max(s['peak_process_tree_rss_bytes'] for s in stages),reference_depth_maps=len(maps),
            allocated_reference_pixels=pixels,allocated_pixel_iterations_two_passes=pixels*3*2,
            additional_cells=metrics['valid_cells']-base['valid_cells'])
        rows.append(row)
        added=(baseline_z==-9999)&(z!=-9999)
        with rasterio.open(baseline) as ds:bounds=ds.bounds
        extent=[bounds.left,bounds.right,bounds.bottom,bounds.top]
        axs[0,i].imshow(np.where(z!=-9999,z,np.nan),vmin=10,vmax=25,extent=extent);axs[0,i].set_title(f'{arm}: {metrics["valid_fraction"]:.4%}')
        for action in (work/arm).glob('action-*'):
            decision=json.loads((action/'decision.json').read_text());x0,y0,x1,y1=decision['tile']['bounds']
            axs[0,i].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='red',linewidth=2))
            axs[0,i].text(x0+4,y0+4,str(decision['index']),color='red')
        axs[1,i].imshow(added,cmap='gray',extent=extent);axs[1,i].set_title(f'Newly supported cells: {int(added.sum())}')
        for axis in axs[:,i]:axis.set_xlabel('Easting (m), EPSG:4547');axis.set_ylabel('Northing (m)')
        # Height difference is zero on common support by declared hole-only merge.
        diff=np.where((baseline_z!=-9999)&(z!=-9999),z-baseline_z,-9999).astype('float32')
        with rasterio.open(baseline) as ds:profile=ds.profile
        with rasterio.open(work/arm/'before_after_difference.tif','w',**profile) as ds:ds.write(diff,1)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (work/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    (work/'metrics.json').write_text(json.dumps(rows,indent=2));fig.suptitle(f'{scene}; ROI {list(bounds)} EPSG:4547; fixed-work budget; no independent accuracy claim')
    fig.savefig(work/'comparisons.png',dpi=150)
    print(json.dumps(rows))
