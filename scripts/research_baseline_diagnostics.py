"""Measure COLMAP models and plot input acquisition without using native poses."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from scipy.sparse.csgraph import connected_components

from research_baseline import ROOT, OUT, SCENES, save
sys.path.insert(0, str(ROOT / 'outputs/airzoo-benchmark/geometric-root-cause-recovery/colmap-python'))
sys.path.insert(0, str(ROOT / 'outputs/airzoo-benchmark/geometric-root-cause-recovery/reference-python'))
import pycolmap


def quantiles(values):
    return dict(zip(['p0', 'p5', 'p50', 'p95', 'p100'], np.percentile(values, [0, 5, 50, 95, 100]).tolist())) if len(values) else None


def acquisition():
    # Only read headers of the known local large GeoTIFFs; never decode their pixels.
    Image.MAX_IMAGE_PIXELS = None
    data = json.loads((OUT / 'reports/dataset-audit.json').read_text())
    report = {}
    for scene, dataset in data.items():
        rows = dataset['images']
        llh = np.array([[float(r['xmp'][k]) for k in ['GpsLongitude', 'GpsLatitude', 'AbsoluteAltitude']] for r in rows])
        xy = (llh[:, :2] - llh[0, :2]) * [111195 * np.cos(np.radians(llh[0, 1])), 111195]
        distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
        count, labels = connected_components(distances < 15, directed=False)
        yaw = np.radians([float(r['xmp']['GimbalYawDegree']) for r in rows])
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(*xy.T, c=np.arange(len(rows)), cmap='viridis', s=18)
        ax.quiver(*xy.T, np.sin(yaw), np.cos(yaw), angles='xy', scale_units='xy', scale=.04, alpha=.35)
        for group in range(count):
            center = xy[labels == group].mean(axis=0)
            ax.text(*center, f'C{group+1}: {(labels==group).sum()}', fontsize=8)
        ax.set(aspect='equal', xlabel='Approximate east (m)', ylabel='Approximate north (m)',
               title=f'{scene}: acquisition positions and gimbal yaw (diagnostic only)')
        fig.colorbar(sc, ax=ax, label='Sorted input index; no adjacency assumption')
        fig.savefig(OUT / 'reports' / f'{scene}-acquisition.png', dpi=140)
        plt.close(fig)
        refs = {}
        scene_base=ROOT / 'datasets/airzoo' / scene / SCENES[scene]
        for path in [*(scene_base/'dom_dsm').glob('*.tif'),*(scene_base/'satemap').glob('*.tif')]:
            with Image.open(path) as im:
                refs[str(path.relative_to(scene_base))] = dict(size=im.size, mode=im.mode,
                    geotiff_tags={str(k): str(im.tag_v2.get(k)) for k in [33550,33922,34264,34735,34736,34737,42113]})
        report[scene] = dict(approximate_position_clusters_15m=Counter(labels.tolist()),
            cluster_count=count, horizontal_extent_m=np.ptp(xy, axis=0).tolist(),
            altitude_ellipsoid_m=quantiles(llh[:,2]),
            gimbal_pitch_degrees=quantiles([float(r['xmp']['GimbalPitchDegree']) for r in rows]),
            references=refs, caveat='15m clustering is a diagnostic, not proof of view overlap or connected SfM')
    save(OUT / 'reports/acquisition.json', report)


def references():
    from pyproj import Transformer
    from compare_airzoo_dsm import FloatStripRaster
    Image.MAX_IMAGE_PIXELS=None  # Trusted local TIFFs; process one raster at a time.
    metadata=json.loads((OUT/'reports/dataset-audit.json').read_text())
    transform=Transformer.from_crs('EPSG:4326','EPSG:4547',always_xy=True)
    for scene,folder in SCENES.items():
        base=ROOT/'datasets/airzoo'/scene/folder/'dom_dsm'
        with Image.open(base/'DOM.tif') as image:
            width,height=image.size
            # TIFF extra-sample/alpha semantics are not trusted for this reference.
            # Preserve RGB before thumbnailing (premultiplied alpha can erase RGB).
            rgb=image.convert('RGB')
            mask_size=(1200,max(1,round(height*1200/width)))
            sampled=np.asarray(rgb.resize(mask_size,Image.Resampling.NEAREST))
            mask=Image.fromarray((np.any(sampled>0,axis=2)*255).astype(np.uint8))
            mask=mask.filter(ImageFilter.MinFilter(3))
            mask.save(OUT/'reports'/f'{scene}-dom-footprint-mask.png')
            save(OUT/'reports'/f'{scene}-dom-footprint-mask.json',dict(source=str(base/'DOM.tif'),
                source_size=[width,height],mask_size=mask_size,world_file=np.loadtxt(base/'DOM.tfw').tolist(),
                definition='Nearest RGB samples, any nonblack RGB, eroded by one mask pixel; alpha ignored',
                caveat='Approximate conservative reference footprint, not ground-truth geometry or sensor coverage'))
            rgb.thumbnail((1200,1200))
            thumbnail=np.array(rgb)
        a,d,b,e,c,f=np.loadtxt(base/'DOM.tfw')
        if b or d: raise ValueError('Overview needs explicit rotated-raster handling')
        extent=[c-a/2,c+a*(width-.5),f-e/2,f+e*(height-.5)]
        rows=metadata[scene]['images']
        east,north=transform.transform([float(r['xmp']['GpsLongitude']) for r in rows],
                                       [float(r['xmp']['GpsLatitude']) for r in rows])
        raster=FloatStripRaster(base/'DSM.tif')
        xs=np.linspace(extent[0],extent[1],700)
        ys=np.linspace(extent[2],extent[3],700)
        xx,yy=np.meshgrid(xs,ys)
        heights=raster.sample(np.column_stack([xx.ravel(),yy.ravel()])).reshape(xx.shape)
        fig,axes=plt.subplots(1,2,figsize=(16,8),layout='constrained')
        axes[0].imshow(thumbnail,origin='lower',extent=extent)
        axes[0].scatter(east,north,c='red',s=5)
        axes[0].set_title('Supplied DOM RGB + camera positions\n(sidecar CRS; alpha not interpreted)')
        low,high=np.nanpercentile(heights,[2,98])
        plot=axes[1].imshow(heights,origin='lower',extent=extent,cmap='terrain',vmin=low,vmax=high)
        fig.colorbar(plot,ax=axes[1],label='Supplied DSM height; vertical datum unresolved')
        axes[1].set_title('Supplied DSM, display range P2–P98\n(no reconstruction shown)')
        for ax in axes: ax.set(xlabel='Easting (m)',ylabel='Northing (m)',aspect='equal')
        fig.suptitle(f'{scene}: reference products, not independently certified ground truth')
        fig.savefig(OUT/'reports'/f'{scene}-reference-overview.png',dpi=130)
        plt.close(fig)


def sfm(scene):
    work = OUT / scene / 'sfm-2000'
    input_count = json.loads((OUT / 'reports/dataset-audit.json').read_text())[scene]['count']
    with sqlite3.connect(work / 'database.db') as db:
        ids = [r[0] for r in db.execute('select image_id from images')]
        index = {k:i for i,k in enumerate(ids)}
        graph = np.zeros((len(ids),len(ids)), bool)
        edges = []
        for pair, n in db.execute('select pair_id, rows from two_view_geometries where rows > 0'):
            a, b = divmod(pair, 2147483647)
            graph[index[a],index[b]]=graph[index[b],index[a]]=True
            edges.append([a,b,n])
        _, labels = connected_components(graph,directed=False)
    report = dict(input_count=input_count, database_images=len(ids), verified_edges=len(edges),
                  verified_graph_component_sizes=sorted(Counter(labels.tolist()).values(), reverse=True), models=[])
    for folder in sorted((work / 'models').iterdir()):
        if not (folder / 'cameras.bin').exists():
            continue
        rec = pycolmap.Reconstruction(folder)
        images = dict(rec.images)
        centers = {k:v.projection_center() for k,v in images.items()}
        points = list(rec.points3D.values())
        angles=[]
        for point in points[::max(1,len(points)//10000)]:
            rays=np.array([centers[e.image_id]-point.xyz for e in point.track.elements if e.image_id in centers])
            if len(rays)<2:
                continue
            rays/=np.linalg.norm(rays,axis=1)[:,None]
            angles.append(float(np.degrees(np.arccos(np.clip(np.min(rays@rays.T),-1,1)))))
        registered=list(images)
        metadata=json.loads((OUT/'reports/dataset-audit.json').read_text())[scene]['images']
        lookup={Path(r['path']).name:r['xmp'] for r in metadata}
        from pyproj import Transformer
        gps=np.array([[float(lookup[im.name][k]) for k in ['GpsLongitude','GpsLatitude','AbsoluteAltitude']] for im in images.values()])
        easting,northing=Transformer.from_crs('EPSG:4326','EPSG:4547',always_xy=True).transform(gps[:,0],gps[:,1])
        projected_origin=np.array([easting[0],northing[0]])
        gps[:,:2]=np.column_stack([easting,northing])-projected_origin
        source=np.array(list(centers.values()))
        aa=source-source.mean(axis=0)
        bb=gps-gps.mean(axis=0)
        u,s,vh=np.linalg.svd(bb.T@aa)
        signs=np.array([1.,1.,np.sign(np.linalg.det(u@vh))])
        rotation=(u*signs)@vh
        scale=np.dot(s,signs)/np.square(aa).sum()
        fitted=scale*aa@rotation.T+gps.mean(axis=0)
        position_residual=np.linalg.norm(fitted-gps,axis=1)
        sub=graph[np.ix_([index[i] for i in registered],[index[i] for i in registered])]
        _, lab=connected_components(sub,directed=False)
        report['models'].append(dict(id=folder.name, registered=len(images), ratio=len(images)/input_count,
            points=len(points), mean_reprojection_error_px=rec.compute_mean_reprojection_error(),
            point_mean_error_px=quantiles([p.error for p in points]),
            track_length=quantiles([p.track.length() for p in points]),
            max_triangulation_angle_deg_sampled=quantiles(angles),
            registered_graph_components=sorted(Counter(lab.tolist()).values(),reverse=True),
            diagnostic_ppk_similarity=dict(meters_per_sfm_unit=scale,
                rotation=rotation.tolist(),
                translation=(gps.mean(axis=0)-scale*rotation@source.mean(axis=0)).tolist(),
                projected_origin_easting_northing=projected_origin.tolist(),
                projected_crs='EPSG:4547',
                local_gps_origin_lon_lat=[float(lookup[next(iter(images.values())).name]['GpsLongitude']),
                                          float(lookup[next(iter(images.values())).name]['GpsLatitude'])],
                camera_position_residual_m=quantiles(position_residual),
                caveat='In-sample Sim3 camera fit to XMP RTK positions projected to EPSG:4547; horizontal datum realization and vertical reference unverified; not independent ground surface accuracy',
                positions_used_for_estimation=False),
            cameras=[dict(id=c.camera_id,model=c.model_name, width=c.width,height=c.height,params=c.params.tolist()) for c in rec.cameras.values()],
            image_names=[im.name for im in images.values()], summary=rec.summary()))
        xyz=np.array([p.xyz for p in points])
        rgb=np.array([p.color for p in points])/255
        cam=np.array(list(centers.values()))
        fig=plt.figure(figsize=(15,5))
        # PCA is an explicitly arbitrary display frame, not geospatial alignment.
        origin=np.median(xyz,axis=0)
        _,_,basis=np.linalg.svd(xyz-origin,full_matrices=False)
        cloud=(xyz-origin)@basis.T
        cameras=(cam-origin)@basis.T
        for i,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
            ax=fig.add_subplot(1,3,i+1)
            ax.scatter(cloud[:,a],cloud[:,b],c=rgb,s=.2,rasterized=True)
            ax.scatter(cameras[:,a],cameras[:,b],c='red',marker='^',s=10)
            ax.set(aspect='equal', title=f'PCA axes {a}/{b}; red cameras')
        fig.suptitle(f'{scene} independent COLMAP model {folder.name}: {len(images)}/{input_count} cameras')
        fig.savefig(work/f'model-{folder.name}-sparse-cameras.png',dpi=160)
        plt.close(fig)
    save(work/'metrics.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['acquisition','references','sfm'])
    p.add_argument('--scene',choices=SCENES)
    p.add_argument('--output-root',type=Path,default=OUT)
    a=p.parse_args()
    OUT=a.output_root.resolve()
    if a.action=='acquisition': acquisition()
    elif a.action=='references': references()
    elif a.scene: sfm(a.scene)
    else: p.error('--scene is required')
