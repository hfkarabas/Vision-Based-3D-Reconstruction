"""Measure saved independent COLMAP artifacts; never infer missing stage success."""
import argparse
import json
from pathlib import Path
import sys
import struct

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageOps

from research_baseline import ROOT, OUT, SCENES, save
from research_baseline_diagnostics import quantiles
from compare_airzoo_depth_backends import read_depth
from compare_airzoo_dsm import FloatStripRaster
from audit_surface_topology import audit


def point_views(points, colors, output, label):
    """Whole-cloud orthographic splats; one nearest colored point per pixel."""
    output.mkdir(exist_ok=True)
    center=(points.min(0)+points.max(0))/2
    for name,direction in [('top',(0,0,1)),('oblique',(1,-1,.8)),('side',(0,-1,0))]:
        direction=np.array(direction,dtype=float)
        direction/=np.linalg.norm(direction)
        up=np.array([0.,0.,1.]) if name!='top' else np.array([0.,1.,0.])
        right=np.cross(up,direction)
        right/=np.linalg.norm(right)
        up=np.cross(direction,right)
        view=(points-center)@np.stack([right,up,direction],axis=1)
        view[:,:2]-=(view[:,:2].min(axis=0)+view[:,:2].max(axis=0))/2
        size=640
        span=float(np.ptp(view[:,:2],axis=0).max())*1.1
        xy=np.rint((view[:,:2]/span+.5)*(size-1)).astype(int)
        xy=np.clip(xy,0,size-1)
        key=(size-1-xy[:,1])*size+xy[:,0]
        order=np.lexsort((view[:,2],key))
        ordered=key[order]
        keep=order[np.r_[ordered[1:]!=ordered[:-1],True]]
        pixels=np.full((size*size,3),35,dtype=np.uint8)
        pixels[key[keep]]=np.clip(colors[keep]*255,0,255).astype(np.uint8)
        im=Image.fromarray(pixels.reshape(size,size,3))
        draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,size,34),fill=(25,25,25))
        draw.text((8,6),f'{label}: {name}; full bounds; point splats',fill='white')
        im.save(output/f'{name}.png')


def visibility_counts(path,expected):
    data=memoryview(path.read_bytes())
    n,=struct.unpack_from('<Q',data,0)
    if n!=expected: raise ValueError('Visibility/point count mismatch')
    offset=8
    counts=np.empty(n,dtype=np.int32)
    for i in range(n):
        count,=struct.unpack_from('<I',data,offset)
        offset+=4
        counts[i]=count
        offset+=count*4
        if offset>len(data): raise ValueError('Truncated visibility data')
    if offset!=len(data): raise ValueError('Unexpected trailing visibility data')
    return counts


def review(scene):
    """Full-scene review, with deterministic original photographs and no missing tiles."""
    work=OUT/scene/'dense-900'
    photos=json.loads((OUT/'reports/dataset-audit.json').read_text())[scene]['images']
    tiles=[]
    for index in [0,len(photos)//2,len(photos)-1]:
        tiles.append((f'Original input {index+1}/{len(photos)}',ROOT/photos[index]['path']))
    tiles.extend([('Acquisition positions',OUT/'reports'/f'{scene}-acquisition.png'),
                  ('Supplied DOM/DSM reference',OUT/'reports'/f'{scene}-reference-overview.png'),
                  ('Independent sparse points + cameras',OUT/scene/'sfm-2000/model-0-sparse-cameras.png')])
    for mode in ['dense','neutral','wireframe','textured','poisson']:
        for view in ['top','oblique','side']:
            path=work/'dense-views'/f'{view}.png' if mode=='dense' else work/'asset/blender-review'/(
                f'poisson-{view}-neutral.png' if mode=='poisson' else f'{view}-{mode}.png')
            tiles.append((f'{mode}: {view}; full scene framing',path))
    for mode in ['neutral','wireframe','textured']:
        for view in ['top','oblique','side']:
            tiles.append((f'Poisson {mode}: {view}; own full bounds',
                          work/'asset-poisson/blender-review'/f'{view}-{mode}.png'))
    missing=[str(p) for _,p in tiles if not p.exists()]
    if missing: raise ValueError(f'Review incomplete; missing {missing}')
    width,height=640,520
    board=Image.new('RGB',(width*3,height*((len(tiles)+2)//3)+60),(35,38,43))
    draw=ImageDraw.Draw(board)
    draw.text((12,10),f'{scene.upper()} independent COLMAP baseline — review candidate; user acceptance PENDING',fill='white')
    for i,(label,path) in enumerate(tiles):
        with Image.open(path) as im:
            thumb=ImageOps.contain(im.convert('RGBA'),(width-12,height-40))
            bg=Image.new('RGBA',thumb.size,(35,38,43,255))
            bg.alpha_composite(thumb)
            x=(i%3)*width+(width-thumb.width)//2
            y=60+(i//3)*height+30
            board.paste(bg.convert('RGB'),(x,y))
        draw.text(((i%3)*width+8,60+(i//3)*height+7),label,fill='white')
    board.save(OUT/'reports'/f'{scene}-visual-review.jpg',quality=95)
    save(OUT/'reports'/f'{scene}-visual-review.json',dict(user_acceptance='pending',tiles=[
        dict(label=label,path=str(path)) for label,path in tiles],
        note='Source camera/sparse and point-splat images have their own full bounds; initial Blender Delaunay/Poisson views share union bounds. Additional textured Poisson views show its own full bounds, explicitly labelled. No favorable crop.'))


def evaluate(scene):
    work=OUT/scene/'dense-900'
    workspace=work/'workspace'
    alignment=json.loads((OUT/scene/'sfm-2000/metrics.json').read_text())['models'][0]['diagnostic_ppk_similarity']
    scale=alignment['meters_per_sfm_unit']
    rotation=np.array(alignment['rotation'])
    shift=np.array(alignment['translation'])
    def metric(xyz): return scale*xyz@rotation.T+shift
    report=dict(scene=scene, alignment=alignment, depth_maps=[], meshes={}, stages={})
    for path in work.glob('*.json'):
        if path.name not in ['metrics.json','inputs.json']:
            row=json.loads(path.read_text())
            if 'command' in row: report['stages'][path.stem]=row
    for path in sorted((workspace/'stereo/depth_maps').glob('*.geometric.bin')):
        values=read_depth(path)
        valid=np.isfinite(values)&(values>0)
        report['depth_maps'].append(dict(name=path.name, size=values.shape,
            valid_pixels=int(valid.sum()),valid_fraction=float(valid.mean())))
    if report['depth_maps']:
        report['valid_depth_fraction']=quantiles([r['valid_fraction'] for r in report['depth_maps']])
    fused=workspace/'fused.ply'
    if fused.exists():
        cloud=o3d.io.read_point_cloud(str(fused))
        xyz=np.asarray(cloud.points)
        points=metric(xyz)
        point_views(points,np.asarray(cloud.colors),work/'dense-views',f'{scene} COLMAP dense')
        cells=np.floor(points[:,:2]/.5).astype(np.int64)
        keys,first,inverse,counts=np.unique(cells,axis=0,return_index=True,return_inverse=True,return_counts=True)
        report['dense']=dict(points=len(points),finite=bool(np.isfinite(points).all()),
            bounds_sfm=[xyz.min(axis=0).tolist(),xyz.max(axis=0).tolist()],
            approximate_metric_bounds=[points.min(axis=0).tolist(),points.max(axis=0).tolist()],
            half_meter_occupied_xy_cells=len(keys),occupied_xy_area_m2=len(keys)*.25,
            points_per_occupied_half_meter_cell=quantiles(counts),
            scope='PPK-fitted scale; occupied cells are not independently verified valid completeness')
        vis=fused.with_suffix('.ply.vis')
        if vis.exists():
            counts_vis=visibility_counts(vis,len(points))
            report['dense']['distinct_supporting_images']=quantiles(counts_vis)
            report['dense']['view_support_caveat']='Distinct images, not independent physical camera stations or independent accuracy evidence'
        raster=FloatStripRaster(ROOT/'datasets/airzoo'/scene/SCENES[scene]/'dom_dsm/DSM.tif')
        projected=points[:,:2]+alignment['projected_origin_easting_northing']
        reference=raster.sample(projected)
        mask_meta=json.loads((OUT/'reports'/f'{scene}-dom-footprint-mask.json').read_text())
        mask=np.array(Image.open(OUT/'reports'/f'{scene}-dom-footprint-mask.png'))>0
        pixel=(projected-raster.origin)@np.linalg.inv(raster.affine).T
        pix=np.floor(pixel/[raster.width,raster.height]*[mask.shape[1],mask.shape[0]]).astype(np.int64)
        in_mask=(pix>=0).all(axis=1)&(pix<[mask.shape[1],mask.shape[0]]).all(axis=1)
        reference_valid=np.zeros(len(points),dtype=bool)
        reference_valid[in_mask]=mask[pix[in_mask,1],pix[in_mask,0]]
        good=np.isfinite(reference)&reference_valid
        residual=points[good,2]-reference[good]
        offset=float(np.median(residual)) if len(residual) else None
        report['dsm_reference_agreement']=dict(reference_samples=int(good.sum()),total_points=len(points),
            reference_overlap_fraction=float(good.mean()), fitted_vertical_offset_m=offset,
            footprint_mask=mask_meta,
            raw_signed_residual_m=quantiles(residual),
            offset_adjusted_absolute_residual_m=quantiles(abs(residual-offset)) if offset is not None else None,
            offset_adjusted_rmse_m=float(np.sqrt(np.mean((residual-offset)**2))) if offset is not None else None,
            warning='Height-map reference agreement only, in-sample PPK similarity and fitted DSM vertical offset. Not absolute geospatial accuracy, not independent 3D ground truth; CRS conflict and DSM lineage unresolved.')
        # A DSM is an upper surface: facade samples legitimately lie below rooftops.
        # Report a fixed projected 1 m upper envelope as a separate diagnostic,
        # retaining the original all-point result and exposing outlier sensitivity.
        grid=np.floor(projected[good]).astype(np.int64)
        unique,inverse,count=np.unique(grid,axis=0,return_inverse=True,return_counts=True)
        highest=np.full(len(unique),-np.inf)
        np.maximum.at(highest,inverse,points[good,2])
        reference_grid=raster.sample(unique.astype(float)+.5)
        envelopes={}
        for minimum in [1,3]:
            keep=np.isfinite(reference_grid)&(count>=minimum)
            error=highest[keep]-reference_grid[keep]
            shift=float(np.median(error)) if len(error) else None
            envelopes[f'minimum_{minimum}_points']=dict(cells=int(keep.sum()),
                fitted_vertical_offset_m=shift,raw_signed_residual_m=quantiles(error),
                offset_adjusted_absolute_residual_m=quantiles(abs(error-shift)) if shift is not None else None)
        report['dsm_upper_envelope_diagnostic']=dict(projected_grid_size_m=1,
            grid_origin='EPSG4547 easting/northing zero',results=envelopes,
            caveat='Highest fused point per occupied cell, not interpolated or hole-filled. Same conservative input mask. Sensitive to high outliers and sampling density; minimum-3 cells exclude sparse areas. Each variant fits its own offset. Not full 3D accuracy or comparable to all-point native DSM scores.')
    for name in ['delaunay','poisson']:
        path=work/f'{name}.ply'
        if path.exists():
            data=audit(path)
            stats=data['summary']
            normals_stored=o3d.io.read_triangle_mesh(str(path)).has_vertex_normals()
            stats['vertex_normals_stored']=normals_stored
            if not normals_stored:
                stats['invalid_vertex_normals']=None
                data['normal_caveat']='PLY omits vertex normals; absent normals are not counted as invalid computed normals.'
            stats['largest_component_triangle_share']=stats['largest_component_triangles']/stats['triangles']
            stats['ppk_scaled_surface_area_m2']=stats['area']*scale**2
            stats['supported_surface_area_m2']=None
            data['support_caveat']='Total area is not certified supported area; visibility/independent depth support must be measured separately.'
            report['meshes'][name]=data
    save(work/'metrics.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True,choices=SCENES)
    p.add_argument('--review',action='store_true')
    p.add_argument('--output-root',type=Path,default=OUT)
    args=p.parse_args()
    OUT=args.output_root.resolve()
    (review if args.review else evaluate)(args.scene)
