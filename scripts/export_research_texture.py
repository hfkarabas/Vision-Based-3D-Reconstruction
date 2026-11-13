"""Convert COLMAP's textured triangle PLY to OBJ without changing geometry or UVs.

This is format conversion only. COLMAP performs reconstruction and texturing.
The strict reader targets the documented writer layout and rejects other layouts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from research_baseline import OUT, save


def read_textured_ply(path):
    with path.open('rb') as stream:
        lines=[]
        while True:
            line=stream.readline().decode('ascii').strip()
            if not line: raise ValueError('Incomplete PLY header')
            lines.append(line)
            if line=='end_header': break
        expected=['property float x','property float y','property float z',
                  'property list uchar int vertex_indices','property list uchar float texcoord']
        if [line for line in lines if line.startswith('property ')]!=expected:
            raise ValueError('Unexpected COLMAP textured PLY properties; inspect before adapting')
        n=int(next(l for l in lines if l.startswith('element vertex ')).split()[-1])
        m=int(next(l for l in lines if l.startswith('element face ')).split()[-1])
        texture=next(l for l in lines if l.startswith('comment TextureFile ')).split(' ',2)[2]
        if 'format binary_little_endian 1.0' not in lines:
            raise ValueError('Expected binary little-endian COLMAP output')
        xyz=np.fromfile(stream,dtype='<f4',count=3*n).reshape(n,3)
        dtype=np.dtype([('count','u1'),('vertices','<i4',(3,)),('uvcount','u1'),('uv','<f4',(6,))])
        records=np.fromfile(stream,dtype=dtype,count=m)
        if len(records)!=m or stream.read(1) or not np.all(records['count']==3) or not np.all(records['uvcount']==6):
            raise ValueError('Unexpected PLY face records')
        faces=records['vertices'].copy()
        uv=records['uv'].reshape(-1,3,2).copy()
        if not np.isfinite(xyz).all() or not np.isfinite(uv).all() or faces.min()<0 or faces.max()>=n:
            raise ValueError('Invalid geometry or UVs')
        return xyz,faces,uv,path.parent/texture


def convert(scene, method='delaunay'):
    started=time.perf_counter()
    work=OUT/scene/'dense-900'
    source=work/f'textured-{method}/mesh.ply'
    xyz,faces,uv,texture=read_textured_ply(source)
    source_vertices=len(xyz)
    used=np.unique(faces.ravel())
    remap=np.full(len(xyz),-1,dtype=np.int32)
    remap[used]=np.arange(len(used))
    xyz=xyz[used]
    faces=remap[faces]
    alignment=json.loads((OUT/scene/'sfm-2000/metrics.json').read_text())['models'][0]['diagnostic_ppk_similarity']
    xyz=alignment['meters_per_sfm_unit']*xyz.astype(float)@np.array(alignment['rotation']).T+alignment['translation']
    # Keep metric local east/north and shift elevation near the model for asset precision.
    origin=np.median(xyz,axis=0)
    xyz-=origin
    output=work/('asset' if method=='delaunay' else f'asset-{method}')
    output.mkdir(exist_ok=False)
    shutil.copy2(texture,output/'texture.png')
    (output/'textured.mtl').write_text('newmtl photographic\nKd 1 1 1\nKa 0 0 0\nmap_Kd texture.png\n',encoding='ascii')
    assigned_faces=0
    assigned_area=0.
    total_area=0.
    with (output/'textured.obj').open('w',encoding='ascii') as stream:
        stream.write('mtllib textured.mtl\no COLMAP_independent\n')
        np.savetxt(stream,xyz,fmt='v %.9g %.9g %.9g')
        np.savetxt(stream,uv.reshape(-1,2),fmt='vt %.9g %.9g')
        stream.write('usemtl photographic\n')
        for start in range(0,len(faces),10000):
            f=faces[start:start+10000]+1
            triangles=xyz[f-1]
            area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
            assigned=np.any(uv[start:start+len(f)]!=0,axis=(1,2))
            assigned_faces+=int(assigned.sum())
            assigned_area+=float(area[assigned].sum())
            total_area+=float(area.sum())
            t=np.arange(start*3+1,(start+len(f))*3+1).reshape(-1,3)
            records=np.stack([f,t],axis=2).reshape(-1,6)
            np.savetxt(stream,records,fmt='f %d/%d %d/%d %d/%d')
    save(output/'conversion.json',dict(software='COLMAP reconstruction and texture; project format adapter',
        method=method,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        vertices=len(xyz),triangles=len(faces),source_uv_bounds=[uv.min(axis=(0,1)).tolist(),uv.max(axis=(0,1)).tolist()],
        source_vertices=source_vertices,unreferenced_vertices_omitted=source_vertices-len(xyz),
        bounds_local_fitted_m=[xyz.min(axis=0).tolist(),xyz.max(axis=0).tolist()],
        obj_axes='X=east, Y=north, Z=up; Blender OBJ importer must use forward Y, up Z',
        uv_assigned_faces=assigned_faces,uv_assigned_face_fraction=assigned_faces/len(faces),
        uv_assigned_area_fraction=assigned_area/total_area,
        alignment=alignment,asset_origin_in_alignment_frame=origin.tolist(),units='PPK-fitted metres, local ENU-like axes',
        accuracy_caveat='Scale/registration estimated from RTK cameras; not survey-certified surface accuracy',
        photographic_support_caveat='UV atlas binding does not certify source support for every triangle; inspect untextured/fallback regions',
        runtime_seconds=time.perf_counter()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True,choices=['rural','urban'])
    p.add_argument('--output-root',type=Path,default=OUT)
    p.add_argument('--method',choices=['delaunay','poisson'],default='delaunay')
    args=p.parse_args()
    OUT=args.output_root.resolve()
    convert(args.scene,args.method)
