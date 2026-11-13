"""Established plyfile parser and trimesh OBJ exporter; affine/UV glue only."""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image
from plyfile import PlyData
import trimesh
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mission-omega'
scene=sys.argv[1];variant=sys.argv[2];raw=OUT/'raw'/scene/variant;dest=OUT/'asset'/f'{scene}-{variant}'
texture_dir=dest/('photo-texture-precise' if (dest/'photo-texture-precise/mesh.ply').exists() else 'photo-texture')
source=texture_dir/'mesh.ply';ply=PlyData.read(source)
v=ply['vertex'];xyz=np.column_stack([v[k] for k in ['x','y','z']])
faces=np.stack(ply['face']['vertex_indices']);uv=np.stack(ply['face']['texcoord']).reshape(-1,2)
# Duplicate vertices only to express PLY's face-corner UV coordinates in trimesh.
mesh=trimesh.Trimesh(vertices=xyz[faces.ravel()],faces=np.arange(faces.size).reshape(-1,3),process=False,
    visual=trimesh.visual.texture.TextureVisuals(uv=uv,image=Image.open(texture_dir/'texture.png')))
mesh.merge_vertices(merge_tex=False,merge_norm=True,digits_vertex=7,digits_uv=7)
mat=np.array(json.loads((raw/'geospatial_transform.json').read_text())['matrix_sfm_to_projected'])
b=json.loads((raw/'raster_qa.json').read_text())['roi_bounds'];origin=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2,0.])
mat[:3,3]-=origin;mesh.apply_transform(mat)
output=dest/'photographic-obj';output.mkdir(exist_ok=True)
obj,assets=trimesh.exchange.obj.export_obj(mesh,include_texture=True,return_texture=True,mtl_name='terrain.mtl')
(output/'terrain.obj').write_text(obj,encoding='utf-8')
for name,content in assets.items():(output/name).write_bytes(content)
coverage=np.any(np.stack(ply['face']['texcoord']).reshape(-1,6)!=0,axis=1)
report=dict(parser='plyfile',exporter='trimesh.exchange.obj.export_obj',source=str(source),triangles=len(faces),
    exported_vertices=len(mesh.vertices),uv_seam_merge='trimesh.merge_vertices; coordinates and UVs both retained at 1e-7 precision in SfM frame',
    uv_assigned_faces=int(coverage.sum()),uv_assigned_fraction=float(coverage.mean()),
    geospatial_origin=origin.tolist(),bounds=mesh.bounds.tolist(),reference_geometry_used=False,
    note='COLMAP image-to-DTM projection; visibility/roof displacement limitations; assigned UV is not independent correctness')
(output/'adapter.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
