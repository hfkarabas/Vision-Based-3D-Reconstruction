"""Independent established trimesh topology measurements on final exported GLB."""
from pathlib import Path
import json,sys
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mission-omega'
scene=sys.argv[1];variant=sys.argv[2];dest=OUT/'asset'/f'{scene}-{variant}'
loaded=trimesh.load_scene(dest/f'{scene}_terrain.glb')
mesh=loaded.to_mesh();before=len(mesh.vertices)
_,inverse,xycounts=np.unique(np.round(mesh.vertices[:,[0,2]],4),axis=0,return_inverse=True,return_counts=True)
zmin=np.full(len(xycounts),np.inf);zmax=np.full(len(xycounts),-np.inf)
np.minimum.at(zmin,inverse,mesh.vertices[:,1]);np.maximum.at(zmax,inverse,mesh.vertices[:,1])
seams=zmax[xycounts>1]-zmin[xycounts>1]
# Analysis copy only: welding UV seams must not rewrite the deliverable.
mesh.merge_vertices(merge_tex=True,merge_norm=True,digits_vertex=4)
edges,counts=np.unique(mesh.edges_sorted,axis=0,return_counts=True)
components=trimesh.graph.connected_components(mesh.face_adjacency,nodes=np.arange(len(mesh.faces)),min_len=1,engine='scipy')
areas=sorted([float(mesh.area_faces[c].sum()) for c in components],reverse=True)
report=dict(library='trimesh',source=str(dest/f'{scene}_terrain.glb'),analysis_only_welding_precision_m=.0001,
    vertices_before_weld=before,vertices_after_weld=len(mesh.vertices),faces=len(mesh.faces),
    connected_surface_components=len(components),component_surface_areas_m2=areas,
    largest_component_area_fraction=areas[0]/sum(areas),boundary_edges=int((counts==1).sum()),
    nonmanifold_edges=int((counts>2).sum()),zero_area_triangles=int((mesh.area_faces<1e-10).sum()),
    shared_horizontal_samples=int((xycounts>1).sum()),max_shared_XY_height_deviation_m=float(seams.max()) if seams.size else None,
    watertight=bool(mesh.is_watertight),watertight_expected=False,
    seam_note='Adjacent QGIS blocks are joined in the analysis copy by matching metric positions; open outer/NoData boundaries are expected',
    reference_geometry_used=False)
(dest/'topology_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
