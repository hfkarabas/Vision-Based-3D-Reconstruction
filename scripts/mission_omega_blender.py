"""Native Blender import/export/render/QA and affine coordinate adapter only."""
from pathlib import Path
import argparse, hashlib, json, math, sys, time
import bpy
import numpy as np
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mission-omega'
parser=argparse.ArgumentParser();parser.add_argument('scene');parser.add_argument('variant');parser.add_argument('--prepare',action='store_true');parser.add_argument('--obj',type=Path);parser.add_argument('--verify-glb',action='store_true')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);scene=args.scene;variant=args.variant
raw=OUT/'raw'/scene/variant;dest=OUT/'asset'/f'{scene}-{variant}'
bpy.ops.wm.read_factory_settings(use_empty=True)
if args.obj:bpy.ops.wm.obj_import(filepath=str(args.obj.resolve()),forward_axis='Y',up_axis='Z')
else:bpy.ops.import_scene.gltf(filepath=str(dest/(f'{scene}_terrain_photographic.glb' if args.verify_glb else f'{scene}_terrain.glb')))
objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
assert objects,'Empty import'
qa=json.loads((raw/'raster_qa.json').read_text());b=qa['roi_bounds'];origin=[(b[0]+b[2])/2,(b[1]+b[3])/2,0]
if args.prepare:
    matrix=json.loads((raw/'geospatial_transform.json').read_text())['matrix_sfm_to_projected']
    shift=np.eye(4);shift[:3,3]=origin
    # Compose in float64 BEFORE entering Blender's float32 transform API.
    # This avoids loss of precision from subtracting million-metre northings.
    local_to_sfm=Matrix((np.linalg.inv(np.array(matrix))@shift).tolist())
    for obj in objects:obj.matrix_world=local_to_sfm@obj.matrix_world
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:obj.select_set(True)
    bpy.ops.wm.ply_export(filepath=str(dest/'terrain_sfm_precise.ply'),export_selected_objects=True,
        forward_axis='Y',up_axis='Z',export_normals=False,export_uv=False,export_colors='NONE',apply_modifiers=False)
    print('Prepared DTM mesh in cached COLMAP coordinates');sys.exit(0)

points=[];faces=0;zero=0;maxedge=0;area=0.;textured_faces=0
for obj in objects:
    vertices=np.array([obj.matrix_world@v.co for v in obj.data.vertices]);points.append(vertices)
    obj.data.calc_loop_triangles();indices=np.array([t.vertices for t in obj.data.loop_triangles]);tri=vertices[indices]
    a=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    zero+=int((a<1e-10).sum());area+=float(a.sum());faces+=len(indices)
    maxedge=max(maxedge,float(np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2).max()))
    if obj.data.uv_layers and any(m and m.node_tree and any(n.type=='TEX_IMAGE' and n.image for n in m.node_tree.nodes) for m in obj.data.materials):textured_faces+=len(indices)
allpoints=np.concatenate(points);lo=allpoints.min(0);hi=allpoints.max(0);center=(lo+hi)/2
textures=[im for im in bpy.data.images if im.type=='IMAGE' and min(im.size)>1]
assert textures,'Texture images missing: import is not accepted'
for im in textures:im.pack()
if variant.endswith('-ortho'):
    objdir=dest/'obj';objdir.mkdir(exist_ok=True)
    for index,im in enumerate(textures):
        im.filepath_raw=str(objdir/f'terrain_texture_{index}.png');im.file_format='PNG';im.save()
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:obj.select_set(True)
    bpy.ops.wm.obj_export(filepath=str(objdir/'terrain.obj'),export_selected_objects=True,
        forward_axis='Y',up_axis='Z',export_materials=True,path_mode='RELATIVE')
bpy.context.scene.unit_settings.system='METRIC';bpy.context.scene.unit_settings.scale_length=1
if args.obj:
    bpy.ops.export_scene.gltf(filepath=str(dest/f'{scene}_terrain_photographic.glb'),export_format='GLB',use_selection=False)
    modelpath=dest/f'{scene}_terrain_photographic.glb'
else:modelpath=dest/(f'{scene}_terrain_photographic.glb' if args.verify_glb else f'{scene}_terrain.glb')
validation=dict(source=str(modelpath),sha256=hashlib.file_digest(modelpath.open('rb'),'sha256').hexdigest(),bytes=modelpath.stat().st_size,
    objects=len(objects),vertices=len(allpoints),triangles=faces,zero_area_triangles=zero,max_edge_m=maxedge,surface_area_m2=area,
    bounds_local_ENU_m=[lo.tolist(),hi.tolist()],dimensions_m=(hi-lo).tolist(),uv_textured_face_count=textured_faces,
    images=[dict(name=im.name,size=list(im.size),packed=bool(im.packed_file)) for im in textures],
    geospatial_origin=origin,crs='EPSG:4547',Z_scale=1.,human_acceptance='PENDING',
    texture_lineage='COLMAP mesh_texturer source photos' if args.obj or args.verify_glb else qa.get('texture_status','NOT VERIFIED'),
    blender_version=bpy.app.version_string)
(OUT/'reports').mkdir(exist_ok=True);(dest/'blender_validation.json').write_text(json.dumps(validation,indent=2))
sc=bpy.context.scene;sc.render.engine='BLENDER_EEVEE';sc.render.resolution_x=1400;sc.render.resolution_y=1000;sc.render.resolution_percentage=100
sc.render.image_settings.file_format='PNG';sc.render.film_transparent=False;sc.world=bpy.data.worlds.new('NeutralWorld');sc.world.use_nodes=True
sc.world.node_tree.nodes['Background'].inputs[0].default_value=(.14,.16,.18,1);sc.world.node_tree.nodes['Background'].inputs[1].default_value=.8
bpy.ops.object.light_add(type='SUN',location=(0,0,500));bpy.context.object.data.energy=2.;bpy.context.object.rotation_euler=(.25,-.3,-.3)
bpy.ops.object.camera_add();camera=bpy.context.object;camera.data.type='ORTHO';sc.camera=camera
sc.view_settings.view_transform='Standard';sc.view_settings.look='Medium High Contrast' if 'Medium High Contrast' in [] else 'None'
preview=OUT/'preview';preview.mkdir(exist_ok=True)
prefix=f'{scene}_{variant}'+('_photographic' if args.obj or args.verify_glb else '')
corners=[Vector((x,y,z)) for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]]
radius=float(np.linalg.norm(hi-lo))
for label,direction in [('top',(0,0,1)),('oblique',(1,-1,.8)),('low_angle',(1,-1,.16))]:
    camera.location=Vector(center)+Vector(direction).normalized()*radius*2
    camera.rotation_euler=(Vector(center)-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.clip_start=.1;camera.data.clip_end=radius*10
    inv=camera.matrix_world.inverted() if False else camera.rotation_euler.to_matrix().transposed()
    view=np.array([inv@(p-Vector(center)) for p in corners]);aspect=sc.render.resolution_x/sc.render.resolution_y
    camera.data.ortho_scale=max(np.ptp(view[:,1]),np.ptp(view[:,0])/aspect)*1.15
    # Blender ortho_scale is horizontal camera width.
    camera.data.ortho_scale=max(np.ptp(view[:,0]),np.ptp(view[:,1])*aspect)*1.15
    sc.render.filepath=str(preview/f'{prefix}_{label}.png');bpy.ops.render.render(write_still=True)
# Render a geospatially bounded orthographic photo mosaic using Blender's existing
# renderer. This is a photo-textured DTM projection, not a building-correct true ortho.
if args.obj or args.verify_glb:
    for mat in bpy.data.materials:
        if not mat.node_tree:continue
        nodes=mat.node_tree.nodes;tex=next((n for n in nodes if n.type=='TEX_IMAGE' and n.image),None)
        output=next((n for n in nodes if n.type=='OUTPUT_MATERIAL'),None)
        if tex and output:
            emit=nodes.new('ShaderNodeEmission');mat.node_tree.links.new(tex.outputs['Color'],emit.inputs['Color'])
            mat.node_tree.links.new(emit.outputs[0],output.inputs['Surface'])
    camera.location=(0,0,float(hi[2]+1000));camera.rotation_euler=(0,0,0);camera.data.ortho_scale=b[2]-b[0]
    sc.render.resolution_x=round((b[2]-b[0])*2);sc.render.resolution_y=round((b[3]-b[1])*2)
    sc.render.film_transparent=True;sc.render.image_settings.color_mode='RGBA'
    sc.render.filepath=str(dest/'orthographic_photo_mosaic.png');bpy.ops.render.render(write_still=True)
    (dest/'orthographic_photo_mosaic.json').write_text(json.dumps(dict(bounds_EPSG4547=b,pixel_size_m=.5,
        method='Blender orthographic emission render of COLMAP-photo-textured UAV-derived DTM',
        caveat='Not ODM orthophoto; not DSM building-correct true ortho; occluded ground appearance inferred from image projection; no independent accuracy claim'),indent=2))
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            space=area.spaces.active;space.shading.type='MATERIAL'
            space.region_3d.view_location=Vector(center);space.region_3d.view_distance=radius*1.1
            space.region_3d.view_rotation=Vector((1,-1,.8)).to_track_quat('Z','Y')
bpy.ops.wm.save_as_mainfile(filepath=str(dest/f'{scene}_terrain.blend'))
print(json.dumps(validation))
