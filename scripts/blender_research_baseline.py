"""Run inside Blender: independently import OBJ, export/reimport GLB, render views.

Uses unlit photographic material for texture inspection and neutral illumination for
surface inspection. All views include the full mesh; no geometry is repaired.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector, Matrix
import numpy as np


def clear():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def geometry(objects):
    xyz=[]
    faces=0
    for obj in objects:
        values=np.empty(len(obj.data.vertices)*3,dtype=np.float32)
        obj.data.vertices.foreach_get('co',values)
        xyz.append(values.reshape(-1,3)@np.array(obj.matrix_world)[:3,:3].T+np.array(obj.matrix_world)[:3,3])
        obj.data.calc_loop_triangles()
        faces+=len(obj.data.loop_triangles)
    points=np.concatenate(xyz)
    if not np.isfinite(points).all(): raise ValueError('Nonfinite imported mesh')
    return points,faces


def material(name, mode):
    mat=bpy.data.materials.new(name)
    mat.use_nodes=True
    nodes=mat.node_tree.nodes
    nodes.clear()
    output=nodes.new('ShaderNodeOutputMaterial')
    if mode=='wire':
        wire=nodes.new('ShaderNodeWireframe')
        wire.use_pixel_size=True
        wire.inputs['Size'].default_value=.65
        mix=nodes.new('ShaderNodeMixRGB')
        mix.inputs[1].default_value=(.55,.6,.65,1)
        mix.inputs[2].default_value=(.015,.02,.025,1)
        mat.node_tree.links.new(wire.outputs['Fac'],mix.inputs[0])
        emit=nodes.new('ShaderNodeEmission')
        mat.node_tree.links.new(mix.outputs[0],emit.inputs['Color'])
        mat.node_tree.links.new(emit.outputs[0],output.inputs['Surface'])
    else:
        shader=nodes.new('ShaderNodeBsdfPrincipled')
        shader.inputs['Base Color'].default_value=(.55,.6,.65,1)
        shader.inputs['Roughness'].default_value=.8
        mat.node_tree.links.new(shader.outputs[0],output.inputs['Surface'])
    return mat


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-dir',required=True,type=Path)
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    root=args.asset_dir.resolve()
    output=root/'blender-review'
    if not args.validate_only: output.mkdir(exist_ok=False)
    started=time.perf_counter()
    clear()
    bpy.ops.wm.obj_import(filepath=str(root/'textured.obj'),forward_axis='Y',up_axis='Z')
    objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
    points,faces=geometry(objects)
    expected=json.loads((root/'conversion.json').read_text())
    if faces!=expected['triangles']: raise ValueError('OBJ import changed triangle count')
    if not np.allclose([points.min(0),points.max(0)],expected['bounds_local_fitted_m'],atol=1e-3):
        raise ValueError('OBJ import changed source bounds or axis convention')
    textures=[n.image for o in objects for m in o.data.materials if m and m.node_tree
              for n in m.node_tree.nodes if n.type=='TEX_IMAGE' and n.image]
    if not textures or not all(min(i.size)>1 for i in textures): raise ValueError('Texture import failed')
    if not all(o.data.uv_layers for o in objects): raise ValueError('UV import failed')
    if not args.validate_only:
        bpy.ops.export_scene.gltf(filepath=str(root/'textured.glb'),export_format='GLB',use_selection=False)
    clear()
    bpy.ops.import_scene.gltf(filepath=str(root/'textured.glb'))
    objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
    reimported,glbfaces=geometry(objects)
    if glbfaces!=faces or not np.allclose([points.min(0),points.max(0)],
                                         [reimported.min(0),reimported.max(0)],atol=1e-3):
        raise ValueError('GLB reimport changed geometry count/bounds')
    materials={m for o in objects for m in o.data.materials}
    original=[]
    for mat in materials:
        if not mat or not mat.node_tree: continue
        nodes=mat.node_tree.nodes
        image=next((n.image for n in nodes if n.type=='TEX_IMAGE' and n.image),None)
        if image is None: raise ValueError('GLB lost photograph material')
        original.append(dict(name=mat.name,image=image.name,size=list(image.size),packed=bool(image.packed_file)))
        if args.validate_only: continue
        # Change only the review material: unlit atlas inspection eliminates lighting bias.
        outputnode=next(n for n in nodes if n.type=='OUTPUT_MATERIAL')
        tex=next(n for n in nodes if n.type=='TEX_IMAGE')
        emit=nodes.new('ShaderNodeEmission')
        mat.node_tree.links.new(tex.outputs['Color'],emit.inputs['Color'])
        mat.node_tree.links.new(emit.outputs[0],outputnode.inputs['Surface'])
    if args.validate_only:
        (root/'import-validation.json').write_text(json.dumps(dict(
            obj_triangles=faces,glb_triangles=glbfaces,bounds_match=True,finite_geometry=True,
            uv_and_image_binding=True,materials=original,blender=bpy.app.version_string),indent=2),encoding='utf-8')
        return
    poisson_objects=[]
    poisson_path=root.parent/'poisson.ply'
    comparison_points=points
    if poisson_path.exists() and expected.get('method','delaunay')!='poisson':
        before=set(bpy.context.scene.objects)
        bpy.ops.wm.ply_import(filepath=str(poisson_path))
        poisson_objects=[o for o in bpy.context.scene.objects if o not in before and o.type=='MESH']
        alignment=expected['alignment']
        matrix=np.eye(4)
        matrix[:3,:3]=alignment['meters_per_sfm_unit']*np.array(alignment['rotation'])
        matrix[:3,3]=np.array(alignment['translation'])-expected['asset_origin_in_alignment_frame']
        for obj in poisson_objects: obj.matrix_world=Matrix(matrix.tolist())
        bpy.context.view_layer.update()
        poisson_points,poisson_faces=geometry(poisson_objects)
        comparison_points=np.concatenate([points,poisson_points])
        for obj in poisson_objects: obj.hide_render=True
    scene=bpy.context.scene
    scene.render.engine='CYCLES'
    scene.cycles.device='CPU'
    scene.cycles.samples=12
    scene.render.threads_mode='FIXED'
    scene.render.threads=4
    scene.render.resolution_x=720
    scene.render.resolution_y=720
    scene.render.resolution_percentage=100
    scene.world.color=(.2,.2,.2)
    scene.view_settings.view_transform='Standard'
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGBA'
    scene.render.film_transparent=True
    bpy.ops.object.light_add(type='SUN')
    bpy.context.object.rotation_euler=(.4,-.4,.3)
    bpy.context.object.data.energy=2
    bpy.ops.object.camera_add()
    camera=bpy.context.object
    camera.data.type='ORTHO'
    scene.camera=camera
    neutral=material('Neutral geometry review','neutral')
    wire=material('Wireframe review','wire')
    bounds=np.array([comparison_points.min(0),comparison_points.max(0)])
    center=bounds.mean(0)
    radius=float(np.linalg.norm(bounds[1]-bounds[0]))
    framing={}
    for name,direction in [('top',(0,0,1)),('oblique',(1,-1,.8)),('side',(0,-1,0))]:
        direction=Vector(direction).normalized()
        camera.location=Vector(center)+direction*radius*2
        camera.rotation_euler=(-direction).to_track_quat('-Z','Y').to_euler()
        bpy.context.view_layer.update()
        basis=np.array(camera.matrix_world)[:3,:3]
        view=(comparison_points-center)@basis
        screen_center=(view[:,:2].min(0)+view[:,:2].max(0))/2
        camera.location+=Vector(basis[:,:2]@screen_center)
        camera.data.ortho_scale=float(np.max(np.ptp(view[:,:2],axis=0))*1.12)
        camera.data.clip_end=radius*10
        framing[name]=dict(camera=list(camera.location),ortho_scale=camera.data.ortho_scale)
        for mode,override in [('textured',None),('neutral',neutral),('wireframe',wire)]:
            scene.view_layers[0].material_override=override
            scene.render.filepath=str(output/f'{name}-{mode}.png')
            bpy.ops.render.render(write_still=True)
        if poisson_objects:
            for obj in objects: obj.hide_render=True
            for obj in poisson_objects: obj.hide_render=False
            for mode,override in [('neutral',neutral),('wireframe',wire)]:
                scene.view_layers[0].material_override=override
                scene.render.filepath=str(output/f'poisson-{name}-{mode}.png')
                bpy.ops.render.render(write_still=True)
            for obj in objects: obj.hide_render=False
            for obj in poisson_objects: obj.hide_render=True
    report=dict(blender=bpy.app.version_string,obj_triangles=faces,glb_triangles=glbfaces,
        finite_geometry=True,uv_and_image_binding=True,materials=original,
        bounds_local_fitted_m=bounds.tolist(),framing=framing,
        framing_scope=('Full union of Delaunay and Poisson mesh bounds, consistent per-view camera across methods and modes'
                       if poisson_objects else 'Full selected mesh bounds, consistent per-view camera across modes; separate from raw comparison union'),
        user_acceptance='pending',runtime_seconds=time.perf_counter()-started,
        glb_sha256=hashlib.sha256((root/'textured.glb').read_bytes()).hexdigest())
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__': main()
