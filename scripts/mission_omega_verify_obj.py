"""Native clean Blender OBJ import check; run inside Blender."""
from pathlib import Path
import json,sys
import bpy
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mission-omega'
argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
variant=argv[0] if argv else 'production8-ortho'
dest=OUT/'asset'/f'rural-{variant}';path=dest/'obj/terrain.obj'
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=str(path),forward_axis='Y',up_axis='Z')
objects=[o for o in bpy.context.scene.objects if o.type=='MESH'];triangles=0
for obj in objects:obj.data.calc_loop_triangles();triangles+=len(obj.data.loop_triangles)
images=[im for im in bpy.data.images if min(im.size)>1]
expected=json.loads((dest/'blender_validation.json').read_text())
assert triangles==expected['triangles'] and len(images)==len(expected['images'])
assert all(Path(bpy.path.abspath(im.filepath)).is_file() for im in images)
report=dict(success=True,objects=len(objects),triangles=triangles,
    images=[dict(name=im.name,size=list(im.size),path=bpy.path.abspath(im.filepath)) for im in images],
    method='Clean native Blender OBJ importer; relative MTL/image references resolved')
(dest/'obj_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
