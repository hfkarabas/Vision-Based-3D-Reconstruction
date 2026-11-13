"""Normalize two known exporter-generated MTL references, no geometry/parser code."""
from pathlib import Path
import re,sys
ROOT=Path(__file__).resolve().parents[1]
variant=sys.argv[1] if len(sys.argv)>1 else 'production8-ortho'
p=ROOT/'outputs/mission-omega/asset'/f'rural-{variant}'/'obj/terrain.mtl'
text=p.read_text()
for image in p.parent.glob('terrain_texture_*.png'):
    name=image.name
    assert (p.parent/name).is_file()
    text=re.sub(r'(?m)^(map_Kd|map_d) [^\r\n]*'+re.escape(name)+r'$',lambda m:m.group(1)+' '+name,text)
p.write_text(text)
print('Normalized native exporter MTL texture paths to adjacent verified PNG files')
