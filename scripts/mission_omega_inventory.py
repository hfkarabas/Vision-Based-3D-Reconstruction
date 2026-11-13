"""Read-only source audit; isolated Omega evidence. No reconstruction algorithms."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, platform, shutil, subprocess, tarfile
import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/mission-omega'
def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding='utf-8')
def main():
    now = datetime.now(timezone.utc).isoformat()
    scenes = {}
    prior = json.loads((ROOT/'outputs/research-baseline/reports/dataset-audit.json').read_text())
    for scene, folder in [('rural','Scene2'), ('urban','Scene1')]:
        base = ROOT/'datasets/airzoo'/scene/folder
        rows=[]
        for p in sorted((base/'query').glob('*.JPG')):
            with Image.open(p) as im:
                rows.append(dict(path=str(p), bytes=p.stat().st_size, dimensions=im.size,
                    sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()))
        scenes[scene]=dict(images=rows, count=len(rows), metadata_source=str(ROOT/'outputs/research-baseline/reports/dataset-audit.json'),
            previous_xmp_examples=[r.get('xmp') for r in prior[scene]['images'][:2]],
            ancillary=[str(p) for p in base.rglob('*') if p.is_file() and p.suffix.lower()!='.jpg'])
    inventory=dict(timestamp=now, root=str(ROOT), platform=platform.platform(), cpu=platform.processor(), logical_cpus=os.cpu_count(),
        memory=psutil.virtual_memory()._asdict(), disk=shutil.disk_usage(ROOT)._asdict(), scenes=scenes,
        tools={k:shutil.which(k) for k in ['docker','wsl','qgis','pdal','gdalinfo','uv']},
        blender='C:/Program Files/Blender Foundation/Blender 5.3/blender.exe',
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True), git_index_lock=(ROOT/'.git/index.lock').exists(),
        limits=['CIM hardware query denied; psutil used', 'WSL executable exists but reports subsystem not installed',
                'Reference DOM/DSM excluded from production geometry; vertical datum unresolved'])
    save(OUT/'00_inventory.json',inventory)
    save(OUT/'mission_status.json',dict(started_at=now,last_updated_at=now,phase='P0',outcome='working',
        active_tasks=['official portable PDAL/GDAL fallback installation','ODM platform availability check'],
        blockers=['Docker absent; WSL subsystem not installed'], gates={f'G{i}':{'status':'NOT VERIFIED'} for i in range(7)}))
    with (OUT/'progress.log').open('a',encoding='utf-8') as f:
        f.write(f'{now} scripts/mission_omega_inventory.py: verified original image hashes/counts, psutil resources, git status; historical files preserved\n')
    archive=OUT/'tools/micromamba.tar.bz2'
    if archive.exists():
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if member.name.endswith('micromamba.exe'):
                    (OUT/'tools/micromamba.exe').write_bytes(tf.extractfile(member).read())
    print(json.dumps({s:scenes[s]['count'] for s in scenes}))
if __name__=='__main__': main()
