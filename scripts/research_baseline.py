"""Mission 0: original-image audit and independent, resumable COLMAP CLI SfM.

No native poses, native intrinsics, or position priors are imported.
Run with the project Python; all commands and stage measurements are preserved.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageOps, ExifTags
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/research-baseline'
COLMAP = ROOT / 'outputs/airzoo-benchmark/geometric-root-cause-recovery/colmap-cuda/bin/colmap.exe'
SCENES = {'rural': 'Scene2', 'urban': 'Scene1'}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def audit():
    save(OUT / 'reports/environment.json', dict(platform=platform.platform(), cpu=platform.processor(),
         logical_cpus=os.cpu_count(), ram=psutil.virtual_memory()._asdict(), colmap=str(COLMAP),
         gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'], text=True),
         version=subprocess.check_output([str(COLMAP), '-h'], text=True)))
    result = {}
    for scene, folder in SCENES.items():
        base = ROOT / 'datasets/airzoo' / scene / folder
        images = sorted((base / 'query').glob('*.JPG'))
        sheet = Image.new('RGB', (8 * 200, ((len(images) + 7) // 8) * 160), '#242424')
        draw = ImageDraw.Draw(sheet)
        rows = []
        for i, path in enumerate(images):
            with Image.open(path) as im:
                tags = im.getexif()
                exif = {ExifTags.TAGS.get(k, str(k)): str(v) for k, v in tags.items()}
                exif.update({ExifTags.TAGS.get(k, str(k)): str(v) for k,v in tags.get_ifd(34665).items()})
                thumb = ImageOps.contain(im, (196, 130))
                sheet.paste(thumb, ((i % 8) * 200, (i // 8) * 160))
                draw.text(((i % 8) * 200, (i // 8) * 160 + 132), path.stem[-12:], fill='white')
                header = path.read_bytes()[:262144]
                xmp = {k.decode(): v.decode() for k, v in re.findall(rb'drone-dji:([A-Za-z]+)="([^\"]+)"', header)}
                rows.append(dict(path=str(path.relative_to(ROOT)), size=im.size, bytes=path.stat().st_size,
                                 sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(), exif=exif, xmp=xmp))
        sheet.save(OUT / 'reports' / f'{scene}-contact-sheet.jpg')
        ancillary = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size) for p in base.rglob('*')
                     if p.is_file() and p.suffix.upper() != '.JPG']
        result[scene] = dict(count=len(rows), resolutions=dict(Counter(str(r['size']) for r in rows)),
                             images=rows, ancillary=ancillary,
                             reference_metadata=(base / 'dom_dsm/metadata.xml').read_text())
    save(OUT / 'reports/dataset-audit.json', result)
    inventory = {}
    for folder in (ROOT / 'datasets').iterdir():
        if folder.is_dir():
            files = list(p for p in folder.rglob('*') if p.is_file())
            inventory[folder.name] = dict(files=len(files), bytes=sum(p.stat().st_size for p in files),
                extension_counts=dict(Counter(p.suffix.lower() for p in files)),
                interpretation='UAV originals' if folder.name=='airzoo' else 'Auxiliary benchmark; UAV provenance not established')
    save(OUT / 'reports/all-datasets.json',inventory)


def stage(work, name, args):
    marker = work / f'{name}.json'
    if marker.exists() and json.loads(marker.read_text())['returncode'] == 0:
        return
    command = [str(COLMAP), *map(str, args)]
    env = os.environ.copy()
    env['PATH'] = str(COLMAP.parent) + os.pathsep + env['PATH']
    start = time.time()
    peak = 0
    gpu_samples = []
    next_gpu_sample = 0
    save(marker, dict(command=command, status='running', started=start, returncode=None))
    with (work / f'{name}.log').open('wb') as log:
        p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        while p.poll() is None:
            try:
                peak = max(peak, psutil.Process(p.pid).memory_info().rss)
            except psutil.Error:
                pass
            if time.time() >= next_gpu_sample:
                try:
                    gpu_samples.append(dict(elapsed=time.time()-start, system_gpu=subprocess.check_output(
                        ['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                        text=True, timeout=5).strip()))
                except (OSError, subprocess.SubprocessError):
                    pass
                next_gpu_sample = time.time() + 5
            time.sleep(0.5)
    save(marker, dict(command=command, returncode=p.returncode, seconds=time.time()-start,
                      peak_process_rss_bytes=peak, peak_sampling_seconds=0.5,
                      gpu_samples=gpu_samples, gpu_scope='Whole GPU, includes unrelated applications'))
    if p.returncode:
        raise RuntimeError(f'{name} failed: see {work}')


def sfm(scene):
    work = OUT / scene / 'sfm-2000'
    work.mkdir(parents=True, exist_ok=True)
    images = ROOT / 'datasets/airzoo' / scene / SCENES[scene] / 'query'
    database = work / 'database.db'
    models = work / 'models'
    models.mkdir(exist_ok=True)
    stage(work, 'features', ['feature_extractor', '--database_path', database, '--image_path', images,
          '--ImageReader.single_camera', 1, '--ImageReader.camera_model', 'SIMPLE_RADIAL',
          '--FeatureExtraction.max_image_size', 2000, '--FeatureExtraction.num_threads', 4,
          '--FeatureExtraction.use_gpu', 1])
    stage(work, 'matching', ['exhaustive_matcher', '--database_path', database,
          '--FeatureMatching.num_threads', 4, '--FeatureMatching.use_gpu', 1])
    stage(work, 'mapping', ['mapper', '--database_path', database, '--image_path', images,
          '--output_path', models, '--Mapper.num_threads', 4])
    for model in models.iterdir():
        if model.is_dir():
            stage(work, f'analyze-{model.name}', ['model_analyzer', '--path', model])


def dense(scene):
    sys.path.insert(0, str(ROOT / 'outputs/airzoo-benchmark/geometric-root-cause-recovery/colmap-python'))
    import pycolmap
    models = OUT / scene / 'sfm-2000/models'
    candidates = [(pycolmap.Reconstruction(p).num_reg_images(), p) for p in models.iterdir()
                  if (p / 'cameras.bin').exists()]
    count, model = max(candidates, key=lambda item: item[0])
    work = OUT / scene / 'dense-900'
    work.mkdir(parents=True, exist_ok=True)
    workspace = work / 'workspace'
    save(work / 'inputs.json', dict(model=str(model), registered_images=count,
          scope='Largest independent model, not necessarily every source image',
          max_dense_image_size=900, source_views=6, no_native_poses=True))
    images = ROOT / 'datasets/airzoo' / scene / SCENES[scene] / 'query'
    stage(work, 'undistort', ['image_undistorter','--image_path',images,'--input_path',model,
          '--output_path',workspace,'--output_type','COLMAP','--max_image_size',900,
          '--num_patch_match_src_images',6,'--num_threads',4])
    stage(work, 'patchmatch', ['patch_match_stereo','--workspace_path',workspace,
          '--PatchMatchStereo.gpu_index',0,'--PatchMatchStereo.geom_consistency',1,
          '--PatchMatchStereo.max_image_size',900,'--PatchMatchStereo.cache_size',2])
    stage(work, 'fusion', ['stereo_fusion','--workspace_path',workspace,'--input_type','geometric',
          '--output_path',workspace/'fused.ply','--StereoFusion.num_threads',4,
          '--StereoFusion.max_image_size',900,'--StereoFusion.cache_size',2])
    # Open-scene comparison: visibility-based Delaunay versus screened Poisson.
    stage(work, 'delaunay', ['delaunay_mesher','--input_path',workspace,
          '--output_path',work/'delaunay.ply','--DelaunayMeshing.num_threads',4])
    stage(work, 'poisson', ['poisson_mesher','--input_path',workspace/'fused.ply',
          '--output_path',work/'poisson.ply','--PoissonMeshing.depth',10,'--PoissonMeshing.num_threads',4])
    stage(work, 'texture-delaunay', ['mesh_texturer','--workspace_path',workspace,
          '--input_path',work/'delaunay.ply','--output_path',work/'textured-delaunay',
          '--MeshTextureMapping.num_threads',4])
    stage(work, 'texture-poisson', ['mesh_texturer','--workspace_path',workspace,
          '--input_path',work/'poisson.ply','--output_path',work/'textured-poisson',
          '--MeshTextureMapping.num_threads',4])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'sfm', 'dense'])
    parser.add_argument('--scene', choices=SCENES)
    parser.add_argument('--output-root',type=Path,default=OUT)
    args = parser.parse_args()
    OUT=args.output_root.resolve()
    if args.action == 'audit':
        audit()
    else:
        if not args.scene:
            parser.error('--scene required for sfm')
        (sfm if args.action == 'sfm' else dense)(args.scene)
