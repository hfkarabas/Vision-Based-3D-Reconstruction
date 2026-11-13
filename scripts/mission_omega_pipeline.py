"""Small orchestration adapter over COLMAP caches, PDAL and GDAL.

No photogrammetry, ground-filter, interpolation or meshing implementations.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, os, subprocess, time, shutil
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/mission-omega'
GIS=OUT/'tools/gis'
PDAL=GIS/'Library/bin/pdal.exe'
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def env():
    e=os.environ.copy()
    e.update(PATH=os.pathsep.join(map(str,[GIS/'Library/bin',GIS,GIS/'Scripts']))+os.pathsep+e['PATH'],
             PROJ_DATA=str(GIS/'Library/share/proj'),GDAL_DATA=str(GIS/'Library/share/gdal'),
             OMP_NUM_THREADS='2',GDAL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    return e
def stage(name,cmd):
    p=OUT/'reports/stages'/f'{name}.json'
    if p.exists() and json.loads(p.read_text()).get('returncode')==0:
        return
    start=time.time(); peak=0; cpu=0
    record=dict(command=list(map(str,cmd)),started_at=datetime.now(timezone.utc).isoformat(),status='running',
        available_ram_at_start_bytes=psutil.virtual_memory().available,free_disk_at_start_bytes=shutil.disk_usage(OUT).free)
    save(p,record)
    with p.with_suffix('.log').open('wb') as log:
        try:
            proc=subprocess.Popen(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,env=env())
        except OSError as error:
            record.update(returncode=127,status='blocked',seconds=time.time()-start,error=str(error),ended_at=datetime.now(timezone.utc).isoformat())
            save(p,record);log.write(str(error).encode());raise
        while proc.poll() is None:
            try:
                group=[psutil.Process(proc.pid),*psutil.Process(proc.pid).children(recursive=True)]
                peak=max(peak,sum(q.memory_info().rss for q in group))
                cpu=max(cpu,sum(sum(q.cpu_times()[:2]) for q in group))
            except psutil.Error: pass
            time.sleep(.25)
    record.update(returncode=proc.returncode,status='finished',seconds=time.time()-start,
        peak_process_tree_rss_bytes=peak,sampled_cpu_seconds=cpu,ended_at=datetime.now(timezone.utc).isoformat())
    save(p,record)
    with (OUT/'progress.log').open('a',encoding='utf-8') as f:
        f.write(f"{record['ended_at']} {name}: {record['seconds']:.3f}s rc={proc.returncode} peakRSS={peak}; command={record['command']}\n")
    if proc.returncode: raise RuntimeError(p.with_suffix('.log'))
def pipeline(scene,variant='v2-elm'):
    raw=OUT/'raw'/scene/variant; raw.mkdir(parents=True,exist_ok=True)
    baseline=ROOT/'outputs/research-baseline'/scene
    align=json.loads((baseline/'sfm-2000/metrics.json').read_text())['models'][0]['diagnostic_ppk_similarity']
    mat=np.eye(4); mat[:3,:3]=np.array(align['rotation'])*align['meters_per_sfm_unit']
    mat[:3,3]=align['translation']; mat[:2,3]+=align['projected_origin_easting_northing']
    # Existing historical diagnostic transform is reused and explicitly labelled in-sample.
    save(raw/'geospatial_transform.json',dict(matrix_sfm_to_projected=mat.tolist(),source=align,
        horizontal='CGCS2000 EPSG:4547 easting/northing metres; RTK WGS84-to-CGCS realization unresolved',
        vertical='XMP RTK altitude fit, ellipsoid/orthometric relation unresolved; not survey-certified',
        reference_rasters_used=False))
    jobs={
      'classify':[
        dict(type='readers.ply',filename=str(baseline/'dense-900/workspace/fused.ply')),
        dict(type='filters.transformation',matrix=' '.join(map(str,mat.ravel()))),
        dict(type='filters.assign',value='Classification = 0'),
        dict(type='filters.outlier',method='statistical',mean_k=16,multiplier=2.5),
        dict(type='filters.expression',expression='Classification != 7'),
        dict(type='filters.elm',cell=10.,threshold=1.),
        dict(type='filters.expression',expression='Classification != 7'),
        dict(type='filters.smrf',cell=1.,slope=.15,scalar=1.25,threshold=.5,window=18.),
        dict(type='writers.las',filename=str(raw/'classified.laz'),a_srs='EPSG:4547',
             offset_x='auto',offset_y='auto',offset_z='auto',scale_x=.001,scale_y=.001,scale_z=.001)]}
    if variant.endswith('csf'):
        bounds=json.loads((OUT/'raw'/scene/'v3-mad/raster_qa.json').read_text())['roi_bounds']
        x0,y0,x1,y1=bounds
        jobs['classify'].insert(2,dict(type='filters.crop',bounds=f'([{x0},{x1}],[{y0},{y1}])'))
        jobs['classify']=[dict(type='filters.csf',resolution=1.,rigidness=3,iterations=500,threshold=.5)
            if item['type']=='filters.smrf' else item for item in jobs['classify']]
    for name,job in jobs.items():
        path=raw/f'{name}.json';save(path,job);stage(f'{scene}-{variant}-{name}',[PDAL,'pipeline',path])
    stage(f'{scene}-{variant}-stats',[PDAL,'info','--stats','--dimensions','X,Y,Z,Classification',raw/'classified.laz'])
    # Full classified cloud bounds are retained. A fixed camera-footprint + 50 m
    # envelope is a declared pilot ROI, not a certificate of full-scene coverage.
    meta=json.loads((ROOT/'outputs/research-baseline/reports/dataset-audit.json').read_text())[scene]
    save(raw/'source_manifest.json',dict(count=len(meta['images']),images=[{k:r[k] for k in ['path','sha256','size']} for r in meta['images']],
        reused_dense_stage=str(baseline/'dense-900/patchmatch.json'),scope='all original registered images, cached 900px dense reconstruction'))
    # Project camera positions with official pyproj in the installed GIS interpreter.
    stage(f'{scene}-{variant}-rasters',[GIS/'python.exe',ROOT/'scripts/mission_omega_rasters.py',scene,variant])
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('scene',choices=['rural','urban']);args=parser.parse_args();pipeline(args.scene)
