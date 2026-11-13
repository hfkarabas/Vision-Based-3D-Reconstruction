"""Measured regional scheduling over existing COLMAP/PDAL/GDAL, no new MVS.

Budget unit: complete fixed-size reference depth maps, not cropped tile pixels.
"""
from pathlib import Path
import argparse,json,os,shutil,sys
import numpy as np
from mission_omega_pipeline import ROOT,OUT,GIS,PDAL,stage,save
sys.path.insert(0,str(ROOT/'outputs/airzoo-benchmark/geometric-root-cause-recovery/colmap-python'))
import pycolmap
COLMAP=ROOT/'outputs/airzoo-benchmark/geometric-root-cause-recovery/colmap-cuda/bin/colmap.exe'
def prepare(scene):
    work=OUT/'research'/scene;work.mkdir(parents=True,exist_ok=True)
    raw=OUT/'raw'/scene/'v3-mad'
    protocol=dict(title='Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon',
        dataset=scene,seed=42,tiles=[4,4],actions_per_arm=2,reference_images_per_action=4,
        max_image_size=1200,iterations=3,num_samples=10,source_views_per_reference=3,
        budget='8 entire 1200x900 reference maps per arm; photometric and geometric passes; 3 iterations and 10 samples; full image work charged, NOT tile crop work',
        common_cost='Frozen cached 900px baseline + shared 1200px undistortion and sparse-support measurement; separately charged and disclosed',
        arms=['B0_uniform','B1_random_seed42','B2_feedback'],
        quality_score='0.7*NoData_fraction + 0.3*min(P95 adjacent DTM difference / 2m,1)',
        eligibility='At least 100 sparse points in tile, at least 4 supported registered source images',
        update='GDAL merge on fixed 1m grid within chosen tile; fill baseline holes only, existing valid cells unchanged; all new elevations carry inferred-ground uncertainty',
        feedback='After each executed action recompute score from actual merged raster; choose highest-score unused eligible tile',
        accuracy='No independent bare-earth DTM or resolved reference DSM lineage available; NO RMSE or absolute accuracy claim',
        stopping='Two actions per arm, no tuning on held-out reference; no external reference consumed',
        novelty='Engineering experiment, not exact Mostegel/RQF reproduction and not a proven novel method')
    save(work/'protocol.json',protocol)
    stage(f'{scene}-research-tile-state',[GIS/'python.exe',ROOT/'scripts/mission_omega_research_gis.py',scene,'prepare'])
    tiles=json.loads((work/'tile_state.json').read_text())
    rec=pycolmap.Reconstruction(ROOT/'outputs/research-baseline'/scene/'sfm-2000/models/0')
    transform=np.array(json.loads((raw/'geospatial_transform.json').read_text())['matrix_sfm_to_projected'])
    support={t['id']:{} for t in tiles}
    for point in rec.points3D.values():
        xy=(transform[:3,:3]@point.xyz+transform[:3,3])[:2]
        for t in tiles:
            x0,y0,x1,y1=t['bounds']
            if x0<=xy[0]<x1 and y0<=xy[1]<y1:
                for e in point.track.elements:
                    name=rec.images[e.image_id].name
                    support[t['id']][name]=support[t['id']].get(name,0)+1
                t['sparse_points']=t.get('sparse_points',0)+1
                break
    for t in tiles:
        ranked=sorted(support[t['id']].items(),key=lambda p:(-p[1],p[0]))
        # Existing point tracks define eligible imagery; this is a transparent
        # support-count approximation, not a claim to confidence prediction.
        t.update(ranked_images=ranked,eligible=t.get('sparse_points',0)>=100 and len(ranked)>=4)
    save(work/'tile_state.json',tiles)
    workspace=work/'workspace-1200';workspace.mkdir(exist_ok=True)
    folder='Scene2' if scene=='rural' else 'Scene1'
    stage(f'{scene}-research-undistort1200',[COLMAP,'image_undistorter','--image_path',ROOT/'datasets/airzoo'/scene/folder/'query',
        '--input_path',ROOT/'outputs/research-baseline'/scene/'sfm-2000/models/0','--output_path',workspace,
        '--output_type','COLMAP','--max_image_size','1200','--num_threads','2'])
    print('prepared',len([t for t in tiles if t['eligible']]),'eligible tiles')
def action(scene,arm,index,tile):
    work=OUT/'research'/scene;dest=work/arm/f'action-{index:02d}-{tile["id"]}';dest.mkdir(parents=True,exist_ok=True)
    source=work/'workspace-1200'
    for name in ['images','sparse']:
        if not (dest/name).exists():shutil.copytree(source/name,dest/name,copy_function=os.link)
    stereo=dest/'stereo';stereo.mkdir(exist_ok=True)
    for name in ['depth_maps','normal_maps','consistency_graphs']:(stereo/name).mkdir(exist_ok=True)
    offset=tile.get('image_batch_offset',0)
    images=[x[0] for x in tile['ranked_images'][offset:offset+4]]
    config=''.join(name+'\n'+', '.join(other for other in images if other!=name)+'\n' for name in images)
    (stereo/'patch-match.cfg').write_text(config);(stereo/'fusion.cfg').write_text('\n'.join(images)+'\n')
    save(dest/'decision.json',dict(tile=tile,arm=arm,index=index,reference_images=images,
        scope='4 complete source image depth maps, NOT pixel-level local MVS; only merge restricted to tile'))
    stage(f'{scene}-{arm}-{index}-patchmatch',[COLMAP,'patch_match_stereo','--workspace_path',dest,
        '--PatchMatchStereo.max_image_size','1200','--PatchMatchStereo.num_iterations','3','--PatchMatchStereo.num_samples','10',
        '--PatchMatchStereo.geom_consistency','1','--PatchMatchStereo.cache_size','1','--PatchMatchStereo.num_threads','2',
        '--default_random_seed','42'])
    stage(f'{scene}-{arm}-{index}-fusion',[COLMAP,'stereo_fusion','--workspace_path',dest,'--input_type','geometric',
        '--output_path',dest/'fused.ply','--StereoFusion.num_threads','2','--StereoFusion.min_num_pixels','3',
        '--StereoFusion.cache_size','1','--StereoFusion.use_cache','1'])
    mat=np.array(json.loads((OUT/'raw'/scene/'v3-mad/geospatial_transform.json').read_text())['matrix_sfm_to_projected'])
    x0,y0,x1,y1=tile['bounds']
    job=[dict(type='readers.ply',filename=str(dest/'fused.ply')),
        dict(type='filters.transformation',matrix=' '.join(map(str,mat.ravel()))),
        dict(type='filters.assign',value='Classification = 0'),
        dict(type='filters.outlier',method='statistical',mean_k=16,multiplier=2.5),
        dict(type='filters.expression',expression='Classification != 7'),
        dict(type='filters.elm',cell=10.,threshold=1.),dict(type='filters.expression',expression='Classification != 7'),
        dict(type='filters.smrf',cell=1.,slope=.15,scalar=1.25,threshold=.5,window=18.),
        dict(type='filters.expression',expression='Classification == 2'),dict(type='filters.mad',dimension='Z',k=10.),
        dict(type='writers.gdal',filename=str(dest/'dtm.tif'),resolution=1.,radius=1.5,output_type='idw',
            window_size=0,nodata=-9999,data_type='float',origin_x=x0,origin_y=y0,width=x1-x0,height=y1-y0,
            override_srs='EPSG:4547',allow_empty=True)]
    save(dest/'ground.json',job)
    stage(f'{scene}-{arm}-{index}-ground',[PDAL,'pipeline',dest/'ground.json'])
    stage(f'{scene}-{arm}-{index}-merge',[GIS/'python.exe',ROOT/'scripts/mission_omega_research_gis.py',scene,'merge',arm,str(dest)])
def run(scene):
    work=OUT/'research'/scene;tiles=json.loads((work/'tile_state.json').read_text());eligible=[t for t in tiles if t['eligible']]
    if len(eligible)<2:raise RuntimeError('Not enough eligible tiles')
    rng=np.random.default_rng(42)
    choices={'B0_uniform':[eligible[len(eligible)//4]['id'],eligible[3*len(eligible)//4]['id']],
        'B1_random_seed42':[t['id'] for t in rng.choice(eligible,size=2,replace=False)]}
    for arm in ['B0_uniform','B1_random_seed42','B2_feedback']:
        selected=[]
        for i in range(2):
            previous=list((work/arm).glob(f'action-{i:02d}-*/decision.json'))
            if previous:
                t=json.loads(previous[0].read_text())['tile']
            elif arm=='B2_feedback':
                state=work/arm/'feedback_tile_state.json'
                current=json.loads(state.read_text()) if state.exists() else tiles
                scores={t['id']:t['score'] for t in current}
                t=max((t for t in eligible if t['id'] not in selected),key=lambda t:(scores[t['id']],t['id']))
                t=dict(t,feedback_score_at_decision=scores[t['id']])
            else:t=next(t for t in tiles if t['id']==choices[arm][i])
            selected.append(t['id']);action(scene,arm,i,t)
        save(work/arm/'selected.json',selected)
    stage(f'{scene}-research-summary',[GIS/'python.exe',ROOT/'scripts/mission_omega_research_gis.py',scene,'summary'])
def revisit(scene):
    """Exploratory correction: allow a deficient region to receive another view batch.

    The original no-revisit trial is preserved, since disjoint tile updates cannot
    change the ranking of unused tiles. This is a limitation, not concealed novelty.
    """
    work=OUT/'research'/scene;tiles=json.loads((work/'tile_state.json').read_text())
    save(work/'feedback_v2_protocol.json',dict(arm='B2_feedback_revisit',reason='Original disjoint-tile no-revisit loop is rank-equivalent to a frozen score; explicitly test genuine revisitation',
        declared_before_v2=True,selection='Highest recomputed NoData/edge score among actions with at least four unused eligible source images',
        action='Next unused batch of four support-ranked registered images for that region; entire images processed',
        budget='Same 8 complete 1200x900 reference maps, same three iterations/ten samples/three source views',
        status='exploratory correction after engineering pilot; no independent evaluation reference used'))
    arm='B2_feedback_revisit';visits={t['id']:0 for t in tiles}
    for i in range(2):
        previous=list((work/arm).glob(f'action-{i:02d}-*/decision.json'))
        if previous:t=json.loads(previous[0].read_text())['tile']
        else:
            state=work/arm/'feedback_tile_state.json';current=json.loads(state.read_text()) if state.exists() else tiles
            scores={t['id']:t['score'] for t in current}
            eligible=[t for t in tiles if t['eligible'] and len(t['ranked_images'])>=4*(visits[t['id']]+1)]
            t=dict(max(eligible,key=lambda t:(scores[t['id']],t['id'])))
            t.update(image_batch_offset=4*visits[t['id']],feedback_score_at_decision=scores[t['id']])
        action(scene,arm,i,t);visits[t['id']]+=1
    stage(f'{scene}-research-summary-with-revisit',[GIS/'python.exe',ROOT/'scripts/mission_omega_research_gis.py',scene,'summary'])
def confidence(scene):
    work=OUT/'research'/scene;tiles=json.loads((work/'tile_state.json').read_text())
    eligible=[dict(t,mean_sparse_observation_support=sum(x[1] for x in t['ranked_images'])/t['sparse_points']) for t in tiles if t['eligible']]
    chosen=sorted(eligible,key=lambda t:(t['mean_sparse_observation_support'],t['id']))[:2]
    save(work/'B3_support_proxy_protocol.json',dict(arm='B3_support_proxy',
        rule='Ascending mean registered sparse observations per point within tile; four support-ranked image references per action',
        relationship='Geometry-support cue inspired by preliminary geometry/view-constellation reasoning in Mostegel 2018; NOT learned confidence prediction, NOT exact reproduction, NOT RQF',
        budget='Same 8 complete 1200x900 maps and all processing settings; no reference raster used',selected=chosen))
    for i,t in enumerate(chosen):action(scene,'B3_support_proxy',i,t)
    stage(f'{scene}-research-summary-all-controls',[GIS/'python.exe',ROOT/'scripts/mission_omega_research_gis.py',scene,'summary'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scene',choices=['rural','urban']);p.add_argument('mode',choices=['prepare','run','revisit','confidence','one-action'])
    p.add_argument('--tile');p.add_argument('--arm',default='manual_roi_replay');p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.mode=='one-action':
        tiles=json.loads((OUT/'research'/a.scene/'tile_state.json').read_text())
        tile=next(t for t in tiles if t['id']==a.tile)
        action(a.scene,a.arm,a.index,tile)
    else:{'prepare':prepare,'run':run,'revisit':revisit,'confidence':confidence}[a.mode](a.scene)
