"""Checkpoint measured Mission 0 results without committing generated assets."""
import hashlib
import json
from pathlib import Path
import re

from research_baseline import OUT, ROOT, save


def main():
    evidence=ROOT/'Documentation/research'
    evidence.mkdir(exist_ok=True)
    result={'status':'baseline checkpoint; user visual acceptance pending; mission gates remain open',
            'scenes':{},'openmvs':'not run: executable unavailable',
            'literature_records':len(json.loads((evidence/'literature-register.json').read_text(encoding='utf-8'))),'literature_target_met':False,
            'native_equal_input_comparison_completed':False,
            'independent_3d_accuracy_available':False}
    lines=['# Independent baseline benchmark checkpoint', '',
           'Measured 12 November 2026. Full camera sets, 900-pixel dense pilot. No native poses imported.',
           'User visual acceptance remains pending. This is not a completed Mission 0 verdict.', '',
           '| Measure | Rural | Urban |','|---|---:|---:|']
    for scene in ['rural','urban']:
        work=OUT/scene/'dense-900'
        dense=json.loads((work/'metrics.json').read_text())
        sfm=json.loads((OUT/scene/'sfm-2000/metrics.json').read_text())
        stages={}
        for folder in [OUT/scene/'sfm-2000',work]:
            for path in folder.glob('*.json'):
                data=json.loads(path.read_text())
                if 'command' not in data: continue
                row={k:v for k,v in data.items() if k!='gpu_samples'}
                if 'gpu_samples' in data:
                    memory=[float(s['system_gpu'].split(',')[0]) for s in data['gpu_samples']]
                    row['sampled_whole_gpu_peak_mib']=max(memory) if memory else None
                stages[path.stem]=row
        native_log=(work/'patchmatch.log').read_text(errors='replace')
        elapsed=re.findall(r'Elapsed time: ([\d.]+) \[minutes\]',native_log)
        stages['patchmatch']['native_log_elapsed_minutes']=float(elapsed[-1]) if elapsed else None
        stages['patchmatch']['timing_caveat']='Shared GPU with other scene. Urban observer queue inflates marker wall time; native elapsed is preferable, still not standalone throughput.'
        assets={}
        for name in ['asset','asset-poisson']:
            asset=work/name
            conversion=json.loads((asset/'conversion.json').read_text())
            validation=json.loads((asset/'import-validation.json').read_text())
            review=json.loads((asset/'blender-review/report.json').read_text())
            files=[]
            for filename in ['textured.obj','textured.mtl','textured.glb','texture.png']:
                path=asset/filename
                files.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
                                  sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()))
            assets[name]=dict(conversion=conversion,blender_validation=validation,review=review,files=files)
        compact={k:v for k,v in dense.items() if k not in ['stages','depth_maps']}
        compact['depth_map_count']=len(dense['depth_maps'])
        compact['stage_measurements']=stages
        compact['assets']=assets
        compact['sfm']=sfm
        result['scenes'][scene]=compact
        save(evidence/f'{scene}-baseline-results.json',compact)
    r,u=[result['scenes'][s] for s in ['rural','urban']]
    def row(label,fn): lines.append(f'| {label} | {fn(r)} | {fn(u)} |')
    row('Registered / original',lambda x:f"{x['sfm']['models'][0]['registered']} / {190 if x['scene']=='rural' else 161}")
    row('Sparse points',lambda x:x['sfm']['models'][0]['points'])
    row('Mean reprojection error (original px)',lambda x:round(x['sfm']['models'][0]['mean_reprojection_error_px'],4))
    row('Fused points',lambda x:x['dense']['points'])
    row('Median per-map valid depth fraction',lambda x:round(x['valid_depth_fraction']['p50'],4))
    row('Median distinct supporting images',lambda x:x['dense']['distinct_supporting_images']['p50'])
    row('Occupied 0.5 m XY cell area (not valid completeness)',lambda x:x['dense']['occupied_xy_area_m2'])
    row('Delaunay faces / components',lambda x:f"{x['meshes']['delaunay']['summary']['triangles']} / {x['meshes']['delaunay']['summary']['components']}")
    row('Poisson faces / components',lambda x:f"{x['meshes']['poisson']['summary']['triangles']} / {x['meshes']['poisson']['summary']['components']}")
    row('Poisson largest component fraction',lambda x:round(x['meshes']['poisson']['summary']['largest_component_triangle_share'],5))
    row('Poisson boundary / nonmanifold edges',lambda x:f"{x['meshes']['poisson']['summary']['boundary_edges']} / {x['meshes']['poisson']['summary']['nonmanifold_edges']}")
    row('Poisson positive local intersection probes / 8',lambda x:sum(t['intersection_detected'] for t in x['meshes']['poisson']['bounded_intersection_probes']))
    row('Poisson nominal photographic UV area fraction',lambda x:round(x['assets']['asset-poisson']['conversion']['uv_assigned_area_fraction'],5))
    row('Fused-cloud centred DSM absolute median / P95 (m)',lambda x:f"{x['dsm_reference_agreement']['offset_adjusted_absolute_residual_m']['p50']:.3f} / {x['dsm_reference_agreement']['offset_adjusted_absolute_residual_m']['p95']:.3f}")
    row('Fused-cloud centred DSM RMSE (m)',lambda x:round(x['dsm_reference_agreement']['offset_adjusted_rmse_m'],3))
    row('1 m upper-envelope DSM P95, all occupied cells (m)',lambda x:round(x['dsm_upper_envelope_diagnostic']['results']['minimum_1_points']['offset_adjusted_absolute_residual_m']['p95'],3))
    row('Stereo native log minutes (shared GPU)',lambda x:x['stage_measurements']['patchmatch']['native_log_elapsed_minutes'])
    lines += ['', '## Interpretation and limits','',
        'Rural Poisson visibly recovers fields, roads and low buildings; Urban recovers roads, gardens and high-rise structures but retains important holes and disconnected fragments. Raw Delaunay is rejected as a usable whole asset because of peripheral stretched/extrapolated faces. Poisson is the preferred review candidate, not a certified watertight or intersection-free environment. Both raw methods are preserved.',
        '', 'Poisson has positive bounded self-intersection probes (4 Rural, 3 Urban); zero nonmanifold edges does not prove intersection freedom. Each probe examines at most 12000 faces in one of eight dense neighborhoods. Negative probes are not global certificates. Missing stored PLY normals are reported separately from invalid computed normals.',
        '', 'Nominal photo UV assignment is not independent texture correctness. Total mesh area is not supported reconstructed area; the latter remains unmeasured. Occupied XY cells can include outliers and overlapping vertical surfaces. No 3D precision/completeness/F-score or absolute geospatial accuracy is established.',
        '', 'DSM figures use all fused points inside a conservative DOM mask, a similarity fitted to camera RTK positions and a fitted median vertical offset. They are height-reference agreement only. Urban has a serious tail and Rural has rare extreme outliers; do not hide these behind the median. DSM lineage, vertical datum and the Urban DOM CRS conflict remain unresolved. Historical native DSM results use a different mask/input set and cannot establish superiority.',
        '', 'A facade can correctly lie below a rooftop DSM. The separately reported fixed 1 m upper-envelope diagnostic reduces Urban P95 from 25.834 to 6.386 m over 269220 occupied cells, consistent with a representation effect as well as outliers; it does not prove the entire tail was erroneous geometry. Minimum-three-point cells have P95 4.142 m but cover only 154291 cells, so this must not replace the complete result. Rural upper-envelope P95 is 2.555 m over 393868 cells (1.468 m over 167431 minimum-three-point cells). Each diagnostic fits its own offset and is sensitive to high outliers. No metric is absolute accuracy.',
        '', 'The fusion cache_size argument is 2 GB but installed use_cache defaults false; this is not a hard memory cap. Measured per-process peak RSS and whole-device GPU samples are in JSON. Stereo ran concurrently; downstream Urban was queued for memory and resumed. No observer remains suspended.',
        '', 'COLMAP 4.2.1 produced SfM, depth, fusion, both surfaces and photographic textures. The project adapter only serializes existing vertices/faces/UVs to OBJ, omitting unreferenced vertices. Blender 5.3 independently imported OBJ, exported/reimported GLB and verified bounds, faces, UV and image binding. All four native validation invocations returned 0; initial failed import-bounds evidence is retained separately.',
        '', '## Remaining gates','',
        '- Explicit user inspection/acceptance of both full montages.',
        '- Matched-input native/backend experiment and common-footprint comparison.',
        '- OpenMVS comparison (unavailable locally; not performed).',
        '- Supported-area and spatial-hole validation with independent reference geometry.',
        f"- Literature expansion: {result['literature_records']} screened records, not 200–300; stronger novelty claim prohibited.",
        '- Local commit: .git is read-only in this session; git add failed creating index.lock. No commit or push occurred.',
        '', 'See Documentation/RESEARCH_FOUNDATION.md and the five companion research documents for decisions and replay.']
    save(OUT/'reports/final-benchmark-summary.json',result)
    save(evidence/'baseline-summary.json',result)
    text='\n'.join(lines)+'\n'
    (OUT/'reports/FINAL_BENCHMARK_SUMMARY.md').write_text(text,encoding='utf-8')
    (evidence/'BASELINE_RESULTS.md').write_text(text,encoding='utf-8')


if __name__=='__main__': main()
