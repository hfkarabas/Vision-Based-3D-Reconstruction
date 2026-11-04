"""Traceable figures/tables from real baseline and controlled recovery outputs."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vision3d.evaluation.poses import evaluate_poses
from vision3d.features.cache import cached_extract
from vision3d.features.matching import match
from vision3d.geometry import verify
from vision3d.geometry.camera import Camera
from vision3d.io import load_image
from vision3d.preprocessing import preprocess
from vision3d.reconstruction.pair import source_hash, write_json
from vision3d.reconstruction.graph import components


def read(path):
    return json.loads(Path(path).read_text())


def feature_figure(run, output):
    manifest, intrinsics = read(run / 'manifest.json'), read(run / 'intrinsics.json')
    config = manifest['config']
    ids = (0, 29)
    images, features, cameras = [], [], []
    for i in ids:
        record = manifest['inputs'][i]
        if source_hash(record['source']) != record['sha256']:
            raise ValueError('Source image changed since experiment')
        image, _ = preprocess(load_image(record['source']).image, config['max_dimension'])
        images.append(image)
        features.append(cached_extract(image, record['sha256'], config['max_dimension'], config['detector'], config['max_features'])[0])
        model = dict(intrinsics[i]['processed'])
        model.pop('K')
        cameras.append(Camera(**model))
    cv2.setRNGSeed((config['seed'] + ids[0] * 1009 + ids[1] * 9176) % 2147483647)
    matches = match(*features, ratio=config['ratio'], mutual=True, geometric=False).consistent
    focal = min(cameras[0].fx, cameras[0].fy, cameras[1].fx, cameras[1].fy)
    _, mask, _ = verify(cameras[0].normalized(features[0].points[[m.queryIdx for m in matches]]) * focal,
                         cameras[1].normalized(features[1].points[[m.trainIdx for m in matches]]) * focal,
                         threshold=config['geometry']['essential_threshold_px'], confidence=config['geometry']['confidence'])
    inliers = [m for m, keep in zip(matches, mask) if keep]
    saved = next(p for p in read(run / 'pairs.json') if p['pair'] == list(ids))
    if len(inliers) != saved['undistorted_verified']:
        raise ValueError('Recomputed pair verification differs from recorded experiment')
    selected = [inliers[k] for k in np.linspace(0, len(inliers) - 1, min(50, len(inliers)), dtype=int)]
    canvas = cv2.drawMatches(images[0], features[0].keypoints, images[1], features[1].keypoints,
                            selected, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Real discovered match: Temple views 1 and 30\n{len(inliers)} verified correspondences; {len(selected)} shown', fontsize=16)
    ax.axis('off')
    fig.tight_layout(); fig.savefig(output / '01_feature_matching.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    return {'pair_image_ids': ids, 'verified_correspondences': len(inliers),
            'shown_feature_indices': [[m.queryIdx, m.trainIdx] for m in selected], 'source_run': str(run.resolve())}


def cloud_figure(run, output):
    landmarks, poses = read(run / 'landmarks.json'), read(run / 'poses.json')
    xyz = np.array([p['xyz'] for p in landmarks]); colors = [p['color'] for p in landmarks]
    cameras = [p for p in poses if p['world_to_camera'] is not None]
    fig = plt.figure(figsize=(13, 6))
    for index, show_cameras in enumerate((False, True), 1):
        ax = fig.add_subplot(1, 2, index, projection='3d')
        ax.scatter(*xyz.T, c=colors, s=2, depthshade=False)
        if show_cameras:
            centers = np.array([p['center_world'] for p in cameras])
            ax.scatter(*centers.T, c='crimson', marker='^', s=35, label=f'{len(cameras)} registered cameras')
            ax.legend(fontsize=9)
        ax.set(xlabel='X (baseline units)', ylabel='Y (baseline units, down)', zlabel='Z (baseline units)',
               title='Actual sparse triangulation' if not show_cameras else 'Landmarks and estimated camera centers')
    fig.suptitle(f'Temple: {len(xyz)} reconstructed points; {len(cameras)}/{len(poses)} cameras registered', fontsize=17)
    fig.tight_layout(); fig.savefig(output / '02_sparse_scene_and_cameras.png', dpi=160, bbox_inches='tight'); plt.close(fig)


def comparison_figures(baseline, combined, output, reference):
    names = ('castle6', 'temple15', 'temple30')
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(3)
    before = [read(baseline/n/'metrics.json')['registered_images'] for n in names]
    after = [read(combined/('castle_combined' if n == 'castle6' else n+'_combined')/'metrics.json')['registered_images'] for n in names]
    ax.bar(x-.18, before, .36, label='Reproduced baseline')
    ax.bar(x+.18, after, .36, label='Discovery + attempted joint BA')
    ax.set(xticks=x, xticklabels=['Castle (6 inputs)', 'Temple (15 inputs)', 'Temple (30 inputs)'], ylabel='Registered cameras',
           title=f'Registration coverage: {after[-1]-before[-1]} additional Temple 30 cameras')
    for i, (a, b) in enumerate(zip(before, after)):
        ax.text(i-.18, a+.3, str(a), ha='center'); ax.text(i+.18, b+.3, str(b), ha='center')
    ax.set_ylim(0, 32); ax.legend(); fig.tight_layout()
    fig.savefig(output/'03_registration_before_after.png', dpi=160); plt.close(fig)
    base = read(baseline/'temple30/poses.json'); new = read(combined/'temple30_combined/poses.json')
    common = {r['name'] for r in base if r['world_to_camera'] is not None} & {r['name'] for r in new if r['world_to_camera'] is not None}
    evaluations = {label: evaluate_poses([r for r in rows if r['name'] in common], reference) for label, rows in [('baseline',base),('combined',new)]}
    write_json(output/'common_camera_accuracy.json', evaluations)
    fig, ax = plt.subplots(figsize=(11,5))
    for label, rows in [('baseline',base),('combined',new)]:
        ids=[r['image_id'] for r in rows if r['name'] in common]
        ax.plot(ids, evaluations[label]['aligned_rotation_errors_degrees'], 'o-',
                label=f"{label}: median {evaluations[label]['median_aligned_rotation_degrees']:.2f} degrees")
    direction='regression' if evaluations['combined']['median_aligned_rotation_degrees'] > evaluations['baseline']['median_aligned_rotation_degrees'] else 'improvement'
    ax.set(xlabel='Original image ID', ylabel='Aligned rotation error (degrees)',
           title=f'Same {len(common)} cameras: measured rotation {direction}')
    ax.legend(); fig.tight_layout(); fig.savefig(output/'04_common_camera_pose_accuracy.png', dpi=160);plt.close(fig)


def graph_figure(baseline, run, output):
    angles=np.linspace(0,2*np.pi,30,endpoint=False)
    xy=np.c_[np.cos(angles),np.sin(angles)]
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,directory,label in [(axes[0],baseline/'temple30','Baseline'),(axes[1],run,'Bounded discovery')]:
        rows=read(directory/'pairs.json')
        groups=components(range(30),[tuple(p['pair']) for p in rows if p['undistorted_verified']>=20])
        for edge in rows:
            if edge['undistorted_verified']<20:continue
            a,b=edge['pair'];extra=edge.get('discovery_distance') is not None
            ax.plot(xy[[a,b],0],xy[[a,b],1],color='crimson' if extra else 'steelblue',alpha=.85 if extra else .25,lw=2 if extra else .7)
        ax.scatter(*xy.T,s=50,color='white',edgecolors='black',zorder=3)
        for i,(x,y) in enumerate(xy):ax.text(x*1.1,y*1.1,str(i+1),ha='center',va='center',fontsize=8)
        ax.set_title(f'{label}: {len(groups)} components');ax.set_aspect('equal');ax.axis('off')
    count=next(p['undistorted_verified'] for p in read(run/'pairs.json') if p['pair']==[0,29])
    fig.suptitle(f'Verified image matching graphs; circular layout is not camera geometry\nRed: discovered 1–30 edge ({count} verified matches)',fontsize=15)
    fig.tight_layout();fig.savefig(output/'05_matching_graph.png',dpi=160,bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('outputs/recovery-baseline'))
    parser.add_argument('--experiments', type=Path, default=Path('outputs/recovery-final'))
    parser.add_argument('--output', type=Path, default=Path('outputs/progress-presentation'))
    args=parser.parse_args();args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    reference=read('datasets/temple/reference_poses.json')
    matching=feature_figure(args.experiments/'temple30_combined',args.output)
    cloud_figure(args.experiments/'temple30_combined',args.output)
    comparison_figures(args.baseline,args.experiments,args.output,reference)
    graph_figure(args.baseline,args.experiments/'temple30_combined',args.output)
    rows=[]
    all_results=read(args.baseline/'results.json')+read(args.experiments/'results.json')
    for result in all_results:
        summary=result['summary'];metrics=result['metrics']
        rows.append({k:summary.get(k) for k in ['case','registered','images','landmarks','median_reprojection_px','median_aligned_rotation_deg',
                                              'median_translation_direction_deg','runtime_seconds','peak_memory_bytes']})
        rows[-1]['tracks']=metrics.get('tracks',{}).get('tracks')
        rows[-1]['track_observations']=metrics.get('tracks',{}).get('observations')
        rows[-1]['matching_seconds']=metrics.get('stage_seconds',{}).get('matching')
    with (args.output/'benchmark_comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['| Experiment | Registered | Points | Median image error (px) | Median aligned rotation (deg) | Runtime (s) |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['case'] not in ['castle6','temple15','temple30','castle_combined','temple15_combined','temple30_combined']:continue
        rotation='unavailable' if row['median_aligned_rotation_deg'] is None else f"{row['median_aligned_rotation_deg']:.2f}"
        lines.append(f"| {row['case']} | {row['registered']}/{row['images']} | {row['landmarks']} | {row['median_reprojection_px']:.3f} | {rotation} | {row['runtime_seconds']:.2f} |")
    (args.output/'presentation_table.md').write_text('\n'.join(lines)+'\n\nMore registered cameras did not imply better rotation accuracy.\n',encoding='utf-8')
    write_json(args.output/'provenance.json',{'purpose':'Technical evidence for November 9, 2026; no slides or paper',
        'baseline':str(args.baseline.resolve()),'experiments':str(args.experiments.resolve()),'matching':matching,
        'source_fingerprint':read(args.experiments/'execution.json')['source_sha256'],
        'figures':{'01':'Recomputed verified pair; checked against saved pairs.json and source hashes',
                   '02':'Actual combined landmarks.json and poses.json', '03':'Baseline/combined registered camera counts',
                   '04':'Same common camera names, independently Sim(3) aligned to reference after reconstruction',
                   '05':'Saved verified pairs only; circular index layout is schematic'},
        'note':'All estimated points/cameras are exported reconstruction data; reference poses used only for evaluation. No full-registration or improved-rotation claim.'})
    (args.output/'README.md').write_text('Real experiment evidence. Read provenance.json and benchmark_comparison.csv.\nRegistration improves on Temple30, but common-camera rotation accuracy worsens; joint BA regresses Temple15 rotation.\nDense reconstruction remains not ready. Figures contain actual reconstructed points and estimated poses.\n',encoding='utf-8')
    print(args.output.resolve())


if __name__=='__main__':
    main()
