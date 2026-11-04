"""Reproducible bounded sparse-quality ablations and genuine exported-data figures."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from vision3d.evaluation.sparse import summarize, quality_mask
from vision3d.evaluation.poses import evaluate_poses

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    'castle_baseline': ('castle', []),
    'castle_features12000': ('castle', ['--max-features', '12000']),
    'castle_features8000': ('castle', ['--max-features', '8000']),
    'castle_resolution2000': ('castle', ['--max-dimension', '2000']),
    'castle_ratio070': ('castle', ['--ratio', '0.70']),
    'castle_ba1200': ('castle', ['--quality-config', 'bundle1200.json']),
    'castle_refined': ('castle', ['--refine-points']),
    'castle_features_refined': ('castle', ['--max-features', '12000', '--refine-points']),
    'castle_delivery': ('castle', ['--refine-points']),
    'temple_baseline': ('temple', []),
    'temple_refined': ('temple', ['--refine-points']),
    'temple_features12000': ('temple', ['--max-features', '12000']),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def figures(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from vision3d.io import load_image
    from vision3d.preprocessing import preprocess
    from vision3d.geometry.camera import Camera
    from vision3d.geometry.two_view import project
    directory = Path(directory)
    out = directory / 'quality'
    out.mkdir(exist_ok=True)
    points = read(directory / 'landmarks.json')
    poses = [p for p in read(directory / 'poses.json') if p['status'] == 'registered']
    manifest = read(directory / 'manifest.json')
    xyz = np.array([p['xyz'] for p in points])
    colors = np.array([p['color'] for p in points])
    centers = np.array([p['center_world'] for p in poses])
    rotation = np.array(poses[0]['world_to_camera'])[:3, :3]
    origin = np.median(xyz, axis=0)
    aligned = (xyz - origin) @ rotation.T
    cameras = (centers - origin) @ rotation.T
    keep = quality_mask(points)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, mask, title in zip(axes, [np.ones(len(xyz), bool), keep], ['All exported landmarks', 'Diagnostic subset: >=3 views, angle >=3°, max error <=1 px']):
        ax.scatter(aligned[mask, 0], -aligned[mask, 1], c=colors[mask], s=3)
        ax.set(xlabel='Camera-aligned X (arbitrary units)', ylabel='Camera-aligned up (arbitrary units)', title=f'{title}\n{mask.sum()} / {len(xyz)} points')
        ax.set_aspect('equal'); ax.grid(alpha=.2)
    # Identical limits explicitly avoid a tighter crop masquerading as more geometry.
    axes[1].set_xlim(axes[0].get_xlim()); axes[1].set_ylim(axes[0].get_ylim())
    fig.tight_layout(); fig.savefig(out / 'cloud_and_subset.png', dpi=180); plt.close(fig)
    fig = plt.figure(figsize=(11, 7)); ax = fig.add_subplot(projection='3d')
    ax.scatter(*xyz.T, c=colors, s=2); ax.scatter(*centers.T, c='black', marker='^', label='Estimated camera centers')
    directions = np.array([np.asarray(p['world_to_camera'])[:3, :3].T @ [0, 0, 1] for p in poses])
    ax.quiver(*centers.T, *directions.T, length=1, color='black')
    for p, c in zip(poses, centers): ax.text(*c, str(p['image_id']))
    ax.set(xlabel='World X', ylabel='World Y', zlabel='World Z', title='Actual sparse landmarks, camera centers and forward directions\nArbitrary monocular scale; arrows length 1 world unit'); ax.legend(); ax.view_init(25, -65)
    fig.tight_layout(); fig.savefig(out / 'cameras_and_cloud.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    arrays = [[v for p in points for v in p['depths'].values()], [p['max_triangulation_angle_degrees'] for p in points],
              [v for p in points for v in p['reprojection_errors_px'].values()], [len(p['observations']) for p in points]]
    labels = ['Observation depth (world units)', 'Maximum triangulation angle (degrees)', 'Observation reprojection error (px)', 'Accepted observation count per landmark']
    for ax, values, label in zip(axes.flat, arrays, labels): ax.hist(values, bins=30, color='gray', edgecolor='white'); ax.set(xlabel=label, ylabel='Count')
    fig.tight_layout(); fig.savefig(out / 'quality_distributions.png', dpi=180); plt.close(fig)
    intrinsics = {p['image_id']: p['processed'] for p in read(directory / 'intrinsics.json')}
    features = np.load(directory / 'features.npz', allow_pickle=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 9)); coverage = []
    for ax, item in zip(axes.flat, manifest['inputs'][:6]):
        i = item['image_id']; image, _ = preprocess(load_image(item['source']).image, manifest['config']['max_dimension'])
        pixels = features[f'image_{i}_points']; used = [p['observations'][str(i)] for p in points if str(i) in p['observations']]
        ax.imshow(image[..., ::-1]); ax.scatter(pixels[:, 0], pixels[:, 1], s=.4, c='cyan', alpha=.45)
        if used: ax.scatter(pixels[used, 0], pixels[used, 1], s=1, c='yellow', alpha=.6)
        hist, _, _ = np.histogram2d(pixels[:, 0], pixels[:, 1], bins=8, range=[[0, image.shape[1]], [0, image.shape[0]]])
        coverage.append({'image_id': i, 'features': len(pixels), 'landmark_observations': len(used), 'occupied_8x8_cells': int((hist > 0).sum()), 'cell_counts': hist.tolist()})
        ax.set_title(f'{item["name"]}: {len(pixels)} features, {len(used)} 3D observations', fontsize=9); ax.axis('off')
    fig.suptitle('Actual feature coverage: cyan = detected, yellow = exported landmark observations'); fig.tight_layout(rect=[0, 0, 1, .95], h_pad=2); fig.savefig(out / 'feature_coverage.png', dpi=180); plt.close(fig)
    write(out / 'feature_coverage.json', coverage)
    # Overlay measured and reprojected observations in one genuine image.
    pose = poses[0]; i = pose['image_id']; item = next(x for x in manifest['inputs'] if x['image_id'] == i)
    image, _ = preprocess(load_image(item['source']).image, manifest['config']['max_dimension'])
    selected = [p for p in points if str(i) in p['observations']]
    observed = features[f'image_{i}_points'][[p['observations'][str(i)] for p in selected]]
    matrix = np.asarray(pose['world_to_camera']); model = dict(intrinsics[i]); model.pop('K', None)
    predicted = project(np.array([p['xyz'] for p in selected]), Camera(**model), matrix[:3, :3], matrix[:3, 3])
    fig, ax = plt.subplots(figsize=(10, 7)); ax.imshow(image[..., ::-1]); ax.scatter(*observed.T, s=4, c='yellow', label='Measured')
    ax.quiver(*observed.T, *(predicted-observed).T, color='red', angles='xy', scale_units='xy', scale=1)
    ax.set_title('Actual reprojection residuals (red arrows at 1:1 pixel scale)'); ax.legend(); fig.tight_layout(); fig.savefig(out / 'reprojection.png', dpi=180); plt.close(fig)
    write(out / 'summary.json', summarize(directory))


def comparison_figures(output):
    """Identical view, scale, axes and point size for genuine before/after data."""
    import matplotlib.pyplot as plt
    a, b = output/'castle_baseline', output/'castle_refined'
    if not (a/'landmarks.json').is_file() or not (b/'landmarks.json').is_file(): return
    first, second = read(a/'landmarks.json'), read(b/'landmarks.json')
    pose = next(p for p in read(a/'poses.json') if p['status']=='registered')
    rotation = np.asarray(pose['world_to_camera'])[:3, :3]
    origin = np.median([p['xyz'] for p in first], axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for col, (landmarks, title) in enumerate([(first, 'Baseline'), (second, 'Fixed-camera point refinement')]):
        xyz = (np.array([p['xyz'] for p in landmarks])-origin) @ rotation.T
        colors = np.array([p['color'] for p in landmarks])
        axes[0, col].scatter(xyz[:, 0], -xyz[:, 1], c=colors, s=3)
        axes[0, col].set(title=title+' — all 3,915 points', xlabel='Camera-aligned X', ylabel='Camera-aligned up')
        axes[1, col].scatter(xyz[:, 0], xyz[:, 2], c=colors, s=3)
        axes[1, col].set(xlabel='Camera-aligned X', ylabel='Camera-aligned depth')
    for row in axes:
        limits = row[0].get_xlim(), row[0].get_ylim()
        for ax in row: ax.set_xlim(limits[0]); ax.set_ylim(limits[1]); ax.set_aspect('equal'); ax.grid(alpha=.2)
    fig.suptitle('Real before/after geometry — same observations, colors, scale and display settings\nSparse facade remains incomplete; no interpolated or filtered points')
    fig.tight_layout(); fig.savefig(output/'castle_before_after.png', dpi=180); plt.close(fig)
    # Identity-matched displacement diagnostics and exact camera invariance.
    by_id = {p['id']: p for p in second}; shifts = [np.linalg.norm(np.array(p['xyz'])-by_id[p['id']]['xyz']) for p in first]
    from vision3d.evaluation.sparse import percentile
    poses_a, poses_b = read(a/'poses.json'), read(b/'poses.json')
    write(output/'refinement_comparison.json', {'point_shift_world_units_percentiles': percentile(shifts),
        'max_camera_matrix_difference': max(float(np.max(np.abs(np.array(x['world_to_camera'])-np.array(y['world_to_camera'])))) for x,y in zip(poses_a,poses_b)),
        'same_landmark_ids': set(p['id'] for p in first)==set(by_id),
        'same_observations': all(p['observations']==by_id[p['id']]['observations'] for p in first)})


def matching_diagnostic(output):
    import cv2
    from vision3d.io import load_image
    from vision3d.preprocessing import preprocess
    from vision3d.features import extract
    from vision3d.features.matching import match
    from vision3d.geometry import verify
    from vision3d.geometry.camera import load_cameras
    baseline = output/'castle_baseline'
    if not (baseline/'metrics.json').is_file(): return
    metrics, manifest = read(baseline/'metrics.json'), read(baseline/'manifest.json')
    i, j = metrics['initial_pair']; inputs = {p['image_id']: p for p in manifest['inputs']}
    images = [preprocess(load_image(inputs[k]['source']).image, 1600)[0] for k in (i,j)]
    features = [extract(im, 'sift', 6000) for im in images]
    model = load_cameras(ROOT/'configs/sceaux.json')[0]
    cameras = [model.resized(im.shape[1], im.shape[0]) for im in images]
    result = match(*features, ratio=.75, mutual=True, geometric=False)
    fi = np.array([m.queryIdx for m in result.consistent]); fj = np.array([m.trainIdx for m in result.consistent])
    focal = min(c.fx for c in cameras)
    cv2.setRNGSeed((i*1009+j*9176)%2147483647)
    _, mask, status = verify(cameras[0].normalized(features[0].points[fi])*focal,
        cameras[1].normalized(features[1].points[fj])*focal, threshold=1., confidence=.999)
    verified = [m for m,k in zip(result.consistent,mask) if k]
    shown = [verified[k] for k in np.linspace(0,len(verified)-1,min(100,len(verified)),dtype=int)]
    canvas = cv2.drawMatches(images[0], features[0].keypoints, images[1], features[1].keypoints, shown, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imencode('.jpg', canvas)[1].tofile(output/'castle_matching.jpg')
    write(output/'castle_matching.json', {'pair':[i,j], 'verification_status':status, 'verified_matches':len(verified),
        'displayed':len(shown), 'source':'Recomputed identical stable baseline feature/matching settings',
        'shown_feature_indices':[[m.queryIdx,m.trainIdx] for m in shown]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/reconstruction-quality')
    parser.add_argument('--case', choices=list(CASES), action='append')
    parser.add_argument('--summarize', action='store_true', help='Read completed runs and generate figures; no processing or overwrite of reconstruction files')
    args = parser.parse_args(); args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(args.output / 'matplotlib-cache')
    if not args.summarize:
        for name in args.case or CASES:
            scene, extra = CASES[name]; output = args.output / name
            if output.exists(): raise ValueError(f'Refusing to overwrite {output}')
            if scene == 'castle': folder, camera = ROOT/'datasets/sceaux/images', ROOT/'configs/sceaux.json'
            else: folder, camera = ROOT/'datasets/temple/images30', ROOT/'datasets/temple/cameras.json'
            extra = list(extra)
            if 'bundle1200.json' in extra:
                config = args.output / 'bundle1200.json'; write(config, {'bundle': {'max_cameras': 6, 'max_points': 1200}})
                extra[extra.index('bundle1200.json')] = str(config)
            command = [sys.executable, '-m', 'vision3d.cli', 'reconstruct-sequence', '--input', str(folder), '--camera', str(camera), '--local-ba', '--output', str(output), *extra]
            fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src').rglob('*.py')))).hexdigest()
            write(args.output / f'{name}_execution.json', {'command': command, 'source_sha256': fingerprint, 'seed': 0, 'feature_cache': None})
            with (args.output / f'{name}.log').open('w', encoding='utf-8') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            print(name, 'exit', result.returncode, flush=True)
            if result.returncode not in (0, 3): raise RuntimeError(f'See {name}.log')
    rows = []
    for name in CASES:
        folder = args.output / name
        if not (folder/'landmarks.json').is_file(): continue
        stats = summarize(folder); row = {'case': name, **stats}
        if name.startswith('temple'):
            evaluation = evaluate_poses(read(folder/'poses.json'), read(ROOT/'datasets/temple/reference_poses.json'))
            write(folder/'quality_pose_evaluation.json', evaluation)
            row['median_aligned_rotation_deg'] = evaluation.get('median_aligned_rotation_degrees')
            row['median_translation_direction_deg'] = evaluation.get('median_translation_direction_degrees')
        rows.append(row)
        if args.summarize: figures(folder)
    write(args.output/'comparison.json', rows)
    fields = ['case', 'registered', 'landmarks', 'median_reprojection_px', 'rms_reprojection_px', 'runtime_seconds', 'peak_memory_bytes', 'radial_tail_3iqr', 'median_aligned_rotation_deg', 'median_translation_direction_deg']
    with (args.output/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    if args.summarize:
        comparison_figures(args.output)
        matching_diagnostic(args.output)


if __name__ == '__main__': main()
