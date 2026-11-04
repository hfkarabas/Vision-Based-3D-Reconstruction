"""Geometry diagnostics from actual exported observations, in arbitrary SfM units."""
import json
from pathlib import Path
import numpy as np


def percentile(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {str(q): float(np.percentile(a, q)) for q in (0, 5, 25, 50, 75, 95, 100)} if len(a) else {}


def quality_mask(landmarks, min_views=3, min_angle=3.0, max_error=1.0):
    """Diagnostic subset only: no coordinate modification or automatic export pruning."""
    if min_views < 2 or min_angle < 0 or max_error <= 0:
        raise ValueError("Invalid diagnostic quality limits")
    return np.array([len(l['observations']) >= min_views
        and l['max_triangulation_angle_degrees'] >= min_angle
        and all(np.isfinite(e) and e <= max_error for e in l['reprojection_errors_px'].values())
        and all(np.isfinite(z) and z > 0 for z in l['depths'].values())
        and np.isfinite(l['xyz']).all() for l in landmarks], dtype=bool)


def summarize(directory):
    directory = Path(directory)
    landmarks = json.loads((directory / 'landmarks.json').read_text())
    poses = [p for p in json.loads((directory / 'poses.json').read_text()) if p['status'] == 'registered']
    metrics = json.loads((directory / 'metrics.json').read_text())
    xyz = np.asarray([l['xyz'] for l in landmarks], dtype=float).reshape(-1, 3)
    errors = [e for l in landmarks for e in l['reprojection_errors_px'].values()]
    depth = [z for l in landmarks for z in l['depths'].values()]
    angles = [l['max_triangulation_angle_degrees'] for l in landmarks]
    support = np.array([len(l['observations']) for l in landmarks])
    centers = np.array([p['center_world'] for p in poses])
    directions = np.array([np.asarray(p['world_to_camera'])[:3, :3].T @ [0, 0, 1] for p in poses])
    pairs = []
    for i, a in enumerate(poses):
        for j in range(i + 1, len(poses)):
            pairs.append({'images': [a['image_id'], poses[j]['image_id']],
                'baseline_world_units': float(np.linalg.norm(centers[i] - centers[j])),
                'view_direction_angle_degrees': float(np.rad2deg(np.arccos(np.clip(directions[i] @ directions[j], -1, 1))))})
    # Robust radial diagnostic, not a claim that these points are incorrect.
    radius = np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1) if len(xyz) else np.array([])
    if len(radius):
        q1, q3 = np.percentile(radius, [25, 75])
        tail = radius > q3 + 3 * (q3 - q1)
        singular = np.linalg.svd(xyz - np.median(xyz, axis=0), compute_uv=False)
    else:
        tail, singular = np.array([], bool), np.zeros(3)
    return {'registered': len(poses), 'landmarks': len(landmarks), 'status': metrics['status'],
        'runtime_seconds': metrics['runtime_seconds'], 'peak_memory_bytes': metrics.get('peak_memory_bytes'),
        'median_reprojection_px': float(np.median(errors)) if errors else None,
        'rms_reprojection_px': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        'error_px_percentiles': percentile(errors), 'angle_degrees_percentiles': percentile(angles),
        'observed_depth_world_units_percentiles': percentile(depth), 'radius_percentiles': percentile(radius),
        'support_histogram': {str(k): int((support == k).sum()) for k in sorted(set(support.tolist()))},
        'radial_tail_3iqr': int(tail.sum()), 'nonpositive_depth_observations': sum(z <= 0 for z in depth),
        'spatial_singular_values': singular.tolist(), 'camera_pairs': pairs,
        'diagnostic_support3_angle3_error1_points': int(quality_mask(landmarks).sum()),
        'point_refinement': metrics.get('point_refinement', {'status': 'disabled'}),
        'bundle_statuses': [b['status'] for b in metrics.get('bundle_adjustment', [])]}
