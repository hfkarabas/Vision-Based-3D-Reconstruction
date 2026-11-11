"""Compare native and recentered stereo on identical verified AirZoo pairs."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_airzoo_geometry import percentiles, read
from vision3d.geometry.camera import parse_camera
from vision3d.io import load_image
from vision3d.reconstruction.multiview import make_depth, plan_depth, relative_pose
from vision3d.reconstruction.surface_validation import surface_views


def project(matrix, xyz):
    homogeneous = np.column_stack((xyz, np.ones(len(xyz)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def audit(sparse, pairs, output):
    output.mkdir(parents=True, exist_ok=False)
    views = {view.id: view for view in surface_views(sparse)}
    cameras = {row["image_id"]: parse_camera(row["processed"]) for row in read(sparse / "intrinsics.json")}
    landmarks = read(sparse / "landmarks.json")
    records = []
    with np.load(sparse / "features.npz") as features, np.load(sparse / "verified_pairs.npz") as matches:
        for i, j in pairs:
            views_pair = [views[i], views[j]]
            size = (900, round(900 * views[i].camera.height / views[i].camera.width))
            dense_cameras = [view.camera.resized(*size) for view in views_pair]
            images = [cv2.resize(load_image(view.source).image, size, interpolation=cv2.INTER_AREA)
                      for view in views_pair]
            key = f"{min(i, j)}_{max(i, j)}"
            indices = matches[key] if key in matches else np.empty((0, 2), dtype=int)
            if i > j:
                indices = indices[:, ::-1]
            observed = [features[f"image_{k}_points"][ids] for k, ids in zip((i, j), indices.T, strict=True)]
            xyz = np.array([p["xyz"] for p in landmarks if str(i) in p["observations"]
                            and str(j) in p["observations"]])
            a, b = (view.pose for view in views_pair)
            fig, panels = plt.subplots(1, 2, figsize=(8, 3.3), layout="constrained")
            depths = []
            for recenter in (False, True):
                plan = None
                row = {"reference": i, "source": j, "recentered": recenter, "verified_matches": len(indices),
                       "shared_landmarks": len(xyz)}
                records.append(row)
                try:
                    plan = plan_depth(a, b, *dense_cameras, xyz, recenter=recenter)
                    (r1, r2), (p1, p2) = plan.rotations, plan.projections
                except (ValueError, cv2.error) as exc:
                    row["rejected"] = str(exc)
                    if recenter:
                        depths.append(None)
                        continue
                    r, t = relative_pose(a, b)
                    r1, r2, p1, p2, *_ = cv2.stereoRectify(
                        dense_cameras[0].K, dense_cameras[0].D, dense_cameras[1].K, dense_cameras[1].D,
                        size, r, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
                axis = int(abs(p2[1, 3]) > abs(p2[0, 3]))
                measured = [
                    cv2.undistortPoints(points.reshape(-1, 1, 2), camera.K, camera.D, R=r, P=p[:, :3]).reshape(-1, 2)
                    if len(points) else np.empty((0, 2)) for points, camera, r, p in
                    zip(observed, (cameras[i], cameras[j]), (r1, r2), (p1, p2), strict=True)
                ]
                within = np.logical_and.reduce([((uv >= 0) & (uv < size)).all(axis=1) for uv in measured])
                maps = [cv2.initUndistortRectifyMap(c.K, c.D, r, p[:, :3], size, cv2.CV_32FC1)
                        for c, r, p in zip(dense_cameras, (r1, r2), (p1, p2), strict=True)]
                valid = np.logical_and.reduce([(x >= 2) & (y >= 2) & (x < size[0] - 3) & (y < size[1] - 3)
                                              for x, y in maps])
                row.update(overlap_fraction=float(valid.mean()), verified_matches_in_both_canvases=int(within.sum()),
                           epipolar_px=percentiles(abs(measured[0][within, 1 - axis] - measured[1][within, 1 - axis])),
                           principal_point_disparity_px=float(p1[axis, 2] - p2[axis, 2]))
                if plan is None:
                    depths.append(None)
                    continue
                local = (xyz @ a.rotation.T + a.translation) @ r1.T
                left, right = project(p1, local), project(p2, local)
                disparity = left[:, axis] - right[:, axis]
                reconstructed = np.column_stack((left, disparity, np.ones(len(xyz)))) @ plan.disparity_to_depth.T
                error = np.linalg.norm(reconstructed[:, :3] / reconstructed[:, 3:] - local, axis=1)
                row["known_sparse_point_q_error"] = percentiles(error)
                try:
                    result = make_depth(i, j, a, b, *dense_cameras, *images, xyz, plan=plan)
                except (ValueError, cv2.error) as exc:
                    row["rejected"] = str(exc)
                    depths.append(None)
                    continue
                d = result.depth.T if result.transposed else result.depth
                row["positive_depth_fraction"] = float((d > 0).mean())
                row["depth_percentiles"] = percentiles(d[d > 0])
                neighbors = (d[:, :-1] > 0) & (d[:, 1:] > 0)
                jump = abs(d[:, :-1][neighbors] - d[:, 1:][neighbors])
                row["horizontal_depth_jump_percentiles"] = percentiles(jump)
                depths.append(d)
            positive = [depth[depth > 0] for depth in depths if depth is not None]
            limits = np.percentile(np.concatenate(positive), [2, 98]) if positive else [0, 1]
            for ax, name, depth, row in zip(panels, ("Native", "Recentered"), depths, records[-2:], strict=True):
                ax.set_title(name)
                ax.axis("off")
                if depth is None:
                    ax.text(.5, .5, row.get("rejected", "Unavailable"), ha="center", va="center", wrap=True,
                            transform=ax.transAxes, fontsize=8)
                else:
                    ax.imshow(np.ma.masked_less_equal(depth, 0), cmap="viridis", vmin=limits[0], vmax=limits[1])
            fig.suptitle(f"Fixed stereo pair {i} → {j} · common depth scale")
            fig.savefig(output / f"rectification-{i:04d}-{j:04d}.png", dpi=125)
            plt.close(fig)
    (output / "rectification.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--pair", nargs=2, type=int, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.sparse, args.pair, args.output)
