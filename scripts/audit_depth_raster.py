"""Visualize stored stereo support and original-image reprojections without re-running MVS."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_airzoo_geometry import original_pixels, read
from vision3d.io import load_image
from vision3d.reconstruction.surface_validation import surface_views


def audit(sparse, dense, output):
    output.mkdir(parents=True, exist_ok=False)
    root = dense / "depth-observations"
    pairs = sorted(read(root / "manifest.json")["pairs"], key=lambda row: row["accepted"])
    views = {view.id: view for view in surface_views(sparse)}
    results = []
    # Choose contrasting maps by recorded support, prior to looking at the render.
    for record in (pairs[0], pairs[-1]):
        i, j = record["reference"], record["source"]
        with np.load(root / record["file"]) as data:
            depth, accepted = data["depth"], data["accepted_raster"]
            if record["transposed"]:
                depth, accepted = depth.T, accepted.T
            p1, p2 = data["projection_a"], data["projection_b"]
            axis = int(record["transposed"])
            valid_depth = depth > 0
            disparity = np.zeros_like(depth)
            disparity[valid_depth] = p1[axis, 2] - p2[axis, 2] - p2[axis, 3] / depth[valid_depth]
            world = data["accepted_xyz"][::max(1, len(data["accepted_xyz"]) // 1000)]
            rectified = []
            for key, projection in ((i, p1), (j, p2)):
                view = views[key]
                rgb = load_image(view.source).image
                rect = data["rotation"] @ view.pose.rotation.T
                mx, my = cv2.initUndistortRectifyMap(view.camera.K, view.camera.D, rect, projection[:, :3],
                                                      depth.shape[::-1], cv2.CV_32FC1)
                bounds = (mx >= 2) & (my >= 2) & (mx < rgb.shape[1] - 3) & (my < rgb.shape[0] - 3)
                rectified.append((cv2.remap(rgb, mx, my, cv2.INTER_LINEAR)[:, :, ::-1], bounds))
            source = load_image(views[j].source).image
            source_preview = cv2.resize(source, (800, round(800 * source.shape[0] / source.shape[1])),
                                        interpolation=cv2.INTER_AREA)[:, :, ::-1]
            projected = (original_pixels(world, views[j]) if len(world) else np.empty((0, 2)))
            projected *= [source_preview.shape[1] / source.shape[1], source_preview.shape[0] / source.shape[0]]
        admitted = np.zeros((*depth.shape, 3), np.uint8)
        admitted[rectified[0][1]] = [43, 48, 56]
        admitted[valid_depth] = [235, 173, 76]
        admitted[accepted] = [43, 188, 150]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
        images = (rectified[0][0], rectified[1][0], source_preview, np.ma.masked_where(~valid_depth, disparity),
                  np.ma.masked_where(~valid_depth, depth), admitted)
        labels = ("Rectified reference", "Rectified source", "Accepted samples in source photograph",
                  "Disparity implied by stored depth", "Reciprocal stereo depth", "Valid canvas / reciprocal / corroborated")
        for ax, image, title in zip(axes.ravel(), images, labels, strict=True):
            ax.imshow(image, cmap="viridis")
            ax.set_title(title)
            ax.set_axis_off()
        axes[0, 2].scatter(*projected.T, s=2, color="#28e9c5")
        axes[0, 2].set_xlim(0, source_preview.shape[1])
        axes[0, 2].set_ylim(source_preview.shape[0], 0)
        fig.suptitle(f"Reference {i} / source {j}")
        fig.savefig(output / f"{i:04d}-{j:04d}.png", dpi=125)
        plt.close(fig)
        results.append({"reference": i, "source": j, "image_pixels": int(depth.size),
                        "valid_reference_canvas": int(rectified[0][1].sum()), "stored_reciprocal_pixels": int(valid_depth.sum()),
                        "corroborated_pixels": int(accepted.sum()),
                        "reciprocal_rejected_by_corroboration": int((valid_depth & ~accepted).sum()),
                        "reprojected_samples": len(projected)})
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sparse", "dense", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    audit(args.sparse, args.dense, args.output)
