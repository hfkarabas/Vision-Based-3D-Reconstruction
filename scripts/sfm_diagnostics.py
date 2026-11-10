"""Export reconstruction diagnostics without changing the source photographs."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from vision3d.io import load_image


def export_diagnostics(run, output, keypoints=False, pair=None):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def read(name):
        return json.loads((run / name).read_text(encoding="utf-8"))

    manifest, pairs, metrics = read("manifest.json"), read("pairs.json"), read("metrics.json")
    inputs = manifest["inputs"]
    ids = [item["image_id"] for item in inputs]
    shape = (len(ids), len(ids))
    counts = np.full(shape, np.nan)
    for record in pairs:
        a, b = record["pair"]
        counts[a, b] = counts[b, a] = record["undistorted_verified"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    plot = ax.imshow(np.ma.masked_invalid(counts), interpolation="nearest", cmap="viridis")
    ax.set_facecolor("#e5e5e5")
    ax.set(title="Geometrically verified inlier counts (gray = not evaluated)", xlabel="Image ID", ylabel="Image ID")
    fig.colorbar(plot, ax=ax)
    fig.savefig(output / "matching.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    poses_file = run / "poses.json"
    if poses_file.exists():
        poses = read("poses.json")
        intrinsics = {item["image_id"]: item["original"] for item in read("intrinsics.json")}
        with (output / "registration.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["image_id", "name", "status", "attempts", "pnp_attempted", "correspondences",
                      "candidate_coverage", "ransac_inliers", "inliers", "inlier_ratio",
                      "coverage", "median_px", "reason", "intrinsics", "focal_px"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for pose in poses:
                attempts = pose["attempts"]
                last = attempts[-1] if attempts else {}
                camera = intrinsics[pose["image_id"]]
                writer.writerow(dict(image_id=pose["image_id"], name=pose["name"], status=pose["status"],
                                     attempts=len(attempts), correspondences=last.get("correspondences"),
                                     pnp_attempted=last.get("pnp_attempted"),
                                     candidate_coverage=last.get("candidate_image_coverage"),
                                     ransac_inliers=last.get("ransac_inliers"),
                                     inliers=last.get("inliers"), inlier_ratio=last.get("inlier_ratio"),
                                     coverage=last.get("image_coverage"), median_px=last.get("median_error_px"),
                                     reason=last.get("reason", "seed" if pose["status"] == "registered" else "not attempted"),
                                     intrinsics=camera["source"], focal_px=camera["fx"]))
    if keypoints and (run / "features.npz").is_file():
        tiles, originals = [], []
        with np.load(run / "features.npz", allow_pickle=False) as features:
            with (output / "feature_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["image_id", "keypoints", "occupied_grid_cells_8x8", "convex_hull_fraction"])
                for item in inputs:
                    i = item["image_id"]
                    image = load_image(item["source"]).image
                    width, height = item["resize"]["processed_size"]
                    image = cv2.resize(image, (240, 240), interpolation=cv2.INTER_AREA)
                    points = features[f"image_{i}_points"]
                    grid = np.floor(points / [width, height] * 8).astype(int)
                    hull = cv2.contourArea(cv2.convexHull(points)) if len(points) >= 3 else 0
                    writer.writerow([i, len(points), len(np.unique(grid, axis=0)), hull / (width * height)])
                    cv2.putText(image, f"{i}: {len(points)}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
                    originals.append(image.copy())
                    for x, y in (points / [width, height] * 240).astype(int):
                        cv2.circle(image, (x, y), 1, (70, 220, 50), -1)
                    tiles.append(image)
        for name, images in (("keypoints", tiles), ("original_views", originals)):
            images.extend([np.zeros_like(images[0])] * (-len(images) % 6))
            sheet = np.vstack([np.hstack(images[i:i + 6]) for i in range(0, len(images), 6)])
            cv2.imwrite(str(output / f"{name}.jpg"), sheet)
    for a, b in pair or []:
        a, b = sorted((a, b))
        with np.load(run / "verified_pairs.npz", allow_pickle=False) as verified:
            correspondences = verified[f"{a}_{b}"]
        previews, endpoints = [], []
        with np.load(run / "features.npz", allow_pickle=False) as features:
            for i, indices in ((a, correspondences[:, 0]), (b, correspondences[:, 1])):
                item = inputs[i]
                width, height = item["resize"]["processed_size"]
                preview_size = (round(width * 500 / height), 500)
                image = load_image(item["source"]).image
                previews.append(cv2.resize(image, preview_size, interpolation=cv2.INTER_AREA))
                points = features[f"image_{i}_points"][indices]
                endpoints.append(np.rint(points / [width, height] * preview_size).astype(int))
        endpoints[1][:, 0] += previews[0].shape[1]
        canvas = np.hstack(previews)
        for j in np.linspace(0, len(correspondences) - 1, min(100, len(correspondences)), dtype=int):
            cv2.line(canvas, tuple(endpoints[0][j]), tuple(endpoints[1][j]), (65, 220, 75), 1, cv2.LINE_AA)
        cv2.imwrite(str(output / f"pair-{a}-{b}.jpg"), canvas)
    graph = metrics["tracks"]["graph"]
    (output / "summary.json").write_text(json.dumps({
        "registered": metrics["registered_images"], "input": metrics["input_images"],
        "graph": graph, "track_lengths": metrics["tracks"]["length_distribution"],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keypoints", action="store_true")
    parser.add_argument("--pair", action="append", nargs=2, type=int, metavar=("IMAGE_A", "IMAGE_B"))
    export_diagnostics(**vars(parser.parse_args()))
