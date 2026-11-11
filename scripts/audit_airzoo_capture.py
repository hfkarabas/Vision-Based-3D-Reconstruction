"""Show actual UAV inputs, PPK camera stations and verified image connectivity."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from analyze_airzoo_geometry import station_metadata
from vision3d.io import load_image


def audit(sparse, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs, views, stations, positions, attitudes, _, _ = station_metadata(sparse)
    geo = np.stack([positions[row["image_id"]] for row in inputs])
    xy = (geo[:, [1, 0]] - geo[0, [1, 0]]) * [111195 * np.cos(np.radians(geo[0, 0])), 111195]
    ids = [row["image_id"] for row in inputs]
    index = {key: i for i, key in enumerate(ids)}
    width, height = 200, 168
    sheet = Image.new("RGB", (width * 6, height * ((len(inputs) + 5) // 6)), "#edf0f4")
    draw = ImageDraw.Draw(sheet)
    records = []
    for i, row in enumerate(inputs):
        key, path = row["image_id"], Path(row["source"])
        frame = load_image(path).image
        thumb = ImageOps.contain(Image.fromarray(frame[:, :, ::-1]), (width - 10, height - 25))
        left, top = i % 6 * width, i // 6 * height
        sheet.paste(thumb, (left + 5, top))
        draw.text((left + 5, top + height - 21), f"{path.stem.split('_')[-2]} | station {stations[key]} | "
                  f"{'registered' if key in views else 'unregistered'}", fill="#172631")
        small = cv2.resize(frame, (400, 300), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        records.append({"image_id": key, "name": path.name, "station": stations[key], "registered": key in views,
                        "east_north_m": xy[i].tolist(), "ppk_ellipsoid_height_m": geo[i, 2],
                        "gimbal_degrees": attitudes.get(key, {}), "gray_laplacian_variance_400px": cv2.Laplacian(gray, cv2.CV_32F).var().item(),
                        "dark_or_saturated_gray_fraction": float(((gray < 4) | (gray > 251)).mean())})
    sheet.save(output / "input-photos.jpg", quality=91)
    connectivity = np.zeros((len(ids), len(ids)), int)
    geo_edges, counts = [], Counter()
    with np.load(sparse / "verified_pairs.npz") as pairs:
        for name in pairs:
            a, b = map(int, name.split("_"))
            if not len(pairs[name]):
                continue
            i, j = index[a], index[b]
            connectivity[i, j] = connectivity[j, i] = len(pairs[name])
            if stations[a] != stations[b]:
                counts["cross_station"] += 1
                if a in views and b in views:
                    geo_edges.append(xy[[i, j]])
                    counts["cross_station_registered"] += 1
    registered = np.asarray([key in views for key in ids])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    axes[0].add_collection(LineCollection(geo_edges, colors="#718797", alpha=.25, linewidths=.7))
    axes[0].scatter(*xy[registered].T, c=[stations[key] for key in ids if key in views], cmap="tab10", s=22,
                    label="Registered")
    axes[0].scatter(*xy[~registered].T, c="#b24631", marker="x", s=30, label="Unregistered")
    for group in set(stations.values()):
        center = xy[np.array([stations[key] == group for key in ids])].mean(axis=0)
        axes[0].annotate(f"S{group}", center, xytext=(7, 4), textcoords="offset points")
    axes[0].set(aspect="equal", xlabel="Approximate east (m)", ylabel="Approximate north (m)")
    axes[0].legend(loc="best")
    displayed = axes[1].imshow(np.log1p(connectivity), cmap="viridis")
    axes[1].set(xlabel="Input image index", ylabel="Input image index")
    fig.colorbar(displayed, ax=axes[1], label="log(1 + verified matches)")
    fig.savefig(output / "capture-connectivity.png", dpi=145)
    plt.close(fig)
    report = {"images": records, "station_all": dict(Counter(stations.values())),
              "station_registered": dict(Counter(stations[key] for key in views)), "cross_edges": dict(counts),
              "verified_edges": int(np.count_nonzero(np.triu(connectivity, 1))),
              "scope": "PPK is a camera-position prior. Gimbal angles are available for registered images. "
                       "Camera locations alone do not establish ground visibility."}
    (output / "capture.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.sparse, args.output)
