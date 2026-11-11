"""Trace fixed AirZoo dense samples into their rectified and original source views."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.sparse.csgraph import connected_components

from vision3d.geometry.depth import project_depth, unproject_depth
from vision3d.reconstruction.surface_validation import surface_views


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def percentiles(x):
    return np.percentile(x, [1, 5, 50, 95, 99]).tolist() if len(x) else None


def station_metadata(sparse):
    inputs = read(sparse / "manifest.json")["inputs"]
    views = {v.id: v for v in surface_views(sparse)}
    folder = Path(inputs[0]["source"]).parent
    mrk, = folder.glob("*.MRK")
    marks = {}
    for line in mrk.read_text(encoding="utf-8-sig").splitlines():
        row = line.split("\t")
        marks[int(row[0])] = [float(field.split(",")[0]) for field in row[6:9]]
    geodetic = np.asarray([marks[int(Path(item["name"]).stem.split("_")[-2])] for item in inputs])
    east_north = (geodetic[:, [1, 0]] - geodetic[0, [1, 0]]) * [111195 * np.cos(np.radians(geodetic[0, 0])), 111195]
    _, labels = connected_components(np.linalg.norm(east_north[:, None] - east_north, axis=2) < 15, directed=False)
    stations = {item["image_id"]: int(label + 1) for item, label in zip(inputs, labels, strict=True)}
    positions = {item["image_id"]: position for item, position in zip(inputs, geodetic, strict=True)}
    up, attitudes = [], {}
    for i, view in views.items():
        # DJI XMP is in the JPEG header, before the first image scan.
        with view.source.open("rb") as stream:
            head = stream.read(256 * 1024)
        angles = {}
        for name in ("Pitch", "Roll", "Yaw"):
            match = re.search(rb'Gimbal' + name.encode() + rb'Degree="([+0-9.\-]+)"', head)
            if match:
                angles[name.lower()] = float(match[1])
        attitudes[i] = angles
        if "pitch" in angles and abs(angles.get("roll", 90)) <= 2:
            pitch = np.radians(angles["pitch"])
            up.append(view.pose.rotation.T @ [0, -np.cos(pitch), np.sin(pitch)])
    if len(up) < 3:
        raise ValueError("At least three low-roll gimbal pitch observations are required")
    gravity = np.median(up, axis=0)
    gravity /= np.linalg.norm(gravity)
    seed = np.eye(3)[np.argmin(abs(gravity))]
    horizontal = np.cross(gravity, seed)
    horizontal /= np.linalg.norm(horizontal)
    basis = np.stack((horizontal, np.cross(gravity, horizontal), gravity), axis=1)
    deviation = np.degrees(np.arccos(np.clip(np.asarray(up) @ gravity, -1, 1)))
    return inputs, views, stations, positions, attitudes, basis, deviation


def sparse_report(sparse, views, stations, inputs, attitudes, deviation):
    landmarks = read(sparse / "landmarks.json")
    counts = Counter()
    residuals = defaultdict(list)
    station_residuals = defaultdict(list)
    tracks = []
    negative = 0
    for landmark in landmarks:
        ids = [int(i) for i in landmark["observations"] if int(i) in views]
        tracks.append(len(ids))
        counts[len({stations[i] for i in ids})] += 1
        for i in ids:
            pose = views[i].pose
            negative += int((pose.rotation @ landmark["xyz"] + pose.translation)[2] <= 0)
            if str(i) in landmark.get("reprojection_errors_px", {}):
                error = landmark["reprojection_errors_px"][str(i)]
                residuals[i].append(error)
                station_residuals[stations[i]].append(error)
    edges = []
    with np.load(sparse / "verified_pairs.npz") as pairs:
        for name in pairs.files:
            a, b = map(int, name.split("_"))
            if a in views and b in views:
                edges.append([a, b, len(pairs[name])])
    graph = np.zeros((len(views), len(views)), bool)
    ix = {i: j for j, i in enumerate(views)}
    for a, b, _ in edges:
        graph[ix[a], ix[b]] = graph[ix[b], ix[a]] = True
    _, labels = connected_components(graph, directed=False)
    camera = {i: {"name": str(v.source.name), "station": stations[i], "intrinsics": v.camera.to_dict(),
                  "attitude": attitudes[i], "center_world": v.pose.center.tolist(),
                  "forward_world": v.pose.rotation[2].tolist(),
                  "residual_px_p01_p05_p50_p95_p99": percentiles(residuals[i])}
              for i, v in views.items()}
    report = {"input_images": len(inputs), "registered_images": len(views), "cameras": camera,
              "registered_by_station": dict(Counter(stations[i] for i in views)), "track_length_percentiles": percentiles(tracks),
              "landmarks_by_number_of_stations": dict(counts), "cheirality_violations": negative,
              "station_reprojection_px_p01_p05_p50_p95_p99": {i: percentiles(rows)
                                                             for i, rows in station_residuals.items()},
              "input_metadata": [{k: item[k] for k in ("image_id", "source_size", "exif_orientation", "exif_focal", "resize")}
                                 for item in inputs], "gimbal_up_residual_deg": percentiles(deviation),
              "triangulation_angles": percentiles([p["max_triangulation_angle_degrees"] for p in landmarks
                                                   if "max_triangulation_angle_degrees" in p])}
    graph_report = {"verified_edges": edges, "registered_components": [np.asarray(list(views))[labels == group].tolist()
                    for group in np.unique(labels)], "stations": stations,
                    "registered_cross_station_edges": sum(stations[a] != stations[b] for a, b, _ in edges)}
    return report, graph_report, np.asarray([p["xyz"] for p in landmarks])


def normalized_parallax(xyz, first, second):
    a, b = xyz - first, xyz - second
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), (a * b).sum(axis=1)))


def original_pixels(xyz, view):
    local = xyz @ view.pose.rotation.T + view.pose.translation
    pixels, _ = cv2.projectPoints(local, np.zeros(3), np.zeros(3), view.camera.K, view.camera.D)
    return pixels.reshape(-1, 2)


def export_scene(sparse, dense, historic_meshes, output):
    inputs, views, stations, _, attitudes, basis, deviation = station_metadata(sparse)
    camera, graph, landmarks = sparse_report(sparse, views, stations, inputs, attitudes, deviation)
    output.mkdir(parents=True, exist_ok=False)
    origin = np.median(landmarks, axis=0)
    cloud = np.asarray(o3d.io.read_point_cloud(str(dense / "dense.ply")).points).copy()
    def aligned(xyz):
        return (xyz - origin) @ basis
    colors = plt.get_cmap("tab10")
    layers = [("Sparse landmarks", aligned(landmarks), "#236087")]
    cloud_sample = cloud[::max(1, len(cloud) // 45000)]
    depth_root = dense / "depth-observations"
    records = read(depth_root / "manifest.json")["pairs"]
    depth_report, raw_layers, accepted_layers, examples = [], [], [], []
    for row in records:
        i, j = row["reference"], row["source"]
        with np.load(depth_root / row["file"]) as data:
            raw = data["raw_xyz"]
            accepted = data["accepted_xyz"]
            depth = data["depth"]
            pixels, z = project_depth(raw, data["rotation"], data["translation"], data["intrinsic"])
            returned = unproject_depth(data["raw_pixels_yx"][:, ::-1], z, data["rotation"], data["translation"], data["intrinsic"])
            world_displacement = np.linalg.norm(cloud[data["fused_vertex"]] - accepted, axis=1)
            checked_xyz = aligned(accepted)
            raw_layers.append((aligned(raw)[::3], colors(stations[i] - 1)))
            accepted_layers.append((checked_xyz[::3], colors(stations[i] - 1)))
            row.update(reference_station=stations[i], source_station=stations[j],
                       raw_valid_fraction=float(np.count_nonzero(depth) / depth.size),
                       raw_positive_depth_fraction=float((z > 0).mean()),
                       rectified_pixel_roundtrip_max=float(np.linalg.norm(pixels - data["raw_pixels_yx"][:, ::-1], axis=1).max()),
                       world_roundtrip_max=float(np.linalg.norm(raw - returned, axis=1).max()),
                       depth_percentiles=percentiles(z), raw_elevation=percentiles(aligned(raw)[:, 2]),
                       accepted_elevation=percentiles(checked_xyz[:, 2]),
                       sample_to_fused_displacement=percentiles(world_displacement),
                       accepted_parallax_deg=percentiles(normalized_parallax(accepted, views[i].pose.center, views[j].pose.center)))
            if len(accepted):
                yx = data["accepted_pixels_yx"]
                ids = np.argsort(checked_xyz[:, 2])[:3]
                for k in ids:
                    point = accepted[k:k+1]
                    witnesses = int(data["accepted_witness_bits"][k]) if "accepted_witness_bits" in data else 0
                    examples.append({"reference": i, "source": j, "station": stations[i], "world": point[0].tolist(),
                                     "rectified_pixel_yx": yx[k].tolist(), "rectified_depth": float(depth[*yx[k]]),
                                     "reference_original_uv": original_pixels(point, views[i])[0].tolist(),
                                     "source_original_uv": original_pixels(point, views[j])[0].tolist(),
                                     "witnesses_mask": witnesses, "fused_vertex": int(data["fused_vertex"][k]),
                                     "local_elevation": float(checked_xyz[k, 2])})
            shown = np.ma.masked_where(depth <= 0, depth)
            fig, ax = plt.subplots(1, 2, figsize=(8, 3))
            low, high = np.percentile(depth[depth > 0], [2, 98])
            for panel, raster, title in ((ax[0], shown, "Reciprocal stereo"),
                                         (ax[1], np.ma.masked_where(~data["accepted_raster"], depth), "Corroborated")):
                plot = panel.imshow(raster, vmin=low, vmax=high, cmap="viridis")
                panel.set_title(title)
                panel.axis("off")
            fig.colorbar(plot, ax=ax, shrink=.7, label="Rectified depth in SfM units")
            fig.suptitle(f"{i} → {j} · stations {stations[i]} → {stations[j]} · {'recentered' if row['recentered'] else 'native'}")
            fig.savefig(output / f"depth-{i:04d}-{j:04d}.png", dpi=110)
            plt.close(fig)
        depth_report.append(row)
    layers += [("Raw depth by station", raw_layers, None), ("Accepted depth by station", accepted_layers, None),
               ("Fused dense cloud", aligned(cloud_sample), "#236087")]
    for label, path in historic_meshes:
        geometry = o3d.io.read_triangle_mesh(str(path))
        xyz = np.asarray(geometry.vertices)
        if Path(path).suffix.lower() == ".glb":
            xyz = xyz * [1, -1, -1]
        layers.append((label, aligned(xyz[::max(1, len(xyz) // 45000)]), "#236087"))
    cameras = aligned(np.stack([v.pose.center for v in views.values()]))
    displayed = np.vstack((np.percentile(aligned(cloud_sample), [.15, 99.85], axis=0), cameras))
    extent = [displayed.min(axis=0), displayed.max(axis=0)]
    extent = np.asarray(extent)
    directions = np.stack([v.pose.rotation[2] @ basis for v in views.values()])
    fig, axes = plt.subplots(3, len(layers), figsize=(len(layers)*3, 8), layout="constrained", squeeze=False)
    for axis, label, right in ((0, "Top down", np.array([0, 1, 0])), (1, "Oblique", np.array([0, .8, .6])),
                               (2, "Side elevation", np.array([0, 0, 1]))):
        projected = np.concatenate((np.percentile(aligned(cloud_sample) @ right, [.15, 99.85]), cameras @ right))
        bounds = [projected.min(), projected.max()]
        for col, (title, points, color) in enumerate(layers):
            panel = axes[axis, col]
            groups = [(points, color)] if color is not None else points
            for xyz, rgb in groups:
                panel.scatter(xyz[:, 0], xyz @ right, c=[rgb], s=.45, linewidths=0, alpha=.6, rasterized=True)
            if col == 0:
                panel.quiver(cameras[:, 0], cameras @ right, directions[:, 0], directions @ right,
                             color=[colors(stations[i] - 1) for i in views], width=.006, scale=20)
            panel.set_xlim(extent[:, 0])
            panel.set_ylim(bounds)
            panel.set_aspect("equal", adjustable="box")
            if axis == 0:
                panel.set_title(title)
            if col == 0:
                panel.set_ylabel(label + " (SfM units)")
    fig.savefig(output / "stage-comparison.png", dpi=135)
    plt.close(fig)
    for name, document in (("camera_geometry_report", camera), ("overlap_graph_report", graph),
                           ("depth_consistency_report", {"pairs": depth_report, "lowest_examples": sorted(examples,
                            key=lambda p: p["local_elevation"])[:24], "basis": basis.tolist(), "origin": origin.tolist(),
                            "fused_elevation": percentiles(aligned(cloud)[:, 2])})):
        (output / f"{name}.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", required=True, type=Path)
    parser.add_argument("--dense", required=True, type=Path)
    parser.add_argument("--mesh", nargs=2, action="append", default=[], metavar=("LABEL", "PATH"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export_scene(args.sparse, args.dense, args.mesh, args.output)
