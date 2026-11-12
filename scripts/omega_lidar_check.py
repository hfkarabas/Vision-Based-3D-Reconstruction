"""Evaluate a metric reconstruction against withheld LiDAR over a declared polygon."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import Polygon


def sample(pdal, cloud, boundary, voxel, label, output):
    left, bottom, right, top = boundary.bounds
    target = output / f"{label}.csv"
    recipe = output / f"{label}.json"
    pipeline = [
        dict(type="readers.ply", filename=str(cloud.resolve())),
        dict(type="filters.crop", bounds=f"([{left},{right}],[{bottom},{top}])"),
        dict(type="filters.voxeldownsize", cell=voxel, mode="first"),
        dict(type="writers.text", filename=str(target), order="X,Y,Z", keep_unspecified=False, precision=6),
    ]
    recipe.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
    with (output / f"{label}.log").open("wb") as log:
        subprocess.run([str(pdal), "pipeline", str(recipe), "--stream"], stdout=log,
                       stderr=subprocess.STDOUT, timeout=300, check=True)
    points = np.loadtxt(target, skiprows=1, delimiter=",", ndmin=2)
    return points[contains_xy(boundary, points[:, 0], points[:, 1])]


def distances(source, target):
    nearest, _ = cKDTree(target).query(source, workers=4)
    return dict(points=len(nearest), median_m=float(np.median(nearest)),
                p95_m=float(np.percentile(nearest, 95)), rms_m=float(np.sqrt(np.mean(nearest**2))),
                within_0_5m=float(np.mean(nearest <= 0.5)), within_1m=float(np.mean(nearest <= 1.0)))


def cells(points, size):
    return set(map(tuple, np.unique(np.floor(points[:, :2] / size).astype(np.int32), axis=0)))


def in_cells(points, selected):
    return points[np.array([tuple(cell) in selected for cell in np.floor(points[:, :2]).astype(np.int32)])]


def matched_evaluation(lidar, first, second):
    common = cells(lidar, 1.0) & cells(first, 1.0) & cells(second, 1.0)
    if not common:
        raise ValueError("The three clouds share no common horizontal grid cell.")
    reference = in_cells(lidar, common)
    return dict(common_1m_cells=len(common), lidar_points=len(reference), variants=[
        dict(reconstruction_to_lidar=distances(filtered, reference),
             lidar_to_reconstruction=distances(reference, filtered))
        for filtered in (in_cells(first, common), in_cells(second, common))
    ])


def run(args):
    if not 0.1 <= args.voxel <= 2:
        raise ValueError("Use a representative evaluation voxel between 0.1 and 2 metres.")
    with args.lidar.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").hexdigest()
    if digest != args.lidar_md5.lower():
        raise ValueError("The LiDAR bytes do not match the publisher's archive checksum.")
    polygon = Polygon(np.loadtxt(args.polygon))
    if not polygon.is_valid:
        raise ValueError("Invalid publisher-provided evaluation polygon.")
    args.output.mkdir(parents=True, exist_ok=False)
    lidar = sample(args.pdal.resolve(), args.lidar, polygon, args.voxel, "lidar", args.output)
    cloud = sample(args.pdal.resolve(), args.cloud, polygon, args.voxel, "reconstruction", args.output)
    if min(len(cloud), len(lidar)) < 10:
        raise ValueError("Insufficient genuine overlap in the declared evaluation region.")
    lc, cc = cells(lidar, 1.0), cells(cloud, 1.0)
    overlap = lc & cc
    if not overlap:
        raise ValueError("No common supported horizontal grid cells.")
    report = dict(
        input_lidar=str(args.lidar.resolve()), lidar_md5=digest, input_cloud=str(args.cloud.resolve()),
        reference_scope=args.reference_scope,
        evaluation_polygon=str(args.polygon.resolve()), registration="Author-corrected ENU, no LiDAR fit",
        voxel_m=args.voxel, reference_samples=len(lidar), reconstructed_samples=len(cloud),
        reference_occupied_1m_cells=len(lc), reconstructed_occupied_1m_cells=len(cc),
        shared_occupied_1m_cells=len(overlap), reference_cell_coverage=len(overlap) / len(lc),
        reconstruction_cell_coverage=len(overlap) / len(cc),
        reconstruction_to_lidar_shared_cells=distances(in_cells(cloud, overlap), in_cells(lidar, overlap)),
        lidar_to_reconstruction_shared_cells=distances(in_cells(lidar, overlap), in_cells(cloud, overlap)),
        scope="Nearest sampled surface points over common supported XY cells. No vertical or global completeness claim.",
    )
    if args.comparison_cloud:
        second = sample(args.pdal.resolve(), args.comparison_cloud, polygon, args.voxel, "comparison", args.output)
        report["comparison_cloud"] = str(args.comparison_cloud.resolve())
        report["comparison_reference_cell_coverage"] = len(lc & cells(second, 1.0)) / len(lc)
        report["matched_extent_comparison"] = matched_evaluation(lidar, cloud, second)
    (args.output / "lidar_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pdal", "cloud", "lidar", "polygon", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--lidar-md5", required=True)
    parser.add_argument("--voxel", type=float, default=0.3)
    parser.add_argument("--reference-scope", choices=("full", "verified-prefix"), default="full")
    parser.add_argument("--comparison-cloud", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
