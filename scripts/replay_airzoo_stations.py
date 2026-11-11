"""Replay dense fusion with fixed camera poses, isolating capture-station inputs."""
import argparse
import json
from pathlib import Path
import shutil

from analyze_airzoo_geometry import read, station_metadata
from vision3d.reconstruction.multiview import reconstruct_multiview


def replay(sparse, output, voxel):
    _, views, stations, *_ = station_metadata(sparse)
    output.mkdir(parents=True, exist_ok=False)
    poses = read(sparse / "poses.json")
    results = {}
    for station in sorted(set(stations.values())):
        cameras = {i for i in views if stations[i] == station}
        results[station] = result = {"fixed_registered_camera_ids": sorted(cameras)}
        if len(cameras) < 3:
            result.update(status="not_run", reason="fewer than three registered camera poses")
            continue
        subset = output / f"station-{station}-input"
        subset.mkdir()
        for name in ("manifest.json", "intrinsics.json", "landmarks.json"):
            shutil.copyfile(sparse / name, subset / name)
        masked = [pose if pose["image_id"] in cameras else {**pose, "status": "station_excluded"} for pose in poses]
        (subset / "poses.json").write_text(json.dumps(masked), encoding="utf-8")
        for profile in ("uav", "uav-conditioned"):
            dense = output / f"station-{station}-{profile}"
            try:
                result[profile] = reconstruct_multiview(
                    subset, dense, profile=profile, depth_diagnostics=True, fusion_voxel_size=voxel,
                    stage=lambda name: print(name, flush=True))
            except ValueError as exc:
                result[profile] = {"status": "rejected", "reason": str(exc)}
        (output / "stations.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "stations.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel", type=float, required=True)
    args = parser.parse_args()
    replay(args.sparse, args.output, args.voxel)
