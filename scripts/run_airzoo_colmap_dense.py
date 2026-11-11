"""Run a bounded independent COLMAP PatchMatch/fusion benchmark on a saved model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import open3d as o3d


def run(colmap, model, images, output):
    colmap, model, images, output = [path.resolve() for path in (colmap, model, images, output)]
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    env = os.environ.copy()
    env["PATH"] = str(colmap.parent) + os.pathsep + env["PATH"]
    env["QT_PLUGIN_PATH"] = str(colmap.parent.parent / "plugins")
    stages = {
        "undistort": ["image_undistorter", "--image_path", str(images), "--input_path", str(model),
                      "--output_path", str(workspace), "--output_type", "COLMAP", "--max_image_size", "900",
                      "--num_patch_match_src_images", "6", "--num_threads", "4"],
        "patchmatch": ["patch_match_stereo", "--workspace_path", str(workspace), "--workspace_format", "COLMAP",
                       "--PatchMatchStereo.gpu_index", "0", "--PatchMatchStereo.geom_consistency", "1",
                       "--PatchMatchStereo.max_image_size", "900", "--PatchMatchStereo.cache_size", "2",
                       "--PatchMatchStereo.num_iterations", "3"],
        "fusion": ["stereo_fusion", "--workspace_path", str(workspace), "--workspace_format", "COLMAP",
                   "--input_type", "geometric", "--output_path", str(output / "dense.ply"),
                   "--StereoFusion.num_threads", "4", "--StereoFusion.max_image_size", "900",
                   "--StereoFusion.cache_size", "2"],
    }
    metrics = {"commands": {}, "stages_seconds": {}, "status": "incomplete", "input_model": str(model)}
    start = perf_counter()
    try:
        for name, args in stages.items():
            stamp = perf_counter()
            command = [str(colmap), *args]
            metrics["commands"][name] = command
            with (output / f"{name}.log").open("wb") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=360)
            metrics["stages_seconds"][name] = perf_counter() - stamp
        cloud = o3d.io.read_point_cloud(str(output / "dense.ply"))
        if not len(cloud.points) or not np.isfinite(np.asarray(cloud.points)).all():
            raise ValueError("COLMAP fusion did not produce finite points")
        maps = list((workspace / "stereo/depth_maps").glob("*.geometric.bin"))
        metrics.update(status="success", dense_points=len(cloud.points), colored=cloud.has_colors(), depth_maps=len(maps))
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        metrics.update(status="failed", reason=str(exc))
        raise
    finally:
        metrics["runtime_seconds"] = perf_counter() - start
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.colmap, args.model, args.images, args.output)
