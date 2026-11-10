"""Benchmark dense profiles in separate processes using existing sparse exports."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter

from vision3d.reconstruction.multiview import PROFILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", required=True, metavar="NAME=SPARSE_DIR")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", action="append", choices=PROFILES)
    parser.add_argument("--mesh", action="store_true")
    parser.add_argument("--glb", action="store_true")
    args = parser.parse_args()
    if args.glb and not args.mesh:
        parser.error("--glb requires --mesh")
    scenes = {}
    for spec in args.scene:
        name, separator, source = spec.partition("=")
        if not separator or not source or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in scenes:
            parser.error("Use distinct NAME=SPARSE_DIR values")
        scenes[name] = Path(source).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    software = {p: importlib.metadata.version(p) for p in ("vision3d", "opencv-contrib-python", "open3d", "numpy")}
    environment = {"platform": platform.platform(), "processor": platform.processor(),
                   "python": platform.python_version(), "software": software}
    records, rows = [], []
    for scene, source in scenes.items():
        sparse = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
        reference = source / "pose_evaluation.json"
        pose = json.loads(reference.read_text(encoding="utf-8")) if reference.is_file() else {}
        quality = sparse.get("after_final_ba", {})
        for profile in dict.fromkeys(args.profile or PROFILES):
            label = f"{scene}-{profile}"
            output = args.output / label
            command = [sys.executable, "-B", "-m", "vision3d.cli", "dense-sequence", "--sparse-dir", str(source),
                       "--output", str(output), "--profile", profile]
            command += ["--mesh"] if args.mesh else []
            command += ["--glb"] if args.glb else []
            start = perf_counter()
            with (args.output / f"{label}.log").open("w", encoding="utf-8") as log:
                child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            wall = perf_counter() - start
            metrics_path = output / "metrics.json"
            result = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
            mesh = result.get("mesh") or {}
            row = dict(scene=scene, profile=profile, exit_code=child.returncode, status=result.get("status", "failed"),
                       registered=sparse["registered_images"], input_images=sparse["input_images"],
                       registration_percent=100*sparse["registered_images"]/sparse["input_images"],
                       sparse_points=sparse["valid_landmarks"], dense_points=result.get("dense_points"),
                       mesh_faces=mesh.get("triangles"), median_px=quality.get("median_error_px"),
                       rms_px=quality.get("rms_error_px"), rotation_deg=pose.get("median_aligned_rotation_degrees"),
                       translation_direction_deg=pose.get("median_translation_direction_degrees"),
                       raw_samples=result.get("raw_depth_samples"), checked_samples=result.get("checked_samples"),
                       dense_seconds=result.get("runtime_seconds"), sparse_seconds=sparse.get("runtime_seconds"),
                       wall_seconds=wall, peak_bytes=result.get("peak_memory_bytes"),
                       ply_validated=child.returncode == 0 and (output / "dense.ply").is_file(),
                       glb_validated=(mesh.get("glb") or {}).get("status") == "success")
            rows.append(row)
            records.append({"command": command, "summary": row, "metrics": result})
            print(json.dumps(row), flush=True)
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "results.json").write_text(json.dumps({"environment": environment, "runs": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
