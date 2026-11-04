"""Run isolated SfM benchmark processes and export honest fit/pose/resource results."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
from vision3d.evaluation.poses import evaluate_poses, plot_pose_comparison
from vision3d.reconstruction.pair import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=root / "configs/benchmark_suite.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Optional named case filter, repeatable")
    parser.add_argument("--baseline-source", type=Path, help="Optional archived src directory for historical CLI comparison; no feature cache")
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str((args.output / "matplotlib-cache").resolve())
    write_json(args.output / "suite_snapshot.json", suite)
    source = args.baseline_source.resolve() if args.baseline_source else root / "src"
    fingerprint = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        fingerprint.update(path.relative_to(source).as_posix().encode())
        fingerprint.update(path.read_bytes())
    write_json(args.output / "execution.json", {"python": sys.version, "source_directory": str(source),
        "source_sha256": fingerprint.hexdigest(), "baseline": bool(args.baseline_source),
        "memory_measurement": "Fresh child process, Windows lifetime peak working set or sampled RSS, 50 ms polling"})
    rows, results = [], []
    for case in suite["cases"]:
        if args.case and case["name"] not in args.case:
            continue
        output = args.output / case["name"]
        defaults = {**suite["defaults"], **case.get('settings', {})}
        command = [sys.executable, "-m", "vision3d.cli", "reconstruct-sequence", "--input", str(root / case["input"]),
                   "--output", str(output)]
        if not args.baseline_source:
            cache = args.output / "feature-cache" / case["cache_tag"]
            if case.get("cache_state") == "warm" and not any(cache.glob("*.npz")):
                parser.error("Warm cases require the matching cold case earlier in this invocation")
            command += ["--feature-cache", str(args.output / "feature-cache" / case["cache_tag"])]
        for key in ("max_dimension", "max_features", "window", "long_gap", "seed"):
            command += ["--" + key.replace("_", "-"), str(defaults[key])]
        if defaults["local_ba"]:
            command += ["--local-ba"]
        if defaults.get('recovery_pairs', 0):
            command += ['--recovery-pairs', str(defaults['recovery_pairs'])]
        if defaults.get('joint_ba', False):
            command += ['--joint-ba']
        if "camera" in case:
            command += ["--camera", str(root / case["camera"])]
        else:
            command += ["--approximate-intrinsics", "--fov-degrees", str(case["fov_degrees"])]
        env = dict(os.environ, MPLCONFIGDIR=str((args.output / "matplotlib-cache").resolve()))
        env["PYTHONPATH"] = str(source)
        start, peak = time.perf_counter(), 0
        with (args.output / f"{case['name']}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=env)
            probe = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    memory = probe.memory_info()
                    peak = max(peak, getattr(memory, "peak_wset", memory.rss))
                except psutil.Error:
                    pass
                time.sleep(0.05)
        metrics_path = output / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        evaluation = {"status": "unavailable", "reason": "no_reference_poses"}
        if case.get("reference") and (output / "poses.json").exists():
            estimated = json.loads((output / "poses.json").read_text())
            reference = json.loads((root / case["reference"]).read_text())
            evaluation = evaluate_poses(estimated, reference)
            write_json(output / "pose_evaluation.json", evaluation)
            plot_pose_comparison(estimated, reference, evaluation, output / "trajectory.png")
        quality = metrics.get("after_final_ba", {})
        status = metrics.get("status", "failed")
        if metrics.get("registered_images", 0) < metrics.get("input_images", 0) and status == "success":
            status = "partial_registration"
        row = {"case": case["name"], "split": case["split"], "intrinsics": "published" if "camera" in case else "approximate",
               "status": status, "exit_code": process.returncode, "images": metrics.get("input_images"),
               "registered": metrics.get("registered_images", 0), "landmarks": metrics.get("valid_landmarks", 0),
               "median_reprojection_px": quality.get("median_error_px"), "rms_reprojection_px": quality.get("rms_error_px"),
               "runtime_seconds": metrics.get("runtime_seconds"), "wall_seconds": time.perf_counter() - start,
               "peak_memory_bytes": max(peak, metrics.get("peak_memory_bytes") or 0), "feature_cache_hits": metrics.get("feature_cache_hits", 0),
               "feature_extraction_seconds": metrics.get("feature_extraction_seconds"),
               "center_rmse_fraction_extent": evaluation.get("center_rmse_fraction_trajectory_extent"),
               "median_aligned_rotation_deg": evaluation.get("median_aligned_rotation_degrees"),
               "median_translation_direction_deg": evaluation.get("median_translation_direction_degrees"),
               "geometric_accuracy_warning": evaluation.get("geometric_accuracy_warning")}
        row.update({f"seconds_{k}": v for k, v in metrics.get("stage_seconds", {}).items()})
        rows.append(row)
        results.append({"case": case, "command": command, "summary": row, "metrics": metrics, "pose_evaluation": evaluation})
        write_json(args.output / "results.json", results)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
    if not rows:
        parser.error("No matching benchmark cases")


if __name__ == "__main__":
    main()
