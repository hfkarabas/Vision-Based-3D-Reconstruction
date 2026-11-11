"""Run the independent Blender benchmark and compare rendered pixels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


def compare_render(reference, current):
    with Image.open(reference) as image:
        first = np.asarray(image.convert("RGBA"))
    with Image.open(current) as image:
        second = np.asarray(image.convert("RGBA"))
    if first.shape != second.shape:
        raise ValueError("Regression image dimensions differ")
    a, b = first[:, :, 3] >= 128, second[:, :, 3] >= 128
    shared = a & b
    if not shared.any():
        raise ValueError("Rendered model foreground is empty or disjoint")
    delta = np.abs(first[shared, :3].astype(np.int16) - second[shared, :3])
    return {"silhouette_iou": float(shared.sum() / (a | b).sum()), "foreground_rgb_mae": float(delta.mean()),
            "foreground_pixels_changed_over_4": float((delta.max(axis=1) > 4).mean()),
            "pixel_identical": bool(np.array_equal(first, second))}


def replay_passed(result):
    return (result["silhouette_iou"] >= .999 and result["foreground_rgb_mae"] <= .25
            and result["foreground_pixels_changed_over_4"] <= .005)


def run(blender, scenes, output, reference, resolution, timeout):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inspector = Path(__file__).with_name("blender_inspect_asset.py").resolve()
    report = {"runs": [], "cross_format": [], "visual_regression": [], "passed": True}
    for scene, bundle in scenes.items():
        root = output / scene
        root.mkdir()
        for format_name in ("glb", "obj"):
            folder = root / format_name
            log = root / f"{format_name}.log"
            command = [str(blender.resolve()), "--background", "--factory-startup", "--python-exit-code", "2",
                       "--python", str(inspector), "--", "--asset", str(bundle / f"textured.{format_name}"),
                       "--output", str(folder), "--resolution", str(resolution)]
            frame = (reference / scene / format_name / "report.json" if reference else
                     root / "glb/report.json" if format_name == "obj" else None)
            if frame is not None:
                command.extend(["--framing", str(frame)])
            with log.open("w", encoding="utf-8") as stream:
                try:
                    exit_code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout).returncode
                except subprocess.TimeoutExpired:
                    exit_code = 124
                    stream.write(f"Blender exceeded its {timeout:g}-second validation timeout.\n")
            complete = (folder / "report.json").is_file()
            report["runs"].append({"scene": scene, "format": format_name, "exit_code": exit_code, "complete": complete,
                                   "report": str(folder / "report.json"), "log": str(log)})
            report["passed"] &= exit_code == 0 and complete
            if reference and complete:
                for view in ("top", "oblique"):
                    try:
                        comparison = compare_render(reference / scene / format_name / f"{view}.png", folder / f"{view}.png")
                        passed = replay_passed(comparison)
                    except (ValueError, OSError) as exc:
                        comparison, passed = {"error": str(exc)}, False
                    report["visual_regression"].append({"scene": scene, "format": format_name, "view": view,
                                                         "passed": passed, **comparison})
                    report["passed"] &= passed
        for view in ("top", "oblique"):
            glb, obj = root / "glb" / f"{view}.png", root / "obj" / f"{view}.png"
            if not glb.is_file() or not obj.is_file():
                continue
            try:
                comparison = compare_render(glb, obj)
                # The OBJ fallback is neutral and the GLB fallback is vertex-colored.
                passed = comparison["silhouette_iou"] >= .995
            except (ValueError, OSError) as exc:
                comparison, passed = {"error": str(exc)}, False
            report["cross_format"].append({"scene": scene, "view": view, "silhouette_passed": passed, **comparison})
            report["passed"] &= passed
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--rural", type=Path, required=True)
    parser.add_argument("--urban", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--timeout-seconds", type=float, default=180.)
    args = parser.parse_args()
    if not 64 <= args.resolution <= 2048 or not np.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("Resolution must be 64–2048 and timeout must be positive and finite")
    return 0 if run(args.blender, {"rural": args.rural, "urban": args.urban}, args.output,
                    args.reference, args.resolution, args.timeout_seconds) else 2


if __name__ == "__main__":
    sys.exit(main())
