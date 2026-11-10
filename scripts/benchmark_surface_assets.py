"""Reproduce fixed or expanded AirZoo surface assets from Mission II dense runs."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("rural", "urban", "rural-48"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-dir", type=Path, help="Override the saved dense input")
    args = parser.parse_args()
    default = "rural-48-uav" if args.scene == "rural-48" else f"{args.scene}-uav-validated"
    dense = args.dense_dir or Path("outputs/airzoo-benchmark/major-dense-reconstruction") / default
    surface, texture = args.output / "surface", args.output / "texture"
    target = {"rural": 450000, "urban": 900000, "rural-48": 260000}[args.scene]
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-u", "-m", "vision3d.cli"]
    subprocess.run([*command, "surface-asset", "--dense-dir", str(dense), "--output", str(surface),
                    "--target-faces", str(target)], check=True)
    subprocess.run([*command, "texture-asset", "--surface-dir", str(surface), "--output", str(texture),
                    "--holdout-stride", "7", "--obj"], check=True)
    subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("audit_surface_depth.py")),
                    "--surface-dir", str(surface), "--output", str(args.output / "reserved-depth.json")], check=True)


if __name__ == "__main__":
    main()
