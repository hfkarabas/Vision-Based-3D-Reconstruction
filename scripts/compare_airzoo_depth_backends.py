"""Compare project depth pixels to COLMAP geometric depth at identical camera poses."""
import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap

from analyze_airzoo_geometry import percentiles, read


def read_depth(path):
    with path.open("rb") as stream:
        fields = []
        for _ in range(3):
            digits = bytearray()
            for _ in range(10):
                byte = stream.read(1)
                if byte == b"&":
                    break
                if not b"0" <= byte <= b"9":
                    raise ValueError(f"Invalid COLMAP depth header in {path}")
                digits.extend(byte)
            else:
                raise ValueError(f"Invalid COLMAP depth header in {path}")
            fields.append(int(digits))
        width, height, channels = fields
        if channels != 1 or min(width, height) <= 0:
            raise ValueError(f"Expected one nonempty depth channel in {path}")
        depth = np.fromfile(stream, dtype="<f4", count=width * height + 1)
    return depth.reshape(height, width)


def compare(workspace, frame, variants, probe=None):
    model = pycolmap.Reconstruction(workspace / "sparse")
    views = {int(Path(i.name).stem): i for i in model.images.values() if i.has_pose}
    results = {}
    for label, directory in variants:
        root = directory / "depth-observations"
        views_report = []
        for record in read(root / "manifest.json")["pairs"]:
            i = record["reference"]
            if i not in views:
                continue
            view = views[i]
            depth_path = workspace / "stereo/depth_maps" / (view.name + ".geometric.bin")
            if not depth_path.is_file():
                continue
            depth = read_depth(depth_path)
            camera = view.camera
            if camera.model_name != "PINHOLE" or depth.shape != (camera.height, camera.width):
                raise ValueError("External depth needs a matching PINHOLE image model")
            pose = view.cam_from_world().matrix()
            fx, fy, cx, cy = camera.params
            row = {"reference": i, "source": record["source"]}
            with np.load(root / record["file"]) as data:
                for name in ("raw_xyz", "accepted_xyz"):
                    points = data[name]
                    local = points @ pose[:, :3].T + pose[:, 3]
                    visible = np.flatnonzero(local[:, 2] > 0)
                    uv = local[visible, :2] / local[visible, 2:3] * [fx, fy] + [cx, cy]
                    pixel = np.floor(uv).astype(int)
                    inside = ((pixel >= 0) & (pixel < [camera.width, camera.height])).all(axis=1)
                    visible, pixel = visible[inside], pixel[inside]
                    expected = depth[pixel[:, 1], pixel[:, 0]]
                    finite = np.isfinite(expected) & (expected > 0)
                    rows, expected = visible[finite], expected[finite]
                    relative = abs(local[rows, 2] - expected) / expected
                    row[name] = {"total": len(points), "valid_colmap_depth": len(rows),
                                 "valid_depth_fraction": len(rows) / len(points) if len(points) else None,
                                 "relative_depth_error_percentiles": percentiles(relative),
                                 "matched_2pct_fraction": float((relative <= .02).mean()) if len(rows) else None,
                                 "matched_5pct_fraction": float((relative <= .05).mean()) if len(rows) else None}
                    if probe is not None:
                        x, y, z = ((points - frame["origin"]) @ frame["basis"]).T
                        lo, hi = probe["localized_y"]
                        selected = (x < probe["x_max"]) & (z > probe["z_min"]) & (y >= lo) & (y <= hi)
                        row[name]["probe"] = {"project_samples": int(selected.sum()),
                                              "valid_colmap_depth": int(selected[rows].sum()),
                                              "matched_5pct": int((selected[rows] & (relative <= .05)).sum())}
            views_report.append(row)
        results[label] = views_report
    return {"method": "Same-pose nearest-pixel comparison with external geometric depth. Agreement is conditioned on valid "
                      "COLMAP depths and can be affected by occlusion or different surface visibility.", "variants": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--strip-probe", type=Path, help="JSON with bounds of a previously measured spatial probe")
    parser.add_argument("--variant", nargs=2, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.workspace, read(args.frame), [(label, Path(path)) for label, path in args.variant],
                     read(args.strip_probe)["bounds"] if args.strip_probe else None)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
