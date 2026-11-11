"""Trace a fixed gravity-aligned elevation region to depth samples and their witnesses."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from analyze_airzoo_geometry import original_pixels, read, station_metadata


def trace(sparse, frame, variants, x_max, z_min):
    _, views, stations, *_ = station_metadata(sparse)
    origin = np.array(frame["origin"])
    basis = np.array(frame["basis"])

    def region(world):
        xyz = (world - origin) @ basis
        return (xyz[:, 0] < x_max) & (xyz[:, 2] > z_min)

    runs = {}
    for label, dense in variants:
        cloud = o3d.io.read_point_cloud(str(dense / "dense.ply"))
        root = dense / "depth-observations"
        provenance = read(root / "manifest.json")
        records = []
        for row in provenance["pairs"]:
            i, j = row["reference"], row["source"]
            with np.load(root / row["file"]) as data:
                raw = int(region(data["raw_xyz"]).sum())
                accepted = np.flatnonzero(region(data["accepted_xyz"]))
                if not raw and not len(accepted):
                    continue
                combinations = Counter()
                for mask in data["accepted_witness_bits"][accepted]:
                    combinations[tuple(k for bit, k in enumerate(provenance["witness_bit_order"])
                                       if int(mask) & (1 << bit))] += 1
                examples = []
                for k in accepted[::max(1, len(accepted) // 3)][:3]:
                    world = data["accepted_xyz"][k:k + 1]
                    yx = data["accepted_pixels_yx"][k]
                    examples.append({"world": world[0].tolist(), "reference_original_uv": original_pixels(world, views[i])[0].tolist(),
                                     "source_original_uv": original_pixels(world, views[j])[0].tolist(),
                                     "rectified_yx": yx.tolist(), "rectified_depth": float(data["depth"][tuple(yx)]),
                                     "fused_vertex": int(data["fused_vertex"][k])})
                records.append({"reference": i, "source": j, "reference_station": stations[i],
                                "source_station": stations[j], "baseline": row["baseline"], "raw_samples": raw,
                                "accepted_samples": len(accepted), "examples": examples,
                                "witness_combinations": [{"references": list(key), "samples": count}
                                                         for key, count in combinations.most_common()]})
        runs[label] = {"fused_vertices_in_region": int(region(np.asarray(cloud.points)).sum()), "views": records}
    return {"scope": "Post hoc spatial probe on a fixed shared frame. Per-view counts are bounded samples; fused counts are exact.",
            "x_upper_exclusive": x_max, "z_lower_exclusive": z_min, "runs": runs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--variant", nargs=2, action="append", required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--z-min", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = trace(args.sparse, read(args.frame), [(label, Path(path)) for label, path in args.variant],
                 args.x_max, args.z_min)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
