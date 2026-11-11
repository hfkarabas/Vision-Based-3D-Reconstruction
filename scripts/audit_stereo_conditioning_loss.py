"""Measure which raw pixels disappear when conditioning is applied to unchanged stereo pairs."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_airzoo_geometry import read
from compare_airzoo_dsm import q, survey_alignment
from vision3d.geometry.depth import unproject_depth


def compare(sparse, dsm, baseline, conditioned, offset):
    raster, georeference, _ = survey_alignment(sparse, dsm)
    baseline, conditioned = (directory / "depth-observations" for directory in (baseline, conditioned))
    maps_before, maps_after = [{entry["reference"]: entry for entry in read(path / "manifest.json")["pairs"]}
                               for path in (baseline, conditioned)]
    records = []
    errors_by_status = {"retained": [], "below_effective_disparity": [], "lost_witness_support": []}
    threshold = read(conditioned.parent / "metrics.json")["min_effective_disparity_px"]
    if not threshold or not np.isfinite(offset):
        raise ValueError("Expected a conditioned run and a finite fixed DSM offset")

    def summarize(error):
        valid = error[np.isfinite(error)]
        return {"pixels": len(error), "valid_dsm": len(valid), "signed_residual_m": q(valid),
                "absolute_residual_m": q(abs(valid))}

    for i, entry in maps_before.items():
        other = maps_after.get(i)
        row = {"reference": i, "source": entry["source"]}
        records.append(row)
        if other is None or entry["source"] != other["source"]:
            row["status"] = "stereo_pair_changed"
            continue
        with np.load(baseline / entry["file"]) as before, np.load(conditioned / other["file"]) as after:
            depth = before["depth"]
            if any(not np.array_equal(before[key], after[key])
                   for key in ("depth", "rotation", "translation", "intrinsic", "projection_b")):
                row["status"] = "rectified_geometry_changed"
                continue
            retained = before["accepted_raster"] & after["accepted_raster"]
            dropped = before["accepted_raster"] & ~after["accepted_raster"]
            effective = abs(before["projection_b"][:2, 3]).max()
            weak = depth > effective / threshold
            row.update(status="compared", gained_pixels=int((after["accepted_raster"] & ~before["accepted_raster"]).sum()))
            for name, mask in (("retained", retained), ("below_effective_disparity", dropped & weak),
                               ("lost_witness_support", dropped & ~weak)):
                y, x = np.nonzero(mask)
                xyz = unproject_depth(np.column_stack((x, y)), depth[y, x], before["rotation"],
                                      before["translation"], before["intrinsic"])
                geo = georeference(xyz)
                error = geo[:, 2] - raster.sample(geo[:, :2]) - offset
                row[name] = summarize(error)
                errors_by_status[name].append(error)
    return {"minimum_effective_disparity_px": threshold, "fixed_baseline_offset_m": offset,
            "summary": {name: summarize(np.concatenate(parts)) for name, parts in errors_by_status.items() if parts},
            "pairs": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--dsm", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--conditioned", type=Path, required=True)
    parser.add_argument("--fixed-offset", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.sparse, args.dsm, args.baseline, args.conditioned, args.fixed_offset)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
