"""Optional per-pair depth maps and bounded links to fused observations."""
import numpy as np

from .pair import write_json


def save_depth_diagnostics(output, maps, accepted, inverse, plans, poses, limit=6000, *, recovery=None):
    folder = output / "depth-observations"
    folder.mkdir()
    checked = {}
    offset = 0
    for depth, mask, _ in accepted:
        count = int(mask.sum())
        checked[depth.reference] = (mask, inverse[offset:offset + count])
        offset += count
    records = []
    for depth in maps:
        raw = np.linspace(0, len(depth.xyz) - 1, min(len(depth.xyz), limit), dtype=int)
        mask, fused = checked.get(depth.reference, (np.zeros(len(depth.xyz), bool), np.empty(0, dtype=int)))
        kept = np.flatnonzero(mask)
        sample = np.linspace(0, len(kept) - 1, min(len(kept), limit), dtype=int)
        raster_mask = np.zeros(depth.depth.shape, bool)
        y, x = depth.pixels[kept].T
        raster_mask[y, x] = True
        plan = plans[depth.reference]
        name = f"view-{depth.reference:04d}.npz"
        observations = {"depth": depth.depth, "accepted_raster": raster_mask, "raw_xyz": depth.xyz[raw],
                        "raw_pixels_yx": depth.pixels[raw], "accepted_xyz": depth.xyz[kept[sample]],
                        "accepted_pixels_yx": depth.pixels[kept[sample]], "fused_vertex": fused[sample],
                        "rotation": depth.rotation, "translation": depth.translation, "intrinsic": depth.intrinsic,
                        "projection_a": plan.projections[0], "projection_b": plan.projections[1],
                        "disparity_to_depth": plan.disparity_to_depth}
        if depth.witness_bits is not None:
            observations["accepted_witness_bits"] = depth.witness_bits[kept[sample]]
        if recovery and depth.reference in recovery:
            votes = []
            for reference, source, rows in recovery[depth.reference]:
                matched = np.searchsorted(kept, rows)
                votes.append(np.column_stack((depth.pixels[rows], fused[matched],
                                              np.full(len(rows), reference), np.full(len(rows), source))))
            observations["recovered_yx_vertex_reference_source"] = np.concatenate(votes)
        np.savez_compressed(folder / name, **observations)
        records.append({"reference": depth.reference, "source": depth.source, "file": name,
                        "recentered": depth.recentered, "transposed": depth.transposed,
                        "rectified_overlap_pixels": depth.overlap_pixels,
                        "raw": len(depth.xyz), "accepted": len(kept), "baseline": float(np.linalg.norm(
                            poses[depth.reference].center - poses[depth.source].center)),
                        "disparity_range": [plan.minimum, plan.minimum + plan.count - 1],
                        "depth_bounds": plan.depth_bounds})
    write_json(folder / "manifest.json", {"coordinate_frame": "input SfM world and rectified pixel coordinates",
                                         "sampling": f"At most {limit} uniformly indexed raw and accepted pixels per map",
                                         "witness_bit_order": [depth.reference for depth in maps], "pairs": records})
