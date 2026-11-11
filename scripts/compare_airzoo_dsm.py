"""Compare AirZoo elevations with a PPK/gimbal-aligned DSM using strip reads."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image

from analyze_airzoo_geometry import read, station_metadata


class FloatStripRaster:
    def __init__(self, path):
        path = Path(path)
        limit = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = None  # Read metadata only. Access values through the memory map below.
            with Image.open(path) as image:
                tags = image.tag_v2
                self.width, self.height = image.size
                if tags.get(258) != (32,) or tags.get(339) != (3,) or tags.get(259) != 1 or 273 not in tags:
                    raise ValueError("Expected an uncompressed stripped float32 DSM")
                self.offsets = np.asarray(tags[273], dtype=np.int64)
                self.rows_per_strip = tags[278]
                self.nodata = float(tags[42113]) if 42113 in tags else None
        finally:
            Image.MAX_IMAGE_PIXELS = limit
        with path.open("rb") as stream:
            order = stream.read(2)
        self.data = np.memmap(path, mode="r", dtype=np.uint8)
        self.dtype = "<f4" if order == b"II" else ">f4"
        a, d, b, e, c, f = np.loadtxt(path.with_suffix(".tfw"))
        self.affine = np.array([[a, b], [d, e]])
        self.origin = np.array([c, f])

    def sample(self, xy):
        pixels = np.rint((xy - self.origin) @ np.linalg.inv(self.affine).T).astype(np.int64)
        valid = ((pixels >= 0) & (pixels < [self.width, self.height])).all(axis=1)
        x, y = pixels[valid].T
        offset = self.offsets[y // self.rows_per_strip] + 4 * ((y % self.rows_per_strip) * self.width + x)
        output = np.full(len(xy), np.nan)
        values = np.asarray(self.data[offset[:, None] + np.arange(4)]).reshape(-1).view(self.dtype)
        values = values.astype(np.float64)
        # An untagged extreme Float32 sentinel is not an Earth elevation.
        values[(~np.isfinite(values)) | (abs(values) > 1e6) | (values == self.nodata)] = np.nan
        output[valid] = values
        return output


def fit_horizontal(world, projected):
    a = world[:, 0] + 1j * world[:, 1]
    b = projected[:, 0] + 1j * projected[:, 1]
    weights = np.ones(len(a))
    for _ in range(5):
        aa, bb = np.average(a, weights=weights), np.average(b, weights=weights)
        coefficient = np.sum(weights * (b - bb) * np.conj(a - aa)) / np.sum(weights * np.abs(a - aa) ** 2)
        offset = bb - coefficient * aa
        error = abs(coefficient * a + offset - b)
        weights = 1 / np.maximum(error, .25)
    return coefficient, offset, error


def q(x):
    return np.percentile(x, [5, 50, 90, 95, 99]).tolist() if len(x) else None


def cell_medians(xy, values):
    """Summarize residuals once per sampled half-meter cell to limit density bias."""
    cells, inverse, counts = np.unique(np.floor(xy / .5).astype(np.int64), axis=0,
                                      return_inverse=True, return_counts=True)
    order = np.lexsort((values, inverse))
    middle = np.cumsum(counts) - 1 - (counts - 1) // 2
    medians = (values[order[middle]] + values[order[middle - (counts % 2 == 0)]]) / 2
    return dict(zip(map(tuple, cells), medians, strict=True))


def evaluation(sparse, dsm, variants, output, fixed_offset=None):
    from pyproj import Transformer

    _, views, stations, geodetic, _, basis, tilt = station_metadata(sparse)
    xml = ET.parse(dsm.parent / "metadata.xml").getroot()
    crs = xml.findtext("SRS")
    if not crs:
        raise ValueError("The dataset must specify the DSM CRS")
    raster = FloatStripRaster(dsm)
    positions = np.stack([geodetic[i] for i in views])
    east, north = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(positions[:, 1], positions[:, 0])
    xy = np.column_stack((east, north))
    local = np.stack([v.pose.center for v in views.values()]) @ basis
    station = np.array([stations[i] for i in views])
    coefficient, offset, residual = fit_horizontal(local, xy)
    scale = abs(coefficient)
    z_offset = np.median(positions[:, 2] - scale * local[:, 2])
    z_residual = abs(scale * local[:, 2] + z_offset - positions[:, 2])
    held = {}
    for number in np.unique(station):
        train = station != number
        if len(np.unique(station[train])) < 2:
            continue
        value, shift, _ = fit_horizontal(local[train], xy[train])
        test = ~train
        measured = xy[test, 0] + 1j * xy[test, 1]
        held[int(number)] = q(abs(value * (local[test, 0] + 1j * local[test, 1]) + shift - measured))
    def geo(points):
        local = points @ basis
        xy_complex = coefficient * (local[:, 0] + 1j * local[:, 1]) + offset
        return np.column_stack((xy_complex.real, xy_complex.imag, scale * local[:, 2] + z_offset))

    if fixed_offset is not None and not np.isfinite(fixed_offset):
        raise ValueError("The DSM offset must be finite")
    result = {"horizontal_crs": crs, "srs_origin": xml.findtext("SRSOrigin"), "pixel_transform": raster.affine.tolist(),
              "raster_shape": [raster.height, raster.width], "nodata_tag": raster.nodata,
              "meters_per_sfm_unit": scale, "alignment_up_tilt_deg": q(tilt), "ppk_fit_horizontal_m": q(residual),
              "ppk_fit_ellipsoidal_height_m": q(z_residual), "leave_one_station_out_horizontal_m": held,
              "note": "Alignment uses only PPK and gimbal metadata. The DSM vertical datum is unverified. Centered residuals use the fixed baseline median as a datum proxy and do not establish absolute accuracy.",
              "variants": {}}
    output.mkdir(parents=True, exist_ok=False)
    clouds = []
    cell_counts = {}
    residuals_by_cell = {}
    baseline = fixed_offset
    for name, directory in variants:
        directory = Path(directory)
        points = np.asarray(o3d.io.read_point_cloud(str(directory / "dense.ply")).points)
        xy = geo(points)[:, :2]
        cells = np.unique(np.floor(xy / .5).astype(np.int64), axis=0)
        cell_counts[name] = set(map(tuple, cells))
        samples = geo(points[::max(1, (len(points) + 179999) // 180000)])
        depth = raster.sample(samples[:, :2])
        mask = np.isfinite(depth)
        delta = samples[mask, 2] - depth[mask]
        if not mask.any():
            result["variants"][name] = {"samples": len(samples), "valid_dsm_samples": 0}
            continue
        if baseline is None:
            baseline = float(np.median(delta))
        centered = delta - baseline
        residuals_by_cell[name] = cell_medians(samples[mask, :2], abs(centered))
        xy_cells = np.unique(np.floor(samples[mask, :2] / .5), axis=0)
        result["variants"][name] = {"samples": len(samples), "valid_dsm_samples": int(mask.sum()),
             "occupied_half_meter_cells": len(cells), "sampled_half_meter_cells": len(xy_cells),
             "delta_height_m": q(delta),
             "absolute_delta_after_fixed_baseline_offset_m": q(abs(centered)),
             "over_10m_from_centered_dsm_fraction": float((abs(centered) > 10).mean()),
             "below_dsm_by_10m_fraction": float((centered < -10).mean()),
             "above_dsm_by_10m_fraction": float((centered > 10).mean()), "views": []}
        provenance_path = directory / "depth-observations/manifest.json"
        clouds.append((name, samples[mask], centered))
        if not provenance_path.is_file():
            continue
        provenance = read(provenance_path)
        for record in provenance["pairs"]:
            with np.load(directory / "depth-observations" / record["file"]) as data:
                row = {k: record[k] for k in ("reference", "source", "recentered")}
                for key in ("raw_xyz", "accepted_xyz"):
                    locations = geo(data[key])
                    height = raster.sample(locations[:, :2])
                    error = locations[:, 2] - height - baseline
                    valid = np.isfinite(error)
                    row[key] = {"valid_samples": int(valid.sum()), "signed_centered_residual_m": q(error[valid]),
                                "over_10m_fraction": float((abs(error[valid]) > 10).mean()) if valid.any() else None}
                    if key == "accepted_xyz" and valid.any():
                        indices = np.flatnonzero(valid)
                        extremes = indices[np.argsort(error[valid])[[0, -1]]]
                        details = []
                        for index in extremes:
                            witness = int(data["accepted_witness_bits"][index])
                            yx = data["accepted_pixels_yx"][index]
                            details.append({"offset_residual_m": float(error[index]), "world": data[key][index].tolist(),
                                            "rectified_pixel_yx": yx.tolist(),
                                            "rectified_depth": float(data["depth"][tuple(yx)]),
                                            "fused_vertex": int(data["fused_vertex"][index]),
                                            "witness_references": [i for bit, i in enumerate(provenance["witness_bit_order"])
                                                                   if witness & (1 << bit)]})
                        row["signed_extremes"] = details
                result["variants"][name]["views"].append(row)
    result["fixed_baseline_dsm_offset_m"] = baseline
    result["offset_source"] = "supplied" if fixed_offset is not None else "median of the first valid variant"
    baseline_cells = cell_counts[next(iter(cell_counts))]
    baseline_residuals = next(iter(residuals_by_cell.values()), {})
    for name, cells in cell_counts.items():
        overlap = baseline_cells & cells
        result["variants"][name].update(
            baseline_half_meter_cells_retained=len(overlap),
            new_half_meter_cells=len(cells - baseline_cells),
            baseline_half_meter_cell_retention=len(overlap) / len(baseline_cells))
        if name in residuals_by_cell:
            residuals = residuals_by_cell[name]
            shared = sorted(baseline_residuals.keys() & residuals.keys())
            result["variants"][name]["common_sampled_cell_residual_m"] = {
                "cells": len(shared), "baseline": q([baseline_residuals[k] for k in shared]),
                "variant": q([residuals[k] for k in shared])}
    if clouds:
        fig, panels = plt.subplots(1, len(clouds), figsize=(5 * len(clouds), 4), squeeze=False, layout="constrained")
        limits = [clouds[0][1][:, :2].min(axis=0), clouds[0][1][:, :2].max(axis=0)]
        for panel, (name, samples, centered) in zip(panels[0], clouds, strict=True):
            handle = panel.scatter(samples[:, 0], samples[:, 1], c=centered, cmap="coolwarm", s=1, vmin=-15, vmax=15)
            panel.set_title(name)
            panel.set_aspect("equal")
            panel.set_xlim(limits[0][0], limits[1][0]); panel.set_ylim(limits[0][1], limits[1][1])
        fig.colorbar(handle, ax=panels[0].tolist(), label="Elevation residual after fixed baseline offset (m)")
        fig.savefig(output / "dsm-residuals.png", dpi=150)
        plt.close(fig)
    (output / "reference_dsm_comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", required=True, type=Path)
    parser.add_argument("--dsm", required=True, type=Path)
    parser.add_argument("--variant", nargs=2, action="append", required=True, metavar=("LABEL", "PATH"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixed-offset", type=float, help="Reuse a baseline vertical offset in meters")
    args = parser.parse_args()
    evaluation(args.sparse, args.dsm, args.variant, args.output, args.fixed_offset)
