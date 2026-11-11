"""Compare frozen stage renders and quantify connected image-space coverage."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

from analyze_airzoo_geometry import read
from audit_reconstruction_visuals import VIEWS


def summarize(mask):
    labels, count = label(mask, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    return {"pixels": int(mask.sum()), "components_8_connected": count,
            "largest_component_pixels": int(sizes.max(initial=0))}


def compare(before, after, output):
    if read(before / "frame.json") != read(after / "frame.json"):
        raise ValueError("Compare audits rendered on the exact same frozen frame")
    output.mkdir(parents=True, exist_ok=False)
    report = {"baseline": str(before.resolve()), "candidate": str(after.resolve()), "views": {}}
    modes = ("dense", "raw-neutral", "photo-dark")
    size = 250
    sheet = Image.new("RGB", (size * 6, (size + 22) * 3), "#e8edf2")
    draw = ImageDraw.Draw(sheet)
    with np.load(before / "occupancy.npz") as old, np.load(after / "occupancy.npz") as new:
        for j, view in enumerate(VIEWS):
            report["views"][view] = {}
            for stage in ("dense", "raw", "optimized", "glb"):
                a, b = old[f"{stage}_{view}"], new[f"{stage}_{view}"]
                if a.shape != b.shape:
                    raise ValueError("Audit raster resolutions differ")
                report["views"][view][stage] = {"baseline": summarize(a), "candidate": summarize(b),
                                               "common_pixels": int((a & b).sum()), "new_pixels": int((b & ~a).sum()),
                                               "lost_pixels": int((a & ~b).sum())}
            for group, root in enumerate((before, after)):
                for i, mode in enumerate(modes):
                    x, y = (group * len(modes) + i) * size, j * (size + 22)
                    with Image.open(root / f"{view}-{mode}.png") as panel:
                        sheet.paste(panel.resize((size, size)), (x, y + 22))
                    draw.text((x + 5, y + 5), f"{('baseline', 'candidate')[group]} · {view} · {mode}", fill="#14202e")
    sheet.save(output / "before-after.png")
    (output / "coverage.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    compare(args.before, args.after, args.output)
