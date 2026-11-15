"""Before/after terrain figure for the report: ODM's DTM next to the cleaned DTM of the same run, both shown
as the deviation of every cell from its local median - the reference-free pit/spike measure of
uav3d.terrain.health (same cell, window and depth from configs/defaults.json).

usage: python bench/make_dtm_figure.py <run dir> <out.pdf> [--clip 10] [--window 300]
Pits are single cells, invisible over a whole block: both panels show the same --window (m) centred on the
deepest pit of ODM's DTM.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

from uav3d import plan

# diverging: blue (below the local median: pits) <-> neutral gray <-> red (above: spikes)
CMAP = LinearSegmentedColormap.from_list(
    "pits", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3b6b5", "#e34948", "#8f1f1f"])


def deviation(dtm_path: Path, tcfg: dict):
    with rasterio.open(dtm_path) as ds:
        step = max(1, int(round(tcfg["health_cell_m"] / ds.res[0])))
        a = ds.read(1, masked=True)[::step, ::step]
        res = ds.res[0] * step
        b = ds.bounds
    filled = a.filled(np.ma.median(a))
    win = max(3, int(round(tcfg["local_window_m"] / res)) | 1)
    dev = np.ma.masked_array(filled - ndimage.median_filter(filled, size=win), mask=np.ma.getmaskarray(a))
    return dev, (b.left, b.right, b.bottom, b.top)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--window", type=float, default=300.0, help="side of the shown square [m]; 0 = whole block")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tcfg = plan.defaults()["terrain"]
    rec = json.loads((a.run / "run.json").read_text(encoding="utf-8"))
    st = next(s for s in rec["stages"] if s["name"] == "terrain")
    pairs = [("ODM DTM", a.run / "odm" / "project" / "odm_dem" / "dtm.tif", st["health_odm_dtm"]),
             ("Temizlenmiş DTM", a.run / "terrain" / "dtm.tif", st["health_dtm"])]
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    devs = [deviation(p, tcfg) for _, p, _ in pairs]
    d0, e0 = devs[0]
    iy, ix = np.unravel_index(np.ma.argmin(d0), d0.shape)  # deepest pit of ODM's DTM
    cx = e0[0] + (ix + 0.5) * (e0[1] - e0[0]) / d0.shape[1]
    cy = e0[3] - (iy + 0.5) * (e0[3] - e0[2]) / d0.shape[0]
    half = a.window / 2 if a.window else None
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.6), sharex=True, sharey=True)
    for ax, (title, path, h), (dev, ext) in zip(axes, pairs, devs):
        # metres relative to the window's lower-left corner (or the block's, for the whole block)
        ox, oy = (cx - half, cy - half) if half else (e0[0], e0[2])
        im = ax.imshow(dev, cmap=CMAP, vmin=-a.clip, vmax=a.clip, interpolation="nearest", rasterized=True,
                       extent=(ext[0] - ox, ext[1] - ox, ext[2] - oy, ext[3] - oy))
        if half:
            ax.set_xlim(0, a.window)
            ax.set_ylim(0, a.window)
        pit = f"{100 * h['pit_fraction']:.1f}".replace(".", ",")
        deep = f"{h['deepest_pit_m']:.0f}".replace("-", "−")
        ax.set_title(f"{title}\nbütün blok: çukur hücre %{pit},\nen derin çukur {deep} m", fontsize=8)
        ax.set_xlabel("Doğu [m]")
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Kuzey [m]")
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.18, extend="both")
    cb.set_label("Yerel medyandan sapma [m] (mavi: çukur, kırmızı: sivrilik)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", dpi=300)
    fig.savefig(a.out.with_suffix(".png"), bbox_inches="tight", dpi=160)
    print("wrote", a.out, "window centre (projected):", round(cx, 1), round(cy, 1), "deepest", float(d0.min()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
