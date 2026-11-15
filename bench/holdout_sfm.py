"""Unsplit ODM SfM of a block (hold-out input of the adaptive experiment).

The scheduler and all baselines must share one SfM and one camera model. The production run of a large
UseGeo strip falls back to ODM's split-merge (one SfM per submodel), so the experiment re-runs ODM up to and
including OpenSfM (features, matching, reconstruction, undistortion, statistics) without --split, with the
same flags otherwise. Dense matching is then done by bench/fine_pass.py (coarse and fine), whose memory is
bounded by OpenMVS sub-scenes.

usage: python bench/holdout_sfm.py <uav3d input dir> <out dir> [--threads 10]
"""
import argparse
import json
from pathlib import Path

from uav3d import odm, tools


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="dataset uav3d_input (images/ with geo.txt)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    # ODM runs from its install dir: every path handed to it must be absolute (a relative --geo is silently
    # ignored and the SfM ends up unreferenced)
    a.input, a.out = a.input.resolve(), a.out.resolve()
    images_dir = a.input / "images"
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"))
    geo = images_dir / "geo.txt"
    o = tools.odm()
    workdir = a.out / "odm"
    project = odm.prepare(images, workdir)
    flags = ["--dsm", "--dtm", "--skip-3dmodel", "--auto-boundary", "--pc-quality", "medium",
             "--max-concurrency", str(a.threads), "--end-with", "opensfm"]
    if geo.is_file():
        flags += ["--geo", str(geo)]
    rec = {"input": str(a.input), "images": len(images), "flags": flags}
    rec["odm"] = odm.run(o, workdir, flags, a.out / "logs" / "odm.log", cpus=a.threads)
    if not (project / "odm_georeferencing" / "coords.txt").is_file():
        raise SystemExit("SfM is not georeferenced (no odm_georeferencing/coords.txt); check --geo / EXIF GPS")
    rec["gsd"] = odm.gsd_from_reconstruction(project / "opensfm" / "reconstruction.json")
    (a.out / "sfm.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("images", "gsd")}), json.dumps({k: rec["odm"].get(k) for k in ("wall_s", "peak_rss_gb")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
