"""Run ODM natively and read its products. ODM does the photogrammetry; this module only
prepares the project, calls it in phases and computes the metric GSD from its SfM result."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from uav3d.record import run_measured
from uav3d.tools import Odm

PROJECT = "project"
PRODUCTS = {
    "point_cloud": "odm_georeferencing/odm_georeferenced_model.laz",
    "dsm": "odm_dem/dsm.tif",
    "dtm": "odm_dem/dtm.tif",
    "orthophoto": "odm_orthophoto/odm_orthophoto.tif",
    "reconstruction": "opensfm/reconstruction.json",
}
OPTIONAL_PRODUCTS = ("reconstruction",)
OBLIQUE_WARNING = ("Oblique capture: most images look sideways rather than down. The pipeline targets nadir "
                   "mapping flights; terrain and buildings from this image set are out of scope and unreliable.")


class OdmError(RuntimeError):
    def __init__(self, message: str, result: dict | None = None):
        super().__init__(message)
        self.result = result or {}


def prepare(images: list[Path], workdir: Path) -> Path:
    """Create <workdir>/project/images with hard links (copies across volumes). Inputs stay untouched."""
    target = workdir / PROJECT / "images"
    target.mkdir(parents=True, exist_ok=True)
    for src in images:
        dst = target / src.name
        if dst.exists():
            if dst.stat().st_size == src.stat().st_size:
                continue
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return workdir / PROJECT


FINISHED = "ODM app finished"


def run(odm: Odm, workdir: Path, flags: list[str], log: Path, cpus: int | None = None) -> dict:
    """Run ODM; success requires exit code 0 AND ODM's completion line in this run's log.
    ``cpus`` pins ODM and its children to N CPUs (ODM's own thread flag misses some sub-steps)."""
    # ODM runs with cwd = its install dir, so a relative project path would resolve there
    cmd = odm.run_cmd(["--project-path", str(Path(workdir).resolve()), PROJECT, *flags])
    start = log.stat().st_size if log.exists() else 0
    result = run_measured(cmd, log, cwd=odm.home, env=odm.env(), cpus=cpus)
    result["command"] = cmd
    with open(log, "rb") as fh:
        fh.seek(start)
        this_run = fh.read().decode("utf-8", errors="replace")
    if result["returncode"] != 0 or FINISHED not in this_run:
        raise OdmError(f"ODM failed (exit {result['returncode']}); see {log}: {tail(log)}", result)
    return result


def tail(log: Path, lines: int = 3) -> str:
    try:
        text = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    errors = [s for s in text if "error" in s.lower() or "exception" in s.lower()]
    return " | ".join((errors or text)[-lines:])


def _rotation(angle_axis: list[float]) -> np.ndarray:
    v = np.asarray(angle_axis, dtype=float)
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def gsd_from_reconstruction(path: Path) -> dict:
    """Median ground sampling distance (m/px) of the largest OpenSfM reconstruction.

    Per shot: (camera height above the median point height) / (normalised focal * max(w, h)).
    Coordinates are metric only when the reconstruction is georeferenced; the caller decides.
    """
    recs = json.loads(path.read_text(encoding="utf-8"))
    if not recs:
        raise OdmError(f"Empty reconstruction: {path}")
    rec = max(recs, key=lambda r: len(r.get("shots", {})))
    points = np.array([p["coordinates"] for p in rec.get("points", {}).values()], dtype=float)
    if len(points) == 0:
        raise OdmError("Reconstruction has no points.")
    ground = float(np.median(points[:, 2]))
    gsds = []
    for shot in rec["shots"].values():
        cam = rec["cameras"][shot["camera"]]
        focal = cam.get("focal", cam.get("focal_x"))
        if not focal:
            continue
        center = -_rotation(shot["rotation"]).T @ np.asarray(shot["translation"], dtype=float)
        height = center[2] - ground
        if height > 0:
            gsds.append(height / (focal * max(cam["width"], cam["height"])))
    if not gsds:
        raise OdmError("Could not compute GSD: no shot above the reconstructed ground.")
    return {"gsd_m": round(float(np.median(gsds)), 4), "shots": len(rec["shots"]),
            "points": len(points), "reconstructions": len(recs)}


def view_angles(path: Path, oblique_deg: float = 20.0) -> dict:
    """Off-nadir angle of every shot's optical axis in the largest OpenSfM reconstruction (topocentric,
    z up; the axis in world coordinates is R^T [0, 0, 1]). Nadir mapping flights stay within a few
    degrees; oblique capture (facades, towers) does not. Works for any camera, with or without
    gimbal metadata."""
    recs = json.loads(path.read_text(encoding="utf-8"))
    rec = max(recs, key=lambda r: len(r.get("shots", {})))
    off = []
    for shot in rec["shots"].values():
        axis = _rotation(shot["rotation"]).T @ np.array([0.0, 0.0, 1.0])
        off.append(float(np.degrees(np.arccos(np.clip(-axis[2], -1.0, 1.0)))))
    off = np.array(off)
    return {"shots": len(off), "median_off_nadir_deg": round(float(np.median(off)), 1) if len(off) else None,
            "share_oblique": round(float((off > oblique_deg).mean()), 3) if len(off) else 0.0}


def is_oblique(angles: dict, ocfg: dict) -> bool:
    return angles.get("share_oblique", 0.0) >= ocfg["min_share"]


KEEP_ALWAYS = ("opensfm/reconstruction.json", "opensfm/stats/", "odm_report/", "images/", "options.json",
               "cameras.json", "odm_georeferencing/coords.txt", "odm_georeferencing/proj.txt")
KEEP_SFM = ("opensfm/reconstruction.json", "opensfm/stats/")  # also kept inside submodels/ (split runs)
SMALL_FILE = 1 << 20


def prune(project: Path) -> dict:
    """Delete ODM's intermediate files (undistorted images, depth maps, features, matches,
    meshing/texturing work) after a run; keep the products every later stage and --reuse-odm
    need, the SfM result, statistics/report, the hard-linked inputs and small text files.
    Intermediates are 5-50 GB per scene; the products a few GB."""
    keep = set(PRODUCTS.values())
    freed = 0
    for path in sorted(project.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        rel = path.relative_to(project).as_posix()
        if path.is_file():
            # split runs keep their SfM under submodels/opensfm/ and submodels/submodel_*/opensfm/
            sub = None
            if rel.startswith("submodels/"):
                sub = rel[len("submodels/"):]
                if not sub.startswith("opensfm/") and "/" in sub:
                    sub = sub.split("/", 1)[1]
            if (rel in keep or any(rel == k or rel.startswith(k) for k in KEEP_ALWAYS)
                    or (sub is not None and any(sub == k or sub.startswith(k) for k in KEEP_SFM))
                    or path.stat().st_size < SMALL_FILE):
                continue
            freed += path.stat().st_size
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    missing = [k for k, v in products(project).items() if v is None and k not in OPTIONAL_PRODUCTS]
    return {"freed_gb": round(freed / 2**30, 2), "missing_products_after_prune": missing}


def products(project: Path) -> dict[str, str | None]:
    return {k: (str(project / rel) if (project / rel).is_file() else None) for k, rel in PRODUCTS.items()}
