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


class OdmError(RuntimeError):
    pass


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


def run(odm: Odm, workdir: Path, flags: list[str], log: Path) -> dict:
    cmd = [str(odm.run_bat), "--project-path", str(workdir), PROJECT, *flags]
    result = run_measured(cmd, log, cwd=odm.home)
    result["command"] = cmd
    if result["returncode"] != 0:
        raise OdmError(f"ODM exited with {result['returncode']}; see {log}: {tail(log)}")
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


def products(project: Path) -> dict[str, str | None]:
    return {k: (str(project / rel) if (project / rel).is_file() else None) for k, rel in PRODUCTS.items()}
