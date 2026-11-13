"""Locate and verify the pinned external tools (tools.lock.json).

Lookup order per tool: explicit environment variable, then ``$UAV3D_TOOLS/<tool>``
(default ``<repo>/.tools``). Nothing is looked up under ``outputs/``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class ToolError(RuntimeError):
    """A required external tool is missing or has the wrong version."""


def lock() -> dict:
    return json.loads((REPO / "tools.lock.json").read_text(encoding="utf-8"))


def tools_root() -> Path:
    return Path(os.environ.get("UAV3D_TOOLS", REPO / ".tools")).expanduser()


@dataclass(frozen=True)
class Odm:
    home: Path
    version: str

    @property
    def bin(self) -> Path:
        return self.home / "SuperBuild" / "install" / "bin"

    @property
    def run_bat(self) -> Path:
        return self.home / "run.bat"

    def exe(self, name: str) -> Path:
        path = self.bin / f"{name}.exe"
        if not path.is_file():
            raise ToolError(f"{name}.exe not found in ODM bundle {self.bin}")
        return path

    def env(self) -> dict[str, str]:
        """Environment for ODM's bundled GDAL/PDAL executables (mirrors win32env.bat)."""
        env = os.environ.copy()
        env.update(
            PATH=os.pathsep.join([str(self.bin), str(self.bin / "opensfm" / "bin"), env.get("PATH", "")]),
            GDAL_DATA=str(self.bin / "data" / "gdal"),
            GDAL_DRIVER_PATH=str(self.bin / "gdalplugins"),
            PDAL_DRIVER_PATH=str(self.bin),
            PROJ_LIB=str(self.bin / "data" / "proj"),
        )
        for key in ("PYTHONHOME", "PYTHONPATH"):
            env.pop(key, None)
        return env


def odm() -> Odm:
    expected = lock()["odm"]["version"]
    home = Path(os.environ["UAV3D_ODM"]) if os.environ.get("UAV3D_ODM") else tools_root() / "odm"
    version_file = home / "VERSION"
    if not version_file.is_file():
        raise ToolError(f"ODM not found at {home}. Run setup.ps1 or set UAV3D_ODM.")
    version = version_file.read_text(encoding="utf-8").strip()
    if version != expected:
        raise ToolError(f"ODM at {home} is {version}; tools.lock.json requires {expected}.")
    return Odm(home=home, version=version)


def roofer() -> Path:
    if os.environ.get("UAV3D_ROOFER"):
        exe = Path(os.environ["UAV3D_ROOFER"])
    else:
        found = sorted((tools_root() / "roofer").rglob("roofer.exe"))
        exe = found[0] if found else tools_root() / "roofer" / "roofer.exe"
    if not exe.is_file():
        raise ToolError(f"Roofer not found ({exe}). Run setup.ps1 or set UAV3D_ROOFER.")
    return exe


def blender() -> Path | None:
    """Optional. Returns None when Blender is not installed."""
    if os.environ.get("UAV3D_BLENDER"):
        return Path(os.environ["UAV3D_BLENDER"])
    if which := shutil.which("blender"):
        return Path(which)
    root = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Blender Foundation"
    found = sorted(root.glob("Blender */blender.exe"), reverse=True)
    return found[0] if found else None


def doctor() -> list[tuple[str, bool, str]]:
    """Check every tool; never raises."""
    rows: list[tuple[str, bool, str]] = []
    try:
        o = odm()
        pdal = subprocess.run([str(o.exe("pdal")), "--version"], capture_output=True, text=True,
                              env=o.env(), timeout=60)
        line = next((s for s in pdal.stdout.splitlines() if s.startswith("pdal")), "?")
        rows.append(("odm", True, f"{o.version} at {o.home}; {line.strip()}"))
    except (ToolError, OSError, subprocess.SubprocessError) as exc:
        rows.append(("odm", False, str(exc)))
    try:
        exe = roofer()
        out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60)
        rows.append(("roofer", out.returncode == 0, (out.stdout or out.stderr).strip() or str(exe)))
    except (ToolError, OSError, subprocess.SubprocessError) as exc:
        rows.append(("roofer", False, str(exc)))
    b = blender()
    rows.append(("blender (optional)", True, str(b) if b else "not installed; Blender export disabled"))
    return rows
