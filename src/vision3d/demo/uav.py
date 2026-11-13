"""Launch the shared geospatial workflow and read its verified run record."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

from .support import allocate_run, child_options, reported_stage


UAV_PROFILE = "UAV Blender scene"


def scene_command(run, selected):
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if (record["status"] not in {"verified", "partial"}
            or Path(record["input"]) != Path(selected).resolve()):
        raise ValueError("The current photo folder has no verified Blender scene.")
    blend = Path(record["blend"])
    if blend.resolve() != run.resolve() / "scene/terrain_white_buildings.blend":
        raise ValueError("The scene is outside the current run or is not its published BLEND.")
    with blend.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != record["blend_sha256"]:
        raise ValueError("The scene changed after Blender verification.")
    executable = Path(record["blender"])
    if not executable.is_file():
        raise FileNotFoundError(f"Blender executable is missing: {executable}")
    return [str(executable), str(blend)]


def run_uav(root, folder, base, emit, stop):
    run = None
    log = None
    try:
        run = allocate_run(base)
        emit("run", str(run))
        log = run / "workflow.log"
        legacy = root / "outputs/mission-omega/building-recovery"
        python = Path(os.environ.get("VISION3D_GIS_PYTHON", legacy / "runtime/venv/Scripts/python.exe"))
        if not python.is_file():
            raise FileNotFoundError("Set VISION3D_GIS_PYTHON to the local GIS/GeoAI Python executable.")
        libraries = [root / "outputs/mission-omega/reusable-uav/gis-extra",
                     legacy / "terrain-validation/alternate-python"]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([*map(str, libraries), env.get("PYTHONPATH", "")])
        env["PYTHONIOENCODING"] = "utf-8"
        command = [python.resolve(), "-u", root / "scripts/omega_desktop.py", Path(folder).resolve(), "--output", run]
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, **child_options())
            stage = None
            while process.poll() is None:
                current = reported_stage(log, "Starting UAV reconstruction")
                if current != stage:
                    emit("stage", current)
                    stage = current
                if stop.wait(0.2):
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       stdout=stream, stderr=subprocess.STDOUT, **child_options())
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    emit("notice", f"Cancelled. Incomplete run: {run}")
                    return
        path = run / "run.json"
        record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if process.returncode:
            raise ValueError(f"{record.get('stage', 'Workflow startup')}: "
                             f"{record.get('error', f'process exited with status {process.returncode}')}"
                             f"\nStage log: {record.get('log', log)}")
        scene_command(run, folder)
        emit("uav_result", record)
    except (OSError, ValueError, KeyError) as exc:
        if log:
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"{exc}\n")
        emit("error", f"{exc}\nWorkflow log: {log or base}")
