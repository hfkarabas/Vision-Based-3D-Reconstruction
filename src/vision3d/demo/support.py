"""Testable desktop orchestration; no reconstruction algorithms or Tk calls."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import uuid

from PIL import Image, ImageOps


SUFFIXES = {".jpg", ".jpeg", ".png"}


def repository_root():
    """Resolve a source checkout independently of the process working directory."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    return Path.cwd()


def supported_images(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError("Select an existing image folder.")
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES),
                  key=lambda p: (p.name.casefold(), p.name))


def validate_inputs(folder, camera):
    """Decode one image at a time; validate EXIF-oriented dimensions against K."""
    from vision3d.geometry.camera import Camera, load_cameras
    files = supported_images(folder)
    if not 2 <= len(files) <= 30:
        raise ValueError("The demo accepts 2–30 JPEG/PNG files. Select a smaller subset if necessary.")
    if not camera or not Path(camera).is_file():
        raise ValueError("Select a camera calibration JSON before reconstruction.")
    try:
        data = json.loads(Path(camera).read_text(encoding="utf-8"))
        mapping = data.get("images") if isinstance(data, dict) else None
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError("The images calibration field must be an object.")
        if mapping is None:
            a, b = load_cameras(camera)
            if a.to_dict() != b.to_dict():
                raise ValueError("Use shared intrinsics or a per-image mapping, not pair calibration.")
        valid, skipped = [], []
        for path in files:
            try:
                with Image.open(path) as im:
                    if im.format not in {"JPEG", "PNG"}:
                        raise OSError("Not a JPEG or PNG")
                    im.load()
                    size = ImageOps.exif_transpose(im).size
            except (OSError, ValueError) as exc:
                skipped.append(f"{path.name}: {exc}")
                continue
            if mapping is not None:
                if path.name not in mapping:
                    raise ValueError(f"Missing calibration for {path.name}.")
                entry = dict(mapping[path.name])
                matrix = entry.pop("K", None)
                model = Camera(**entry)
                if matrix is not None:
                    import numpy as np
                    if not np.allclose(matrix, model.K):
                        raise ValueError(f"Conflicting K for {path.name}.")
            else:
                model = a
            if model.approximate:
                raise ValueError("This stable demo requires calibrated intrinsics; approximate mode remains CLI-only.")
            if size != (model.width, model.height):
                raise ValueError(f"Calibration dimensions do not match EXIF-oriented image {path.name}.")
            valid.append(path)
    except (TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot validate camera calibration: {exc}") from exc
    if len(valid) < 2:
        raise ValueError(f"At least two readable images are required; found {len(valid)}.")
    return {"supported": len(files), "usable": len(valid), "skipped": skipped}


def resolve_castle(root=None):
    root = Path(root) if root else repository_root()
    folder, camera = root / "datasets/sceaux/images", root / "configs/sceaux.json"
    if not folder.is_dir() or not camera.is_file():
        raise ValueError("Castle demo not found. Expected datasets/sceaux/images and configs/sceaux.json "
                         "in the project checkout. Browse to existing local data; no download is performed.")
    if len(supported_images(folder)) != 6:
        raise ValueError("Castle demo expects the existing six-photo subset; check the dataset folder.")
    return folder.resolve(), camera.resolve()


def allocate_run(base):
    """Atomically reserve a parent; the engine creates its own new result directory."""
    base = Path(base).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = base / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    run.mkdir(exist_ok=False)
    return run


def reconstruction_command(folder, camera, output):
    # Stable registration/BA; independently checked final fixed-camera point refinement.
    # The demo supports pairs as well as longer sequences; a pair requires two registrations.
    minimum = min(5, len(supported_images(folder)))
    return [sys.executable, "-u", "-m", "vision3d.cli", "reconstruct-sequence",
            "--input", str(Path(folder).resolve()), "--camera", str(Path(camera).resolve()),
            "--output", str(Path(output).resolve()), "--local-ba", "--refine-points", "--min-registered", str(minimum)]


@dataclass(frozen=True)
class Result:
    directory: Path
    registered: int
    landmarks: int
    seconds: float
    status: str

    @property
    def ply(self):
        return self.directory / "sparse.ply"


def read_result(directory):
    directory = Path(directory).resolve()
    try:
        data = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if data.get("status") not in {"success", "partial"}:
            raise ValueError(data.get("reason", "Reconstruction was rejected."))
        counts = [data["registered_images"], data["valid_landmarks"]]
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in counts):
            raise ValueError("Invalid camera or landmark counts in metrics.")
        seconds = float(data["runtime_seconds"])
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid execution time in metrics.")
        ply = directory / "sparse.ply"
        if not ply.is_file() or ply.stat().st_size == 0:
            raise ValueError("The result has no nonempty sparse.ply file.")
        return Result(directory, *counts, seconds, data["status"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read reconstruction result: {exc}") from exc


def artifact_stage(output):
    """These files are only written after the named engine operations finish."""
    output = Path(output)
    def complete(name):
        try:
            json.loads((output / name).read_text(encoding="utf-8"))
            return True
        except (OSError, ValueError):
            return False
    if complete("initial_candidates.json"):
        return "Initial pair selected; registration / triangulation / optimization running"
    if complete("pairs.json"):
        return "Geometric matches saved; selecting initial pair"
    if complete("manifest.json"):
        return "Features prepared; matching / track construction running"
    return "Engine running; loading / preprocessing / feature extraction"


def child_options():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def run_job(folder, camera, base, emit, stop):
    """Blocking worker function; only messages cross into the Tk main thread."""
    run = None
    try:
        emit("stage", "Validating readable images and calibration")
        validation = validate_inputs(folder, camera)
        emit("validated", validation)
        if stop.is_set():
            return
        run = allocate_run(base)
        output = run / "reconstruction"
        emit("run", str(run))
        command = reconstruction_command(folder, camera, output)
        (run / "demo_request.json").write_text(json.dumps({"command": command, "validation": validation}, indent=2), encoding="utf-8")
        started = perf_counter()
        with (run / "reconstruction.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **child_options())
            try:
                previous = None
                while process.poll() is None:
                    stage = artifact_stage(output)
                    if stage != previous:
                        emit("stage", stage)
                        previous = stage
                    if stop.wait(0.2):
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        return
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        elapsed = perf_counter() - started
        if process.returncode not in (0, 3):
            tail = (run / "reconstruction.log").read_text(encoding="utf-8", errors="replace")[-2400:]
            raise ValueError(f"Engine exited with code {process.returncode}.\n{tail}\nLog: {run / 'reconstruction.log'}")
        result = read_result(output)
        if (process.returncode == 0) != (result.status == "success"):
            raise ValueError("Engine exit code and saved result status disagree; inspect reconstruction.log.")
        emit("result", (result, elapsed))
    except Exception as exc:
        emit("error", f"{type(exc).__name__}: {exc}" + (f"\nRun folder: {run}" if run else ""))
    finally:
        emit("done", None)


def open_directory(path):
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError(f"Output folder does not exist: {path}")
    if sys.platform == "win32":
        os.startfile(str(path))
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)], **child_options())
