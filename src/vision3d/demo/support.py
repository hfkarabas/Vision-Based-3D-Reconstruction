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

from vision3d.io import image_paths


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
    return image_paths(folder)


def validate_inputs(selection, camera):
    """Check readable files and the explicit model, if supplied, before starting a child."""
    from vision3d.geometry.camera import load_calibration, named_camera
    files = supported_images(selection) if isinstance(selection, (str, Path)) else image_paths(selection)
    if not 2 <= len(files) <= 30:
        raise ValueError("The demo accepts 2–30 JPEG/PNG files. Select a smaller subset if necessary.")
    valid, skipped, sizes = [], [], []
    for path in files:
        try:
            with Image.open(path) as im:
                if im.format not in {"JPEG", "PNG"}:
                    raise OSError("Not a JPEG or PNG")
                im.load()
                sizes.append(ImageOps.exif_transpose(im).size)
                valid.append(path)
        except (OSError, ValueError, SyntaxError) as exc:
            skipped.append(f"{path.name}: {exc}")
    if len(valid) < 2:
        raise ValueError(f"At least two readable images are required; found {len(valid)}.")
    if camera:
        try:
            mapping, models = load_calibration(camera, pair=len(valid) == 2)
            for index, (path, size) in enumerate(zip(valid, sizes)):
                model = named_camera(mapping, path.name) if mapping is not None else models[min(index, 1)]
                if size != (model.width, model.height):
                    raise ValueError(f"Calibration dimensions do not match EXIF-oriented image {path.name}.")
        except (TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot validate camera calibration: {exc}") from exc
    return {"supported": len(files), "usable": len(valid), "skipped": skipped, "files": [str(p) for p in valid]}


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


def reconstruction_command(selection, camera, output, *, max_dimension=1600):
    folder = isinstance(selection, (str, Path))
    files = supported_images(selection) if folder else image_paths(selection)
    if len(files) == 2:
        args = ["reconstruct-pair", "--image1", str(files[0]), "--image2", str(files[1])]
    else:
        args = ["reconstruct-sequence"]
        args += ["--input", str(Path(selection).resolve())] if folder else ["--images", *map(str, files)]
        args += ["--local-ba", "--refine-points", "--min-registered", str(min(5, len(files)))]
    calibration = ["--camera", str(Path(camera).resolve())] if camera else ["--auto-intrinsics"]
    return [sys.executable, "-u", "-m", "vision3d.cli", *args, *calibration,
            "--output", str(Path(output).resolve()), "--max-dimension", str(max_dimension)]


@dataclass(frozen=True)
class Result:
    directory: Path
    registered: int
    landmarks: int
    seconds: float
    status: str
    representation: str = "sparse"

    @property
    def ply(self):
        return self.directory / ("dense.ply" if self.representation == "dense_two_view_stereo" else "sparse.ply")


def read_result(directory):
    directory = Path(directory).resolve()
    try:
        data = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if data.get("status") not in {"success", "partial"}:
            raise ValueError(data.get("reason", "Reconstruction was rejected."))
        representation = data.get("representation", "sparse")
        dense = representation == "dense_two_view_stereo"
        counts = [data["registered_images"], data["dense_points" if dense else "valid_landmarks"]]
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in counts):
            raise ValueError("Invalid camera or landmark counts in metrics.")
        seconds = float(data["runtime_seconds"])
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid execution time in metrics.")
        result = Result(directory, *counts, seconds, data["status"], representation)
        if not result.ply.is_file() or result.ply.stat().st_size == 0:
            raise ValueError(f"The result has no nonempty {result.ply.name} file.")
        return result
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


def run_job(folder, camera, base, emit, stop, max_dimension=1600):
    """Blocking worker function; only messages cross into the Tk main thread."""
    run = None
    try:
        emit("stage", "Validating readable images and calibration")
        validation = validate_inputs(folder, camera)
        emit("validated", {**validation, "automatic_intrinsics": not bool(camera)})
        if stop.is_set():
            return
        run = allocate_run(base)
        output = run / "reconstruction"
        emit("run", str(run))
        command = reconstruction_command(validation["files"], camera, output, max_dimension=max_dimension)
        (run / "demo_request.json").write_text(json.dumps({"command": command, "validation": validation}, indent=2), encoding="utf-8")
        started = perf_counter()
        with (run / "reconstruction.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **child_options())
            try:
                previous = None
                while process.poll() is None:
                    stage = artifact_stage(output)
                    if validation["usable"] == 2 and stage.startswith("Features prepared"):
                        stage = "Pair features matched; estimating relative pose and triangulating"
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
        if stop.is_set():
            return
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


def write_camera_visibility(path, visible):
    """Atomically send a desired state; no renderer or reconstruction runs here."""
    if not isinstance(visible, bool):
        raise ValueError("Camera visibility must be boolean")
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'show_cameras': visible, 'request_id': uuid.uuid4().hex}), encoding='utf-8')
    temporary.replace(path)
