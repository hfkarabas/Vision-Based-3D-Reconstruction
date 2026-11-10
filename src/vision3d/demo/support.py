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


def validate_inputs(selection, camera, max_images=30):
    """Check readable files and the explicit model, if supplied, before starting a child."""
    from vision3d.geometry.camera import load_calibration, named_camera
    files = supported_images(selection) if isinstance(selection, (str, Path)) else image_paths(selection)
    if not 2 <= len(files) <= max_images:
        raise ValueError(f"The demo accepts 2–{max_images} JPEG/PNG files. Select a smaller subset if necessary.")
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


def reconstruction_command(selection, camera, output, *, max_dimension=1600, enhanced=False, cache=None, video=False):
    folder = isinstance(selection, (str, Path))
    files = supported_images(selection) if folder else image_paths(selection)
    if len(files) == 2:
        args = ["reconstruct-pair", "--image1", str(files[0]), "--image2", str(files[1])]
    else:
        args = ["reconstruct-sequence"]
        args += ["--input", str(Path(selection).resolve())] if folder else ["--images", *map(str, files)]
        args += ["--local-ba", "--refine-points", "--min-registered", str(min(5, len(files)))]
        if video:
            args += ["--window", "4", "--long-gap", "8", "--max-images", "50"]
            if not enhanced:
                args += ["--recovery-pairs", "12"]
        if enhanced:
            args += ["--exhaustive"] if len(files) <= 30 else ["--recovery-pairs", "30"]
            args += ["--verifier", "magsac", "--probe-seeds", "--ratio", "0.80"]
            if cache is not None:
                args += ["--pair-cache", str(Path(cache).resolve())]
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
    def is_dense(self):
        return self.representation in {"dense_two_view_stereo", "dense_multiview"}

    @property
    def ply(self):
        return self.directory / ("dense.ply" if self.is_dense else "sparse.ply")


def read_result(directory):
    directory = Path(directory).resolve()
    try:
        data = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if data.get("status") not in {"success", "partial"}:
            raise ValueError(data.get("reason", "Reconstruction was rejected."))
        representation = data.get("representation", "sparse")
        dense = representation in {"dense_two_view_stereo", "dense_multiview"}
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


def run_child(command, log_path, stage, emit, stop):
    if stop.is_set():
        return None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **child_options())
        try:
            previous = None
            while process.poll() is None:
                current = stage()
                if current != previous:
                    emit("stage", current)
                    previous = current
                if stop.wait(0.2):
                    break
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        return process.returncode


def reported_stage(log, fallback):
    try:
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 4096))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        return next((line[7:] for line in reversed(lines) if line.startswith("Stage: ")), fallback)
    except OSError:
        return fallback


def run_surface_workflow(dense, run, triangles, emit, stop):
    surface, texture = run / "surface-asset", run / "texture-asset"
    stages = [("surface-asset", ["--dense-dir", str(dense), "--output", str(surface),
                                 "--target-faces", str(max(450000, (triangles + 2) // 3))]),
              ("texture-asset", ["--surface-dir", str(surface), "--output", str(texture)])]
    for name, arguments in stages:
        command = [sys.executable, "-u", "-m", "vision3d.cli", name, *arguments]
        log = run / f"{name}.log"
        code = run_child(command, log, lambda: reported_stage(log, f"Starting {name}"), emit, stop)
        if stop.is_set():
            emit("stage", "Asset processing cancelled. Completed reconstruction files are available.")
            return
        if code != 0:
            tail = log.read_text(encoding="utf-8", errors="replace")[-1600:]
            emit("error", f"{name} failed. The dense mesh is available.\n{tail}")
            return
    emit("texture", str(texture / "textured.glb"))


def run_job(folder, camera, base, emit, stop, max_dimension=1600, profile=None, mesh=False, glb=False,
            enhanced=False, video_options=None, texture=False):
    """Blocking worker function; only messages cross into the Tk main thread."""
    run = None
    try:
        video = video_options is not None
        automatic = not bool(camera)
        if video:
            run = allocate_run(base)
            emit("run", str(run))
            keyframes = run / "keyframes"
            log = run / "keyframes.log"
            extraction = [sys.executable, "-u", "-m", "vision3d.cli", "keyframes",
                          "--video", str(Path(folder).resolve()), "--output", str(keyframes)]
            if camera:
                extraction += ["--camera", str(Path(camera).resolve())]
            for name, value in video_options.items():
                extraction += ["--" + name.replace("_", "-"), str(value)]
            code = run_child(extraction, log, lambda: reported_stage(log, "Starting MP4 extraction"), emit, stop)
            if stop.is_set():
                return
            if code != 0:
                raise ValueError("Video extraction failed. " + log.read_text(encoding="utf-8", errors="replace")[-1600:])
            report = json.loads((keyframes / "keyframes.json").read_text(encoding="utf-8"))
            decisions = report.get("decision_counts", {})
            emit("notice", f"Selected {len(report['selected'])} keyframes from {report['decoded_frames']} decoded frames. "
                           f"Rejected {decisions.get('near_duplicate', 0)} near duplicates and "
                           f"{decisions.get('blur', 0)} blurred samples. Details: {keyframes / 'keyframes.json'}")
            folder = keyframes / "frames"
            camera = keyframes / "camera.json"
        emit("stage", "Validating readable images and calibration")
        validation = validate_inputs(folder, camera, max_images=50 if video else 30)
        emit("validated", {**validation, "automatic_intrinsics": automatic, "video": video})
        if stop.is_set():
            return
        if run is None:
            run = allocate_run(base)
            emit("run", str(run))
        output = run / "reconstruction"
        command = reconstruction_command(validation["files"], camera, output, max_dimension=max_dimension,
                                         enhanced=enhanced, cache=Path(base) / "pair-cache", video=video)
        (run / "demo_request.json").write_text(json.dumps({"command": command, "validation": validation,
                                                              "keyframe_manifest": str(keyframes / "keyframes.json") if video else None}, indent=2), encoding="utf-8")
        started = perf_counter()
        def sparse_stage():
            stage = artifact_stage(output)
            if validation["usable"] == 2 and stage.startswith("Features prepared"):
                return "Pair features matched; estimating relative pose and triangulating"
            return stage

        code = run_child(command, run / "reconstruction.log", sparse_stage, emit, stop)
        if stop.is_set():
            return
        elapsed = perf_counter() - started
        if code not in (0, 3):
            tail = (run / "reconstruction.log").read_text(encoding="utf-8", errors="replace")[-2400:]
            raise ValueError(f"Engine exited with code {code}.\n{tail}\nLog: {run / 'reconstruction.log'}")
        result = read_result(output)
        if (code == 0) != (result.status == "success"):
            raise ValueError("Engine exit code and saved result status disagree; inspect reconstruction.log.")
        emit("result", (result, elapsed))
        data = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        if video:
            quality = data.get("after_final_ba", {})
            median = quality.get("median_error_px", data.get("median_reprojection_error_px"))
            if isinstance(median, (float, int)):
                emit("notice", f"Registered {result.registered}/{validation['usable']} keyframes. "
                               f"Median reprojection error: {median:.3f} px (image fit, not absolute accuracy).")
        graph = data.get("tracks", {}).get("graph")
        if graph:
            emit("notice", f"Matching graph: {graph['verified_edges']} verified pairs, "
                           f"{len(graph['components_after'])} components. See pairs.json and poses.json for details.")
        if profile is not None:
            if result.registered < 3:
                emit("error", "Dense sequence needs at least three registered views. The sparse result is available.")
                return
            dense = run / "dense"
            log = run / "dense.log"
            command = [sys.executable, "-u", "-m", "vision3d.cli", "dense-sequence",
                       "--sparse-dir", str(output), "--output", str(dense), "--profile", profile]
            if mesh:
                command.append("--mesh")
            if glb:
                command.append("--glb")

            code = run_child(command, log, lambda: reported_stage(log, "Starting dense reconstruction"), emit, stop)
            if stop.is_set():
                emit("stage", "Dense stage cancelled. Sparse result remains available.")
            elif code != 0:
                tail = log.read_text(encoding="utf-8", errors="replace")[-1600:]
                emit("error", f"Dense stage failed. Sparse result remains available.\n{tail}")
            else:
                dense_result = read_result(dense)
                emit("result", (dense_result, None))
                data = json.loads((dense / "metrics.json").read_text(encoding="utf-8"))
                surface = data.get("mesh") or {}
                for item in (surface, surface.get("glb", {})):
                    if "reason" in item:
                        emit("notice", item["reason"])
                if texture and surface.get("status") == "success":
                    run_surface_workflow(dense, run, surface["triangles"], emit, stop)
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
