"""Exercise the real Tk worker and exports with a bounded UAV photograph set."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import tkinter as tk

from vision3d.demo.__main__ import DemoApp
from vision3d.demo.viewer import load_textured_meshes


def validate(files, output):
    output.mkdir(parents=True, exist_ok=False)
    root = tk.Tk()
    root.withdraw()
    app = DemoApp(root)
    app.output_base = output / "runs"
    app.selected_files = tuple(str(path.resolve()) for path in files)
    app.folder.set(str(files[0].parent))
    app.dense_profile.set("uav-recovery")
    app.texture_requested.set(True)
    report = {"status": "running", "input_images": len(files), "files": list(app.selected_files)}
    start = perf_counter()
    heartbeat = 0
    restart_at = None
    cancellation_log = None

    def observe():
        nonlocal heartbeat, restart_at, cancellation_log
        heartbeat += 1
        if app.worker is not None:
            if perf_counter() - start > 360:
                app.cancel()
            if restart_at is not None and cancellation_log is None and app.run_folder is not None:
                log = app.run_folder / "reconstruction.log"
                if log.is_file():
                    cancellation_log = log
                    app.cancel()
            root.after(100, observe)
            return
        if restart_at is not None:
            report["cancelled_restart"] = cancellation_log is not None and app.stop.is_set() and app.result is None
            report["cancellation_log"] = str(cancellation_log)
            root.quit()
            return
        if app.textured_path is None or app.result is None:
            report["error"] = app.stage.get()
            root.quit()
            return
        report.update(registered=app.result.registered, dense_points=app.result.landmarks,
                      glb=str(app.textured_path), full_run_seconds=perf_counter() - start,
                      worker_active_heartbeat_count=heartbeat)
        app.start()
        restart_at = perf_counter()
        root.after(100, observe)

    app.start()
    root.after(100, observe)
    try:
        root.mainloop()
        if "glb" not in report or not report.get("cancelled_restart"):
            raise RuntimeError(f"Desktop job or cancellation failed: {report.get('error', app.stage.get())}")
        bundle = Path(report["glb"]).parent
        report["preview_mesh_groups"] = len(load_textured_meshes(report["glb"]))
        report["exports"] = {name: (bundle / name).stat().st_size
                             for name in ("textured.glb", "textured.obj", "textured.mtl", "atlas.jpg")}
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "desktop-validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 6 <= len(args.images) <= 16:
        parser.error("Select 6–16 overlapping UAV photographs")
    validate(args.images, args.output)
