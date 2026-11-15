"""Desktop app for the uav3d pipeline (no terminal needed): python -m uav3d.app, or run_app.bat.

The app is a thin front end. It starts the same command a terminal user would run
(`python -m uav3d run ...`) as a separate process and follows its progress through run.json, which the
pipeline rewrites after every step. Nothing about the processing itself lives here, so the app and the
command line always give the same result. One job at a time: heavy jobs side by side have crashed the
development machine.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

REPO = Path(__file__).resolve().parents[2]

# Display phases and the run.json stage names that belong to each (prefix match)
PHASES = [
    ("images", "Check images", ()),
    ("sfm", "Camera positions (SfM)", ("odm:sfm", "odm:split", "odm:reused")),
    ("dense", "Dense point cloud and surfaces", ("odm:dense", "odm:split", "odm:reused")),
    ("adaptive", "Adaptive refinement", ("adaptive",)),
    ("detect", "Building evidence from images", ("detect",)),
    ("terrain", "Terrain clean-up", ("terrain",)),
    ("buildings", "Building models", ("buildings",)),
    ("export", "Scene export", ("export",)),
    ("prune", "Clean-up of temporary files", ("prune",)),
]
EARLY_STAGES = ("setup", "tools", "ingest", "plan")  # failure stages recorded before any ODM stage


# ----------------------------------------------------------------------------------------------- logic

def build_command(images: str, out: str, *, threads: int | None = None, buildings: bool = True,
                  export: bool = True, refine_budget: float | None = None, ram_gb: float | None = None,
                  keep_intermediates: bool = False, reuse_odm: bool = False, python: str | None = None) -> list[str]:
    cmd = [python or sys.executable, "-m", "uav3d", "run", str(images), "--out", str(out)]
    if threads:
        cmd += ["--threads", str(threads)]
    if ram_gb:
        cmd += ["--budget-ram-gb", f"{ram_gb:g}"]
    if not buildings:
        cmd.append("--skip-buildings")
    if not export:
        cmd.append("--skip-export")
    if refine_budget:
        cmd += ["--refine-budget", f"{refine_budget:g}"]
    if keep_intermediates:
        cmd.append("--keep-odm-intermediates")
    if reuse_odm:
        cmd.append("--reuse-odm")
    return cmd


def quiet_threads(cpus: int | None = None) -> int:
    """Leave a few cores free so the machine stays usable and quiet."""
    n = cpus or os.cpu_count() or 4
    return max(1, n - 4) if n > 6 else max(1, n - 1)


def _time(s: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(s) if s else None
    except ValueError:
        return None


def _max_key(obj, key: str) -> float | None:
    """Largest value of ``key`` anywhere in a nested dict/list (peak RAM sits at different depths)."""
    best = None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, (int, float)):
                best = v if best is None else max(best, v)
            else:
                sub = _max_key(v, key)
                if sub is not None:
                    best = sub if best is None else max(best, sub)
    elif isinstance(obj, list):
        for v in obj:
            sub = _max_key(v, key)
            if sub is not None:
                best = sub if best is None else max(best, sub)
    return best


def phase_rows(run: dict, now: datetime | None = None, adaptive: bool = False) -> list[dict]:
    """One row per display phase: status (waiting / running / done / failed / skipped / not reached),
    duration [s] and peak RAM [GB], derived from run.json alone."""
    now = now or datetime.now(timezone.utc)
    status = run.get("status", "running")
    stages = run.get("stages", [])
    error = run.get("error") or ""
    failed_at = error.split(":", 1)[0] if status == "failed" else None
    finished = _time(run.get("finished"))
    starts = [_time(s.get("started")) for s in stages]
    rows = []
    for key, label, prefixes in PHASES:
        if key == "adaptive" and not adaptive and not any(s["name"].startswith("adaptive") for s in stages):
            continue
        row = {"key": key, "label": label, "status": "waiting", "seconds": None, "peak_gb": None}
        if key == "images":
            if failed_at in EARLY_STAGES:
                row["status"] = "failed"
            elif "ingest" in run or stages:
                row["status"] = "done"
            elif status == "running":
                row["status"] = "running"
            rows.append(row)
            continue
        idx = [i for i, s in enumerate(stages) if s.get("name", "").startswith(prefixes)]
        if not idx:
            # never started: still to come, left out by the options, or after a failure
            row["status"] = {"running": "waiting", "ok": "skipped"}.get(status, "not_reached")
            rows.append(row)
            continue
        last = idx[-1] == len(stages) - 1
        if last and status == "running":
            row["status"] = "running"
        elif last and status == "failed":
            row["status"] = "failed"
        else:
            row["status"] = "done"
        total = 0.0
        for i in idx:
            t0 = starts[i]
            t1 = starts[i + 1] if i + 1 < len(stages) else (finished or now)
            if t0 and t1:
                total += max(0.0, (t1 - t0).total_seconds())
        row["seconds"] = total
        peaks = [p for i in idx if (p := _max_key(stages[i], "peak_rss_gb")) is not None]
        row["peak_gb"] = max(peaks) if peaks else None
        rows.append(row)
    return rows


def summary(run: dict, out: Path) -> dict:
    """What the result screen shows."""
    st = {s["name"]: s for s in run.get("stages", [])}
    gsd = (st.get("odm:sfm") or st.get("odm:reused") or {}).get("gsd") or {}
    exp = st.get("export") or {}
    b = run.get("buildings") or {}
    t0, t1 = _time(run.get("started")), _time(run.get("finished"))
    warnings = [w for w in (run.get("warning"), (run.get("capture") or {}).get("warning")) if w]
    osm = str((st.get("buildings:footprints") or {}).get("osm") or "")
    if osm.startswith("unavailable"):
        warnings.append("OpenStreetMap could not be reached; building footprints come only from the point cloud "
                        "and the images, so buildings may be missing. Run again later to include OSM.")
    scene = out / "scene"
    return {
        "status": run.get("status"), "error": run.get("error"), "warnings": warnings,
        "images": (run.get("ingest") or {}).get("count"),
        "gsd_cm": round(gsd["gsd_m"] * 100, 2) if gsd.get("gsd_m") and gsd.get("metric", True) else None,
        "buildings": b.get("buildings") if b.get("status") == "ok" else (0 if b.get("status") == "none" else None),
        "buildings_in_scene": exp.get("buildings_in_scene"),
        "flagged": len(exp.get("flagged_buildings") or []),
        "minutes": round((t1 - t0).total_seconds() / 60, 1) if t0 and t1 else None,
        "peak_gb": _max_key(run.get("stages", []), "peak_rss_gb"),
        "disk_gb": run.get("disk_gb"),
        "preview": scene / "preview.png" if (scene / "preview.png").is_file() else None,
        "blend": scene / "scene.blend" if (scene / "scene.blend").is_file() else None,
        "glb": scene / "scene.glb" if (scene / "scene.glb").is_file() else None,
    }


def read_run(out: Path) -> dict | None:
    try:
        return json.loads((out / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # missing, or caught mid-rewrite: try again on the next poll


def folder_info(folder: Path) -> dict:
    """Quick look at an image folder before a run (image count, GPS, GCP / position files)."""
    from uav3d import ingest
    images = ingest.find_images(folder)
    if not images:
        return {"count": 0}
    s = ingest.summarise(images)
    return {"count": s["count"], "unreadable": len(s["unreadable"]), "gps_fraction": s["gps_fraction"],
            "cameras": ", ".join(s["cameras"]) or "unknown",
            "gcp": bool(ingest.find_gcp(folder, images)), "geo": bool(ingest.find_geo(folder, images))}


def other_jobs() -> list[int]:
    """PIDs of uav3d runs already running on this machine (any terminal or app)."""
    import psutil
    pids = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = p.info.get("cmdline") or []
        if "uav3d" in cmd and "run" in cmd and p.info["pid"] != os.getpid():
            pids.append(p.info["pid"])
    return pids


def kill_tree(pid: int) -> None:
    import psutil
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = [*root.children(recursive=True), root]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=10)


def open_path(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa: S606 - opens with the user's default program
    else:
        subprocess.Popen(["xdg-open", str(path)])


# ---------------------------------------------------------------------------------------------------- UI

STATUS_TEXT = {"waiting": "waiting", "running": "running…", "done": "done", "failed": "FAILED", "skipped": "skipped",
               "not_reached": "not reached"}


def _fmt_s(s: float | None) -> str:
    if s is None:
        return ""
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("UAV 3D Reconstruction")
        root.geometry("1000x940")
        root.minsize(820, 760)
        self.proc: subprocess.Popen | None = None
        self.out: Path | None = None
        self.started: datetime | None = None
        self.adaptive_on = False
        self.preview_img = None
        self.blender_path = None
        # worker threads never touch widgets: they queue (callback, args) and the Tk thread runs them
        self.inbox: queue.Queue = queue.Queue()
        root.after(200, self.drain)

        main = ttk.Frame(root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        # 1. tools
        tf = ttk.LabelFrame(main, text="1. Installed tools", padding=8)
        tf.grid(row=0, column=0, sticky="ew")
        tf.columnconfigure(1, weight=1)
        self.tool_labels = {}
        for i, (key, name) in enumerate((("odm", "OpenDroneMap (required)"), ("roofer", "Roofer (required)"),
                                         ("sam3", "SAM 3 building detection (optional)"),
                                         ("blender", "Blender (optional)"))):
            ttk.Label(tf, text=name).grid(row=i, column=0, sticky="w", padx=(0, 10))
            lab = ttk.Label(tf, text="checking…", foreground="gray")
            lab.grid(row=i, column=1, sticky="w")
            self.tool_labels[key] = lab
        bf = ttk.Frame(tf)
        bf.grid(row=0, column=2, rowspan=4, sticky="ne")
        ttk.Button(bf, text="Install / repair tools", command=lambda: self.install(False)).pack(fill="x")
        ttk.Button(bf, text="Install with SAM 3", command=lambda: self.install(True)).pack(fill="x", pady=2)
        ttk.Button(bf, text="Check again", command=self.check_tools).pack(fill="x")

        # 2. input / output
        io = ttk.LabelFrame(main, text="2. Images and output", padding=8)
        io.grid(row=1, column=0, sticky="ew", pady=6)
        io.columnconfigure(1, weight=1)
        self.images = tk.StringVar()
        self.output = tk.StringVar()
        ttk.Label(io, text="Image folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(io, textvariable=self.images).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(io, text="Browse…", command=self.pick_images).grid(row=0, column=2)
        self.info = ttk.Label(io, text="Choose the folder with the UAV photos (nadir mapping flight).", foreground="gray")
        self.info.grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(io, text="Output folder").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(io, textvariable=self.output).grid(row=2, column=1, sticky="ew", padx=6, pady=(4, 0))
        ttk.Button(io, text="Browse…", command=self.pick_output).grid(row=2, column=2, pady=(4, 0))

        # 3. options
        op = ttk.LabelFrame(main, text="3. Options", padding=8)
        op.grid(row=2, column=0, sticky="ew")
        self.speed = tk.StringVar(value="quiet")
        self.buildings = tk.BooleanVar(value=True)
        self.export = tk.BooleanVar(value=True)
        self.refine = tk.StringVar(value="Off")
        ttk.Label(op, text="Speed").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(op, text=f"Quiet ({quiet_threads()} threads)", variable=self.speed, value="quiet").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(op, text=f"Full speed ({os.cpu_count()} threads)", variable=self.speed, value="full").grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(op, text="Building models", variable=self.buildings).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(op, text="Scene files (glb, Blender, preview)", variable=self.export).grid(row=1, column=2, sticky="w")
        ttk.Label(op, text="Adaptive refinement").grid(row=2, column=0, sticky="w")
        ttk.Combobox(op, textvariable=self.refine, state="readonly", width=8,
                     values=["Off", "25%", "50%", "75%"]).grid(row=2, column=1, sticky="w")
        ttk.Label(op, text="research feature: re-densifies the tiles with the highest predicted gain; adds time",
                  foreground="gray").grid(row=2, column=2, columnspan=2, sticky="w")
        self.adv_shown = tk.BooleanVar(value=False)
        ttk.Checkbutton(op, text="Show advanced options", variable=self.adv_shown,
                        command=self.toggle_advanced).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.adv = ttk.Frame(op)
        self.ram = tk.StringVar()
        self.keep = tk.BooleanVar(value=False)
        self.reuse = tk.BooleanVar(value=False)
        ttk.Label(self.adv, text="RAM budget [GB] (blank = automatic)").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.adv, textvariable=self.ram, width=8).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Checkbutton(self.adv, text="Keep ODM intermediate files (large)", variable=self.keep).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(self.adv, text="Reuse the photogrammetry of an earlier run in the output folder",
                        variable=self.reuse).grid(row=2, column=0, columnspan=2, sticky="w")

        # 4. run
        rf = ttk.Frame(main)
        rf.grid(row=3, column=0, sticky="ew", pady=6)
        self.run_btn = ttk.Button(rf, text="Run", command=self.start)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(rf, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)
        ttk.Button(rf, text="Open an earlier result…", command=self.open_result).pack(side="left", padx=6)
        self.elapsed = ttk.Label(rf, text="")
        self.elapsed.pack(side="right")

        # 5. progress
        pf = ttk.LabelFrame(main, text="4. Progress", padding=8)
        pf.grid(row=4, column=0, sticky="nsew")
        main.rowconfigure(4, weight=1)
        pf.columnconfigure(0, weight=1)
        self.table = ttk.Treeview(pf, columns=("status", "time", "ram"), height=9)
        self.table.heading("#0", text="Step")
        self.table.heading("status", text="Status")
        self.table.heading("time", text="Time")
        self.table.heading("ram", text="Peak RAM")
        self.table.column("#0", width=330)
        for c, w in (("status", 110), ("time", 90), ("ram", 90)):
            self.table.column(c, width=w, anchor="center")
        self.table.tag_configure("failed", foreground="red")
        self.table.tag_configure("running", foreground="blue")
        self.table.tag_configure("skipped", foreground="gray")
        self.table.tag_configure("not_reached", foreground="gray")
        self.table.grid(row=0, column=0, sticky="nsew")
        pf.rowconfigure(0, minsize=230)
        self.warn = ttk.Label(pf, text="", foreground="#9a6700", wraplength=900)
        self.warn.grid(row=1, column=0, sticky="w")
        self.err = ttk.Label(pf, text="", foreground="red", wraplength=900)
        self.err.grid(row=2, column=0, sticky="w")
        self.log_shown = tk.BooleanVar(value=False)
        ttk.Checkbutton(pf, text="Show detailed log", variable=self.log_shown, command=self.toggle_log).grid(row=3, column=0, sticky="w")
        self.log = tk.Text(pf, height=8, wrap="none", font=("Consolas", 8))

        # 6. result
        res = ttk.LabelFrame(main, text="5. Result", padding=8)
        res.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        res.columnconfigure(1, weight=1)
        self.preview = ttk.Label(res, text="(no result yet)")
        self.preview.grid(row=0, column=0, rowspan=2, sticky="nw")
        self.result_text = ttk.Label(res, text="", justify="left")
        self.result_text.grid(row=0, column=1, sticky="nw", padx=10)
        btns = ttk.Frame(res)
        btns.grid(row=1, column=1, sticky="sw", padx=10)
        self.btn_blender = ttk.Button(btns, text="Open in Blender", command=self.open_blender, state="disabled")
        self.btn_scene = ttk.Button(btns, text="Open 3D scene", command=lambda: self._open("glb"), state="disabled")
        self.btn_folder = ttk.Button(btns, text="Open output folder", command=lambda: self._open("folder"), state="disabled")
        self.btn_report = ttk.Button(btns, text="Open run report", command=lambda: self._open("report"), state="disabled")
        for b in (self.btn_blender, self.btn_scene, self.btn_folder, self.btn_report):
            b.pack(side="left", padx=(0, 6))
        self.result = None

        root.protocol("WM_DELETE_WINDOW", self.close)
        self.check_tools()

    def drain(self):
        while True:
            try:
                fn, args = self.inbox.get_nowait()
            except queue.Empty:
                break
            fn(*args)
        self.root.after(200, self.drain)

    # tools ------------------------------------------------------------------------------------------
    def check_tools(self):
        for lab in self.tool_labels.values():
            lab.config(text="checking…", foreground="gray")

        def work():
            from uav3d import tools
            rows = tools.doctor()
            self.blender_path = tools.blender()
            self.inbox.put((self.show_tools, (rows,)))
        threading.Thread(target=work, daemon=True).start()

    def show_tools(self, rows):
        for name, ok, detail in rows:
            key = name.split()[0]
            lab = self.tool_labels.get(key)
            if not lab:
                continue
            missing = "not installed" in detail or "NO -" in detail
            if not ok:
                lab.config(text="✗ " + detail[:110], foreground="red")
            elif missing:
                lab.config(text="– " + detail[:110], foreground="#9a6700")
            else:
                lab.config(text="✓ " + detail.split(";")[0][:110], foreground="green")

    def install(self, with_ml: bool):
        script = REPO / "setup.ps1"
        if not script.is_file():
            messagebox.showerror("Install", f"{script} not found.")
            return
        if not messagebox.askyesno("Install tools", "This downloads and installs the processing tools "
                                   f"({'about 6 GB with SAM 3' if with_ml else 'about 1 GB'}) in a separate "
                                   "window. Continue?"):
            return
        cmd = ["powershell", "-NoExit", "-ExecutionPolicy", "Bypass", "-File", str(script)] + (["-WithML"] if with_ml else [])
        flags = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        subprocess.Popen(cmd, cwd=REPO, creationflags=flags)
        messagebox.showinfo("Install tools", "Installation runs in the new window. Press 'Check again' when it has finished.")

    # input ------------------------------------------------------------------------------------------
    def pick_images(self):
        d = filedialog.askdirectory(title="Folder with the UAV photos")
        if not d:
            return
        self.images.set(d)
        if not self.output.get():
            p = Path(d)
            self.output.set(str(p.parent / f"{p.name}_3d"))
        self.info.config(text="reading photos…", foreground="gray")

        def work():
            try:
                info = folder_info(Path(d))
            except Exception as exc:  # noqa: BLE001 - shown to the user
                info = {"error": str(exc)}
            self.inbox.put((self.show_info, (info,)))
        threading.Thread(target=work, daemon=True).start()

    def show_info(self, info):
        if "error" in info:
            self.info.config(text=f"Could not read the folder: {info['error']}", foreground="red")
        elif not info["count"]:
            self.info.config(text="No photos found in this folder (or in an 'images' subfolder).", foreground="red")
        else:
            gps = f"GPS in {info['gps_fraction']:.0%} of photos"
            extra = [x for x, ok in (("control point file found", info["gcp"]), ("position file found", info["geo"])) if ok]
            text = f"{info['count']} photos ({info['cameras']}); {gps}" + ("; " + ", ".join(extra) if extra else "")
            warn = info["gps_fraction"] < 1.0 and not (info["gcp"] or info["geo"])
            self.info.config(text=text + (" — without GPS the result has no real-world position" if warn else ""),
                             foreground="#9a6700" if warn else "green")

    def pick_output(self):
        d = filedialog.askdirectory(title="Output folder")
        if d:
            self.output.set(d)

    def toggle_advanced(self):
        self.adv.grid(row=4, column=0, columnspan=4, sticky="w") if self.adv_shown.get() else self.adv.grid_remove()

    def toggle_log(self):
        self.log.grid(row=4, column=0, sticky="nsew") if self.log_shown.get() else self.log.grid_remove()

    # run --------------------------------------------------------------------------------------------
    def start(self):
        images, out = self.images.get().strip(), self.output.get().strip()
        if not images or not Path(images).is_dir():
            messagebox.showerror("Run", "Choose an existing image folder.")
            return
        if not out:
            messagebox.showerror("Run", "Choose an output folder.")
            return
        if others := other_jobs():
            messagebox.showerror("Run", "Another reconstruction is already running on this computer "
                                 f"(process {', '.join(map(str, others))}). Run one job at a time.")
            return
        outp = Path(out)
        if (outp / "run.json").is_file() and not self.reuse.get():
            if not messagebox.askyesno("Run", "The output folder already holds a result. Replace it?"):
                return
        ram = None
        if self.ram.get().strip():
            try:
                ram = float(self.ram.get().replace(",", "."))
            except ValueError:
                messagebox.showerror("Run", "RAM budget must be a number (GB).")
                return
        beta = None if self.refine.get() == "Off" else int(self.refine.get().rstrip("%")) / 100
        cmd = build_command(images, out, threads=quiet_threads() if self.speed.get() == "quiet" else None,
                            buildings=self.buildings.get(), export=self.export.get(), refine_budget=beta,
                            ram_gb=ram, keep_intermediates=self.keep.get(), reuse_odm=self.reuse.get())
        outp.mkdir(parents=True, exist_ok=True)
        (outp / "logs").mkdir(exist_ok=True)
        self.console = open(outp / "logs" / "app_console.log", "w", encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        if not self.reuse.get():  # --reuse-odm reads the earlier run.json; otherwise it would show stale progress
            try:
                (outp / "run.json").unlink(missing_ok=True)
            except OSError:
                pass
        self.proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=self.console, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, creationflags=flags)
        self.out, self.started, self.adaptive_on = outp, datetime.now(timezone.utc), beta is not None
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.err.config(text="")
        self.warn.config(text="")
        self.clear_result()
        self.poll()

    def cancel(self):
        if self.proc and self.proc.poll() is None:
            if messagebox.askyesno("Cancel", "Stop the running reconstruction? The partial result is kept but incomplete."):
                kill_tree(self.proc.pid)

    def poll(self):
        if not self.out:
            return
        run = read_run(self.out)
        if run:
            self.show_progress(run)
        if self.started:
            self.elapsed.config(text="Elapsed " + _fmt_s((datetime.now(timezone.utc) - self.started).total_seconds()))
        if self.log_shown.get():
            self.show_log()
        if self.proc and self.proc.poll() is not None:
            code = self.proc.returncode
            self.proc = None
            self.console.close()
            self.run_btn.config(state="normal")
            self.cancel_btn.config(state="disabled")
            run = read_run(self.out) or {}
            if run.get("status") == "running":  # killed: the pipeline could not record it
                self.err.config(text="Cancelled (or stopped unexpectedly). See logs/app_console.log in the output folder.")
            elif code != 0 and not run:
                self.err.config(text=f"The run did not start (exit {code}). See logs/app_console.log in the output folder.")
            self.show_result(self.out, run)
            return
        self.root.after(1000, self.poll)

    def show_progress(self, run: dict):
        self.table.delete(*self.table.get_children())
        for r in phase_rows(run, adaptive=self.adaptive_on):
            ram = f"{r['peak_gb']:.1f} GB" if r["peak_gb"] else ""
            self.table.insert("", "end", text=r["label"], values=(STATUS_TEXT[r["status"]], _fmt_s(r["seconds"]), ram),
                              tags=(r["status"],))
        s = summary(run, self.out)
        self.warn.config(text="\n".join("Warning: " + w for w in s["warnings"]))
        if s["error"]:
            self.err.config(text="Failed at " + s["error"])

    def show_log(self):
        logs = sorted((self.out / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime) if self.out else []
        if not logs:
            return
        try:
            with open(logs[-1], "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 6000))
                tail = fh.read().decode("utf-8", errors="replace").splitlines()[-40:]
        except OSError:
            return
        self.log.delete("1.0", "end")
        self.log.insert("end", f"[{logs[-1].name}]\n" + "\n".join(tail))
        self.log.see("end")

    # result -----------------------------------------------------------------------------------------
    def clear_result(self):
        self.preview.config(image="", text="(running…)")
        self.result_text.config(text="")
        for b in (self.btn_blender, self.btn_scene, self.btn_folder, self.btn_report):
            b.config(state="disabled")

    def open_result(self):
        d = filedialog.askdirectory(title="Output folder of an earlier run")
        if not d:
            return
        run = read_run(Path(d))
        if not run:
            messagebox.showerror("Open result", "This folder has no run.json (not a uav3d output folder).")
            return
        self.out = Path(d)
        self.output.set(d)
        self.show_progress(run)
        self.show_result(self.out, run)

    def show_result(self, out: Path, run: dict):
        s = summary(run, out)
        self.result = s
        lines = [{"ok": "Finished successfully.", "failed": "Failed.", "running": "Not finished."}.get(s["status"], "")]
        if s["images"]:
            lines.append(f"Photos: {s['images']}" + (f"   ground sampling: {s['gsd_cm']} cm/pixel" if s["gsd_cm"] else ""))
        if s["buildings"] is not None:
            lines.append(f"Buildings: {s['buildings']} found" + (f", {s['buildings_in_scene']} in the scene"
                         if s["buildings_in_scene"] is not None else "") +
                         (f", {s['flagged']} unreliable (magenta)" if s["flagged"] else ""))
        if s["minutes"] is not None:
            lines.append(f"Total time: {s['minutes']} min   peak RAM: {s['peak_gb'] or 0:.1f} GB" +
                         (f"   disk: {s['disk_gb']} GB" if s["disk_gb"] else ""))
        self.result_text.config(text="\n".join(lines))
        if s["preview"]:
            try:
                from PIL import Image, ImageTk
                im = Image.open(s["preview"])
                im.thumbnail((360, 190))
                self.preview_img = ImageTk.PhotoImage(im)
                self.preview.config(image=self.preview_img, text="")
            except Exception:  # noqa: BLE001 - the preview is optional
                self.preview.config(text="(preview could not be shown)")
        else:
            self.preview.config(image="", text="(no preview)")
        self.btn_folder.config(state="normal")
        self.btn_report.config(state="normal" if (out / "run.json").is_file() else "disabled")
        self.btn_scene.config(state="normal" if s["glb"] else "disabled")
        self.btn_blender.config(state="normal" if s["blend"] else "disabled")

    def _open(self, what):
        if not self.out:
            return
        target = {"folder": self.out, "report": self.out / "run.json",
                  "glb": self.result["glb"] if self.result else None}[what]
        if target:
            try:
                open_path(target)
            except OSError as exc:
                messagebox.showerror("Open", f"Could not open {target}: {exc}")

    def open_blender(self):
        if not self.result or not self.result["blend"]:
            return
        if self.blender_path:
            subprocess.Popen([str(self.blender_path), str(self.result["blend"])])
        else:
            open_path(self.result["blend"])

    def close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Quit", "A reconstruction is running. Stop it and quit?"):
                return
            kill_tree(self.proc.pid)
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if sys.platform == "win32" else "clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
