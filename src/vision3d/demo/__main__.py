"""Launch with python -m vision3d.demo in the existing project environment."""
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
from time import perf_counter
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from .support import (allocate_run, child_options, open_directory, read_result,
                      repository_root, resolve_castle, run_job, supported_images, write_camera_visibility)


class DemoApp:
    def __init__(self, root):
        self.root = root
        root.title("Vision-Based 3D Reconstruction")
        root.geometry("890x690")
        root.minsize(760, 620)
        root.configure(background="white")
        root.option_add("*Font", "{Calibri} 11")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=("Calibri", 11), background="white", foreground="#171717")
        style.configure("TButton", padding=(8, 5), background="#eeeeee")
        style.map("TButton", background=[("active", "#dddddd"), ("disabled", "#f4f4f4")])
        style.configure("TEntry", fieldbackground="white")
        self.folder = tk.StringVar()
        self.camera = tk.StringVar()
        self.selected_files = ()
        self.video_path = None
        self.video_options = {}
        self.quality = tk.StringVar(value="1600")
        self.enhanced_matching = tk.BooleanVar(value=False)
        self.dense_profile = tk.StringVar(value="Sparse only")
        self.mesh_requested = tk.BooleanVar(value=True)
        self.glb_requested = tk.BooleanVar(value=False)
        self.texture_requested = tk.BooleanVar(value=False)
        self.textured_path = None
        self.job_results = {}
        self.count = tk.StringVar(value="No input selected")
        self.stage = tk.StringVar(value="Ready")
        self.elapsed = tk.StringVar(value="Elapsed: —")
        self.statistics = tk.StringVar(value="Registered cameras: —    Landmarks: —")
        self.result_path = tk.StringVar(value="No result loaded")
        self.root_dir = repository_root()
        self.output_base = self.root_dir / "outputs/demo"
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.worker = None
        self.started = None
        self.result = None
        self.run_folder = None
        self.viewer = None
        self.viewer_status = None
        self.viewer_control = None
        self.camera_pending = None
        self.show_cameras = tk.BooleanVar(value=False)
        self.closing = False
        self.close_started = None
        self.actions = []
        panel = ttk.Frame(root, padding=16)
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(1, weight=1)
        ttk.Label(panel, text="Input").grid(row=0, column=0, sticky="w", pady=4)
        self.folder_entry = ttk.Entry(panel, textvariable=self.folder, state="readonly")
        self.folder_entry.grid(row=0, column=1, sticky="ew", padx=8)
        image_buttons = ttk.Frame(panel)
        image_buttons.grid(row=0, column=2)
        self.button(image_buttons, "Folder…", self.browse_folder).pack(side="left")
        self.button(image_buttons, "Files…", self.browse_files).pack(side="left")
        self.button(image_buttons, "Video…", self.browse_video).pack(side="left")
        ttk.Label(panel, textvariable=self.count).grid(row=1, column=1, sticky="w", padx=8, pady=(0, 8))
        ttk.Label(panel, text="Camera JSON (optional)").grid(row=2, column=0, sticky="w")
        ttk.Entry(panel, textvariable=self.camera, state="readonly").grid(row=2, column=1, sticky="ew", padx=8)
        camera_buttons = ttk.Frame(panel)
        camera_buttons.grid(row=2, column=2)
        self.button(camera_buttons, "Browse…", self.browse_camera).pack(side="left")
        self.button(camera_buttons, "Clear", lambda: self.camera.set("")).pack(side="left")
        choices = ttk.Frame(panel)
        choices.grid(row=3, column=0, columnspan=3, sticky="w", pady=12)
        self.button(choices, "Load Castle Demo", self.load_castle).pack(side="left")
        self.button(choices, "Open Previous Result", self.previous).pack(side="left", padx=8)
        self.start_button = self.button(choices, "Start Reconstruction", self.start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(choices, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=8)
        options = ttk.Frame(panel)
        options.grid(row=4, column=0, columnspan=3, sticky="w")
        self.button(options, "Output location…", self.choose_output).pack(side="left")
        ttk.Label(options, text="Longest side (px)").pack(side="left", padx=(12, 4))
        self.quality_box = ttk.Combobox(options, textvariable=self.quality, values=("1000", "1600", "2400"),
                                        width=6, state="readonly")
        self.quality_box.pack(side="left")
        self.enhanced_check = ttk.Checkbutton(options, text="Enhanced matching (experimental)",
                                             variable=self.enhanced_matching)
        self.enhanced_check.pack(side="left", padx=8)
        self.video_settings_button = ttk.Button(options, text="Video settings…", command=self.video_settings, state="disabled")
        self.video_settings_button.pack(side="left")
        dense_options = ttk.Frame(panel)
        dense_options.grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(dense_options, text="Multi-view").pack(side="left", padx=(0, 6))
        self.profile_box = ttk.Combobox(dense_options, textvariable=self.dense_profile,
                                        values=("Sparse only", "fast", "balanced", "high", "uav", "uav-conditioned",
                                                "uav-recovery"),
                                        state="readonly", width=16)
        self.profile_box.pack(side="left")
        self.profile_box.bind("<<ComboboxSelected>>", lambda event: self.busy(False))
        self.mesh_check = ttk.Checkbutton(dense_options, text="Colored mesh", variable=self.mesh_requested,
                                          command=lambda: self.busy(False), state="disabled")
        self.mesh_check.pack(side="left", padx=8)
        self.glb_check = ttk.Checkbutton(dense_options, text="GLB export", variable=self.glb_requested, state="disabled")
        self.glb_check.pack(side="left")
        self.texture_check = ttk.Checkbutton(dense_options, text="Photo texture", variable=self.texture_requested,
                                             state="disabled")
        self.texture_check.pack(side="left", padx=(8, 0))
        self.switch_button = ttk.Button(dense_options, text="Switch Sparse / Dense",
                                        command=self.switch_result, state="disabled")
        self.switch_button.pack(side="left", padx=8)
        ttk.Separator(panel).grid(row=6, column=0, columnspan=3, sticky="ew", pady=12)
        ttk.Label(panel, textvariable=self.stage, wraplength=700).grid(row=7, column=0, columnspan=3, sticky="w")
        ttk.Label(panel, textvariable=self.elapsed).grid(row=8, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Label(panel, textvariable=self.statistics).grid(row=9, column=0, columnspan=3, sticky="w")
        ttk.Label(panel, textvariable=self.result_path, wraplength=700).grid(row=10, column=0, columnspan=3, sticky="w", pady=8)
        result_buttons = ttk.Frame(panel)
        result_buttons.grid(row=11, column=0, columnspan=3, sticky="w")
        self.view_button = ttk.Button(result_buttons, text="Open 3D View", command=self.open_view, state="disabled")
        self.view_button.pack(side="left")
        self.mesh_button = ttk.Button(result_buttons, text="Open Mesh", command=lambda: self.open_view(mesh=True), state="disabled")
        self.mesh_button.pack(side="left", padx=(8, 0))
        self.texture_button = ttk.Button(result_buttons, text="Open UV GLB", command=self.open_texture)
        self.texture_button.pack(side="left", padx=(8, 0))
        self.output_button = ttk.Button(result_buttons, text="Open Output Folder", command=self.open_output, state="disabled")
        self.output_button.pack(side="left", padx=8)
        self.export_button = ttk.Button(result_buttons, text="Save PLY As…", command=self.export_ply, state="disabled")
        self.export_button.pack(side="left")
        self.camera_checkbox = ttk.Checkbutton(result_buttons, text="Show Cameras (C)", variable=self.show_cameras,
                                               command=self.set_camera_visibility, state="disabled")
        self.camera_checkbox.pack(side="left", padx=8)
        ttk.Label(panel, text="3D view: left-drag rotates · wheel zooms · Ctrl + left-drag pans").grid(
            row=12, column=0, columnspan=3, sticky="w", pady=8)
        self.log = tk.Text(panel, height=9, wrap="word", background="white", foreground="black",
                           font=("Calibri", 10), relief="solid", borderwidth=1, state="disabled")
        self.log.grid(row=13, column=0, columnspan=3, sticky="nsew")
        panel.rowconfigure(13, weight=1)
        self.note("Select two photographs for pair mode or more for incremental SfM. Without Camera JSON, intrinsics are estimated per image.")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def button(self, parent, text, command):
        button = ttk.Button(parent, text=text, command=command)
        self.actions.append(button)
        return button

    def note(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def error(self, text):
        self.note(text)
        messagebox.showerror("Vision-Based 3D Reconstruction", text, parent=self.root)

    def show_count(self):
        files = supported_images(self.folder.get())
        self.count.set(f"{len(files)} supported JPEG/PNG images detected; readability checked at start")

    def browse_folder(self):
        value = filedialog.askdirectory(title="Select ordered JPEG/PNG image folder", parent=self.root)
        if value:
            self.selected_files = ()
            self.video_path = None
            self.video_settings_button.configure(state="disabled")
            self.folder.set(value)
            self.camera.set("")
            try:
                self.show_count()
            except (ValueError, OSError) as exc:
                self.error(str(exc))

    def browse_files(self):
        files = filedialog.askopenfilenames(title="Select JPEG/PNG photographs",
                                            filetypes=[("Photographs", "*.jpg *.jpeg *.png")], parent=self.root)
        if files:
            self.selected_files = files
            self.video_path = None
            self.video_settings_button.configure(state="disabled")
            self.folder.set(f"{len(files)} selected files")
            self.camera.set("")
            self.count.set(f"{len(files)} selected; filenames determine processing order")

    def browse_video(self):
        from vision3d.io.video import inspect_video

        path = filedialog.askopenfilename(title="Select MP4 video", filetypes=[("MP4 video", "*.mp4")], parent=self.root)
        if not path:
            return
        try:
            info = inspect_video(path)
        except (ValueError, OSError) as exc:
            self.error(str(exc))
            return
        self.video_path = path
        self.selected_files = ()
        self.camera.set("")
        self.folder.set(path)
        self.count.set(f"MP4: {info.width}×{info.height} · {info.fps:.2f} fps · {info.seconds:.1f} s")
        self.video_settings_button.configure(state="normal")

    def video_settings(self):
        sample = simpledialog.askfloat("Video settings", "Candidate frames per second (0.5–12)",
                                       initialvalue=self.video_options.get("sample_hz", 4),
                                       minvalue=.5, maxvalue=12, parent=self.root)
        if sample is None:
            return
        count = simpledialog.askinteger("Video settings", "Maximum retained keyframes (2–50)",
                                        initialvalue=self.video_options.get("max_keyframes", 40),
                                        minvalue=2, maxvalue=50, parent=self.root)
        if count is None:
            return
        aware = messagebox.askyesnocancel(
            "Video settings", "Use overlap-aware keyframe selection?\n\nYes: overlap-aware\nNo: time-uniform",
            default="yes" if self.video_options.get("strategy") == "overlap-aware" else "no", parent=self.root)
        if aware is not None:
            strategy = "overlap-aware" if aware else "time-uniform"
            self.video_options = {"sample_hz": sample, "max_keyframes": count, "strategy": strategy}
            self.note(f"Video settings: {sample:g} candidates/s, at most {count} keyframes, {strategy} selection.")

    def choose_output(self):
        path = filedialog.askdirectory(title="Select parent folder for new reconstruction runs", parent=self.root)
        if path:
            self.output_base = Path(path)
            self.note(f"New runs will be saved under {path}")

    def export_ply(self):
        path = filedialog.asksaveasfilename(title="Save colored point cloud", defaultextension=".ply",
                                           filetypes=[("PLY point cloud", "*.ply")], parent=self.root)
        if path:
            try:
                shutil.copyfile(self.result.ply, path)
                self.note(f"Saved colored PLY: {path}")
            except OSError as exc:
                self.error(str(exc))

    def switch_result(self):
        other = self.job_results.get(not self.result.is_dense) if self.result else None
        if other:
            self.show_result(other, previous=True)
            self.busy(False)

    def cancel(self):
        self.stop.set()
        self.cancel_button.configure(state="disabled")
        self.stage.set("Stopping reconstruction…")

    def browse_camera(self):
        value = filedialog.askopenfilename(title="Select calibrated camera JSON", filetypes=[("JSON", "*.json")], parent=self.root)
        if value:
            self.camera.set(value)

    def load_castle(self):
        try:
            folder, camera = resolve_castle(self.root_dir)
            self.selected_files = ()
            self.video_path = None
            self.video_settings_button.configure(state="disabled")
            self.folder.set(str(folder))
            self.camera.set(str(camera))
            self.show_count()
            self.note("Castle selected: six genuine photographs and published camera intrinsics. Click Start Reconstruction.")
        except (ValueError, OSError) as exc:
            self.error(str(exc))

    def busy(self, active):
        for button in self.actions:
            button.configure(state="disabled" if active else "normal")
        self.view_button.configure(state="normal" if not active and self.result and self.viewer is None else "disabled")
        self.quality_box.configure(state="disabled" if active else "readonly")
        self.enhanced_check.configure(state="disabled" if active else "normal")
        self.video_settings_button.configure(state="normal" if self.video_path and not active else "disabled")
        self.profile_box.configure(state="disabled" if active else "readonly")
        dense = not active and self.dense_profile.get() != "Sparse only"
        self.mesh_check.configure(state="normal" if dense else "disabled")
        self.glb_check.configure(state="normal" if dense and self.mesh_requested.get() else "disabled")
        self.texture_check.configure(state="normal" if dense and self.mesh_requested.get() else "disabled")
        self.switch_button.configure(state="normal" if not active and len(self.job_results) == 2 and self.viewer is None else "disabled")
        self.cancel_button.configure(state="normal" if active and not self.closing and not self.stop.is_set() else "disabled")
        self.export_button.configure(state="normal" if not active and self.result else "disabled")
        mesh_available = self.result and self.result.is_dense and (self.result.directory / "mesh.ply").is_file()
        self.mesh_button.configure(state="normal" if not active and mesh_available and self.viewer is None else "disabled")
        self.texture_button.configure(state="normal" if not active and self.viewer is None else "disabled")
        self.output_button.configure(state="normal" if not active and self.run_folder else "disabled")

    def start(self):
        if self.worker is not None or self.closing:
            return
        if not self.folder.get():
            self.error("Select a video, an image folder or at least two photographs first.")
            return
        self.result = None
        self.textured_path = None
        self.run_folder = None
        self.job_results.clear()
        self.result_path.set("New reconstruction — validating input")
        self.statistics.set("Registered cameras: —    Landmarks: —")
        self.stop.clear()
        self.started = perf_counter()
        self.busy(True)
        profile = self.dense_profile.get()
        mesh = profile != "Sparse only" and self.mesh_requested.get()
        self.worker = threading.Thread(target=run_job, args=(self.selected_files or self.folder.get(), self.camera.get(), self.output_base,
            lambda kind, value: self.events.put((kind, value)), self.stop, int(self.quality.get()),
            None if profile == "Sparse only" else profile, mesh, mesh and self.glb_requested.get(), self.enhanced_matching.get(),
            self.video_options.copy() if self.video_path else None, mesh and self.texture_requested.get()),
            name="reconstruction-worker", daemon=False)
        self.worker.start()

    def show_result(self, result, previous=False, elapsed=None):
        self.result = result
        self.run_folder = result.directory
        label = "Previous result (not computed now)" if previous else "New reconstruction"
        self.stage.set(f"{label}: {result.status}")
        point_label = "Dense points" if result.is_dense else "Sparse landmarks"
        self.statistics.set(f"Registered cameras: {result.registered}    {point_label}: {result.landmarks:,}")
        self.result_path.set(str(result.directory))
        self.elapsed.set(f"Saved engine runtime: {result.seconds:.2f} s" + (f" · process wall time: {elapsed:.2f} s" if elapsed is not None else ""))
        self.note(f"{label}: {result.registered} cameras, {result.landmarks:,} {point_label.lower()}. {result.directory}")
        if result.status == "partial":
            self.note("Partial result: the engine quality target was not reached. Inspect metrics.json and rejected camera records.")

    def previous(self):
        folder = filedialog.askdirectory(title="Open a result folder containing metrics.json and sparse.ply or dense.ply", parent=self.root)
        if folder:
            try:
                result = read_result(folder)
                self.job_results.clear()
                asset = result.directory.parent / "texture-asset" / "textured.glb"
                self.textured_path = asset if asset.is_file() else None
                self.show_result(result, previous=True)
                self.busy(False)
            except (ValueError, OSError) as exc:
                self.error(str(exc))

    def open_output(self):
        try:
            open_directory(self.run_folder)
            self.note(f"Opened output folder: {self.run_folder}")
        except (OSError, ValueError, TypeError) as exc:
            self.error(str(exc))

    def open_texture(self):
        path = self.textured_path
        if path is None or not path.is_file():
            filename = filedialog.askopenfilename(title="Open a saved photographic GLB", parent=self.root,
                                                   filetypes=[("glTF binary model", "*.glb")])
            if not filename:
                return
            path = Path(filename)
        self.open_view(textured=path)

    def open_view(self, mesh=False, textured=None):
        if self.viewer is not None or (textured is None and not self.result):
            return
        try:
            path = textured if textured is not None else (self.result.directory / "mesh.ply" if mesh else self.result.ply)
            if not path.is_file():
                raise ValueError(f"Model file is missing: {path}")
            mesh_flag = ["--textured"] if textured is not None else ["--mesh"] if mesh else []
            session = allocate_run(self.output_base / "viewers")
            self.viewer_status = session / "status.json"
            self.viewer_control = session / "control.json"
            self.show_cameras.set(False)
            self.camera_pending = None
            with (session / "viewer.log").open("w", encoding="utf-8") as log:
                self.viewer = subprocess.Popen([sys.executable, "-m", "vision3d.demo.viewer", "--ply", str(path), *mesh_flag,
                    "--status-file", str(self.viewer_status), "--control-file", str(self.viewer_control)], stdout=log, stderr=subprocess.STDOUT, **child_options())
            self.view_button.configure(state="disabled")
            self.mesh_button.configure(state="disabled")
            self.texture_button.configure(state="disabled")
            self.note(f"Opening {path.name} in Open3D.")
            self.viewer_reported = False
        except (OSError, ValueError) as exc:
            self.error(f"Could not start 3D view: {exc}")

    def set_camera_visibility(self):
        if self.viewer is None or self.viewer.poll() is not None or self.closing:
            return
        try:
            desired = self.show_cameras.get()
            write_camera_visibility(self.viewer_control, desired)
            self.camera_pending = desired
        except (OSError, ValueError) as exc:
            self.camera_pending = None
            self.error(f"Could not change camera visibility: {exc}")

    def poll(self):
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "stage":
                self.stage.set(value)
                self.note(value)
            elif kind == "validated":
                self.count.set(f"{value['usable']} readable images / {value['supported']} supported files")
                if value["automatic_intrinsics"]:
                    source = "a video focal-length prior" if value.get("video") else "per-image EXIF or fallback estimates"
                    self.note(f"Using {source}. Geometry may be biased; see intrinsics.json for provenance.")
                for skipped in value["skipped"]:
                    self.note("Skipped unreadable input: " + skipped)
            elif kind == "run":
                self.run_folder = Path(value)
                self.result_path.set(value)
            elif kind == "result":
                self.job_results[value[0].is_dense] = value[0]
                self.show_result(value[0], elapsed=value[1])
                self.started = None
            elif kind == "notice":
                self.note(value)
            elif kind == "texture":
                self.textured_path = Path(value)
                self.run_folder = self.textured_path.parent
                self.stage.set("Photographic GLB ready")
                self.result_path.set(str(self.run_folder))
                self.note(f"Saved photographic GLB: {value}")
            elif kind == "error":
                self.stage.set("Failed — see details below")
                self.note(value)
                self.started = None
            # The done event may precede thread exit. Controls are re-enabled below only after exit.
        if self.worker is not None and not self.worker.is_alive():
            self.worker.join()
            self.worker = None
            self.started = None
            if not self.closing:
                self.busy(False)
                if self.stop.is_set() and self.result is None:
                    self.stage.set("Cancelled — incomplete run files may remain")
        if self.started is not None:
            self.elapsed.set(f"Elapsed: {perf_counter() - self.started:.1f} s")
        if self.viewer is not None:
            status = {}
            try:
                if self.viewer_status.is_file():
                    status = json.loads(self.viewer_status.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass  # A status-file write may be in progress.
            if status.get("status") == "open" and not self.viewer_reported:
                self.note(f"Open3D view opened: {status['points']} colored PLY vertices.")
                self.viewer_reported = True
            if status.get("status") == "open" and not self.closing:
                self.camera_checkbox.configure(state="normal" if status.get("cameras_available") else "disabled")
                visible = bool(status.get("cameras_visible"))
                if self.camera_pending is None or self.camera_pending == visible:
                    self.show_cameras.set(visible)
                    self.camera_pending = None
            if self.viewer.poll() is not None:
                if not self.closing:
                    if self.viewer.returncode:
                        self.error("Open3D viewer failed: " + status.get("message", f"exit {self.viewer.returncode}; see {self.viewer_status.parent / 'viewer.log'}"))
                    else:
                        self.note("3D viewer closed. Reconstruction remains available.")
                self.viewer = None
                self.show_cameras.set(False)
                self.camera_pending = None
                self.camera_checkbox.configure(state="disabled")
                if not self.closing:
                    self.busy(self.worker is not None)
        if self.closing:
            if self.worker is None and self.viewer is None:
                self.root.destroy()
                return
            if self.viewer is not None and perf_counter() - self.close_started > 5:
                self.viewer.kill()
        self.root.after(100, self.poll)

    def close(self):
        if self.closing:
            return
        if self.worker is not None and not messagebox.askyesno("Close application?",
                "Reconstruction is running. Close the application and stop its child process? Incomplete outputs will remain for diagnosis.", parent=self.root):
            return
        self.closing = True
        self.close_started = perf_counter()
        self.stop.set()
        self.busy(True)
        self.camera_checkbox.configure(state="disabled")
        self.stage.set("Closing child processes…")
        if self.viewer is not None and self.viewer.poll() is None:
            self.viewer.terminate()


def main():
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
