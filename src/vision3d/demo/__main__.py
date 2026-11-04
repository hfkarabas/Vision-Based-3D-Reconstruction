"""Launch with python -m vision3d.demo in the existing project environment."""
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
from time import perf_counter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
        self.count = tk.StringVar(value="No image folder selected")
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
        ttk.Label(panel, text="Image Folder").grid(row=0, column=0, sticky="w", pady=4)
        self.folder_entry = ttk.Entry(panel, textvariable=self.folder, state="readonly")
        self.folder_entry.grid(row=0, column=1, sticky="ew", padx=8)
        self.button(panel, "Browse…", self.browse_folder).grid(row=0, column=2)
        ttk.Label(panel, textvariable=self.count).grid(row=1, column=1, sticky="w", padx=8, pady=(0, 8))
        ttk.Label(panel, text="Camera JSON").grid(row=2, column=0, sticky="w")
        ttk.Entry(panel, textvariable=self.camera, state="readonly").grid(row=2, column=1, sticky="ew", padx=8)
        self.button(panel, "Browse…", self.browse_camera).grid(row=2, column=2)
        choices = ttk.Frame(panel)
        choices.grid(row=3, column=0, columnspan=3, sticky="w", pady=12)
        self.button(choices, "Load Castle Demo", self.load_castle).pack(side="left")
        self.button(choices, "Open Previous Result", self.previous).pack(side="left", padx=8)
        self.start_button = self.button(choices, "Start Reconstruction", self.start)
        self.start_button.pack(side="left")
        ttk.Label(panel, text="Stable SIFT · local BA · fixed-camera point refinement · experimental pose modes off").grid(
            row=4, column=0, columnspan=3, sticky="w")
        ttk.Separator(panel).grid(row=5, column=0, columnspan=3, sticky="ew", pady=12)
        ttk.Label(panel, textvariable=self.stage, wraplength=700).grid(row=6, column=0, columnspan=3, sticky="w")
        ttk.Label(panel, textvariable=self.elapsed).grid(row=7, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Label(panel, textvariable=self.statistics).grid(row=8, column=0, columnspan=3, sticky="w")
        ttk.Label(panel, textvariable=self.result_path, wraplength=700).grid(row=9, column=0, columnspan=3, sticky="w", pady=8)
        result_buttons = ttk.Frame(panel)
        result_buttons.grid(row=10, column=0, columnspan=3, sticky="w")
        self.view_button = ttk.Button(result_buttons, text="Open 3D View", command=self.open_view, state="disabled")
        self.view_button.pack(side="left")
        self.output_button = ttk.Button(result_buttons, text="Open Output Folder", command=self.open_output, state="disabled")
        self.output_button.pack(side="left", padx=8)
        self.camera_checkbox = ttk.Checkbutton(result_buttons, text="Show Cameras (C)", variable=self.show_cameras,
                                               command=self.set_camera_visibility, state="disabled")
        self.camera_checkbox.pack(side="left", padx=8)
        ttk.Label(panel, text="3D view: left-drag rotates · wheel zooms · Ctrl + left-drag pans").grid(
            row=11, column=0, columnspan=3, sticky="w", pady=8)
        self.log = tk.Text(panel, height=9, wrap="word", background="white", foreground="black",
                           font=("Calibri", 10), relief="solid", borderwidth=1, state="disabled")
        self.log.grid(row=12, column=0, columnspan=3, sticky="nsew")
        panel.rowconfigure(12, weight=1)
        self.note("Select calibrated photographs or load the existing Castle demonstration. No datasets are downloaded.")
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
            self.folder.set(value)
            self.camera.set("")
            try:
                self.show_count()
            except (ValueError, OSError) as exc:
                self.error(str(exc))

    def browse_camera(self):
        value = filedialog.askopenfilename(title="Select calibrated camera JSON", filetypes=[("JSON", "*.json")], parent=self.root)
        if value:
            self.camera.set(value)

    def load_castle(self):
        try:
            folder, camera = resolve_castle(self.root_dir)
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
        self.output_button.configure(state="normal" if not active and self.run_folder else "disabled")

    def start(self):
        if self.worker is not None or self.closing:
            return
        if not self.folder.get() or not self.camera.get():
            self.error("Select an image folder and camera calibration JSON first.")
            return
        self.result = None
        self.run_folder = None
        self.result_path.set("New reconstruction — validating input")
        self.statistics.set("Registered cameras: —    Landmarks: —")
        self.stop.clear()
        self.started = perf_counter()
        self.busy(True)
        self.worker = threading.Thread(target=run_job, args=(self.folder.get(), self.camera.get(), self.output_base,
            lambda kind, value: self.events.put((kind, value)), self.stop), name="reconstruction-worker", daemon=False)
        self.worker.start()

    def show_result(self, result, previous=False, elapsed=None):
        self.result = result
        self.run_folder = result.directory
        label = "Previous result (not computed now)" if previous else "New reconstruction"
        self.stage.set(f"{label}: {result.status}")
        self.statistics.set(f"Registered cameras: {result.registered}    Landmarks: {result.landmarks:,}")
        self.result_path.set(str(result.directory))
        self.elapsed.set(f"Saved engine runtime: {result.seconds:.2f} s" + (f" · process wall time: {elapsed:.2f} s" if elapsed is not None else ""))
        self.note(f"{label}: {result.registered} cameras, {result.landmarks} landmarks. {result.directory}")
        if result.status == "partial":
            self.note("Partial result: the engine quality target was not reached. Inspect metrics.json and rejected camera records.")

    def previous(self):
        folder = filedialog.askdirectory(title="Open previous reconstruction folder containing metrics.json and sparse.ply", parent=self.root)
        if folder:
            try:
                self.show_result(read_result(folder), previous=True)
                self.busy(False)
            except (ValueError, OSError) as exc:
                self.error(str(exc))

    def open_output(self):
        try:
            open_directory(self.run_folder)
            self.note(f"Opened output folder: {self.run_folder}")
        except (OSError, ValueError, TypeError) as exc:
            self.error(str(exc))

    def open_view(self):
        if self.viewer is not None or not self.result:
            return
        try:
            if not self.result.ply.is_file():
                raise ValueError(f"PLY file is missing: {self.result.ply}")
            session = allocate_run(self.output_base / "viewers")
            self.viewer_status = session / "status.json"
            self.viewer_control = session / "control.json"
            self.show_cameras.set(False)
            self.camera_pending = None
            with (session / "viewer.log").open("w", encoding="utf-8") as log:
                self.viewer = subprocess.Popen([sys.executable, "-m", "vision3d.demo.viewer", "--ply", str(self.result.ply),
                    "--status-file", str(self.viewer_status), "--control-file", str(self.viewer_control)], stdout=log, stderr=subprocess.STDOUT, **child_options())
            self.view_button.configure(state="disabled")
            self.note("Starting separate Open3D viewer; loading the saved PLY.")
            self.note("All exported points are displayed. R resets camera-aligned view; C toggles camera outlines; A toggles axes when camera metadata is available.")
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
                for skipped in value["skipped"]:
                    self.note("Skipped unreadable input: " + skipped)
            elif kind == "run":
                self.run_folder = Path(value)
                self.result_path.set(value)
            elif kind == "result":
                self.show_result(value[0], elapsed=value[1])
                self.started = None
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
                self.note(f"Open3D view opened: {status['points']} actual colored PLY points.")
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
