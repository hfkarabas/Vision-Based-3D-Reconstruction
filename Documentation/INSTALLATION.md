# Installation and verification

Use 64-bit Windows with Python 3.12 and a working OpenGL driver for the Open3D viewer. The reconstruction runs on the CPU. The release was exercised on Windows with CPython 3.12.14, OpenCV 4.11.0 and Open3D 0.19.0. Blender 5.3 was used for independent import and rendering.

From the repository root, create an environment if the checkout has none.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m pip check
```

The lock file records the Windows validation environment. The repository's Python package can also be installed with `pip install -e ".[dev]"`. For an existing uv environment without pip, use `uv pip check --python .venv/Scripts/python.exe`. Blender is required only for building the independently validated release bundle. Dataset downloads, original photos and generated assets live outside Git. The AirZoo comparisons require the original local photos and separately prepared benchmark metadata.

```powershell
.venv\Scripts\python.exe -m tkinter
.venv\Scripts\python.exe -m vision3d.cli reconstruct-asset --help
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -m vision3d.demo
```

Close the Tk test window before running subsequent commands. You can also double-click `run_demo.bat`. The launch script sets the repository working directory and uses its own virtual environment.

The fresh Rural run reached about 1.81 GiB peak Python working set across all stages on the development workstation. Leave additional memory for Windows and the preview, and disk space for PLY, GLB, OBJ, logs and optional Blender renders. Larger or different image sets can use more resources.

If the preview fails to open, check the viewer log under the chosen output location and confirm the process has access to a Windows desktop and a working graphics driver. If reconstruction fails, read the stage log or `workflow.json`. An incompatible camera JSON, too little overlap, insufficient depth corroboration or a failed surface-quality check can each stop the pipeline. Use fresh output paths for retries.

See the [demonstration guide](DEMO_GUIDE.md) and [release evaluation](FINAL_RELEASE_REPORT.md).
