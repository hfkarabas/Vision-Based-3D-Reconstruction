# Desktop demonstration implementation and validation

Validation date: November 4, 2026. Intended presentation: November 9, 2026.

**Delivery verdict:** ready for the demonstrated Castle sparse-reconstruction workflow on the tested local Windows desktop. Two live GUI runs reconstructed all six Castle cameras and 3,915 colored landmarks, and the generated PLY was opened interactively in Open3D. This verdict does not certify other scenes, hardware, remote desktops or dense reconstruction.

## Implementation and scope

Added a lightweight Tkinter/ttk application titled **Vision-Based 3D Reconstruction**. It uses a white background, dark text, compact controls and Calibri. The application provides image-folder and camera-JSON selection, supported/readable image counts, a Castle preset, reconstruction execution, verified stage messages, elapsed time, final camera/landmark counts, output-folder opening and a separate interactive point-cloud viewer. Previously computed results can be loaded with an explicit **Previous result (not computed now)** label.

The existing SfM engine and reconstruction mathematics are unchanged. No datasets were downloaded, dependencies installed, experimental algorithms enabled, or dense reconstruction, mesh, texture, web or cloud features added.

| File | Responsibility |
|---|---|
| `src/vision3d/demo/__init__.py` | Desktop package |
| `src/vision3d/demo/__main__.py` | Tk main window, controls, queue polling and child-process lifecycle |
| `src/vision3d/demo/support.py` | Input/calibration validation, Castle resolution, unique output allocation, CLI orchestration and result parsing |
| `src/vision3d/demo/viewer.py` | Isolated Open3D loading and interactive rendering of the actual colored PLY |
| `tests/unit/test_demo.py` | Focused tests for orchestration and data/visualization boundaries |
| `run_demo.bat` | Windows launcher using the checkout's existing virtual environment |
| `README.md` | Launch instructions, workflow, supported inputs and troubleshooting |

## Architecture and execution

The Tk event loop owns all UI updates. A non-daemon worker thread performs input validation and launches the existing command in a separate process using the same Python executable:

```text
python -u -m vision3d.cli reconstruct-sequence
  --input <image folder> --camera <calibration JSON>
  --output <new run>/reconstruction --local-ba --min-registered 5
```

The registration target is `min(5, supported image count)` for directories containing fewer than five images. The demo accepts 2–30 supported JPEG/PNG files. All other sequence defaults are retained: SIFT, 6,000 feature target, 1,600-pixel maximum dimension, matching window 2, long gap 4, ratio 0.75 and seed 0. Established bounded local BA is enabled, as in the Castle benchmark configuration. Recovery-pair discovery and joint BA remain off. No geometric quality threshold is relaxed.

The worker sends messages through a thread-safe queue; the main thread polls every 100 ms. Conflicting controls are disabled during reconstruction, and a worker-state guard prevents duplicate launches. The engine's stdout/stderr are captured in a run-specific log rather than piped into a potentially blocking GUI reader.

Progress is deliberately coarse. Successfully parsed engine artifacts establish genuine transitions: `manifest.json` indicates prepared features, `pairs.json` saved geometric matches, and `initial_candidates.json` selected initialization. The GUI does not invent percentages or intermediate camera counts. Final counts and engine runtime come from `metrics.json` after the child exits. Process wall time is measured separately and excludes the preceding validation step. Partial engine results are labeled as partial; rejected or failed runs show errors and available log locations.

Each run atomically reserves `outputs/demo/run_<UTC timestamp>_<unique id>/`. The engine creates a new `reconstruction/` child; it never receives an existing output directory. The parent contains `demo_request.json` and `reconstruction.log`. The child contains the original engine's PLY, metrics, manifest, intrinsics, poses, tracks, landmarks and diagnostics. Failed or interrupted run folders remain available for diagnosis.

Open3D runs in a separate process, not an embedded Tk widget. It reads the actual `sparse.ply`, rejects missing/empty/non-finite or uncolored point clouds, preserves point colors, resets the viewpoint and uses a white background. Viewer status and logs are kept under `outputs/demo/viewers/`. Closing a viewer leaves the Tk application and results available. Only one viewer process is launched at a time.

Application shutdown asks before stopping an active reconstruction. The worker terminates its child, waits up to five seconds and kills it if necessary. Tk remains alive until the worker exits. Viewer processes are also stopped on application shutdown. No Cancel button is advertised. Worker cleanup is covered by controlled process tests; not every OS/driver failure mode was exercised interactively.

## Inputs and calibration

Image candidates are deterministically sorted by case-insensitive filename, with an exact-name tie-break. Validation decodes one image at a time, applies EXIF orientation for dimension checks, reports corrupt files and requires at least two readable images. Supported-file count and readable-image count are shown separately.

The desktop demo requires a calibrated camera JSON, either the existing shared-camera format or an `images` mapping keyed by exact filenames. Missing entries, inconsistent dimensions, conflicting K values and explicitly approximate camera models are rejected. Calibration must describe original EXIF-oriented pixels; the engine retains responsibility for resize transformations and intrinsic scaling. Approximate calibration remains an explicit CLI-only feature.

**Load Castle Demo** resolves `datasets/sceaux/images` and `configs/sceaux.json` relative to the source checkout, independent of the launch directory. It checks that the six-photo subset and camera file exist; full readability/calibration checks run before reconstruction. No machine-specific dataset path is hardcoded. Missing data produces an explanation and manual browsing remains available.

Castle provenance is unchanged: the local OpenMVG Sceaux Castle sample with published K, no supplied distortion coefficients and an explicit zero-distortion assumption. Dataset source/licensing details remain in the earlier reconstruction reports. Images are not redistributed in this commit.

## Launch and presentation workflow

Prerequisites: the configured Python 3.12 virtual environment and existing project dependencies, Tkinter (verified Tk 8.6), the local Castle subset and an interactive desktop with a functioning OpenGL driver for Open3D. No administrator privileges, browser or server are required.

From `C:\projects\Vision-Based-3D-Reconstruction`:

```powershell
.venv\Scripts\python.exe -m vision3d.demo
```

Alternatively, double-click `run_demo.bat`. It changes to its own directory, checks for `.venv\Scripts\python.exe` and launches the same module. A missing virtual environment produces setup guidance. The launcher was reviewed for path quoting and entry-point consistency; double-click behavior was not a separate interactive validation claim.

Presentation sequence:

1. Launch the application and select **Load Castle Demo**.
2. Confirm six detected images and the populated calibration path.
3. Select **Start Reconstruction**. Explain the genuine stage updates while the engine runs.
4. Show the final registered-camera and landmark counts, runtime and result directory.
5. Select **Open 3D View**, rotate the genuine colored sparse cloud and zoom.
6. Close the viewer; use **Open Output Folder** to inspect the saved artifacts if needed.

Open3D's documented controls are left-drag to rotate, wheel to zoom, and Ctrl + left-drag or wheel-button drag to pan. **Panning was not tested.** The original on-screen Shift-pan instruction was corrected after inspecting Open3D's help: Shift + left-drag rolls rather than pans.

For a backup, **Open Previous Result** selects an existing engine result directory containing `metrics.json` and `sparse.ply`. The application identifies it as previously computed. Do not describe this backup as a live reconstruction.

For custom photographs, browse to an ordered image folder and select its calibration JSON. The format is supported, but no custom-scene end-to-end validation is claimed for this delivery.

## Actual Castle validation

Both recorded live runs used the GUI's Castle preset and unchanged stable engine configuration. Counts are from saved metrics and the PLYs were reloaded with the demo's Open3D loader.

| Measurement | First live run | Second live run |
|---|---:|---:|
| Input photographs | 6 | 6 |
| Registered cameras | 6/6 | 6/6 |
| Valid colored landmarks / reloaded PLY points | 3,915 | 3,915 |
| Engine runtime | 15.4545 s | 15.4239 s |
| Displayed child-process wall time | 16.39 s | 16.59 s |
| Engine peak working set | 559.11 MiB | 558.82 MiB |
| Final status | success | success |

The engine peak is its Windows process lifetime peak working set, not total application-plus-viewer memory. These are two observed timings, not a statistical performance benchmark or a claim about other machines.

The first run records median reprojection error **0.285393 px**, RMS error **0.586640 px**, median triangulation angle **10.066965°**, and zero invalid final observations. These describe the existing reconstruction output; reprojection fit alone does not establish absolute scene or pose accuracy.

Recorded result directories, relative to the repository:

```text
outputs/demo/run_20261104T140750758312Z_ffa98385/reconstruction/
outputs/demo/run_20261104T140933771445Z_691b6898/reconstruction/
```

Both PLY files have SHA-256:

```text
c4da8692a315174cfac23197b0830a18d9f0b2012128a2221be44c0657ffd4c4
```

The distinct run directories and matching independently reloaded outputs verify repeat execution without overwriting the earlier result. Local evidence is recorded in `outputs/demo-validation/castle_runs.json`, each run's request/log/metrics files, and viewer status files. Generated artifacts remain ignored by Git.

## Test and interaction coverage

The original baseline was 77 passing tests. The desktop additions bring the suite to **104 passing tests**; no existing test was removed or weakened. The final delivery rerun completed with **104 passed, 0 failed, 14 existing Matplotlib/Pyparsing deprecation warnings in 6.73 s**:

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp outputs/demo-validation/delivery-final-temp
```

Captured output: `outputs/demo-validation/delivery-final-tests.log`. Test execution used normal OS temporary-directory access because sandboxed pytest setup encountered directory-permission failures; those setup failures were not algorithm failures or hidden as successful runs.

Focused tests cover supported-file ordering, missing/corrupt/insufficient images, calibration validation, per-image mapping, Castle resolution/missing data, unique run allocation, stable command flags, metadata extraction, partial/rejected/missing/corrupt results, complete-artifact stage transitions, worker validation/launch/exit failures, child cleanup, colored PLY reload and output-folder validation.

| Check | Evidence and scope |
|---|---|
| Real reconstruction through GUI | Two successful Castle runs, saved metrics and actual PLY files |
| UI responsiveness | Live elapsed/stage updates observed during reconstruction; conflicting buttons disabled |
| Colored PLY display | Actual 3,915-point Castle output opened in a separate Open3D window |
| Rotation and zoom | Mouse rotation and wheel zoom exercised interactively |
| Viewer closure | Viewer closed and application controls/results remained available |
| Output directory | GUI action opened the actual reconstruction directory in Windows Explorer |
| Duplicate reconstruction launch | Guard verified in the real Tk smoke harness and controls disabled during live runs |
| Error recovery | Tk smoke harness submitted an invalid input, observed failure and re-enabled controls |
| Previous-result labeling | Tk smoke harness loaded a genuine saved Castle result with explicit previous-result status and correct counts |
| Shutdown cleanup | Controlled worker-process test verifies termination and wait; Tk harness closes after idle validation |
| Panning | Not gesture-tested; documented Open3D functionality only |

The one-off Tk harness results are saved in `outputs/demo-validation/gui_checks.json`. Its previous-result check substitutes the file-dialog return value; it is not a claim that every native dialog interaction was manually tested. Computer Use was interrupted by the user after the recorded successful interactions; final delivery uses those existing records and does not claim additional interactive validation.

## Known limitations and readiness

- This is a source-checkout desktop demonstration, not a packaged installer. Local Castle images and calibration must already exist.
- Input is an ordered calibrated photograph directory, capped at 30 supported files. Video reconstruction and approximate intrinsics are not exposed in this GUI.
- Progress granularity follows saved engine artifacts; per-image live camera/landmark counts and percentage completion are unavailable.
- Output remains a colored sparse PLY with monocular scale ambiguity. Dense surfaces, mesh generation, textures and OBJ/GLB export are not implemented.
- Existing registration gaps and pose drift remain. The Temple coverage/rotation tradeoff is unchanged; experimental recovery and joint refinement stay disabled.
- Panning, custom-scene reconstruction, other hardware, the intended minimum laptop, remote/headless viewing and all shutdown/driver fault combinations have not been validated.
- Closing during input validation waits for that bounded validation work to finish before the worker exits. Interrupted reconstruction can leave incomplete diagnostic folders.

**Presentation readiness:** the verified live Castle path is ready for the November 9 demonstration on the tested desktop, with an explicitly labeled genuine previous-result backup. This does not imply readiness for dense reconstruction or reliable recovery of every arbitrary scene.

## Repository review

Only the desktop package, its focused tests, launcher, README changes and this report belong to this delivery. Existing deletions of the two bilingual Word deliverables were present before desktop work and are deliberately left unstaged and untouched. Datasets, PLYs, logs, generated outputs, caches and `.venv` remain excluded. No reconstruction-engine files, calibration inputs, benchmark results or existing tests were changed. No push, pull or fetch is performed.
