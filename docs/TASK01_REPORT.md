# TASK 01 implementation and validation report

Validation date: 2026-11-04 (Europe/Istanbul). This milestone implements preprocessing and feature matching, **not full 3D reconstruction**.

## Implementation

The audited repository contained only README.md, .gitignore, .gitattributes and the initial Git commit. No existing application or dataset was replaced. An installable src-layout Python package now provides streamed JPEG/PNG directory and MP4 input, EXIF orientation, explicit resize transforms, grayscale/quality statistics, SIFT/ORB extraction, L2/Hamming KNN matching, ratio filtering, optional reciprocal ratio agreement, fundamental matrix RANSAC, diagnostic JPEGs and JSON/CSV exports. Degenerate/insufficient correspondences produce explicit statuses and empty inlier sets. No pose estimation, triangulation, SfM, bundle adjustment, dense reconstruction, mesh, UI, Unity code or CUDA integration was introduced.

## Final structure

```text
src/vision3d/
  __init__.py
  io/__init__.py
  preprocessing/__init__.py
  features/__init__.py
  features/matching.py
  geometry/__init__.py
  cli/__init__.py
  cli/__main__.py
tests/
  unit/test_core.py
  integration/test_pipeline.py
docs/TASK01_REPORT.md
pyproject.toml
requirements-lock.txt
README.md
.gitignore
.gitattributes                       # existing, unchanged
.venv/                               # local, ignored
.python/ and .uv-cache/               # local runtime/cache, ignored
configs/                             # local reserved directory, no placeholder files
datasets/opencv-basketball/           # ignored photographs and upstream license
outputs/real-sift/ and real-orb/      # ignored diagnostics and metrics
experiments/                         # local reserved directory, ignored
```

No empty reconstruction or evaluation packages were created. The CLI currently owns the small orchestration/export workflow; those concerns can move into dedicated modules if later reuse requires it.

## Environment and dependencies

Windows x64, CPython **3.12.14**, isolated `.venv` with system-site-packages disabled. Direct pins successfully installed:

| Dependency | Version |
|---|---|
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| opencv-contrib-python | 4.11.0.86 |
| open3d | 0.19.0 |
| matplotlib | 3.10.3 |
| pandas | 2.2.3 |
| Pillow | 11.2.1 |
| pytest (development extra) | 8.3.5 |

All 75 installed distributions, including editable vision3d, passed `uv pip check`. Exact third-party versions are in `requirements-lock.txt`; this snapshot omits the machine-specific editable URL. Imports of Open3D, SciPy, pandas, Matplotlib and OpenCV succeeded. Open3D's published dependencies install Dash 4.5.0 and Flask 3.1.3 transitively, conflicting with the requested avoidance of web-framework installation. We retained the explicitly requested Open3D dependency rather than creating an incomplete `--no-deps` install. No web application was built. No PyTorch, TensorFlow, CUDA toolkit or COLMAP is installed in this environment.

Python 3.12 Windows wheel support was verified by actual installation, rather than inferred from the system Python 3.14. Open3D also publishes its [supported installation platforms](https://www.open3d.org/docs/release/getting_started.html).

## Automated validation

Final command: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=outputs/pytest-final`. Result: **16 passed, 0 failed, 0 skipped in 0.90 seconds**. Coverage includes JPEG/PNG sorting and corrupt-file handling; invalid and missing inputs; EXIF rotation; aspect-preserving resize, grayscale and blur; both native descriptor types; ratio and reciprocal matching; zero/single-descriptor safety; degenerate/insufficient geometry; successful nonplanar synthetic epipolar verification; full image pipeline and metrics; MP4 sampling, timestamps, frame maximum and full CLI outputs; unreadable MP4 handling. Synthetic shifted textures only validate plumbing; the separate photographs below validate real-image operation.

Both CLI entry points were exercised:

```powershell
.venv\Scripts\python.exe -m vision3d.cli --help
.venv\Scripts\vision3d.exe analyze --help
.venv\Scripts\python.exe -m vision3d.cli analyze --input datasets/opencv-basketball --detector sift --mutual --output outputs/real-sift
.venv\Scripts\vision3d.exe analyze --input datasets/opencv-basketball --detector orb --mutual --output outputs/real-orb
```

## Real-world smoke test

Two genuine overlapping photographic frames, `basketball1.png` and `basketball2.png`, from the official OpenCV **4.11.0** sample tree. Each is 640x480; no resizing occurred. They contain moving people, so the scene is not a rigid-scene reconstruction benchmark. No camera calibration or ground-truth pose was used.

Source URLs:

- https://raw.githubusercontent.com/opencv/opencv/4.11.0/samples/data/basketball1.png
- https://raw.githubusercontent.com/opencv/opencv/4.11.0/samples/data/basketball2.png
- [Upstream repository Apache-2.0 license](https://github.com/opencv/opencv/blob/4.11.0/LICENSE)

Downloaded 2026-11-04 using PowerShell `Invoke-WebRequest <URL> -OutFile datasets/opencv-basketball/<filename>`. Only the two images (265,589 bytes combined) and upstream license were downloaded, not the full repository or dataset. They remain local and ignored. License recorded from the upstream repository; the sample directory does not supply separate image-specific licensing in this download.

| File | Bytes | SHA-256 |
|---|---:|---|
| basketball1.png | 132826 | ba06f6701f7260998b430c39b6557f775497e6ce7b1a74f0b7ea6af371bf54a6 |
| basketball2.png | 132763 | b3047772e0dc6c9d83c26f64894ca1a897f799f855c5574ac24b2ef934c2978e |

Configuration: feature target 4000, Lowe ratio 0.75, reciprocal ratio agreement enabled, RANSAC threshold 1.5 processed pixels, confidence 0.99, OpenCV RNG seed 0, one OpenCV thread, consecutive images.

| Measurement | SIFT | ORB |
|---|---:|---:|
| Images processed | 2 | 2 |
| Keypoints image 1 / image 2 | 539 / 554 | 2460 / 2437 |
| Raw nearest matches | 539 | 2460 |
| Ratio-filtered matches | 350 | 1114 |
| Mutual matches submitted to RANSAC | 341 | 906 |
| RANSAC inliers | 258 | 715 |
| Inlier ratio (inliers / mutual matches) | 0.756598 | 0.789183 |
| Feature seconds image 1 / image 2 | 0.047572 / 0.046375 | 0.992667 / 0.005331 |
| Matching and verification seconds | 0.007403 | 0.064237 |
| Pipeline seconds | 0.159419 | 1.182305 |
| Process lifetime peak working set (bytes) | 121589760 | 199245824 |

Single observed runs, not averages or a detector speed ranking. ORB's first-call overhead dominates this run. Total timing includes input, preprocessing, extraction, matching and diagnostic encoding, but excludes interpreter startup, final metrics serialization and memory instrumentation. Windows peak working set includes interpreter/import allocations and excludes child processes. Other platforms may report null. CPU model/RAM could not be independently queried because sandbox CIM access was denied; no minimum-machine or GPU performance claim is made.

Example artifacts are under `outputs/real-sift/` and `outputs/real-orb/`: manifest.json, metrics.json, summary.csv, two keypoint images, and raw/ratio/inlier pair images. `frames/` is intentionally empty for image input. The SIFT inlier diagnostic was visually inspected: genuine shared scene content is linked, but moving subjects and possible epipolar-compatible mismatches remain. A large inlier count does not prove valid camera geometry or successful 3D reconstruction.

## Problems and resolutions

- Only Python 3.14 was listed by the launcher. Found a bundled CPython 3.12.14 and created `.venv` with it; a project-local uv-managed 3.12.14 download also succeeded with network access. The active venv uses the bundled 3.12 runtime, not global 3.14.
- Sandboxed network connections failed. Dependency and tiny sample downloads succeeded with normal network permissions. No conflicting package versions required upgrades; pinned versions installed together.
- The first `venv` pip bootstrap failed in the sandbox. `uv pip` successfully installed into the isolated environment without relying on pip there. Standard user setup can use pip or uv as documented.
- Pytest temporary directory access was denied by the sandbox; reran with normal filesystem access and a project-local `--basetemp`. Initial setup errors were environmental and are not counted as test passes.
- Matplotlib's user cache path was restricted during import verification; the import succeeded with a temporary cache. Set `MPLCONFIGDIR` to a writable location in similarly restricted environments.
- Required Open3D brings web frameworks transitively, as disclosed above. It stays out of the runtime processing path.

## Remaining technical risks and recommended TASK 02

Nominal frame-index/fps timestamps do not represent variable-frame-rate presentation times. Decoder EOF cannot always distinguish truncation from a clean ending. Image decoding still allocates one full source image before resizing. Brute-force matching and large diagnostic canvases can be costly; defaults and streaming bound the working set for this baseline. Metrics grow linearly with sequence length. Source paths/configuration are recorded but input file hashing is not yet automatic. JSON configuration files, nonconsecutive pair selection and adaptive frame selection are not implemented.

Planar scenes, pure rotation, repeated texture, nonrigid motion and small baselines require explicit model/degeneracy checks before pose recovery. Current prechecks reject obviously collinear/duplicate points; a successful fundamental-matrix status is not a comprehensive scene-validity test. EXIF and resize conventions must be incorporated into future intrinsics. The minimum development machine remains untested. The dependency snapshot is Windows-specific and has no wheel hashes.

Recommended next task: define and validate camera intrinsics/metadata and coordinate conventions, introduce calibration input with EXIF/resize-aware intrinsics, and add robust two-view essential geometry with rigid-scene, homography/pure-rotation and parallax checks on static real photographs. Pose recovery and triangulation can follow those checks in the next milestone, with numerical validation before incremental SfM. No such reconstruction work was started here.
