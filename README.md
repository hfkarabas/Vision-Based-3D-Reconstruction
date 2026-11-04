# Vision-Based-3D-Reconstruction
Camera-independent 3D reconstruction from real-world images and videos using computer vision and Structure-from-Motion.

This Computer Engineering graduation project aims to produce usable 3D assets from photographs or video without ground-truth camera poses, Unity, simulators, or COLMAP as its processing engine. The current implementation covers **input, preprocessing, features and pairwise matching only**. It does not reconstruct 3D geometry.

## Setup (Windows / Python 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m pytest -q
```

Alternatively, `pip install -e ".[dev]"` resolves the pinned direct dependencies and current compatible transitive dependencies. `requirements-lock.txt` records the exact successful Windows CPython 3.12.14 environment, excluding the local editable package. It is an environment snapshot, without artifact hashes; other operating systems may require a separate resolution. With uv, `uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt` and `uv pip install --python .venv\Scripts\python.exe -e .` work without pip bootstrapping.

The stack is NumPy 2.2.6, SciPy 1.15.3, opencv-contrib-python 4.11.0.86, Open3D 0.19.0, Matplotlib 3.10.3, pandas 2.2.3, Pillow 11.2.1, and pytest 8.3.5. Windows Python 3.12 wheels installed and imports passed. Open3D is reserved for later geometry/visualization work and is not imported by the pipeline. Its distribution brings Dash/Flask transitively; no server or web application is implemented. No CUDA toolkit or neural framework is required or installed for this project.

## CLI

```powershell
.venv\Scripts\python.exe -m vision3d.cli analyze --input datasets/sample --detector sift --mutual
.venv\Scripts\vision3d.exe analyze --input datasets/clip.mp4 --detector orb --interval 10 --max-frames 30 --max-dimension 1280
.venv\Scripts\python.exe -m vision3d.cli analyze --help
```

Use `--output outputs/my-run` for an explicit **new** directory. Existing directories are rejected to prevent overwrites. Default output paths use UTC timestamps. Images are sorted lexicographically by case-insensitive filename with a deterministic tie-break; use zero-padded numbering. `--interval` selects every Nth candidate image or source video frame, starting at zero. The maximum counts successfully loaded frames. Directory input is nonrecursive and accepts JPEG/PNG. Corrupt images are skipped with warnings and recorded in the manifest; no readable images is an error. MP4 decoding streams through OpenCV and releases its capture on exit.

`--max-dimension` shrinks the longest side without upscaling; `--max-features` controls OpenCV's detector target (SIFT can retain tied responses). `--ratio` defaults to 0.75. `--mutual` requires ratio-filtered agreement in both directions. `--threshold` is the fundamental matrix RANSAC distance in processed-image pixels (default 1.5); `--confidence` defaults to 0.99. `--no-blur` disables Laplacian scoring. `--seed` and `--threads` default to 0 and 1 for reproducibility within the same build. Python callers can pass detector-specific OpenCV parameters to `features.extract`, such as SIFT `contrastThreshold` or ORB `fastThreshold`.

Each run writes:

```text
outputs/<run>/
  manifest.json       # input, configuration, software versions, skipped files
  metrics.json        # per-image quality/features/resize, per-pair stages, timing/memory
  summary.csv         # pairwise matching metrics
  frames/            # sampled, processed video frames only; source index in name
  keypoints/         # detected keypoint diagnostics
  matches/           # raw nearest, ratio-filtered and RANSAC inlier diagnostics
```

Original photographs are referenced by absolute path, not duplicated. Frame IDs join records with diagnostic filenames. Features returned by the Python API include coordinates, native float32 SIFT or uint8 ORB descriptors, keypoints, dimensions, detector overrides, and elapsed extraction time. Raw matches mean the nearest candidate per query descriptor from KNN (k=2), before ratio filtering; the second neighbor is used for the ratio test. The mutual stage is recorded separately. Inlier ratio uses the post-mutual candidate count as denominator, or zero for an empty set.

Pillow applies EXIF orientation before preprocessing. Metrics retain raw source size and EXIF orientation, oriented original size, processed size, actual x/y scales, and a 3x3 pixel-center transform `T`: `(x',y') = (sx*(x+0.5)-0.5, sy*(y+0.5)-0.5)`. Future intrinsics in the oriented coordinate system must use `K' = T @ K`; raw sensor intrinsics also require the EXIF orientation transform. Rounding may make x/y scales slightly different. Video timestamps are nominal `source_frame_index / fps`, explicitly labeled; variable-frame-rate presentation timestamps are not extracted.

## Architecture

The installable `src/vision3d` package contains `io` (streamed source frames), `preprocessing` (validation/resize/quality), `features` (SIFT/ORB and staged matching), `geometry` (fundamental matrix RANSAC), and `cli` (orchestration/export). Unit and integration tests live in `tests/`. `docs/TASK01_REPORT.md` records validation and risks. Local `datasets/`, `outputs/`, and `experiments/` are ignored. Modules for reconstruction/evaluation and configuration loading will be introduced when they have actual functionality; there are no empty scaffolding modules.

Only two processed images and their feature sets are retained for consecutive matching. Per-image/pair metric records grow with input length. CPU brute-force matching grows with the product of feature counts; limits matter on 16 GB machines. Diagnostic canvases can be large for high-resolution input.

## Validation and limitations

See [the implementation report](docs/TASK01_REPORT.md) for automated tests, the real OpenCV basketball photo-pair smoke test, exact measurements, source/license details, and reproduction commands. No data or generated images are committed.

RANSAC can fit misleading geometry for moving subjects, planar scenes, pure rotation, repeated texture, or small baselines. High match counts are not evidence of usable camera geometry or successful reconstruction. Blur scores are descriptive and do not automatically reject frames. Truncated videos may appear as early decoder EOF; codec support depends on the OpenCV build. Failed runs may leave partial diagnostics. Timing excludes interpreter startup and final JSON/CSV serialization and is not a controlled benchmark. The target i5-11400H/GTX 1650 machine has not been separately tested.

Next stages: camera metadata/intrinsic calibration and coordinate conventions; robust two-view essential geometry with degeneracy checks; then pose estimation, triangulation and incremental SfM with bundle adjustment. Dense reconstruction, meshing, OBJ/GLB export, adaptive frame selection and interactive visualization are later work.
