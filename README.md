# Vision-Based-3D-Reconstruction
Camera-independent 3D reconstruction from real-world images and videos using computer vision and Structure-from-Motion.

This Computer Engineering graduation project aims to produce usable 3D assets from photographs or video without ground-truth camera poses, Unity, simulators, or COLMAP as its processing engine. The current implementation covers input, preprocessing, features, matching and **calibrated two-view sparse reconstruction**. It does not yet implement full multi-view SfM, bundle adjustment, dense reconstruction or meshing.

## Setup (Windows / Python 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m pytest -q
```

Alternatively, `pip install -e ".[dev]"` resolves the pinned direct dependencies and current compatible transitive dependencies. `requirements-lock.txt` records the exact successful Windows CPython 3.12.14 environment, excluding the local editable package. It is an environment snapshot, without artifact hashes; other operating systems may require a separate resolution. With uv, `uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt` and `uv pip install --python .venv\Scripts\python.exe -e .` work without pip bootstrapping.

The stack is NumPy 2.2.6, SciPy 1.15.3, opencv-contrib-python 4.11.0.86, Open3D 0.19.0, Matplotlib 3.10.3, pandas 2.2.3, Pillow 11.2.1, and pytest 8.3.5. Windows Python 3.12 wheels installed and imports passed. Open3D exports and reloads sparse PLY files; Matplotlib provides optional headless previews. Its distribution brings Dash/Flask transitively; no server or web application is implemented. No CUDA toolkit or neural framework is required or installed for this project.

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

The installable `src/vision3d` package contains `io` (streamed source frames), `preprocessing` (validation/resize/quality), `features` (SIFT/ORB and staged matching), `geometry` (camera models, fundamental/essential geometry and triangulation), `reconstruction` (pair orchestration and cloud export), and `cli`. Unit and integration tests live in `tests/`. Camera JSON examples live in `configs/`; `scripts/motorcycle_experiment.py` reproduces the real reconstruction and evaluates pose separately. Local `datasets/`, `outputs/`, and `experiments/` are ignored.

Only two processed images and their feature sets are retained for consecutive matching. Per-image/pair metric records grow with input length. CPU brute-force matching grows with the product of feature counts; limits matter on 16 GB machines. Diagnostic canvases can be large for high-resolution input.

## Validation and limitations

See [the matching report](docs/TASK01_REPORT.md) and [the two-view reconstruction report](docs/TASK02_REPORT.md) for automated tests, real photographs, source/permission details and measured results. No data or generated images are committed.

RANSAC can fit misleading geometry for moving subjects, planar scenes, pure rotation, repeated texture, or small baselines. High match counts are not evidence of usable camera geometry or successful reconstruction. Blur scores are descriptive and do not automatically reject frames. Truncated videos may appear as early decoder EOF; codec support depends on the OpenCV build. Failed runs may leave partial diagnostics. Timing excludes interpreter startup and final JSON/CSV serialization and is not a controlled benchmark. The target i5-11400H/GTX 1650 machine has not been separately tested.

Next stages: static unrectified camera validation and persistent feature tracks, then incremental multi-view SfM with bundle adjustment. Dense reconstruction, meshing, OBJ/GLB export, adaptive frame selection and interactive visualization are later work.

## Reconstruct two photographs

```powershell
.venv\Scripts\python.exe -m vision3d.cli reconstruct-pair --image1 datasets/a.jpg --image2 datasets/b.jpg --camera configs/camera.json --output outputs/my-pair --preview
# Explicit approximation, only when calibration is unavailable:
.venv\Scripts\vision3d.exe reconstruct-pair --image1 datasets/a.jpg --image2 datasets/b.jpg --approximate-intrinsics --fov-degrees 60 --output outputs/approximate-pair
```

`--camera` takes either one camera object used for both views or `{"camera1": {...}, "camera2": {...}}`. Required scalar fields are `width`, `height`, `fx`, `fy`, `cx`, `cy`. Optional fields include `distortion` (OpenCV coefficient order), `source`, `approximate`, `assumptions`, and `coordinates` (must be `"exif_oriented"`). An optional `K` must agree with the scalar fields. Example for an independently calibrated 800x600 camera:

```json
{
  "width": 800, "height": 600,
  "fx": 650.0, "fy": 650.0, "cx": 399.5, "cy": 299.5,
  "distortion": [-0.08, 0.01, 0.0, 0.0, 0.0],
  "source": "replace with your actual calibration source",
  "coordinates": "exif_oriented"
}
```

These example values are illustrative, not a calibration for arbitrary photographs. Calibration dimensions must match each original EXIF-oriented image. Raw sensor-coordinate calibration must be converted externally, including distortion/axis conventions, before loading; it is explicitly rejected as `coordinates: "raw"`. Resizing applies the recorded pixel-center transform to intrinsics and leaves distortion coefficients unchanged. Supported distortion is OpenCV's standard pinhole model, not fisheye. The program loads calibration; it does not estimate calibration from a checkerboard.

Approximation requires `--approximate-intrinsics` and exactly one of `--focal-px` (original oriented pixels) or `--fov-degrees` (horizontal FOV). It assumes square pixels, image-center principal point and zero distortion. A JSON camera marked `approximate: true` is also an explicit approximation and requires assumptions. Both paths warn and record reduced reliability; they do not turn guessed calibration into ground truth.

OpenCV camera axes are x right, y down, z forward. Camera 1 is the world frame. `X2 = R @ X1 + t`, camera 2 center is `-R.T @ t`, and `||t||=1`. Cloud coordinates are in arbitrary baseline units, not meters. The essential matrix uses undistorted normalized observations. Its default RANSAC tolerance is `1 / min(fx1, fy1, fx2, fy2)` in normalized units; pixel reprojection errors are computed with each camera's own distortion model. See the report for all rejection thresholds.

Runs write `sparse.ply`, `poses.json`, `intrinsics.json`, `manifest.json`, `metrics.json`, `correspondences.npz`, and optional `sparse_preview.png`. NPZ tracks contain processed distorted observations, original feature indices, essential/pose masks, triangulated points, filtering diagnostics and the observation indices corresponding to PLY vertex order. Original observation pixels follow the inverse resize transform. Colors sample image 1's original oriented pixels, not an undistorted or resized copy. Exit codes: **0** success, **3** geometric rejection (diagnostics retained, no cloud), **2** invalid input/configuration or processing error. Existing output directories are rejected. Quality defaults are immutable `geometry.two_view.Quality` parameters for Python callers and are recorded in every geometry result; the CLI does not expose threshold overrides.

Reproduce the small calibrated real-image experiment:

```powershell
.venv\Scripts\python.exe scripts/motorcycle_experiment.py --download --output outputs/motorcycle-run --preview
```

The script downloads only two checksum-verified Middlebury Motorcycle images (~1.3 MB) via scikit-image's official repository. Reference stereo pose is used only after reconstruction for evaluation, never as processing input. The default run produced 859 valid colored points with 0.087 px median reprojection error. For a local Open3D viewer, load the exported file with `open3d.io.read_point_cloud` and pass it to `open3d.visualization.draw_geometries`; this requires a graphical desktop.
