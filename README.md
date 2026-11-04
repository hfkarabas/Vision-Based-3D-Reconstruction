# Vision-Based-3D-Reconstruction
Camera-independent 3D reconstruction from real-world images and videos using computer vision and Structure-from-Motion.

This Computer Engineering graduation project aims to produce usable 3D assets from photographs or video without ground-truth camera poses, Unity, simulators, or COLMAP as its processing engine. The current implementation covers input, preprocessing, features, matching, two-view reconstruction and **incremental sparse multi-view SfM with bounded local bundle adjustment**. Dense reconstruction, meshing, texturing and an application GUI are not implemented.

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

The installable `src/vision3d` package contains `io` (streamed source frames), `preprocessing` (validation/resize/quality), `features` (SIFT/ORB and staged matching), `geometry` (camera models, fundamental/essential geometry and triangulation), `reconstruction` (pair/sequence orchestration, tracks/state, incremental registration, bounded BA and export), and `cli`. Unit and integration tests live in `tests/`. Camera/quality JSON examples live in `configs/`; `scripts/motorcycle_experiment.py` and `scripts/sceaux_experiment.py` reproduce real experiments. Local `datasets/`, `outputs/`, and `experiments/` are ignored.

Only two processed images and their feature sets are retained for consecutive matching. Per-image/pair metric records grow with input length. CPU brute-force matching grows with the product of feature counts; limits matter on 16 GB machines. Diagnostic canvases can be large for high-resolution input.

## Validation and limitations

See [the matching report](docs/TASK01_REPORT.md), [the two-view report](docs/TASK02_REPORT.md) and [the incremental SfM report](docs/TASK03_REPORT.md) for automated tests, real photographs, source/permission details and measured results. No data or generated images are committed.

RANSAC can fit misleading geometry for moving subjects, planar scenes, pure rotation, repeated texture, or small baselines. High match counts are not evidence of usable camera geometry or successful reconstruction. Blur scores are descriptive and do not automatically reject frames. Truncated videos may appear as early decoder EOF; codec support depends on the OpenCV build. Failed runs may leave partial diagnostics. Timing excludes interpreter startup and final JSON/CSV serialization and is not a controlled benchmark. The target i5-11400H/GTX 1650 machine has not been separately tested.

Next stages: broader camera/dataset validation, track refinement and stronger local optimization coverage. Dense reconstruction, meshing, OBJ/GLB export, adaptive frame selection and interactive visualization are later work.

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

## Reconstruct an ordered photograph sequence

```powershell
.venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence --input datasets/scene --camera configs/camera.json --output outputs/scene --local-ba
.venv\Scripts\vision3d.exe reconstruct-sequence --input datasets/scene --camera configs/camera.json --quality-config configs/sequence_quality.json --window 2 --long-gap 4 --max-features 6000 --max-dimension 1600 --output outputs/scene-custom
# Reproduce six genuine unrectified Castle photographs and separate pair-mode checks:
.venv\Scripts\python.exe scripts/sceaux_experiment.py --download --local-ba --output outputs/sceaux-experiment
```

Inputs are JPEG/PNG files sorted deterministically by filename, not EXIF capture time. Original filename order, capture timestamps when present (without inventing a timezone), EXIF orientation, hashes and resize transforms are retained separately from registration order. Corrupt images are skipped and listed. Use one shared camera JSON or `{"images": {"photo01.jpg": {...camera fields...}, "photo02.jpg": {...}}}` with exact filenames. Dimensions must describe the original EXIF-oriented images. Approximation is also explicit for sequences through `--approximate-intrinsics` and one focal/FOV assumption, with warnings and metadata as for pairs.

Default matching covers neighboring offsets 1 and 2 plus offset 4. `--window` changes the neighboring range; `--long-gap 0` disables extra links. Matches undergo reciprocal Lowe filtering and fundamental RANSAC in undistorted virtual pixels before entering tracks. Conflict-free track merging allows only one feature per image. Stable observation identifiers are `(image_id, keypoint row index)`; track/landmark links and keypoint arrays are exported. Geometrically valid initial pairs are scored using retained point count, parallax, cheirality and filter survival. Later cameras use tracked 2D–3D correspondences with PnP RANSAC and LM refinement, never chained pairwise translations. Unregisterable images are deferred until new landmarks become available, then rejected if no progress remains; their exported pose is null.

One landmark belongs to each triangulated track. The widest registered ray pair seeds new geometry, and all additional observations must pass depth/reprojection checks. Final retained points need at least two accepted views, >=1° triangulation angle and finite bounded coordinates. Camera axes and world-to-camera poses follow the two-view convention. The first selected anchor becomes the world reference, which may differ from input image 0. Absolute scale remains unknown.

`--local-ba` enables SciPy sparse-Jacobian `soft_l1` optimization after successful registrations. Defaults: at most five cameras including two fixed covisible boundary cameras, 400 landmarks visible in both boundary cameras, and 30 function evaluations. Intrinsics and global initial anchors remain fixed. Fixed boundary poses constrain local gauge/scale; fixed initial poses preserve initial-pair bias. Updates are transactional: robust local cost and global RMS must not increase, previously valid geometry must remain valid, and camera/point movement bounds must pass. Other proposals roll back. A finite budget-limited solution can be accepted if every quality check passes; convergence status is reported separately.

`configs/sequence_quality.json` demonstrates adjustable `geometry`, `registration` and `bundle` settings. Defaults preserve the two-view 1px virtual RANSAC, 2px reprojection and 1° angle gates; PnP requires 20 inliers and 40% support. Sequence resource bounds are 30 images by default (configurable 2–50), at most 20,000 requested features/image, and explicit BA limits. These are small-sequence limits, not a claim of unbounded SfM scalability. `--min-registered` defaults to five; fewer registered images produce partial status and exit 3. It is an explicit user target, not an automatic threshold reduction. Success also requires sufficient valid landmarks and final support for registered cameras.

Sequence outputs: `sparse.ply`, `poses.json`, `intrinsics.json`, `tracks.json`, `landmarks.json` (per-view errors/depths and angle), `features.npz` (coordinates indexed by stable feature IDs), `pairs.json`, `initial_candidates.json`, `manifest.json`, `metrics.json`, and `overview.png`. Open3D reloads exported PLY files to check coordinates and colors. Exit codes remain 0 success, 3 geometric rejection/partial reconstruction and 2 invalid input or processing failure. No geometry is invented for gaps. There is no reconstruction resume/import command yet.

Six Sceaux Castle photographs registered successfully in one shared scene. The upstream sample contains published K values but no distortion coefficients or reference poses; zero distortion is an explicit assumption. Its copyright notice names Pierre Moulon, and no explicit redistribution license is provided. The experiment uses the upstream SfM testing sample locally and does not redistribute its images. See the report for hashes, source URLs, calibrated-versus-approximate validation, BA acceptance/rollback results and exact performance measurements.

## Reliability benchmarks and feature caching

```powershell
# Downloads only missing pinned samples; prepares separate camera-only and reference files:
.venv\Scripts\python.exe scripts/prepare_benchmarks.py --download
# Fresh child process per case; calibrated and approximate experiments remain separate:
.venv\Scripts\python.exe scripts/benchmark_sfm.py --output outputs/sfm-benchmark
# Optional individual cases (repeat --case to select several):
.venv\Scripts\python.exe scripts/benchmark_sfm.py --output outputs/temple-check --case temple30
# Reuse descriptors during repeated reconstruction without changing matching or poses:
.venv\Scripts\vision3d.exe reconstruct-sequence --input datasets/scene --camera configs/camera.json --feature-cache outputs/feature-cache --local-ba --output outputs/cached-scene
```

`configs/benchmark_suite.json` declares Castle as the tuning/regression scene and Middlebury TempleRing as held-out evaluation, including 6-, 15- and 30-image subsets and cold/warm-cache cases. The scripts pin downloads/checksums, retain natural filename order and keep all photographs outside Git. Source/usage notices and calibration assumptions are documented in [the reliability report](docs/TASK04_REPORT.md). The Temple archive contains calibrated camera references, but only K enters processing; reference R/t is read for evaluation after the reconstruction process exits. The reference laser-scanned object model is not publicly distributed by this benchmark, so sparse surface accuracy is not claimed.

Benchmark outputs include `results.json`, `summary.csv`, `suite_snapshot.json`, source fingerprint in `execution.json`, individual run logs and reconstruction directories. Results separate target-based CLI success from full image coverage: a run meeting `--min-registered 5` can still be labeled `partial_registration` by the benchmark if it registers fewer than all inputs. Every rejection is retained in per-image attempts. Stage times cover loading/preprocessing, features/cache, matching, tracks, initial geometry, registration, triangulation, BA, validation and export. Memory is a fresh child-process peak, not an assumption about 16 GB laptop compatibility.

Feature caches are optional, numeric NPZ files with pickle disabled. Keys cover source SHA-256, detector/limits, resize/shape, OpenCV/NumPy versions and extraction/preprocessing implementation hashes. Arrays retain SIFT float32 and ORB uint8 descriptors and keypoint metadata. Writes are atomic; corrupt or inconsistent entries warn and are recomputed. Source images still load for validation, EXIF and colors. Cache hits report zero new extraction time and retain measured cache-loading wall time. Cold and warm CLI runs are tested for identical poses/landmarks. Caching does not cache geometric verification or make a disconnected scene reconstructable.

Reliability corrections include conflict-aware initial selection, spatially distributed PnP support (default >=1% image-area convex hull), exclusion of ambiguous 2D–3D associations, reranking after map growth, and up to three triangulation-pair hypotheses with per-track rejection reasons. The sequence matcher runs a single undistorted fundamental verification; existing `analyze` and pair matching keep their original defaults. Track conflict examples and two-view-only track counts are exported. These checks reduce known failure paths but cannot guarantee correct associations in repeated texture.

Local BA now selects two fixed, noncoincident covisible boundary cameras when the active window no longer sees both initial anchors. Initial anchors remain fixed globally; boundary poses preserve the current world frame and scale. Window/point/evaluation bounds stay unchanged. Updates must preserve the identities of previously valid observations and landmarks, not just aggregate invalid counts. Candidate pruning completes before atomic commit. Both local same-observation RMS/robust cost and global same-observation metrics are recorded; new geometric invalidity causes rollback even if an error summary improves.

Temple evaluation demonstrates why reprojection fit alone is insufficient: the 30-input run registered 18 cameras and had about 0.26px median reprojection error but about 6.6° median aligned rotation error. Similarity-aligned camera-center errors and relative rotation/translation-direction errors are exported separately, with reporting warnings for poor reference agreement. Approximate intrinsics can fit image observations while producing substantially biased camera orientation. The current engine remains a bounded sparse research pipeline; the measured registration gaps and drift need resolution before dense reconstruction. See the report for exact results and failure cases.
