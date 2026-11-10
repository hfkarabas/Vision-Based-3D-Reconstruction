# Vision-Based-3D-Reconstruction
Camera-independent 3D reconstruction from real-world images and videos using computer vision and Structure-from-Motion.

This Computer Engineering graduation project aims to produce usable 3D assets from photographs or video without ground-truth camera poses, Unity, simulators, or COLMAP as its processing engine. The current implementation covers input, preprocessing, features, matching, two-view reconstruction and **incremental sparse multi-view SfM with bounded local bundle adjustment**. A local desktop GUI handles sparse reconstruction. Opt-in multi-view and calibrated two-view stereo modes produce dense clouds and open vertex-colored meshes.

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

Next stages: broader camera/dataset validation, track refinement and stronger local optimization coverage. Multi-view dense fusion, UV texturing, OBJ/GLB export and adaptive frame selection are later work. An interactive local sparse-reconstruction demo is available below.

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

## Experimental registration recovery and progress evidence

```powershell
.venv\Scripts\python.exe scripts/benchmark_sfm.py --suite configs/recovery_experiments.json --output outputs/recovery-final
.venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence --input datasets/scene --camera configs/camera.json --local-ba --recovery-pairs 12 --joint-ba --output outputs/recovery-example
.venv\Scripts\python.exe scripts/progress_evidence.py --baseline outputs/recovery-baseline --experiments outputs/recovery-final --output outputs/progress-presentation-final
```

These flags are **experimental and off by default**. `--recovery-pairs` bounds additional appearance-ranked matching (0–30 attempts, at most three new attempts per image), prioritizing disconnected components. Only geometrically verified edges enter tracks. `--joint-ba` attempts one final sparse refinement capped at 30 cameras, 2000 landmarks and the existing evaluation budget, with actual anchor-connectivity checks, fixed initial poses/intrinsics, unchanged quality gates and atomic rollback.

On Temple 30, discovery registered 23/30 cameras versus 18/30, but rotation accuracy worsened, including on the same 18-camera comparison set. Joint refinement reduced Temple 15 image residuals while worsening reference rotation accuracy; other joint proposals rolled back. These are controlled negative/tradeoff findings, not a successful accuracy-improving default. Full registration and safe drift reduction remain unresolved; dense reconstruction is not ready.

See [the recovery report](docs/TASK05_REPORT.md) for root causes, all individual/combined experiments, exact metrics and reproducibility. `outputs/progress-presentation-final/` contains traceable real-image matches, reconstructed points/cameras, registration and pose-accuracy comparisons, graph diagnostics and compact tables for the November 9, 2026 university update. No slides or academic paper are generated. Keep the pre-change baseline run (or archived preceding source) for the evidence script, and use fresh output directories when reproducing.

## Local desktop demonstration

Launch from the configured checkout (Python 3.12, existing dependencies and Tkinter):

```powershell
.venv\Scripts\python.exe -m vision3d.demo
```

Alternatively, double-click `run_demo.bat`. The launcher uses this checkout's `.venv` and does not require administrator privileges. Standard Windows Python includes Tkinter; verify it with `.venv\Scripts\python.exe -m tkinter` if needed. No new Python dependencies, browser, server or dataset download is required.

For the presentation, select **Load Castle Demo → Start Reconstruction → Open 3D View**. The existing six photographs in `datasets/sceaux/images` and published intrinsics in `configs/sceaux.json` must already be present. With the validated fixed-camera point refinement, a real run takes about 20–23 seconds on the tested desktop; allow additional time on other hardware. Mouse left-drag rotates, the wheel zooms, and Ctrl + left-drag or wheel-button drag pans. Close the viewer with Q or the window close button; the application remains available. R resets the camera-aligned view; C toggles estimated camera outlines and A toggles coordinate axes when camera metadata exists. Overlays are display context, not reconstructed landmarks. Every exported point is displayed; no filtering or interpolation is applied by the viewer.

After the 3D window opens, use **Show Cameras (C)** beside **Open Output Folder** in the desktop application to show or hide estimated camera positions and viewing directions as frustums. Pressing **C** in the Open3D window controls the same overlay and synchronizes the checkbox. Toggle freely without rerunning reconstruction or reopening the PLY. Camera visibility is independent of the A-key coordinate-axis overlay. The checkbox is disabled until the viewer has loaded usable pose and intrinsic metadata; standalone PLY files without that metadata remain viewable.

For custom input, choose **Folder…** or **Files…** and select 2–30 JPEG/PNG photographs. Two readable photos use pair reconstruction, while three or more use incremental SfM. Filenames determine order. **Camera JSON** is optional. Use a shared camera, an `images` mapping keyed by exact filenames, or a `camera1`/`camera2` file for pair mode in sorted filename order. Calibration must describe original EXIF-oriented dimensions. If no file is selected, each image gets an independent focal estimate from EXIF or a fallback. Unreadable files are reported and skipped. Small baselines, scene changes, weak overlap and inaccurate focal estimates can still cause rejection or biased geometry.

Choose **Output location…** to set the parent folder for new runs. **Longest side** bounds processed resolution. **Cancel** stops the active worker and retains any incomplete diagnostics. **Save PLY As…** copies the current point cloud. The original run also keeps camera poses and intrinsics in JSON.

Each run gets a new `outputs/demo/run_<UTC timestamp>_<id>/` directory. `reconstruction.log` and `demo_request.json` sit beside the engine's `reconstruction/` folder, which contains `metrics.json`, `sparse.ply`, poses, tracks and intrinsics. Existing results are not overwritten. Stage messages come from completed engine artifacts; camera and landmark counts appear only after final metrics exist. A partial result is clearly labeled and does not mean all input images registered. Sparse PLY is not a mesh or a textured asset.

**Open Output Folder** opens the actual run/result folder. **Open Previous Result** loads an existing reconstruction folder containing `metrics.json` and `sparse.ply`; it is explicitly labeled as previously computed, making it a presentation backup rather than a claimed live run. **Open 3D View** launches a separate Open3D process using the actual PLY. No unstable rendering widget is embedded in Tkinter. Duplicate jobs and duplicate viewers are prevented; closing the app asks before stopping an active reconstruction and reaps its child process. Cancellation also leaves any completed prior run files available through **Open Previous Result**.

Troubleshooting:

- Missing Castle data: select existing local photographs manually; the app does not download replacements.
- Unexpected geometry with automatic intrinsics: inspect `intrinsics.json`. If available, supply calibration with matching EXIF-oriented image dimensions. See `configs/sceaux.json` for the shared-camera format.
- Reconstruction rejected/failed: read the visible error and the run's `reconstruction.log` / `metrics.json`; select a connected, translated, nonplanar sequence. Quality thresholds are unchanged.
- Missing/corrupt PLY: select an intact result or rerun reconstruction. The viewer validates nonempty finite points and colors.
- Open3D window failure: use a local graphical desktop with a working OpenGL driver. Reconstruction is CPU-based; visualization still needs graphics support. Remote/headless rendering and the target laptop are not validated.

See [the desktop validation report](docs/TASK06_REPORT.md) for exact real-run results, automated tests and the limits of interactive validation. Experimental graph discovery and joint BA are not enabled by the demo; the established bounded local BA configuration is retained.

## Sparse quality investigation

The desktop now adds a bounded final **fixed-camera multi-view point refinement**. It optimizes existing landmarks with at least three accepted observations, preserving camera poses, intrinsics, associations, colors and point count. Finite/depth/reprojection/angle/movement gates reject unsafe proposals. Two-view points are unchanged. The standalone CLI retains its previous defaults; request refinement explicitly:

```powershell
.venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence --input datasets/sceaux/images --camera configs/sceaux.json --local-ba --refine-points --output outputs/castle-refined-new
# Reproduce individual ablations in a NEW directory; existing run folders are rejected:
.venv\Scripts\python.exe scripts/reconstruction_quality.py --output outputs/quality-new --case castle_baseline --case castle_refined --case temple_baseline --case temple_refined
.venv\Scripts\python.exe scripts/reconstruction_quality.py --output outputs/quality-new --summarize
```

Recorded Castle results retain 6/6 cameras and 3,915 landmarks: RMS reprojection error decreases from 0.587 to 0.477 px, while median error increases from 0.285 to 0.306 px. Temple retains 18/30 cameras, 2,664 landmarks and exactly the same reference camera errors, with lower RMS but slightly higher median reprojection error. These results improve aggregate point fit conditional on fixed cameras; they do not establish absolute surface accuracy or recover missing architectural detail. Larger feature targets, stricter matching, higher resolution and expanded BA were not adopted. Quality subsets are labeled diagnostic plots only and are not automatically pruned from PLY outputs.

See [the sparse-quality report](docs/TASK07_REPORT.md) for all ablations, genuine before/after figures, independent validation, runtime costs and remaining limitations. The original demonstration measurements and historical benchmark/presentation artifacts remain unchanged.


## Automatic intrinsics and calibrated stereo

Without Camera JSON the desktop uses the same opt-in mode as CLI `--auto-intrinsics`. EXIF 35 mm equivalent focal length scales by the oriented image diagonal relative to the 36 × 24 mm reference frame. Physical focal length and focal-plane resolution are used when the recorded pixel grid and units are usable. Suspect or conflicting metadata can fall back to focal length equal to image diagonal (about 53° diagonal FOV). The inferred model assumes a centered principal point and zero distortion. Source, confidence and assumptions are recorded per image, before and after resize. Explicit JSON takes precedence in the desktop. Automatic initialization estimates focal length only. It does not calibrate a lens or optimize intrinsics.

```powershell
.venv\Scripts\python.exe -m vision3d.cli reconstruct-pair --image1 datasets/sceaux/images/100_7101.JPG --image2 datasets/sceaux/images/100_7103.JPG --auto-intrinsics --output outputs/my-auto-pair
.venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence --input datasets/sceaux/images --auto-intrinsics --local-ba --refine-points --output outputs/my-auto-sequence
```

The experimental CPU stereo stage takes a successful **explicitly calibrated** pair result and its unchanged original photos. It requires equal processed sizes (64–1600 px per side), horizontal rectification and a bounded disparity search. It exports `dense.ply` and optionally `mesh.ply` to a separate directory. The mesh connects neighboring accepted disparity samples, rejects depth jumps, and uses image-derived vertex colors. It is an open visible surface with holes at occlusions or unmatched regions. Open a completed output through **Open Previous Result**, then **Open 3D View** or **Open Mesh**.

```powershell
.venv\Scripts\python.exe scripts/motorcycle_experiment.py --output outputs/motorcycle-sparse
.venv\Scripts\python.exe -m vision3d.cli dense-pair --pair-dir outputs/motorcycle-sparse --output outputs/motorcycle-dense --mesh
```

PLY exports retain the sparse world frame and arbitrary baseline scale. Camera poses and intrinsics are available as JSON. See the [completion report](docs/COMPLETION_REPORT.md) for validation and limitations.

## Multi-view dense reconstruction and GLB

In the desktop, choose **fast**, **balanced** or **high** under **Multi-view** and optionally enable **Colored mesh** and **GLB export**. Run with three or more overlapping registered photographs. The sparse stage finishes first and remains available if the dense stage fails or is cancelled. Use **Switch Sparse / Dense**, **Open 3D View** and **Open Mesh** to inspect the results. **Open Output Folder** exposes the validated PLY and optional GLB. Sparse-only remains the default. The separate calibrated two-photo stereo mode is still available through the CLI.

```powershell
.venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence --input datasets/sceaux/images --camera configs/sceaux.json --local-ba --output outputs/castle-sparse
.venv\Scripts\python.exe -m vision3d.cli dense-sequence --sparse-dir outputs/castle-sparse --output outputs/castle-dense --profile fast --mesh --glb
.venv\Scripts\python.exe scripts/benchmark_multiview.py --scene castle=outputs/castle-sparse --output outputs/castle-profiles
```

The dense stage estimates bidirectionally checked stereo depth, requires corroboration from a distinct reference and at least three cameras across the stereo edges, and fuses accepted samples in world voxels. It checks source image hashes against the sparse run. Profile settings bound resolution, attempted neighbors and reference count. Results are sensitive to overlap, pose quality and calibration. The mesh is an open visible surface with colors sampled from depth-consistent photographs. GLB stores vertex colors and normals, with a right/up/back axis transform. UV textures, OBJ export and metric scale are unavailable.

Start with **fast** for meshes. Higher profiles use more memory, and optional CLI `--target-faces` simplification can take substantially longer and may retain more faces than requested. See the [evaluation](docs/MULTIVIEW_EVALUATION.md) and [engineering report](docs/MULTIVIEW_COMPLETION_REPORT.md) for measurements and limitations.

### Difficult small photo sets

**Enhanced matching (experimental)** in the desktop tests every pair for sequences of up to 30 photos, applies MAGSAC verification and probes candidate seeds for a third camera using the existing PnP gates. The equivalent CLI flags are `--exhaustive --verifier magsac --probe-seeds --ratio 0.80`. Use `--pair-cache DIR` to reuse reciprocal descriptor matches while recomputing geometry. The original local schedule is the default and is faster on ordered photos. For offline diagnostics, run `python scripts/sfm_diagnostics.py --run SPARSE_DIR --output NEW_DIR --keypoints`. For dense-only foreground selection, `dense-sequence --object-masks FILE.json` accepts normalized polygons keyed by registered filenames. See the [real-world investigation](docs/REAL_WORLD_SFM_IMPROVEMENTS.md) for measured gains and remaining limitations.

## MP4 reconstruction

Choose **Video…** in the desktop app to preview the decoded resolution, frame rate and nominal duration. **Video settings…** controls how often the selector evaluates a candidate and the maximum number of retained keyframes. The selector scores sharpness, feature coverage, overlap and image displacement before passing the selected views to SfM. Supply **Camera JSON** if you have calibration at the decoded video resolution. Otherwise, the run records its estimated focal length and lens assumptions. Select a multi-view profile for dense reconstruction, with optional colored mesh and GLB export when the registered views support it.

Use `python -m vision3d.cli keyframes --help` for standalone extraction. See the [video workflow](docs/VIDEO_RECONSTRUCTION.md), [selection algorithm](docs/VIDEO_KEYFRAME_SELECTION.md) and [evaluation](docs/PHOTO_VS_VIDEO_EVALUATION.md) for reproduction commands and measured results.
