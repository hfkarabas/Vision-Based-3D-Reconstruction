# TASK 03 — feature tracks and incremental sparse SfM

Validation date: 2026-11-04, Europe/Istanbul. **Acceptance target achieved: six genuine unrectified photographs registered in one shared reconstruction**, with 3,913 validated colored 3D landmarks and a PLY exported/reloaded by Open3D. No dense reconstruction, meshing, texturing or GUI was implemented.

## Audit and architecture

Started from clean commit `dad8c6780325c837669d043ad958a4dbc3b07086`. Reviewed existing source, CLI, tests and both earlier reports before implementation. Reused image loading/EXIF handling, preprocessing and resize transforms, native SIFT/ORB descriptors, reciprocal matching, fundamental verification, camera validation, two-view essential/pose estimation, triangulation and PLY export. The original `analyze` and `reconstruct-pair` commands remain operational. No new dependencies or processing engines were installed.

New reconstruction modules:

| Module | Responsibility |
|---|---|
| `state.py` | View, Pose, Track, Landmark and State records; conflict-safe track construction |
| `incremental.py` | Initial-pair scoring, track-based PnP, triangulation, visibility and landmark validation |
| `bundle.py` | Bounded, sparse, transactional local bundle adjustment |
| `sequence.py` | Ordered input, camera/quality configuration, matching schedule, deferred registration and exports |

A `View` retains its image ID, source metadata, processed camera model, keypoint/descriptor `Features` record and original-image sampled colors. An observation is `(image_id, feature_id)`, where feature ID is the extracted keypoint's row index. A `Track` holds at most one feature per image. A `Landmark` has a unique ID equal to its owning track ID, a shared-world 3D position, RGB color and accepted observations. State separates input order from registered poses, registration order and attempt history. No source images are retained as a full sequence in memory; descriptors and point/color arrays are retained.

## Unrectified validation performed before incremental implementation

Used the first six original Sceaux Castle photographs supplied by the official OpenMVG SfM sample repository. These are ordinary photographs from differing camera viewpoints, not a rectified stereo pair. The sample supplies a K matrix. Its calibration accuracy is not independently known, and no lens coefficients are supplied: **zero distortion is an explicit assumption**, not a measured lens calibration.

Published intrinsics: width=2832, height=2128, fx=fy=2905.88, cx=1416, cy=1064. All six JPEGs have EXIF orientation 1. Preprocessing preserves the existing pixel-center convention: original 2832x2128 becomes 1600x1202, with independent actual x/y scales and `K'=T K`. Colors are sampled from original oriented image pixels by reversing T. EXIF capture timestamps are retained as source strings without assuming a timezone; the six times range from `2010:10:12 14:43:07` to `14:43:44`.

Ran the unchanged two-view pipeline on `100_7100.JPG` and `100_7101.JPG`, separately with the supplied K and an explicit 60-degree horizontal-FOV approximation:

| Measurement | Published K | Explicit approximate intrinsics |
|---|---:|---:|
| Features in each image | 6000 / 6000 | 6000 / 6000 |
| Essential / pose inliers | 887 / 887 | 823 / 823 |
| Valid triangulated points | 887 | 823 |
| Median reprojection error | 0.148867 px | 0.231053 px |
| Median triangulation angle | 11.621722 degrees | 11.491950 degrees |

Neither run changed the existing geometric thresholds. The approximation emitted a warning and recorded square pixels, centered principal point and zero distortion assumptions. Its success does not validate the guessed focal length. The existing numerical tests also exercise nonzero lens distortion and EXIF rotations; the real Castle sample cannot validate a nonzero distortion model because none is provided. A same-photograph control was rejected as rotation-dominant (exit 3, no PLY), under `outputs/sceaux-identical-rejected/`. No underlying two-view bug required a fix before proceeding.

## Dataset provenance and licensing

Source: [official OpenMVG Sceaux Castle sample](https://github.com/openMVG/ImageDataset_SceauxCastle), pinned revision **fde7f5faba555e3c54700477c304488613346a19**. Download date: 2026-11-04. Only `100_7100.JPG` through `100_7105.JPG` (six photos, **6,766,631 bytes** total), K.txt and the small upstream notice were downloaded, not a full repository/archive. The [upstream notice](https://raw.githubusercontent.com/openMVG/ImageDataset_SceauxCastle/fde7f5faba555e3c54700477c304488613346a19/Readme.txt) credits **Copyright 2012 Pierre MOULON** and supplies the images as an SfM sample. There is no explicit redistribution license in that notice; no permissive/Creative Commons license is claimed. Images stay local, ignored and uncommitted. Obtain appropriate permission before redistributing this sample in a publication or another repository.

Image URLs use `https://raw.githubusercontent.com/openMVG/ImageDataset_SceauxCastle/fde7f5faba555e3c54700477c304488613346a19/images/<filename>`. The experiment script records the complete URLs, notice URL and checksums in `provenance.json`.

| File | SHA-256 |
|---|---|
| 100_7100.JPG | b475a1f74cd54536b6fe186fb165431ef0ee502f56b0d9895d6da576e364ee1e |
| 100_7101.JPG | 05de696ef02054a3d22af94f48fa3cb88df801cfe48561ad82010a130f408723 |
| 100_7102.JPG | 9c25103cf8411909773a189d524eb959a0f8a9ac74a1632b3ca343b605b7e034 |
| 100_7103.JPG | eebf00b09b773aff229e53c4d8c744c5f64f3c004fa8edece0f3d8c917a19411 |
| 100_7104.JPG | 3b18d6112bb68aa15e1e20d3be234ea4fa88ecf349f42d89bd7830d75ebe4fc6 |
| 100_7105.JPG | e67d29ccf210fab3227fec99e5a32862d3266f81bf4694eeb30cbe984a0b309f |

No reference camera poses are supplied in this sample, so no ground-truth alignment or pose-accuracy claim is made. No externally reconstructed structure or poses are used as input. The application does not run OpenMVG or COLMAP; OpenMVG supplies only test photographs and K.

## Tracking and initial-pair selection

By default, match ordered image offsets 1 and 2, plus selected non-neighbor links at offset 4. Reuse SIFT L2 or ORB Hamming KNN matching, ratio 0.75 and reciprocal ratio agreement. Fundamental RANSAC verification runs on **undistorted normalized observations multiplied by a shared virtual focal length**. Its 1px threshold therefore has an explicit common virtual-pixel interpretation. Only verified edges enter tracks; descriptor similarity alone never merges them.

Edges are processed deterministically in ascending descriptor-distance order, with observation IDs breaking ties. Union-find rejects a proposed merge if it would introduce two different features from one image; existing components remain intact. Duplicate/cycle edges are idempotent. Track IDs follow sorted minimum observations. This is a conservative greedy conflict policy, not a guarantee that every transitive match is correct; later 3D projection checks reject inconsistent observations.

Every candidate pair with enough geometric matches is passed through the existing essential estimation, pose, cheirality, homography/rotation, reprojection and parallax gates. Accepted candidates are ranked by:

`valid_points * min(median_angle_degrees / 5, 2) * cheirality_ratio * valid_point_ratio`.

The chosen pair was **image IDs 1 and 3**, filenames `100_7101.JPG` and `100_7103.JPG`. It had 952 verified input correspondences, 950 essential/pose inliers, 10.686326-degree median angle, 0.135070px median pair reprojection error, 100% cheirality and 0.326316 homography overlap. Conflict-free track linking initialized 949 landmarks; an unmatched/conflicting track association was excluded. All candidate scores and rejection reasons are exported, rather than selecting by raw descriptor count.

## Shared frame, registration and landmark management

Convention remains **world-to-camera**: `Xc=R Xw+t`, x right, y down, z forward. The first selected anchor (image 1) is the world origin with identity pose. Camera center is `C=-R.T@t`; inverse pose is `[R.T|C]`. The second selected anchor fixes a unit baseline. Monocular scale remains arbitrary; coordinates are not meters. Original input order and EXIF timestamps remain available even when registration order differs.

Register pending images in descending count of existing track-linked 2D–3D correspondences, with stable image-ID ties. Use `solvePnPRansac` (EPNP, 2000 iterations, confidence 0.999), followed by `solvePnPRefineLM` on RANSAC inliers. Recompute the consensus after refinement. Defaults require >=20 inliers, >=40% inlier fraction, <=2px reprojection error and positive depth for accepted correspondences, finite proper poses and bounded camera distance. No chained pairwise translation is used. Retry deferred images after new landmarks become available; stop when a complete pass makes no progress. Failed cameras have null exported poses, not fabricated placements.

For a new track observed by registered cameras, select the widest world-frame bearing pair, derive relative pose from their shared-frame poses, and reuse the existing normalized triangulator. Convert the point back into the world frame. Require finite bounded coordinates, positive depth, <=2px error per accepted observation and >=1-degree parallax. Validate all available registered observations; retain a consistent majority including both seed views. Existing landmarks accept new observations only after projection/depth checks, and track IDs prevent duplicate landmarks.

Final validation evaluates every retained observation and each landmark's maximum multi-view triangulation angle. Invalid observations are removed; points with fewer than two surviving views, insufficient angle or extreme coordinates are retired. Each final registered camera must retain at least the configured PnP support count for a success status. Landmark JSON includes per-view errors/depths and triangulation angle. Track observations and accepted landmark visibility are separate: a tracked association need not survive 3D validation.

## Local bundle adjustment

SciPy `least_squares`, trust-region reflective method, sparse finite-difference Jacobian, `soft_l1` loss with 1px scale. Variables are Rodrigues rotation vectors and translations of recent non-anchor cameras plus selected 3D point coordinates. Residuals are distorted processed-pixel reprojection differences; intrinsics and distortion remain fixed.

Default bounds: **five cameras total**, including both fixed anchors; **400 points** visible in both anchors and at least one recent camera; **30 function evaluations**. User bounds are constrained to 3–10 cameras, 20–2000 points and 1–100 evaluations. Each residual depends only on one camera block and one 3D point block. Holding both complete anchor poses fixes the world frame and scale and anchors every selected point; it also preserves initial-pair rotation/translation bias. This is deliberately more restrictive than a global minimal-gauge BA.

Stopping tolerances are ftol=xtol=gtol=1e-6. A budget-exhausted finite solution may be accepted only if all quality checks pass, with convergence reported separately. Updates are computed in separate candidate state and committed only when local robust cost and **global** RMS error do not increase, invalid observation/low-parallax/extreme counts do not increase, and camera movement remains <=1 initial-baseline unit. Check all current landmarks, including those outside the optimization subset, so optimizing a small local subset cannot silently corrupt other observations. Exceptions, non-finite/failed solves or divergent proposals roll back without changing live poses, points or associations.

The median is reported but is not minimized by least squares: balancing errors across views may increase it while reducing RMS and robust cost. No 2px or parallax threshold was loosened to accept BA. In the accepted real update, all retained observations remained valid. Three later converged proposals were rolled back because they introduced invalid observations/low-angle landmarks or worsened global RMS. A solver convergence flag is not sufficient evidence of a better reconstruction.

References: [OpenCV pose estimation](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html), [SciPy least-squares API](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.least_squares.html).

## Real sequence measurements

Default SIFT target 6000 (tied responses may exceed it), maximum dimension 1600, window 2, long gap 4, seed 0, one OpenCV thread. `configs/sequence_quality.json` retains the previous two-view geometric defaults. All six input images registered; no real sequence image was skipped/rejected. Registration order was **1, 3, 2, 4, 5, 0**, demonstrating order-independent PnP registration while retaining source order.

| Measurement | Local BA enabled | BA disabled control |
|---|---:|---:|
| Input / registered / rejected images | **6 / 6 / 0** | **6 / 6 / 0** |
| Features per image, ordered IDs 0–5 | 6000, 6000, 6001, 6000, 6001, 5493 | same |
| Verified feature tracks | 3923 | 3923 |
| Track observations | 11208 | 11208 |
| Track lengths 2 / 3 / 4 / 5 / 6 | 2095 / 866 / 523 / 306 / 133 | same |
| Conflicting edges rejected / duplicate-cycle edges | 9 / 1795 | same |
| Valid colored landmarks | **3913** | **3910** |
| Retained 3D observations | 10478 | 10464 |
| Final median reprojection error | **0.284303 px** | **0.270556 px** |
| Final RMS reprojection error | **0.585669 px** | **0.594777 px** |
| Median maximum multi-view triangulation angle | 10.073405 degrees | 10.087029 degrees |
| Final invalid observations / low-angle / extreme points | 0 / 0 / 0 | 0 / 0 / 0 |
| Accepted / rolled-back BA proposals | 1 / 3 | disabled |
| Runtime | **17.341445 s** | **15.657639 s** |
| Peak working set | **586383360 bytes (~559 MiB)** | **585883648 bytes (~559 MiB)** |

The BA-enabled and disabled final sets differ, so their final RMS values are not a paired same-observation optimization comparison. The accepted local step is the direct comparison on the same **5,872 observations**: global RMS **0.445444 → 0.423286px**, median **0.212737 → 0.238919px**, optimized-subset robust cost **150.573027 → 113.511537**, four evaluations with convergence. Final-stage BA was rolled back, leaving its pre/post state unchanged. Complete per-stage metrics are exported.

Independent final-export inspection reloaded all 3,913 colored finite points, confirmed coordinate equality with landmark metadata and unique landmark/track IDs, and measured maximum retained reprojection error **1.995339px**, minimum retained depth **2.849385** baseline units, and minimum multi-view triangulation angle **3.144196 degrees**.

PnP registration counts:

| Added image ID | 2D–3D candidates | Accepted inliers | Median inlier error |
|---|---:|---:|---:|
| 2 | 659 | 628 | 0.767067 px |
| 4 | 999 | 882 | 0.857967 px |
| 5 | 888 | 599 | 0.905240 px |
| 0 | 606 | 347 | 0.854378 px |

Measurements are single CPU runs on this development environment, not averages or minimum-laptop certification. Runtime includes loading, extraction, pair matching, candidate evaluation, registration, BA, exports/reload and overview rendering, excluding interpreter startup and the final metrics write. Memory is Windows process-lifetime peak working set, including imports. No CUDA processing is required. No ground-truth pose or absolute-scale accuracy was measured.

## Reproduction, exports and diagnostics

```powershell
# Optional writable cache when running inside a restricted environment:
$env:MPLCONFIGDIR="$PWD\outputs\matplotlib-cache"
# Optional small download, followed by pair-mode validation and sequence reconstruction:
.venv\Scripts\python.exe scripts/sceaux_experiment.py --download --local-ba --output outputs/sceaux-reproduce
# Direct sequence commands used for final delivery measurements:
.venv\Scripts\vision3d.exe reconstruct-sequence --input datasets/sceaux/images --camera configs/sceaux.json --quality-config configs/sequence_quality.json --local-ba --output outputs/sceaux-delivery
.venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence --input datasets/sceaux/images --camera configs/sceaux.json --quality-config configs/sequence_quality.json --output outputs/sceaux-delivery-no-ba
```

Use new output paths when reproducing; existing directories are rejected. The downloader uses a pinned revision, bounded reads and SHA-256 checks before saving; existing samples are also checked. Reference poses are neither downloaded nor used. Actual pair artifacts are in `outputs/sceaux-final/calibrated-pair/` and `outputs/sceaux-final/approximate-pair/`. The final sequence is **`outputs/sceaux-delivery/`**, with the independent no-BA control at `outputs/sceaux-delivery-no-ba/`.

Successful sequence exports:

- `sparse.ply`: actual finite triangulated landmarks, with original-image RGB colors; Open3D export/reload checks point count, coordinates and color quantization.
- `poses.json`: world-to-camera matrices, centers, registration status, timestamps and PnP attempts; rejected images have null pose.
- `intrinsics.json`: original/processed calibration, approximation flags, sources and assumptions.
- `tracks.json`: stable track IDs, all observation feature IDs and optional landmark link.
- `landmarks.json`: point/track IDs, coordinates, colors, accepted visibility, depths, reprojection errors and angle.
- `features.npz`: processed keypoint coordinates, row-indexed by stable feature ID. Native descriptors remain in state while running; recomputation requires the original inputs/configuration.
- `pairs.json`, `initial_candidates.json`: matching stages, verification metrics, initial-pair quality/selection.
- `manifest.json`: source hashes/order/timestamps/EXIF/resize, configuration, coordinate convention and software versions.
- `metrics.json`: per-image and overall counts, BA history, error distributions summarized by median/RMS, timing/memory.
- `overview.png`: colored sparse scene, actual estimated camera centers with IDs, track-length histogram. Visually inspected; facades/depth structure and six separate camera centers are present, with no synthetic fill.

The [local overview](../outputs/sceaux-delivery/overview.png) and PLY are generated/ignored artifacts, not committed assets. Exit codes: 0 success, 3 rejected/partial reconstruction and 2 invalid input/configuration/processing errors. The default success target is five cameras; it can be explicitly changed by the user but was **not lowered** in this experiment. Success also requires enough surviving landmarks and per-camera final support. Failed runs may retain partial diagnostic files.

## Automated and regression verification

Command: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=outputs/pytest-incremental-final`.

Result: **58 passed, 0 failed, 0 skipped in 5.02 seconds**. Fourteen third-party Matplotlib/Pyparsing deprecation warnings appeared on first plotting imports; they did not affect rendering or geometry. All 45 earlier tests remain present. New coverage includes correct/conflicting/cyclic/disconnected track construction, geometry-based initial selection over a higher-count zero-baseline pair, PnP with outliers/failure and arbitrary order, deferred image registration after new landmark triangulation, deduplication, multi-view visibility/reprojection pruning, proper pose conventions, BA convergence and fixed gauge, exception/non-finite rollback, quality validation, per-image calibration and full sequence CLI/export/reload. The synthetic CLI test registers five textured views while leaving an additional blank image unregistered with a null pose. Synthetic checks are separate from the real six-photo acceptance result.

Both module and installed executable CLI entry points were exercised. The original `analyze` processed a two-image real input successfully; the two-view commands were used for the unrectified validations and no-motion control. Python 3.12.14 and the previous pinned NumPy/OpenCV/SciPy/Open3D environment remain unchanged. Datasets, `.venv`, runtime/cache directories and outputs remain ignored; no large binaries are committed. Git operations are local only.

## Limitations and next steps

Published K is used as supplied; missing lens distortion and calibration uncertainty may bias poses/structure despite low reprojection errors. This is one six-photo static architectural scene with substantial texture and overlap, not broad camera validation or proof of metric accuracy. There are a few isolated sparse points in the overview; geometric gates reduce outliers but cannot prove semantic point correctness. Approximate calibration was tested separately on a pair and is not claimed equivalent to measured intrinsics.

Track merging is greedy and can reject useful edges or retain an epipolar-compatible false association. Initial-pair selection does not perform a global image-graph optimization. A track whose widest pair is inconsistent is left untriangulated rather than exhaustively trying every pair. Failed registration can require better overlap or a different explicit matching schedule; it never receives an invented pose. The local BA window only refines a capped anchor-visible subset, fixes initial-pair bias and can be conservatively rejected when it would harm landmarks outside the subset. No unbounded/global BA, loop closure, intrinsic refinement, session resume or arbitrary video-sequence reconstruction was introduced.

Sequence inputs default to at most 30 images and can be configured up to 50; features cap at 20,000 requested per view. BA limits prevent unrestricted optimization, but feature retention and pair matching still grow with sequence size. The i5-11400H/GTX1650 target machine has not been independently tested. Source timestamps may lack timezone; raw sensor-coordinate/fisheye calibration limitations from earlier work remain. Open3D's pre-existing web-library transitive dependencies are unchanged and no web service is used.

Recommended next stage: validate additional static unrectified scenes with measured distortion/ground-truth poses, strengthen track consistency and subpixel observations, expand local optimization with well-defined fixed boundary cameras, and benchmark registration/drift under bounded resources. Keep sparse validation separate from later dense reconstruction and surface generation.
