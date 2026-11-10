# Reconstruction engineering report

## Delivered workflows

The desktop accepts a folder or a file selection of 2–30 JPEG/PNG photos. It starts two-view reconstruction for two readable images and incremental SfM for three or more. An explicit shared, filename-keyed or pair calibration file remains supported. When omitted, the engine initializes each image independently from usable EXIF or a documented fallback. The desktop exposes processed resolution, output location, cancellation, sparse viewing, camera overlays and a colored PLY save action. Calibrated two-view dense stereo and its optional vertex-colored mesh are experimental CLI stages. Their results can be opened in the desktop.

Initial state and fresh baseline are recorded in [the baseline report](COMPLETION_BASELINE.md). Existing edits to university deliverables and untracked depth captures were outside this engineering work. No dependency versions changed.

## Implementation

- Pillow reads orientation, capture time and focal metadata when loading the image. EXIF equivalent focal length is converted using the oriented image diagonal. Physical focal length can also be combined with focal-plane sampling if units and pixel-grid dimensions agree, with axes swapped for rotated EXIF orientations. Suspect or contradictory values use a diagonal focal prior. Results record provenance, confidence and assumptions. The same existing pixel-center resize transform applies to explicit and inferred models.
- Two-view and sequence entry points share input and intrinsics helpers. Pair mode retains its own calibrated essential/triangulation pipeline and pair-specific JSON layout. Shared result counts let the desktop present both modes. The viewer reads both pose schemas. The desktop validates readable files before selecting a mode, passes the resulting exact list to the worker, and retains skipped-file diagnostics.
- Existing sparse geometry, registration and bundle-adjustment gates stay in place. Automatic mode does not optimize focal length or distortion. Historical experimental graph recovery and joint BA stay disabled by default because higher registration coverage did not establish better reference-pose accuracy.
- Dense pair mode verifies original image hashes and rectifies two explicitly calibrated images using the sparse estimated poses. OpenCV SGBM computes disparity in both directions, with a 256-disparity upper bound and a search interval estimated from sparse depth. Valid remap pixels must agree within one disparity pixel. Reprojected depth is bounded by sparse support and transformed into the sparse world frame. PLY point colors come from the rectified original image.
- The optional surface connects only adjacent valid disparity samples and drops triangles at relative depth jumps over 2%. Open3D removes degenerate triangles and unused vertices and estimates normals. The PLY is checked after reload for topology, coordinates, normals and vertex colors. This creates an open visible surface with gaps. UV texture mapping, watertight completion and multi-view fusion remain future work.

## Measured regressions and additional cases

Fresh Windows CPU runs used the checkout's existing Python 3.12 environment. The calibrated sequence comparison used identical inputs, seed, thresholds, resolution and local BA, with fresh feature caches. Runtime is one engine measurement per run and does not establish a speed trend.

| Calibrated case | Registered before → after | Points before → after | Median / RMS px before → after | Engine s before → after | Peak MiB before → after |
|---|---|---|---|---|---|
| Castle 6 | 6/6 → 6/6 | 3915 → 3915 | 0.285393 / 0.586640 → same | 16.03 → 16.54 | 563.6 → 566.5 |
| Temple 30 | 18/30 → 18/30 | 2664 → 2664 | 0.260683 / 0.653783 → same | 9.18 → 9.08 | 199.5 → 198.7 |
| Motorcycle pair | 2/2 → 2/2 | 859 → 859 | 0.086653 / — → 0.086653 / 0.180480 | 1.06 → 1.02 | 138.0 → 135.2 |

The two calibrated sequences had identical estimated pose matrices, landmark JSON and PLY bytes. Temple's reference rotation and translation-direction errors remained 6.613361° and 2.844933°. The calibrated Motorcycle reference values at baseline were 0.249754° and 0.476587°. Castle has no reference poses in the local dataset. Reprojection fit measures image consistency. It does not establish absolute 3D accuracy.

| New workflow | Result | Median / RMS px | Engine s |
|---|---|---|---:|
| Castle EXIF, selected two-photo pair | 2/2, 879 points | 0.187114 / 0.288991 | 2.47 |
| Castle EXIF, all six photos with desktop refinement | 6/6, 3911 points | 0.313175 / 0.483944 | 15.86 |
| Motorcycle PNG fallback, two-photo pair | 2/2, 576 points | 0.069090 / 0.173615 | 0.98 |
| Calibrated Motorcycle dense stereo plus mesh | 268359 dense points, 268122 mesh vertices, 520582 triangles | — | 1.09 |

The Motorcycle fallback pair's reference direction errors were 0.216459° rotation and 0.404323° translation direction. It retained fewer points than the calibrated run, so residual summaries compare different accepted point sets. Dense surface accuracy was not measured against a reference. A calibrated Castle pair (876 sparse points) was rejected by the bounded dense disparity search. Sparse export remained intact.

Raw artifacts are ignored under `outputs/completion-before`, `outputs/completion-after`, `outputs/completion-smoke` and `outputs/completion-mesh-motorcycle`. The last folder contains `mesh-preview.png`, captured from the actual Open3D viewer. The preview shows one visible side and holes in the surface.

## Verification and reproduction

The existing 122-test baseline passed. The final suite passed 132 tests with 14 existing dependency deprecation warnings. It covers EXIF, orientation, resize, invalid metadata, calibration precedence and schemas, desktop dispatch, cancellation, point-cloud reload, known synthetic disparity and mesh discontinuities. On the local desktop, a Tk event-loop harness completed automatic Castle pair reconstruction. Another harness opened both actual Motorcycle dense and mesh PLYs in Open3D and toggled camera overlays. This was programmatic desktop validation without manual mouse gestures.

```powershell
Set-Location 'C:\projects\Vision-Based-3D-Reconstruction'
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m vision3d.demo
.venv\Scripts\python.exe scripts/benchmark_sfm.py --output outputs/completion-check --case castle6 --case temple30
.venv\Scripts\python.exe scripts/motorcycle_experiment.py --output outputs/motorcycle-sparse-check
.venv\Scripts\python.exe -m vision3d.cli dense-pair --pair-dir outputs/motorcycle-sparse-check --output outputs/motorcycle-dense-check --mesh
```

For Castle choose **Load Castle Demo**, **Start Reconstruction**, then **Open 3D View**. For experimental stereo select the dense output directory in **Open Previous Result**. Use **Open 3D View** for the cloud or **Open Mesh** for the colored surface. All three exports are PLY. Pose/intrinsics JSON is kept beside geometry. Output coordinates use arbitrary baseline units.

## Open work

EXIF initialization is a prior. Cropped, digitally zoomed, wide-angle, moving, or low-overlap scenes can fail or produce biased geometry. Mixed resolutions and orientation are unit tested, but broad validation across camera models and arbitrary user photographs remains open. Temple still misses 12 cameras. Match scheduling and local optimization were left unchanged after comparison. Castle matching was the largest measured stage, so a reproducible cache and overlap study is a better next performance step than adding parallelism without profiling.

Dense stereo currently supports calibrated, similarly sized pairs with usable horizontal rectification and limited disparity range. Large view changes and vertical stereo are unsupported. It cannot recover occluded or untextured regions reliably. The vertex-colored mesh has no UV texture or unseen surfaces. Next steps are reference-based validation of intrinsics and pose recovery, depth consistency across multiple registered views, and a validated multi-view fusion and texturing pipeline before offering generalized dense reconstruction in the desktop.
