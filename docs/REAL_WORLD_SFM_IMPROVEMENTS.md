# Real-world SfM improvements

The Darth Vader investigation recovered five cameras from the original 30 instead of two, with 645 sparse landmarks. The result remains partial. The dense stage produced reloadable geometry from the recovered views, but visual inspection shows an incomplete surface. The [failure analysis](DARTH_VADER_FAILURE_ANALYSIS.md) and [controlled experiment matrix](DARTH_VADER_EXPERIMENTS.md) document the evidence.

## Changes

- Sequence reconstruction supports bounded exhaustive scheduling for up to 30 images. It tests all remaining pairs after the ordinary local window, records which schedule produced each attempt, and keeps before/after graph components. The existing local schedule and bounded appearance-ranked discovery remain available.
- Reciprocal descriptor matches can be cached by ordered descriptor contents, configuration, ratio, OpenCV version and matcher implementation. Cached descriptors do not provide geometric edges. Each run recomputes geometry with current intrinsics and verification settings. Pair diagnostics include the virtual-pixel fundamental matrix, inlier count and ratio, spatial hull coverage and edge outcome. Exhaustive and discovery runs save the actual verified feature indices.
- MAGSAC is an optional fundamental estimator. Seed probing evaluates up to 12 geometrically accepted candidate seeds and up to five next-view candidates per seed. It uses temporary states and the existing PnP acceptance checks. A third-camera support count alone did not help on Vader. Actual validated PnP probes avoided a seed whose prospective camera support was too concentrated.
- The desktop offers **Enhanced matching (experimental)**. It selects exhaustive scheduling, MAGSAC, ratio 0.80 and seed probing, with a reusable descriptor cache under the chosen output root. It leaves the default stable. The log summarizes verified pairs and graph components. The optional diagnostics script exports a heatmap, feature overlays, verified pair previews and per-image registration CSV. PnP diagnostics distinguish entry into the solver, raw inliers, candidate coverage and post-refinement quality.
- The dense CLI accepts `--object-masks` with normalized polygons keyed by registered filenames. The masks constrain accepted pixels in both rectified images and leave sparse camera poses untouched. They provide foreground selection for the dense stage only.

## Regression and launch

The full suite passed **140 tests** with 14 existing dependency deprecation warnings. Tests cover exhaustive scheduling, cache reuse, seed probing without modifying the live map and polygon validation. A Tk harness ran all 30 original photos with enhanced matching, found five registered cameras, generated Fast dense PLY and GLB, and opened the mesh in Open3D.

| Stable calibrated case | Cameras | Sparse points | Median / RMS px | Dense check |
|---|---:|---:|---|---|
| Castle | 6/6 | 3915 | 0.285393 / 0.586640 | 554068 points, 1194432 faces, PLY and GLB |
| Temple30 | 18/30 | 2664 | 0.260683 / 0.653783 | 513500 points, 1055314 faces, PLY and GLB |
| Motorcycle | 2/2 | 859 | 0.086653 / 0.180480 | 268359 points, 520582 faces, PLY |

Castle and Temple sparse PLY and landmark JSON were byte-identical to the prior calibrated benchmark. Temple's median aligned rotation and translation-direction errors remained 6.613361° and 2.844933°. Motorcycle reproduced the reference direction errors of 0.249754° rotation and 0.476587° translation. These results apply to the default settings. The new profile was not promoted as a calibrated default.

```powershell
& 'C:\projects\Vision-Based-3D-Reconstruction\run_demo.bat'
```

Choose the original Vader folder, enable **Enhanced matching (experimental)**, then choose **fast** for multi-view dense processing and optionally request mesh and GLB. Inspect **Open Output Folder** for the separate sparse and dense results. The first exhaustive pass is slower than an ordinary local run. Reproduce individual runs and masks using the experiment commands.

## Next steps

Capture another object sequence with smaller angular steps, fixed focal setting, even diffuse illumination and more overlap between the high and low orbits. Use larger coverage of distinctive features or a fixed textured surround, while preserving independent source masks for dense processing. To improve on this dataset without weakening geometry, investigate repeatable multi-view correspondences on the glossy helmet and better cross-orbit capture coverage. Validate intrinsics against independent calibration and require comparable reference-pose or reference-surface measurements before claiming better accuracy.

## Map-aware recovery on the Vader video

Sequence reconstruction can try untested pairs against registered views when ordinary pose registration stalls. Enable it with `--map-recovery-pairs 30 --map-recovery-rounds 3`. The default pair budget of zero disables it. The original `--recovery-pairs` still controls discovery before initialization. Map recovery preserves live landmark IDs, records aliases for retired unmapped tracks, and uses the existing pair verifier, PnP thresholds and triangulation gates. The desktop keeps its existing settings.

The controlled comparison used the same 40 overlap-aware keyframes, camera JSON, SfM settings and initial pair `8–9`. Use a fresh output directory for each command.

```powershell
$frames = 'outputs/video-overlap-ab/overlap-aware/keyframes'
$common = @('--input', "$frames/frames", '--camera', "$frames/camera.json", '--max-images', '50', '--window', '4', '--long-gap', '8', '--recovery-pairs', '30', '--verifier', 'magsac', '--probe-seeds', '--ratio', '0.80', '--local-ba', '--refine-points', '--seed', '0')
.venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence @common --output outputs/map-recovery-repeat/baseline
.venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence @common --output outputs/map-recovery-repeat/recovery --map-recovery-pairs 30 --map-recovery-rounds 3
```

The measured runs in `outputs/map-recovery-ab/` registered 5 and 8 cameras, with 388 and 570 landmarks respectively. Recovery verified 12 of 30 additional pairs across three rounds and left six graph components. Median/RMS reprojection error rose from 0.281/0.503 px to 0.325/0.571 px across the respective maps. Recovery itself took 3.08 s. The full runs took 42.20 and 44.44 s. See `metrics.json` for per-round support changes, `pairs.json` for pair outcomes and `poses.json` for PnP outcomes. Ground-truth poses are unavailable.
