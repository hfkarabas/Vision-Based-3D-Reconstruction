# Sparse reconstruction quality investigation

Date: November 4, 2026. Starting commit: `aa3be2d9bcd1279587a32094cf4c34ece6465636`.

## Outcome

Implemented bounded fixed-camera refinement of existing multi-view landmarks and camera-aware viewer framing/context. Castle retains **6/6 cameras and 3,915 landmarks**, while RMS reprojection error decreases **0.586640 → 0.476889 px** on the same 10,488 observations. Median error increases **0.285393 → 0.306497 px**. Temple confirms the same RMS/median tradeoff with **identical camera poses and reference pose errors**. No camera-accuracy improvement is claimed.

The result is more defensible multi-view point consistency and an easier-to-read initial view, not a substantially more complete building. The same-view before/after plots show only modest coordinate changes. Missing surfaces and architectural detail remain missing. No interpolation, synthetic scene points, dense reconstruction, mesh or texture was added.

## Reproducible baseline

Inspected the existing CLI, incremental triangulation, track store, BA acceptance/rollback, feature extraction/cache, desktop subprocess integration and Open3D viewer. Existing deletions of the bilingual Word documents were present before this work; they remain untouched and are excluded from the commit.

The original stable engine was run before source changes, into a new directory:

```powershell
.venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence --input datasets/sceaux/images --camera configs/sceaux.json --local-ba --output outputs/reconstruction-quality/castle_baseline
```

Baseline: 6 registered cameras, 3,915 landmarks, 10,488 accepted observations, 0.285393 px median / 0.586640 px RMS reprojection error, 10.066965° median maximum triangulation angle, 16.743 s engine runtime and 558.71 MiB lifetime peak engine working set. No final observation fails the existing positive-depth/error gates. Baseline PLY, poses, features and metrics remain intact under `outputs/reconstruction-quality/castle_baseline/`.

All experiments use the existing local photographs and published calibration. Castle has six unrectified 2832×2128 images, processed to a 1600-pixel maximum dimension unless explicitly varied. K is published; absent distortion coefficients are treated as zero, with the existing calibration assumption preserved. Temple uses the existing 30-photo Middlebury subset and published camera intrinsics. Reference R/t is used only for post-run evaluation. No new data were downloaded or redistributed; provenance/licensing remains as documented in the earlier reports.

Runs use SIFT, ratio 0.75, reciprocal consistency, seed 0, one OpenCV thread, the existing neighboring-pair schedule and local BA unless explicitly varied. Feature caches are disabled throughout these quality experiments. Each named run has a separate output directory and execution JSON with the command/source fingerprint. Timings are individual sequential observations, not statistical speed comparisons. This is a development-scene investigation, not a fresh held-out study.

## Geometric diagnosis

### Cameras and parallax

The world-to-camera convention remains `Xc = R Xw + t`, camera center `C = -R.T t`; no pose-format/sign defect was demonstrated. The Castle seed pair is IDs **1/3**, with unit baseline. Estimated camera-center separations range from **0.379 to 2.988 arbitrary units**. Forward-direction differences range from **4.668° to 29.866°**, consistent with a lateral sequence observing the same facade rather than coverage around the entire building.

Maximum accepted ray angles per landmark range **3.144°–36.098°**, with quartiles **6.084°, 10.067°, 16.016°**. Thus the output is not uniformly a near-zero-parallax reconstruction; increasing the minimum angle indiscriminately is not justified. Frontal coverage still limits visible surfaces and depth constraints. Without reference Castle poses or a reference surface, estimated depth distortion cannot be separated conclusively from calibration, matching and camera-pose bias.

Camera-direction and baseline values are exported in each `quality/summary.json`; `cameras_and_cloud.png` shows actual centers and unit-length forward arrows, not an illustrative trajectory.

### Landmark support and depth

| Accepted observations per landmark | Baseline landmarks |
|---:|---:|
| 2 | 2,398 |
| 3 | 787 |
| 4 | 399 |
| 5 | 251 |
| 6 | 80 |

**61.25%** of landmarks have only two accepted observations. Existing triangulation seeds a point from a selected two-view hypothesis; later observations can be attached without an independent all-observation point-only optimization. This provides a concrete opportunity to improve consistency for the remaining 1,517 multi-view points without moving cameras or merging tracks.

Observed camera-space depths span **2.767–7.301 units**, median **5.145** and 5th–95th percentiles **3.925–6.059**. There are no nonpositive accepted depths. A diagnostic radial-tail test, `distance from median > Q3 + 3×IQR`, flags **zero** Castle baseline points. Therefore visually scattered points are not evidence of a large extreme-distance population; automatic distance pruning has no measured basis here. Spatial singular values are **78.135, 39.918, 19.357**; they describe dispersion in an arbitrary reconstruction frame, not surface accuracy.

Tracks contain 3,923 components and 11,208 track observations; exported landmark observations are a validated subset. Nine conflicting associations were rejected. One landmark per track and one feature observation per image remain enforced; no duplicate-landmark or descriptor-only merge change was justified. These checks do not exclude all epipolar-compatible repeated-window mismatches.

### Features and matching

The baseline verified graph is connected, with 11 accepted scheduled pair edges. Recomputed seed-pair diagnostics contain **952 fundamental-matrix-verified correspondences**, with 100 displayed. Actual feature/landmark overlays show strong concentration at windows, roof edges and ornaments, alongside features on vegetation and ground. Broad plain walls and unseen building surfaces have little or no reliable support.

The six images contain **6,000 / 6,000 / 6,001 / 6,000 / 6,001 / 5,493** SIFT keypoints. Slightly exceeding the target is existing OpenCV behavior. Occupied 8×8 feature cells are **57 / 50 / 51 / 48 / 49 / 50**, but grid occupancy includes background; it is not a facade-completeness score. Most views already hit the feature target. Contrast/edge thresholds and mutual consistency were left unchanged rather than introducing unverified extra detector controls.

### Bundle adjustment

Baseline local BA accepts the first proposal and rolls back three later proposals, which would introduce **35 / 58 / 127** invalid observations. Acceptance gates are doing useful work. Overriding rollback to obtain a smoother-looking cloud would sacrifice valid geometry. The existing two-anchor gauge, fixed intrinsics, bounds and atomic rollback remain unchanged. Expanded-window BA was explored separately below, not promoted as a new default.

## Implemented geometric improvement

`reconstruction/refinement.py` adds a final fixed-camera multi-view point pass. For each landmark with at least three accepted observations, SciPy robust least squares minimizes the existing distorted processed-pixel residuals with `soft_l1` loss and at most **15 evaluations per point**. Camera poses/intrinsics, landmark IDs, colors and observation associations are fixed. There is no new gauge freedom because no camera variable is optimized.

Accept a proposal only if it:

- Has finite parameters/residuals and a nonfailed solver state.
- Reduces that point's summed squared image residuals and does not increase its robust cost.
- Keeps every existing observation at positive depth and at or below the unchanged **2 px** error limit.
- Preserves the existing minimum **1°** ray angle and **1,000-unit** distance limit.
- Moves by at most **5% of the original median camera-to-point distance**.

Updates are staged until the pass finishes; failed or rejected point proposals remain unchanged. Two-view points are deliberately unchanged. Existing final validation/export still runs afterwards. Metrics include accepted/unchanged reasons, eligible-observation RMS before/after, displacement, time and, in the final integration, full-map before/after quality.

Castle accepts **1,499 / 1,517** multi-view proposals and leaves 18 gate-rejected points plus all 2,398 two-view points unchanged. Eligible-observation RMS decreases **0.699818 → 0.524091 px**. Across all 3,915 points, the 95th-percentile displacement is **0.014058 units**, maximum **0.043607 units**. Landmark IDs and observation associations are identical; maximum camera-matrix difference is **exactly zero**. This is a limited point-estimation improvement conditional on the existing poses, not proof of more accurate absolute depth.

The CLI retains the original default (`refine_points=False`); enable with `--refine-points`. After synthetic tests and Castle/Temple validation, the desktop explicitly requests this pass. Experimental graph discovery and joint BA remain off.

## Controlled ablations

All reprojection summaries below use each run's accepted observations. Only baseline versus its fixed-point refinement counterpart has identical observation/landmark populations; other ablations must not be interpreted as same-population accuracy comparisons.

| Castle experiment | Cameras | Points | Median px | RMS px | Runtime s | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| Stable baseline | 6/6 | 3,915 | 0.2854 | 0.5866 | 16.74 | 558.71 |
| SIFT target 8,000 | 5/6 | 4,068 | 0.3048 | 0.6864 | 20.72 | 562.78 |
| SIFT target 12,000 | 5/6 | 4,094 | 0.3146 | 0.7183 | 22.60 | 563.66 |
| Maximum dimension 2,000 | 6/6 | 3,282 | 0.3241 | 0.6062 | 18.52 | 805.89 |
| Matching ratio 0.70 | 6/6 | 3,664 | 0.2928 | 0.6044 | 17.56 | 558.79 |
| BA window 6 / 1,200 points | 6/6 | 3,913 | 0.3078 | 0.5778 | 18.25 | 558.68 |
| Fixed-camera point refinement | 6/6 | 3,915 | 0.3065 | 0.4769 | 20.75 | 558.86 |
| 12,000 features + point refinement | 5/6 | 4,094 | 0.3260 | 0.6294 | 25.57 | 563.66 |

The final repeat (`castle_delivery`) also produces 6/6 cameras, 3,915 points and identical refined geometric results in **19.78 s**, at **558.70 MiB**. Runtime varies; the first ablation is retained for the before/after table. Memory is engine lifetime peak working set, not aggregate desktop/viewer memory.

Rejected or deferred options:

- **Larger feature targets:** both lose image ID 5 (`100_7105.JPG`), choose different seed pairs (1/2 and 0/1) and worsen image fit. More landmarks is not sufficient justification. Combining 12,000 features with refinement does not recover the camera.
- **Higher resolution:** fewer landmarks and higher fit errors; processed-pixel units also differ, so no absolute-accuracy ranking is inferred. Memory rises substantially.
- **Stricter ratio:** fewer landmarks and worse median/RMS fit. No reliability improvement demonstrated.
- **Expanded BA:** small RMS gain, higher median error, changed point population, no Castle reference accuracy. The two optimizer limits changed together; their individual effects are not attributed. Defaults and rollback are preserved.
- **Quality subset:** >=3 observations, angle >=3° and all errors <=1 px retains only **757/3,915** baseline points or **1,273/3,915** refined points. Inspection shows considerable architectural detail is lost. Subsets remain explicitly labeled diagnostic plots; PLYs and default displays are not pruned. The larger refined subset is reclassification of existing points, not newly recovered geometry.
- **Extreme-depth removal:** no baseline radial-tail population was found by the stated diagnostic. No arbitrary clipping or plane flattening was introduced.

## Independent Temple validation

| Temple 30 experiment | Cameras | Points | Median px | RMS px | Angle median | Runtime s | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stable baseline | 18/30 | 2,664 | 0.2607 | 0.6538 | 12.9916° | 10.82 | 192.50 |
| Fixed-camera refinement | 18/30 | 2,664 | 0.2743 | 0.4477 | 12.9901° | 14.13 | 193.09 |
| SIFT target 12,000 | 18/30 | 2,664 | 0.2607 | 0.6538 | 12.9916° | 11.12 | 192.75 |

Refinement accepts 1,453 points. Observation support histograms, landmark IDs, rejected image IDs and all estimated camera matrices are unchanged. All final observations remain valid. Reference evaluation is identical before/after: **6.613361° median aligned rotation error** and **2.844933° median relative translation-direction error**. Reference poses never enter reconstruction. Temple's measured registration gaps and pose bias are not solved by this point pass. More features has no effect here because existing detection already falls below the target.

This independent scene supports the fixed-pose invariance and aggregate fit result; it does not provide sparse surface ground truth or demonstrate generalization to arbitrary scenes/hardware.

## Visualization and desktop integration

The viewer now derives its front/up directions from actual world-to-camera poses, looks toward the cloud's median position and uses a four-pixel point size. **R** resets this camera-aligned view; **C** toggles estimated camera frustum outlines, and **A** toggles a coordinate frame. Overlays are off initially. Their bounds are included when shown so near-camera overlays do not clip into giant foreground shapes. Frustums illustrate calibrated pinhole image boundaries at a display scale; they are context geometry, not reconstructed surfaces or additional landmarks. Distortion is not represented by curved frustum edges.

All actual exported points and colors are retained. The viewer never rewrites PLY geometry or silently filters counts. Standalone colored PLYs remain viewable when camera metadata is absent, with ordinary default framing. The file-selection, worker subprocess, unique output, count reporting and previous-result workflow are preserved.

The updated Tk workflow was exercised with a **withdrawn window** through its real callbacks and engine child process: **207 heartbeat ticks**, successful 6/6 registration and 3,915 landmarks, **21.010 s engine / 22.660 s harness wall time**, controls re-enabled and a genuine 3,915-point colored PLY reloaded. This is an event-loop/orchestration check, not a claim of new manual interaction with every GUI control. Output:

```text
outputs/reconstruction-quality/desktop_validation/run_20261104T143433114210Z_e70b9454/reconstruction/
```

A real updated Open3D window opened the final Castle PLY, and initial camera-aligned display plus camera/axis toggle behavior were inspected interactively. Final adjusted overlay framing/reset was then checked through Open3D's own **hidden-window rendering API**, executing registered callbacks, saving actual renderer images and verifying the PLY hash was unchanged. That check returned exit 0 and closed status with 3,915 points. Final callback captures were visually inspected. These native API checks are not claimed as manual mouse/keyboard tests. Panning was not newly tested; no new custom-scene or hardware validation is claimed.

Launch remains:

```powershell
.venv\Scripts\python.exe -m vision3d.demo
```

Use **Load Castle Demo → Start Reconstruction → Open 3D View**. Expect roughly 20–23 seconds on the tested desktop with refinement. The previous-result backup remains explicitly labeled. The original benchmark's 15–16 s measurements remain valid historical measurements for its unrefined configuration.

## Genuine evidence and reproduction

All new artifacts are under `outputs/reconstruction-quality/`; older benchmarks and presentation figures are unchanged.

| Artifact | Contents |
|---|---|
| `comparison.json`, `comparison.csv` | All ablation counts, fit, angle/depth/support statistics, runtime, memory and available reference errors |
| `<case>_execution.json`, `<case>.log` | Exact command, source fingerprint, seed/cache policy and process output |
| `<case>/quality/cloud_and_subset.png` | Genuine cloud versus explicitly labeled diagnostic subset, identical limits |
| `<case>/quality/cameras_and_cloud.png` | Actual points, estimated centers and viewing directions |
| `<case>/quality/quality_distributions.png` | Depth, angle, residual and support distributions |
| `<case>/quality/feature_coverage.png` / `.json` | Source-image detected/accepted features and grid coverage |
| `<case>/quality/reprojection.png` | Measured observations with true pixel-scale residual arrows |
| `castle_matching.jpg`, `castle_matching.json` | Genuine recomputed seed-pair matches and displayed feature IDs |
| `castle_before_after.png`, `refinement_comparison.json` | Same-view all-point geometry comparison and exact identity/pose/displacement checks |
| `viewer_initial.png`, `viewer_cameras.png`, `viewer_axes.png`, `viewer_reset.png` | Genuine native Open3D renderer captures, not generated illustrative scenes |
| `viewer_api_validation.json`, `desktop_validation.json` | Rendering callback/file-integrity and real Tk/engine validation records |

Local before/after figure:

![Same-view actual baseline and refined Castle landmarks](../outputs/reconstruction-quality/castle_before_after.png)

To reproduce in a **new** experiment directory:

```powershell
.venv\Scripts\python.exe scripts/reconstruction_quality.py --output outputs/quality-repeat --case castle_baseline --case castle_refined --case temple_baseline --case temple_refined
.venv\Scripts\python.exe scripts/reconstruction_quality.py --output outputs/quality-repeat --summarize
```

Omit `--case` to run all declared ablations. Existing case directories are rejected rather than overwritten. `--summarize` rebuilds only derived diagnostic figures/tables, not engine outputs. The mathematical improvement remains explicitly selectable with `--refine-points` on `reconstruct-sequence`.

## Tests and remaining limitations

**115 tests passed, 0 failed, 14 existing dependency deprecation warnings**, final full-suite time **8.01 s**. All previous tests are retained. New tests cover synthetic 3D noise reduction with distorted observations, fixed poses/associations/colors, two-view preservation, non-finite/failed/extreme proposal rollback, numerical limits, camera convention/frustum reprojection, non-mutating subset logic and explicit CLI flag/default behavior. The desktop command regression additionally verifies the enabled fixed-point pass while experimental pose modes remain off.

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp outputs/reconstruction-quality/tests-final-temp
```

Log: `outputs/reconstruction-quality/tests-final.log`. Tests used normal OS temporary-directory access, as in the preceding desktop delivery.

Remaining technical limits:

- Most Castle landmarks are still two-view points. Refinement cannot create observations, solve unseen surfaces or fill blank walls.
- Fixed-camera optimization cannot repair biased camera poses or uncertain calibration; Castle has no reference surface/pose validation in this investigation.
- Median residual increases slightly even though per-point squared and robust costs improve. No claim that every individual observation improves is made.
- Camera-aware framing and larger point sprites aid interpretation separately from geometry. Before/after coordinate plots reveal that true structural changes are modest.
- Temple remains 18/30 registered, with unchanged pose error. No registration threshold, risky joint BA or graph recovery setting was enabled.
- Dense reconstruction, meshing and texturing remain out of scope. The building is not fully reconstructed, and metric scale is unknown.

Recommended next work: collect better-translated, convergent views with additional oblique coverage and defensible calibration; add independent surface/pose references; investigate track consistency and initialization accuracy before changing camera optimization. Keep the unrefined baseline available and evaluate whether any future change improves reference geometry, not just image fit. Dense reconstruction should follow stronger camera accuracy and coverage, not be used to disguise current sparse limitations.
