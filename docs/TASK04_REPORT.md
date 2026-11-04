# TASK 04 — sparse SfM reliability and benchmarking

Date: 2026-11-04 (Europe/Istanbul). The reliability changes, multi-scene evaluation, 6/15/30-input resource measurements and reference-pose comparisons are implemented. **Complete registration and accurate geometry are not achieved for all moderate sequences. The engine is not ready for dense reconstruction.** No dense reconstruction, mesh, texture, neural network, GUI or CUDA requirement was introduced.

## Audit findings and corrections

Audited clean baseline `965dd677312ca2840895feaea31ad37b57799c16`, all source/tests and earlier reports. Existing camera conventions are consistent: `Xc=R Xw+t`, camera center `-R.T@t`, first selected anchor defines the world, and the initial baseline has arbitrary unit scale. Distorted processed-pixel reprojection uses each fixed camera model. No camera poses or reference structure enter reconstruction.

| Concrete risk / measured weakness | Correction and evidence |
|---|---|
| BA compared counts of invalid observations; repairing one invalid observation could conceal a different newly invalid one | Compare observation identities `(landmark,image,feature)` and invalid landmark ID sets. Targeted equal-count/different-identity regression rejects the proposal |
| BA assigned live poses/landmarks before pruning finished | Prune candidate state before atomic commit. An injected pruning exception leaves original poses, points and links unchanged |
| BA always required visibility in both initial anchors, preventing useful windows farther along a sequence | Select two noncoincident fixed covisible boundary cameras. Initial anchors remain globally fixed. A synthetic window without anchor visibility converges while boundary/anchor poses remain unchanged |
| Non-finite initial parameters could reach validation/residual construction before solver checks | Explicit initial-state, observation, residual and solver-result finite checks; rollback tests cover initial and returned NaNs |
| Initial pair could pass geometry but lack enough conflict-free landmark-track links | Count compatible tracks before ranking; reject unusable candidates and consider alternatives |
| Many tightly clustered PnP inliers can give poorly constrained poses | Default >=1% image-area convex-hull coverage, in addition to existing >=20 inliers, >=40% support, positive depth and <=2px error |
| Corrupt/inconsistent state could associate the same image feature with multiple 3D points | Exclude all ambiguous associations and non-finite points from PnP; export discarded counts |
| Next-image priority remained stale after map growth | Recompute support/spatial coverage priority after each registration; retry a failed image only on a newer map revision |
| A bad widest-ray observation could block an otherwise usable track | Try at most three ordered triangulation hypotheses; each still requires multi-view agreement and existing depth/error/parallax gates |
| Track conflicts and triangulation failures had only aggregate counts | Export capped examples with feature/image IDs and reasons, conflict truncation flags, two-view-track counts and fallback/rejection statistics |
| Sequence matching performed redundant distorted and undistorted fundamental fits | Sequence explicitly skips the irrelevant distorted F stage and retains the calibrated undistorted verification. Existing matcher defaults/CLI commands remain unchanged |
| Validation/export made repeated per-observation OpenCV calls | Batch projections by camera and vectorize ray-angle calculation. Audit profile recorded 122,611 observation-quality calls; batching addresses this specific overhead |
| No stage/resource breakdown or repeat-run feature reuse | Add stage timers, numeric content/configuration/source-keyed feature caching, isolated child-process benchmarks and source fingerprints |

No existing tests were removed or weakened. The two-view 1px virtual RANSAC, 2px reprojection and 1-degree parallax defaults remain unchanged. The new PnP coverage gate is conservative and configurable; small subjects may require a justified explicit setting. No threshold was weakened to connect failed benchmark images.

## Datasets, provenance and split

**Castle: tuning/regression only.** Six original `100_7100.JPG`–`100_7105.JPG` photographs from [OpenMVG's Sceaux Castle sample](https://github.com/openMVG/ImageDataset_SceauxCastle), revision `fde7f5faba555e3c54700477c304488613346a19`. Original resolution 2832x2128; processed 1600x1202. Supplied K: fx=fy=2905.88, cx=1416, cy=1064. EXIF orientation 1; capture timestamps/resize transforms preserved. No lens coefficients or reference poses are supplied; zero distortion remains an explicit assumption. Six images total 6,766,631 bytes. Copyright notice credits Pierre Moulon (2012); it supplies an SfM testing sample but no explicit redistribution license. Images remain local/uncommitted. Original hashes/provenance are retained in TASK03_REPORT and the preparation script.

**TempleRing: held-out geometry evaluation.** [Middlebury Multi-View Stereo benchmark](https://vision.middlebury.edu/mview/data/), TempleRing collection from Seitz, Curless, Diebel, Scharstein and Szeliski, *A Comparison and Evaluation of Multi-View Stereo Reconstruction Algorithms*, CVPR 2006. Different subject and acquisition: a photographed nonplanar plaster temple, rather than an architectural facade. Native 640x480, without stereo rectification; provider-corrected radial distortion, so standard pinhole K with zero residual distortion is used. Published K: fx=1520.4, fy=1525.9, cx=302.32, cy=246.87. PNG inputs have no EXIF orientation. No resizing occurs with the common 1600-pixel limit.

Downloaded the provider's [TempleRing ZIP](https://vision.middlebury.edu/mview/data/data/templeRing.zip), 11,707,443 bytes containing 47 views. SHA-256: `5f871fe96d25f510eac026c66c3a4c38229326260986e9926cba8a64e88c8359`. Extracted only the first 30 filename/parameter rows (`templeR0001.png`–`templeR0030.png`, 7,416,465 image bytes), plus small calibration/notice files. Prepared nested 6/15/30 subsets using hard links, with copy fallback. No 312-view full dataset was downloaded. Archive and image hashes, original selection and URLs are in ignored `datasets/temple/provenance.json`.

The archive README attributes the collection to the benchmark authors and explains calibration/acquisition, but contains no explicit redistribution license. No CC/BSD/Apache image license is asserted. Data is used locally for the provider's benchmark purpose and is not redistributed. Obtain permission before rehosting the photographs. Calibration accuracy is documented by the provider as approximately pixel-level. Its reference laser-scanned surface is [not distributed](https://vision.middlebury.edu/mview/), so sparse surface distance/completeness is **not evaluated**.

Geometry algorithms/defaults were selected on Castle and synthetic regressions before examining Temple results. No held-out pose error or failure was used to retune focal assumptions, geometric thresholds, matching windows or BA budgets. Subsequent cache-integrity, plotting and non-finite-state safeguards do not change valid-image geometry. Natural filename order is retained; no reference-pose-based reordering, rotation correction, pair selection or camera initialization is performed. This is one held-out scene and nested subsets, not an independent large statistical sample.

Temple filenames do not form a uniformly sampled continuous sequence: acquisition gaps are visible in provider metadata, including approximately 46 degrees between views 5/6 and 107 degrees between 12/13. These gaps were not repaired using reference information. Their effect is reported rather than hidden by cherry-picking a successful subset.

## Experiment protocol and reference isolation

Common settings: SIFT target 6000 features (ties may exceed target), maximum dimension 1600, neighboring offsets 1/2 and selected offset 4, reciprocal ratio 0.75, seed 0, one OpenCV thread, original geometry/PnP count/error gates, local BA enabled. BA remains bounded to five cameras, 400 points, 30 evaluations, robust `soft_l1` scale 1px. `configs/benchmark_suite.json` freezes the eight cases, splits and calibration mode. `--min-registered` stays at five.

Calibrated cases use only the provider K. Approximate experiments use the existing explicit **60-degree horizontal FOV**, centered principal point, square pixels and zero distortion, with warnings/assumptions recorded. This generic assumption is intentionally not derived from the reference focal length. Approximate runs reuse image features where available, because feature extraction does not depend on K; pose estimation is recomputed.

`prepare_benchmarks.py` writes `cameras.json` containing intrinsic fields only, and a separate `reference_poses.json`. The benchmark child command receives no reference path/R/t or baseline. Only **after the child exits**, evaluation reads reference cameras. Each reconstruction runs in a fresh Python process; the parent samples memory every 50ms and uses the Windows lifetime peak working set when exposed, otherwise sampled RSS. `execution.json` records interpreter, source directory/fingerprint and memory semantics; suite/commands/results/logs are saved. Historical comparison uses an archived local baseline source tree through PYTHONPATH, not a second processing engine or a Git network operation.

Feature cache keys include source SHA-256, detector/limit, resize/shape, OpenCV/NumPy versions and extraction/preprocessing source hashes. Numeric NPZ entries retain native descriptor dtypes, keypoint metadata and a matching signature; pickle is disabled. Corrupt entries warn and recompute; writes replace atomically. Cold case caches are separate by input subset, followed by explicit warm repeats. A warm-only selection without its cold predecessor is rejected to prevent a mislabeled cold run. Source loading, EXIF and original-image colors still run on cache hits.

## Per-scene results

Canonical machine-readable artifacts: **`outputs/sfm-reliability-delivery/results.json` and `summary.csv`**. Eight cases completed. Benchmark `partial_registration` explicitly means fewer than all input images registered, even where the backwards-compatible CLI exits 0 after meeting its five-camera target. No failed frame has a fabricated pose.

| Case | Registered/input | Landmarks | Median / RMS reprojection px | Interpretation |
|---|---:|---:|---:|---|
| Castle published K, cold | 6/6 | 3915 | 0.285393 / 0.586640 | Complete coverage; no reference pose accuracy available |
| Castle published K, warm | 6/6 | 3915 | 0.285393 / 0.586640 | Same geometry as cold |
| Castle 60-degree approximation | 5/6 | 3386 | 0.363043 / 0.677297 | Incomplete; guessed intrinsics are not equivalent to published K |
| Temple 6, published K | 5/6 | 794 | 0.204040 / 0.482432 | Component of views 1–5 recovered |
| Temple 15, published K | 5/15 | 794 | 0.204040 / 0.482432 | Other components have no 2D–3D link to chosen seed |
| Temple 30, published K | 18/30 | 2664 | 0.260683 / 0.653783 | Moderate shared component; drift/reference warning |
| Temple 30, warm | 18/30 | 2664 | 0.260683 / 0.653783 | Same geometry as cold |
| Temple 15, 60-degree approximation | 5/15 | 794 | 0.189696 / 0.487436 | Low image residual despite strongly biased orientation |

Castle selected anchors remain image IDs 1/3. Temple 6/15 selects 0/1 and registers `[0,1,2,3,4]`. Temple 30 selects 26/27 and registers `[26,27,28,25,24,23,22,21,20,19,18,17,29,16,15,14,13,12]`. The other 12 views are rejected, not joined by an invented transform. This demonstrates an 18-camera shared reconstruction, **not** a successful full 30-view reconstruction.

Track diagnostics:

| Case | Tracks / track observations | Length-2 tracks | Conflict edges rejected | Successful fallback triangulations |
|---|---:|---:|---:|---:|
| Castle 6, published | 3923 / 11208 | 2095 | 9 | 5 |
| Temple 6 | 794 / 2304 | 387 | 1 | 0 |
| Temple 15 | 2177 / 5959 | 1178 | 5 | 0 |
| Temple 30 | 4273 / 14769 | 1903 | 20 | 1 |

Temple 30 track lengths 2–14 have counts `[1903,929,531,374,180,123,72,70,35,19,16,10,11]`. These are feature tracks, not all successfully triangulated landmarks. Two-view-only tracks remain a substantial fraction; rejecting all of them would remove useful scene coverage and was not introduced without supporting evidence. Observation-level 3D validation and explicit conflict/failure diagnostics remain necessary.

## Ground-truth camera comparison

Align estimated camera centers to matched reference centers using a proper-rotation **Sim(3)** fit (rotation, translation and positive scale). Reject collinear/insufficient trajectories instead of claiming an unobservable alignment. Convert aligned world-to-camera rotations consistently with the world transform. Report center RMSE both in provider reference-coordinate units and normalized by reference trajectory extent. Also compare adjacent **registered** cameras' relative rotations and signed translation directions; no absolute monocular scale claim is made.

| Temple case | Center RMSE, reference units | RMSE / trajectory extent | Median aligned rotation | Median relative rotation | Median translation-direction error |
|---|---:|---:|---:|---:|---:|
| 6 / 15, published K, five matched cameras | 0.003868 | 1.3010% | 1.9632 degrees | 1.3897 degrees | 2.3022 degrees |
| 30, published K, 18 matched cameras | 0.069289 | 6.3815% | **6.6134 degrees** | 3.3318 degrees | 2.8449 degrees |
| 15, approximate K, five matched cameras | 0.010169 | 3.4203% | **16.3172 degrees** | 5.1010 degrees | 2.0199 degrees |

Pose comparisons cover registered/reference-matched cameras only; they do not hide unregistered images, whose counts are reported separately. Same-sample alignment is descriptive, not a prediction on missing cameras. Provider calibration has finite accuracy. A reporting flag is raised for >10% extent center RMSE or >5-degree median aligned rotation; it does not feed back into reconstruction or lower acceptance gates. Temple 30 and the approximate case trigger it. The approximate case's lower median reprojection error is **not evidence of better camera geometry**. No reference surface distances are invented.

`pose_evaluation.json` includes full alignment and per-camera/pair errors; `trajectory.png` overlays aligned/reference camera centers and plots rotation error. The 30-camera-input diagnostic was visually reviewed for its recovered 18-camera component. Castle has no reference extrinsics, so its pose-accuracy fields are explicitly unavailable.

Independent PLY reload checks confirmed finite colored vertices and coordinate equality with exported landmark metadata for Castle and all calibrated Temple sizes. Maximum retained image reprojection errors were 1.993556px (Castle), 1.977432px (Temple 6/15) and 1.998156px (Temple 30). Cold/warm pose matrices and landmark coordinates compared **exactly equal** for Castle and Temple 30. The final source-tree fingerprint matches `execution.json`.

## BA comparisons and rollback

Every proposal records local RMS and robust cost on identical optimization observations, and global before/after metrics on identical current map observations **before any pruning**. Newly invalid identities veto the update even if RMS falls. Bounds remain five cameras / 400 points / 30 evaluations; all fixed boundary and initial anchor poses remain unchanged. Accepted changes are committed only after candidate pruning completes.

Castle accepts 1 proposal and rolls back 3; Temple 6/15 accepts 1 and rolls back 2; Temple 30 accepts 1 and rolls back 15. In Temple 30, the first accepted local step reduces same-observation RMS **0.470537 → 0.306581px**. Later examples reduce local RMS but create invalid map observations; they correctly roll back. One proposed boundary window `[21,22]` reduces local RMS **0.849283 → 0.261569px** but creates **179** new invalid observations elsewhere. Local fit improvement alone would therefore be an unsafe acceptance rule.

The covisible boundary changes along the 30-input run (`[26,27]`, `[25,26]`, `[24,25]`, …), so BA remains available beyond initial-anchor visibility. It still often rolls back because a 400-point subset cannot safely move cameras while leaving the rest of the map fixed. This is measured evidence for better bounded boundary/landmark coverage in future work, not justification to relax reprojection/depth gates. Final no-update proposals leave original reconstruction state intact. Aggregate final reprojection values across different pipelines are not substituted for same-observation BA comparisons.

## Runtime, memory and scaling

All measurements are single CPU runs, not averages or a speed guarantee. Native SIFT tie counts and feature density differ substantially across scenes. Castle detects roughly 5500–6000 features per photo; Temple detects 597–967 in this subset. Temple's 640x480 images and incomplete registration are cheaper than 30 high-resolution, fully connected Castle-like photos would be. Do not extrapolate a general linear scaling model from these cases.

| Case | Total seconds | Raw new feature extraction seconds | Cache hits | Peak child working set, MiB |
|---|---:|---:|---:|---:|
| Castle 6 cold | 19.445907 | 2.396025 | 0 | 560.32 |
| Castle 6 warm | 15.542059 | 0 | 6 | 219.27 |
| Castle 6 approximate, cached features | 14.392599 | 0 | 6 | 218.08 |
| Temple 6 cold | 2.343344 | 0.316836 | 0 | 174.62 |
| Temple 15 cold | 4.344975 | 0.860509 | 0 | 179.64 |
| Temple 30 cold | 12.881978 | 1.598771 | 0 | 193.35 |
| Temple 30 warm | 10.076622 | 0 | 30 | 192.16 |
| Temple 15 approximate, cached features | 2.682623 | 0 | 15 | 179.21 |

Cold-case stage seconds (four decimals; full precision in JSON/CSV):

| Stage | Castle 6 | Temple 6 | Temple 15 | Temple 30 |
|---|---:|---:|---:|---:|
| Loading/preprocessing | 0.6602 | 0.0976 | 0.2118 | 0.4363 |
| Features and numeric-cache I/O | 4.0901 | 0.5220 | 1.3956 | 2.7550 |
| Matching / geometric verification | 9.5491 | 0.1499 | 0.4982 | 1.4659 |
| Tracks | 0.0141 | 0.0022 | 0.0062 | 0.0239 |
| Initial-pair evaluation | 0.2304 | 0.0403 | 0.2326 | 0.6056 |
| Camera registration / ranking | 0.0487 | 0.0079 | 0.0194 | 0.3081 |
| Triangulation / observation attachment | 0.9456 | 0.0983 | 0.1133 | 0.9196 |
| Local BA including validation | 1.5258 | 0.2830 | 0.3759 | 4.5502 |
| Final validation | 0.3433 | 0.0475 | 0.0700 | 0.2517 |
| Export / reload / diagnostics | 2.0145 | 1.0786 | 1.4074 | 1.5430 |

Stage timers are nonoverlapping operational buckets: loading/preprocessing (including hashing, EXIF and colors), features (including cache read/write), matching/undistorted verification, track construction, initial geometry, registration/ranking, triangulation, BA, final validation and export/rendering. Raw new extraction time is separately recorded; warm cases report zero new extraction. Bookkeeping/manifest writes need not sum exactly to total runtime. BA includes its own whole-map validation. Process peak memory includes interpreter/native import workspaces; parent reference plots are excluded from child peaks. Wall time also includes startup and post-hoc evaluation/plotting.

Cold Castle feature extraction itself takes approximately 2.4s, with additional numeric cache compression overhead; warm reads avoid extraction. In this run, warming the cache reduces Castle total time by about 20% and Temple 30 time by about 22%, with identical cold/warm poses and landmarks. Cold-cache total time is not faster than the historical baseline: cache construction adds I/O/compression work. Matching remains the dominant Castle bottleneck at approximately 9.5s for eleven scheduled pairs and 6000-feature limits. No FLANN/approximate matcher or CUDA path was substituted just to improve the timing. Temple 30 BA/whole-map validation is a major cost, while matching remains smaller because it detects far fewer features.

Historical comparison uses the exact local baseline source archive, same inputs/settings and a fresh process: 6/6 cameras, 3913 landmarks, median 0.284303px, RMS 0.585669px, **19.413812s** and **587526144 bytes** peak in this session. The earlier 17.34s report is a historical observation, not a guaranteed comparison constant. New cold-cache construction does not promise a large total speedup; useful repeat-run savings are reported separately. Geometry differs slightly because per-pair RNG isolation, track-compatible scoring, registration ranking and fallback checks changed intentionally.

## Reproduction and artifacts

```powershell
.venv\Scripts\python.exe scripts/prepare_benchmarks.py --download
.venv\Scripts\python.exe scripts/benchmark_sfm.py --output outputs/sfm-reproduce
# Select a cold/warm pair together:
.venv\Scripts\python.exe scripts/benchmark_sfm.py --output outputs/repeat30 --case temple30 --case temple30_warm
# Existing public CLI, optional feature reuse:
.venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence --input datasets/scene --camera configs/camera.json --feature-cache outputs/cache --local-ba --output outputs/scene-run
```

Use new output directories. The full suite produces `results.json`, `summary.csv`, `suite_snapshot.json`, `execution.json`, logs, cache files and per-case reconstruction outputs. Actual canonical location: `outputs/sfm-reliability-delivery/`. Each case retains PLY, poses/status/attempts, intrinsics/assumptions, tracks/conflict examples, landmark visibility/depth/errors/angles, initial candidates, manifest and stage/BA metrics. Temple cases also include reference evaluation and trajectory plots. No original images, references, caches or generated results are committed.

Original `analyze`, `reconstruct-pair` and `reconstruct-sequence` commands remain available. The new `--feature-cache` option is additive. Benchmark setup hard-links subsets when possible and never loads reference poses into the application. No Git push, pull or fetch is used; the historical snapshot is a local `git archive`.

## Tests, limitations and readiness

Final automated command: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=outputs/pytest-sparse-reliability-delivery`. Result: **72 passed, 0 failed, 0 skipped in 7.62 seconds**. All 58 earlier tests remain. Added regressions cover track-compatible initial selection, ambiguous PnP associations, clustered support, contaminated-track alternative triangulation, covisible BA gauge, equal-count/different-invalid identities, pruning exception atomicity, non-finite initial/returned optimization states, Sim(3)/pose conventions, collinear/non-finite alignment rejection, native-dtype cache round trips/invalidation/corruption repair, exact cold/warm CLI geometry, and fifteen-camera execution. Fourteen existing third-party plotting deprecation warnings remain; no dependencies were upgraded to suppress them.

The benchmark parent initially attempted a restricted Matplotlib user-cache directory; plots still rendered, but the script now sets a writable benchmark-local cache for parent and child. A subsequent reference-plot smoke run completed without that filesystem warning. This does not change measured child reconstruction geometry.

Remaining weaknesses: incomplete image-graph coverage, locally greedy track merges, weak/repeated-texture associations, initial-pair bias, drift, and conservative local BA rollback. Temple 15 selects a strong seed component rather than globally maximizing reachable image coverage. A disconnected/unlinked component cannot be registered without real connecting observations; it is not merged using GT poses. PnP convex-hull coverage is a safeguard, not a proof of observability or correctness. Fisheye/raw-sensor calibration, rolling shutter, intrinsic refinement, global loop closure and reconstruction resume remain unsupported.

The 16 GB minimum laptop was not tested; no hardware certification is claimed. Resource bounds are retained, but 30 dense high-resolution inputs may cost much more than this benchmark. One held-out object scene and no public surface ground truth cannot establish broad reconstruction accuracy. Data redistribution permissions require care as noted above.

**Dense reconstruction readiness: no.** Before proceeding, address graph-aware seed/overlap handling and registration coverage, validate additional independently calibrated scenes, improve bounded BA coverage without sacrificing map validity, and reduce reference-camera drift. This task delivers a measured sparse baseline with explicit failures, not a claim that every moderate sequence reconstructs reliably.
