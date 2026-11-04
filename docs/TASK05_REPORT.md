# TASK 05 — registration recovery and drift investigation

Validation date: 2026-11-04, Europe/Istanbul. Technical evidence prepared for the university progress presentation on **November 9, 2026**. No slides or academic paper were created.

**Outcome:** reproduced the baseline, diagnosed missing registration, implemented bounded opt-in discovery and joint-refinement experiments, and generated traceable evidence. Temple 30 coverage increases from 18 to 23 cameras, but rotation accuracy regresses. Drift reduction without an accuracy tradeoff is **not achieved**. Experimental options remain disabled by default. Dense reconstruction remains **not ready**.

## Baseline reproduction before changes

Started from clean commit `0452da75405f75059ddf862dbc63c4b5bb375c1d`. Read the implementation and reports, then ran the existing benchmark CLI on Castle 6, Temple 15 and Temple 30 **before editing source**:

```powershell
.venv\Scripts\python.exe scripts/benchmark_sfm.py --output outputs/recovery-baseline --case castle6 --case temple15 --case temple30
```

All counts, geometry and reference errors reproduced the preceding milestone exactly. New measured runtimes vary from older observations:

| Dataset | Registered | Landmarks | Median image error | Median aligned rotation | Runtime | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| Castle 6 | 6/6 | 3915 | 0.285393px | unavailable | 20.029s | 560.01 MiB |
| Temple 15 | 5/15 | 794 | 0.204040px | 1.963206 degrees | 4.577s | 179.68 MiB |
| Temple 30 | 18/30 | 2664 | 0.260683px | 6.613361 degrees | 13.295s | 192.89 MiB |

Original per-image attempts, matching edges, track counts, seed-pair metrics, BA proposals and reference comparisons remain under `outputs/recovery-baseline/`. `diagnosis.json` records the complete failed-image records and connected components.

## Root causes

### Missing registration is primarily a connectivity failure in these runs

Verified image graph components, zero-based IDs:

- Castle: `[0,1,2,3,4,5]`.
- Temple 15: `[0..4]`, `[5..11]`, `[12..14]`.
- Temple 30: `[0..4]`, `[5..11]`, `[12..29]`.

Temple 15 selected seed IDs 0/1 in the first component and registered `[0,1,2,3,4]`. Every rejected image ID 5–14 (`templeR0006.png` through `templeR0015.png`) had **zero** tracked 2D–3D correspondences, rejected as `insufficient_or_collinear_2d3d` at map revision 5. Temple 30 selected 26/27 and registered the 18-camera last component. Every rejected ID 0–11 (`templeR0001.png` through `templeR0012.png`) also had **zero** 2D–3D correspondences, at revision 18. These were not PnP solutions that narrowly missed an error threshold. Redundant retries or weaker PnP thresholds cannot create missing associations.

The original offset-1/2/4 matching schedule misses a strong wraparound edge and cannot span some acquisition gaps. Temple filenames include substantial viewpoint gaps, as documented previously; there is not uniform temporal overlap. A seven-view component remains disconnected even after the bounded search. Initial-pair scoring favors geometric quality within a component, not total reachable image coverage. It can therefore seed a smaller component, as Temple 15 demonstrates.

### Drift has several separate contributors

Post-hoc reference evaluation finds **2.720668 degrees** relative rotation error in Temple 30's initial pair; Temple 15's initial pair has **1.375566 degrees**. Holding both initial poses fixed constrains world frame/scale but also retains initial-pair bias. PnP propagates estimates through imperfect landmarks and changing baselines. Local BA improves a small selected subset while many other landmarks remain fixed; most proposals introduce invalid observations elsewhere and are correctly rolled back. Thus low reprojection error is not sufficient evidence of camera accuracy.

Camera conventions were checked against existing analytical tests and Sim(3) evaluation: `Xc=R Xw+t`, camera center `-R.T@t`, consistent normalized triangulation and pixel reprojection. No sign/inverse-convention defect was found. The observed error is not explained solely by a pose-format mistake, nor solely by graph disconnection.

### Track coverage and fragmentation

Baseline Temple 30 has 4273 tracks and 14769 track observations, including many length-two tracks. Track conflict detection prevents duplicate image observations, but no track bridges the three components. Zero correspondence availability is therefore consistent with the graph, not an index-loss bug. Triangulated landmark visibility is a strict subset of track observations after depth/error/parallax checks. Repeated texture and epipolar-compatible false links remain possible; descriptor similarity alone cannot safely join tracks.

## Implemented experiments

### Bounded pair discovery (`--recovery-pairs 12`)

New `reconstruction/graph.py` computes components and cheap normalized descriptor-population signatures. SIFT signatures use descriptor means and standard deviations; ORB signatures use unpacked binary populations. These **rank candidates only**. Candidates crossing disconnected components are preferred, followed by appearance distance with deterministic image-ID ties. Already attempted pairs are excluded; each view participates in at most three new attempts. The CLI caps the total additional budget at 30; these experiments use 12.

Every attempted pair still needs native descriptor matching, reciprocal ratio filtering and the unchanged calibrated/undistorted fundamental RANSAC verification with >=20 inliers before becoming an edge. Tracks are rebuilt from verified edges through the existing conflict-safe union, before landmarks are initialized. No descriptor-only merge, pose transfer between components or duplicate landmark fill is used. Diagnostics export candidate distance, attempted/verified counts, components before/after and all existing track/triangulation rejection records.

Temple 30 discovers the natural wraparound pair **IDs 0/29**, with **648 verified correspondences**, joining the five-view component to the eighteen-view component. This yields a 23-image connected component and five additional PnP registrations. The remaining seven-view component gets no valid bridge within the budget. Temple 15 gains no verified edge. Candidate rankings are coarse appearance statistics, not a guarantee of overlap.

Camera counts refer to registered **image poses**, not a claim of equally separated viewpoints. Views 1 and 30 revisit a near-identical wraparound viewpoint; the extra link supplies observation connectivity rather than a new triangulation baseline. Low-parallax triangulation remains rejected.

### Bounded joint refinement (`--joint-ba`)

Reuses the established sparse robust solver and atomic candidate-state acceptance. It runs once after registration, not an unbounded periodic/global solver. Limits: **30 registered cameras, 2000 selected landmarks and the existing 30 function-evaluation budget**. More than 30 cameras explicitly skips the attempt. Intrinsics and both initial anchor poses remain fixed. All other registered poses are variables, with landmarks selected by visibility and stable ID. Anchor support and connectivity of the actual selected observation graph are checked; merely keeping unobserved anchor objects in state would not constrain gauge. Each optimized camera needs >=20 selected observations.

Residuals are identical processed distorted-pixel observations before/after. Sparse Jacobian, `soft_l1` loss, finite checks, positive depth, <=2px observation limits, >=1-degree angle checks, bounded movement, invalid-identity checks and atomic pruning/rollback remain active. Full-map error/validity is checked even for points outside the 2000-point optimization cap. This can conservatively reject a locally improved fit; it is not bypassed. Selected landmark IDs and scope are now recorded for debugging.

Existing default commands retain zero discovery budget and no joint pass. No geometry threshold or calibration was relaxed to raise counts. These optional research modes are **not recommended as a new accuracy-improving default** based on the results below.

### Registration and track safeguards retained

Recovery uses persistent feature-to-landmark links, distributed 2D–3D support, PnP RANSAC/refinement, depth/error validation and the existing revision-based retry logic. Failed frames are retried only after map registration/coverage changes; unchanged-data retries are suppressed. Original filenames/order/timestamps remain intact. Recovered images receive estimated PnP poses, not fabricated placements. New tracks retain one observation per image and one landmark per track, with inconsistent observations excluded by multi-view checks. The final joint pass does not invent connecting tracks or automatically register disconnected frames.

## Controlled results

Datasets, original filename order, published K, zero-distortion assumptions, 6000 SIFT target, 1600 maximum dimension, seed 0, one OpenCV thread, offset-1/2/4 schedule and local BA are identical to the reproduced baseline. Each final variant has a separate **cold** feature cache; all final cases report zero cache hits. This avoids the pilot's shared-cache timing confound. Comparisons are single runs, not statistical speed claims. Castle and Temple are now **development scenes**, not fresh held-out evaluation. No new dataset or dependency was downloaded/installed; provenance/licensing remains as documented in TASK04_REPORT.

Canonical variants: `outputs/recovery-final/results.json`, `summary.csv`, `suite_snapshot.json`, `execution.json` and individual run folders. Baseline remains the pre-change `outputs/recovery-baseline/`. The source fingerprint records each experimental version; the final source matches the final experiment snapshot.

| Dataset / mode | Registered | Points | Median image error px | Aligned rotation deg | Translation-direction deg | Runtime s | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Castle baseline | 6/6 | 3915 | 0.2854 | unavailable | unavailable | 20.029 | 560.01 |
| Castle graph | 6/6 | 4032 | 0.2829 | unavailable | unavailable | 23.749 | 560.95 |
| Castle joint | 6/6 | 3915 | 0.2854 | unavailable | unavailable | 20.712 | 560.74 |
| Castle combined | 6/6 | 4032 | 0.2829 | unavailable | unavailable | 25.332 | 560.56 |
| Temple 15 baseline | 5/15 | 794 | 0.2040 | 1.9632 | 2.3022 | 4.577 | 179.68 |
| Temple 15 graph | 5/15 | 794 | 0.2040 | 1.9632 | 2.3022 | 4.784 | 179.33 |
| Temple 15 joint | 5/15 | 794 | 0.1650 | **5.9844** | 0.5345 | 5.048 | 179.68 |
| Temple 15 combined | 5/15 | 794 | 0.1650 | **5.9844** | 0.5345 | 5.145 | 179.59 |
| Temple 30 baseline | 18/30 | 2664 | 0.2607 | 6.6134 | 2.8449 | 13.295 | 192.89 |
| Temple 30 graph | **23/30** | 3175 | 0.3334 | **12.7362** | 5.0477 | 14.122 | 194.42 |
| Temple 30 joint | 18/30 | 2664 | 0.2607 | 6.6134 | 2.8449 | 17.515 | 192.98 |
| Temple 30 combined | **23/30** | 3175 | 0.3334 | **12.7362** | 5.0477 | 20.110 | 195.23 |

Rotation/translation errors use exactly the existing post-hoc reference definitions. Castle has no reference poses. Temple reference poses are read only after each child reconstruction exits; they are not used for candidate ranking, geometry, optimization, scale fixing or result acceptance.

All-camera before/after rotation medians use different registered sets (18 versus 23), so a second comparison uses the **same 18 camera names**. Independently fitting the same Sim(3) definition on that common set gives **6.613361 → 7.452729 degrees** median aligned rotation: a genuine regression even after controlling the evaluation set. The 23-camera center RMSE / reference trajectory extent is 6.4285% versus baseline 6.3815%; more registration does not imply better metric geometry. No successful drift-reduction claim is made.

The baseline order is `[26,27,28,25,24,23,22,21,20,19,18,17,29,16,15,14,13,12]`; discovery changes it to `[26,27,28,25,24,23,22,21,20,19,18,17,29,0,16,1,2,3,4,15,14,13,12]`. Newly reachable images enter before some existing-component images, changing available landmarks and subsequent local windows. This is a measured propagation pathway, not a controlled attribution of all error to registration order alone.

### Connectivity, track coverage and cost

| Dataset | Baseline → graph components | Tested pairs baseline → graph | Tracks baseline → graph | Track observations baseline → graph | Matching seconds baseline → graph |
|---|---|---:|---:|---:|---:|
| Castle 6 | 1 → 1 | 11 → 15 | 3923 → 4046 | 11208 → 11655 | 10.037 → 13.901 |
| Temple 15 | 3 → 3 | 38 → 50 | 2177 → 2177 | 5959 → 5959 | 0.521 → 0.708 |
| Temple 30 | 3 → 2 | 83 → 95 | 4273 → 4283 | 14769 → 15427 | 1.482 → 1.741 |

Castle exhausts its four remaining possible pairs within the fixed budget; the algorithm is not unrestricted all-pairs discovery. Temple 30 tests 95 of 435 possible pairs. Reconnecting tracks increases observation coverage and changes PnP order/local-BA windows; it also permits accumulated uncertainty to propagate into newly reachable cameras. In the combined result, IDs 5–11 (`templeR0006.png`–`templeR0012.png`) remain unregistered with no usable link to the reconstructed component. Failed attempts and per-edge reasons remain in the saved JSON.

### Optimization outcomes and negative evidence

The final joint proposal rolls back for Castle (3 newly invalid observations), graph-augmented Castle (4), baseline Temple 30 (168) and graph-augmented Temple 30 (859). These counts are candidate-state diagnostics, **not** invalid observations retained in exported geometry. Runtime increases despite rollback, and no accuracy benefit is claimed for those cases.

Temple 15's joint proposal is accepted by the unchanged image-space geometric gates: median reprojection improves 0.204040 → 0.164983px and RMS improves 0.482432 → 0.270062px on the same retained map observations; no new invalid observations are introduced. Translation-direction error improves to 0.534540 degrees, but reference rotation worsens to 5.984394 degrees. Initial-pair bias, sparse structure and alignment sensitivity remain. The solver cannot use unavailable ground truth to veto a production update. This is why the mode is experimental and off by default. Improving one metric is not sufficient evidence of a better reconstruction.

These experiments establish a repeatable coverage/accuracy tradeoff; they do not establish full registration, drift reduction, loop closure correctness or broad camera accuracy. Memory peaks are measured fresh-process working sets including native imports; no minimum-laptop certification is claimed. CPU operation remains independent of CUDA.

## Presentation evidence

Canonical directory: **`outputs/progress-presentation-final/`**. Generated from actual baseline/final outputs, visually inspected and suitable as technical source material for a ten-minute progress update:

1. `01_feature_matching.png`: actual Temple views 1/30, 648 verified correspondences, 50 deterministically sampled matches displayed. Recomputed verification must equal the saved experiment; source image hashes are checked.
2. `02_sparse_scene_and_cameras.png`: separate panels for actual triangulated colored points and those points with estimated camera centers. Shows 3175 points and 23/30 registered cameras; axes explicitly use arbitrary initial-baseline units.
3. `03_registration_before_after.png`: registered counts for Castle 6 / Temple 15 / Temple 30; no missing poses are filled in.
4. `04_common_camera_pose_accuracy.png`: the same 18 Temple camera IDs before/after, labeled as a measured rotation regression.
5. `05_matching_graph.png`: verified baseline/discovery graphs; labels use one-based image names. Circular layout is explicitly schematic, **not camera geometry**; the actual recovered edge is highlighted.
6. `presentation_table.md`: compact six-row baseline/combined comparison; `benchmark_comparison.csv` contains all twelve rows and runtime/memory/track measurements.

`provenance.json` links source runs/fingerprint, image pair and exact shown feature indices; `common_camera_accuracy.json` stores the common-set reference evaluation. Feature and scene visuals contain real photographs and estimated reconstruction output, never illustrative/random point clouds. Original datasets and these generated assets remain ignored/uncommitted. Earlier pilot evidence under `outputs/progress-presentation/` is superseded by the final directory. No presentation deck, paper or external publication was created.

## Reproduction

```powershell
# Existing dataset preparation, if needed (no new dataset is required):
.venv\Scripts\python.exe scripts/prepare_benchmarks.py --download
# Reproduced original baseline must be captured before applying changes, or run the preceding commit's archived source.
# Final controlled variants, with separate cold caches:
.venv\Scripts\python.exe scripts/benchmark_sfm.py --suite configs/recovery_experiments.json --output outputs/recovery-final
# Opt-in individual reconstruction:
.venv\Scripts\python.exe -m vision3d.cli reconstruct-sequence --input datasets/temple/images30 --camera datasets/temple/cameras.json --local-ba --recovery-pairs 12 --joint-ba --output outputs/recovery-example
# Traceable figures and tables (use a fresh output directory):
$env:MPLCONFIGDIR="$PWD\outputs\matplotlib-cache"
.venv\Scripts\python.exe scripts/progress_evidence.py --baseline outputs/recovery-baseline --experiments outputs/recovery-final --output outputs/progress-presentation-final
```

Use new output paths when rerunning. Historical baseline source is available from the preceding local commit; the existing benchmark runner's `--baseline-source` supports an archived src tree. No Git push, pull or fetch is needed. The raw baseline and final artifacts are linked by report and evidence provenance rather than overwritten.

## Tests, remaining work and readiness

Final command: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=outputs/pytest-recovery-delivery`. **77 passed, 0 failed, 0 skipped in 7.58 seconds**; all 72 earlier tests remain. Fourteen pre-existing third-party plotting deprecation warnings remain. New regressions test component discovery priority/caps, rejection of descriptor-only unverified links, bounded joint gauge/observation preservation, the 30-camera limit, and rejection of joint windows whose selected points do not actually observe the anchors. Existing conflict, outlier, PnP, deferral, triangulation, rollback, cache determinism and CLI tests pass.

No dependencies were installed or upgraded. Existing commands are preserved; new flags are additive and off by default. Dataset/outputs/cache/venv exclusions are checked before the local commit.

Independent final PLY reload checks confirm finite colored vertices, coordinate equality with landmark JSON and maximum retained reprojection errors below 2px: 1.997982px (Castle combined), 1.646692px (Temple 15 combined) and 1.999566px (Temple 30 combined). The source fingerprint matches the final experimental snapshot. These are image-space/export checks, not substitutes for the reference accuracy regressions above.

**Remaining blockers to the primary engineering goal:** no demonstrated drift reduction without an accuracy tradeoff; seven Temple 30 frames and ten Temple 15 frames remain disconnected; bounded joint optimization leaves many points outside its variable set and often rolls back; fixed initial-pair bias persists; coarse appearance ranking cannot guarantee a useful bridge. Future work should investigate graph-aware initialization, stronger multi-view track validation, uncertainty-aware registration and better anchored optimization coverage on an independent scene. Do not use reference cameras as processing priors to force success.

The investigation, bounded experimental implementation, negative findings and presentation evidence satisfy the delivery scope. They do **not** make dense reconstruction ready, and no dense reconstruction, meshing, texturing or UI work was started.
