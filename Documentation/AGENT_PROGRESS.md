# UAV reconstruction engineering mission

**COMPLETE for this engineering cycle.** Implemented, benchmarked, repeated and documented the opt-in improvements on the fixed AirZoo Rural and Urban 40-view subsets. Rural improved from 16/40 to 28/40 and Urban from 25/40 to 35/40. Rural did not reach the aspirational 30-camera target. Existing outputs, acceptance gates and unrelated changes were preserved. No commits were made.

## Verified starting point

- Baselines under `outputs/airzoo-benchmark/{rural,urban}/reconstruction-primary/` are 16/40 and 25/40 registrations, with 1/3 and 2/3 stations respectively. Per-camera CSVs and geometry evidence are under `registration-diagnostics/`.
- Rural has cross-station matches that rarely reach valid map points. Urban repeatedly fails the full-set PnP consensus gates. One unchanged final-map retry passed Rural trigger 59 and exposed more support inside the already mapped station.
- Neither baseline accepted a local BA update. The current seed probe stops after the first successful third view. Several accepted cross-station seeds were never probed. Existing map-aware matching recovery and other working-tree changes predate this mission.
- Implemented optional independent seed rollouts in `reconstruction/seed_search.py` and wired the sequence CLI and initializer. The current seed competes with geometrically accepted alternatives. Optional metric position hints group stations and rank coverage. PnP and triangulation gates are unchanged.
- `io.load_image` now reads the primary MPO image directly. The AirZoo replay driver is `outputs/airzoo-benchmark/major-sfm-improvement/run_airzoo.py`. It takes exact inputs and flags from the saved run and builds local position hints from MRK.
- Initial Rural seed portfolio finished at **23/40, two stations, 7,428 landmarks** (0.733/1.009 px, 1.66° median angle, 308.7 s). Final saved map has 317 cross-station landmarks, zero invalid observations, and 1.70 m median leave-one-out MRK center residual. It is the current Rural coverage leader, with worse image residual and median angle than baseline.
- Initial Urban seed portfolio finished at **35/40, three stations, 6,066 landmarks** (0.732/0.949 px, 2.96° median angle, 338.0 s). It has 1,493 cross-station landmarks and zero invalid observations. MRK median leave-one-out residual increased from 1.18 to 2.73 m across different camera populations, so coverage is stronger but absolute accuracy is unverified.
- The Rural baseline-seed late-retry ablation finished at **19/40** in one station, 4,227 landmarks and 0.567/0.831 px. It registered IDs 18, 32, 19 after final refinement. Urban quality-PnP-only finished at **25/40** with the same geometry as baseline, no camera gain.
- Rural `deeper-portfolio` regressed from 23 to **18/40**, but improved median angle to 2.97° and left-out MRK median to 1.07 m. Rural `station-balanced` remained at **23/40** with no third-station registration. Urban `combined` remained **35/40** with the same poses. Rural explicit `southern-bridge` registered **10/40**, including six S3 views. It shares four S2 cameras but only one landmark pair with corroborating feature observations with the stronger Rural map, so a geometry-supported merge is blocked.
- Pre-initialization bridge screening is implemented as opt-in `--bridge-pairs`. With 24 extra attempts, Rural verified 23 bridges and reached **28/40 in two stations** (8 S1, 20 S2), 6,715 landmarks, 1,453 cross-station landmarks, 0.598/0.865 px and 2.07° median angle. Urban verified 24 bridges but fell to **30/40** versus the seed-only 35/40. Current preferred profiles are Rural `preinit-bridges`, Urban `seed-search`.
- Both selected profiles repeated byte-for-byte in poses, tracks, landmarks and PLY. Open3D reloaded 6,715 Rural and 6,066 Urban colored points. The selected Rural run took 86.6 s with warm caches (469 MiB). The selected Urban run took 338.0 s cold (718 MiB) and 102.3 s warm.
- Rural `bridges-refined` remained at 28, `bridges40` fell to 24, and `southern-rich` remained at 10 with six S3 views. The main Rural map has zero S3 registrations. Four S3 cameras never reach PnP and the other three return 5–6 RANSAC inliers.
- The richer southern map and the 28-camera winner share four cameras and 24 corroborated landmark ties. Two point-based similarity folds failed all 48 held-out observations at the unchanged 2 px gate. A shared-rotation-constrained fit also failed all tested held-out observations. No map merge was promoted. See `rural/merge-audit.json`.
- Added independent opt-in `--quality-pnp` and `--final-retry N`. The former tries AP3P with spatially balanced, high-angle landmarks only after the default hypothesis fails. It must regain the original full-set count, ratio and coverage before acceptance. The latter revisits stalled views after the fixed-camera refinement and map pruning, with a camera budget. Neither was enabled in the in-flight seed-only runs.
- Resumed full regression suite passed **165 tests** (14 existing dependency warnings, 13.53 s). Tests include synthetic seed continuation, unchanged input state, invalid and missing hints, full-set PnP ratio enforcement, primary MPO loading, station reservation, explicit seed rejection, and bridge screening.
- Added `audit_geometry.py` in the experiment workspace. It checks fixed image hashes and quality settings, positive depth, proper rotations, station coverage, cross-station tracks, unused final-map pair epipolar residuals, and leave-one-camera-out MRK similarity residuals. MRK is supporting telemetry, not independent pose ground truth.

## Reports and reproduction

All eight requested reports are under `outputs/airzoo-benchmark/major-sfm-improvement/`. `final_report.md` summarizes the selected profiles. `experiment_history.csv` links measured successes and regressions to commands and run outputs. Shared-observation RMS increased on both scenes. The Urban winner's shared-camera MRK median increased from 1.18 to 2.02 m. These are coverage improvements with geometry-fit trade-offs and no independent pose ground truth.

```powershell
$root = 'outputs/airzoo-benchmark/major-sfm-improvement'
.venv/Scripts/python.exe -B "$root/run_airzoo.py" rural reproduce-rural --seed-search 7 --seed-search-views 6 --position-hints '@positions' --bridge-pairs 24
.venv/Scripts/python.exe -B "$root/run_airzoo.py" urban reproduce-urban --seed-search 7 --seed-search-views 6 --position-hints '@positions'
.venv/Scripts/python.exe -B "$root/collect_results.py" --rural preinit-bridges --urban seed-search
```

Use new tags for additional runs. Future work starts with obtaining independently corroborated Rural S2–S3 ties and revalidating the geometry on held-out observations. A wider bridge budget and GPS proximity alone did not justify a merge. For Urban, address the residual trade-off before considering default promotion. No pending benchmark needs to be rerun to reproduce this checkpoint's recorded results.

## Mission II — dense UAV reconstruction

**COMPLETE for the validated dense, mesh and export scope.** Real dense clouds and open surfaces were benchmarked on both fixed AirZoo subsets and an expanded 48-image Rural collection. Fixed comparisons use `major-sfm-improvement/rural/preinit-bridges/reconstruction` and `major-sfm-improvement/urban/seed-search/reconstruction`. The previous sparse worlds and separate southern Rural map were preserved. All new evidence is under `outputs/airzoo-benchmark/major-dense-reconstruction/`.

Audit found a working CPU SGBM pipeline with rectification, reciprocal disparity, a third-camera depth and RGB check, voxel fusion, image-grid triangles, mesh cleanup, and PLY/GLB round-trip checks. The reference budget ranks each view by its best pair alone, which can concentrate dense output. Current corroboration uses relative depth at the nearest witness pixel without explicit return reprojection or depth-edge handling. GUI dense execution already uses the CLI subprocess, supports cancellation and preserves sparse outputs on later-stage failure.

Balanced baselines and the two `uav-validated` runs reloaded as colored PLY and GLB. Rural balanced → final: 10 → 17 fused references, 628,982 → 950,775 points, 1,150,737 → 1,420,823 faces, 16.46 → 53.94 s including new OBJ export, 1,226 → 1,739 MiB. Occupied horizontal grid cells increased from 6,344 to 19,590. Urban: 12 → 16 fused references, 1,555,539 → 1,623,418 points, 3,279,231 → 3,107,260 faces, 29.03 → 67.36 s, 2,777 → 3,082 MiB, and 8,347 → 8,708 cells. Both final PLYs have zero edges with over two incident faces, and both retain open boundaries. The final Rural mesh also reloaded as OBJ.

Implementation is in `dense_quality.py`, `multiview.py`, `stereo.py`, `surface.py` and `asset_export.py`, wired through CLI and desktop profile selection. `uav` is opt-in. It uses positive-depth sparse observations with ≤2.5 px saved error, spends the reference budget on novel visible landmarks, preflights and recenters invalid disparity canvases, requires a continuous witness depth footprint and ≤1.5 px round-trip agreement, and removes overly long and nonmanifold faces without filling boundaries. Optional OBJ carries geometry and normals. PLY and GLB carry image-derived vertex colors. Sparse poses and calibration files are reused without edits.

The first full test run had 170 passes and one UAV export failure. Open3D can weld numerically identical vertices. The export fix verifies every indexed triangle corner and all applicable attributes. A genuine Rural OBJ attempt then exposed Windows sandbox permissions on native writes into Python's temporary subdirectory. A sibling temp file resolved the failure, including in the final benchmark. The final suite passed **174 tests**. Tkinter selected the new profile, and both final meshes rendered through the desktop's Open3D mesh loader and framing path. No manual mouse navigation was verified.

The 48-image Rural run used the 40 original triggers plus four immediately before and after. It registered 31/48 with 3,755 sparse landmarks, 0.593/0.846 px median/RMS, 4.51° median angle, and completed sparse in 202.79 s. UAV before recentering fused 16 references and 1,548,295 points with a decimated 339,821-face mesh. Dense runtime was 87.79 s and GLB reloaded.

The `quality-only` run removed nonmanifold topology but lost some point coverage. Preflight alone produced the same geometry as the wider scheduling variant, with lower observed runtime. Recentering with one witness produced 1,095,226 Rural points and several weak patches. The final two-witness requirement kept all three recovered S1 references and 191,854 of their samples. Urban yielded no successful recentered map. Reproduction entry points are `run_dense_benchmarks.py`, `evaluate_dense.py`, and `render_mesh.py` in the Mission II output root. Use new tags and retain previous results. For example:

```powershell
$p='outputs/airzoo-benchmark/major-dense-reconstruction'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='4'
.venv/Scripts/python.exe -B "$p/run_dense_benchmarks.py" rural replay-uav --obj
.venv/Scripts/python.exe -B "$p/run_dense_benchmarks.py" urban replay-uav
.venv/Scripts/python.exe -B "$p/evaluate_dense.py" rural --tag replay-uav --suffix _replay
```

`final_report.md` and `experiment_history.csv` link the model, topology, footprint, runtime and regression evidence. Remaining development priorities are independent reference-based geometry checks, measured calibration, Urban vertical-surface and S2 depth coverage, and larger blockwise reconstruction past the 50-image sparse cap. Current artifacts have arbitrary SfM scale and image-derived vertex color. UV atlases, metric georeferencing and full-collection processing remain unvalidated. No completed experiments need to be rerun for this checkpoint.

## Mission III — validated surfaces and photographic UV assets

**IN PROGRESS for intersecting geometry.** Blender replaces the blocked Unity test for the active import and rendering milestone. The optimized and UV-textured workflows passed real reconstruction, export, desktop worker and Blender validation. All experiments are in `outputs/airzoo-benchmark/major-surface-improvement/`. The fixed Mission II inputs and unrelated deliverable edits were preserved. The latest full suite passed 186 tests.

Previously validated engineering commits recovered at startup:

- `2c97d49` — sparse map recovery, multi-station initialization and MPO input.
- `577eb1f` — overlap-aware video keyframe selection.
- `c38d4b6` — UAV dense stereo, checked open surfaces and mesh export.

Mesh milestone validated on both fixed scenes. Open3D's orientation routine left 14,018/34,989 conflicts on the first 450,000-face candidates. Shared-edge parity repair with camera-facing component votes reduced conflicts to zero, with more disconnected pieces where cycles were cut. Rural accepted 434,645 faces at P99 0.000717 SfM units. Urban's 450,000-face candidate was rejected because its maximum sampled displacement exceeded the bound. Urban accepted 848,146 faces using a 900,000 target, with P99 0.000921. Outputs are `rural-geometry/` and `urban-geometry/`. Full regression after geometry and initial texturing passed 177 tests. The photographic atlas milestone has working real GLBs on both scenes and is undergoing export and desktop integration.

- `8cf9d50` — topology, camera-facing winding, guarded simplification and `surface-asset` CLI.
- Both fixed GLB and OBJ/MTL bundles passed independent reload in `rural-export/` and `urban-export/`. Photographic coverage with appearance holdout is 32.62% Rural and 21.73% Urban by area. Rural's front-only trial reached 25.10%. The remaining GLB faces use vertex colors. Original-photo texture sources, visibility, atlas coordinates, and fallback counts are recorded in the texture metrics.
- Recomputed dense reconstruction for the 48-image Rural sequence using its 31 registered cameras. It reached 1,409,780 dense points and 395,278 mesh faces. Optimization and texturing finished at 248,023 faces with 34.15% photographic area. Outputs are `rural-48-dense/`, `rural-48-geometry/` and `rural-48-export/`.
- The reserved-depth audit measured four Urban pairs and found a small loss of depth agreement after simplification. Rural had no viable pair whose cameras were both excluded from all successful dense depth maps. Open3D rendered both actual textured GLBs. The Unity 6000.6.1f1 native OBJ import attempt exited 198 with no valid editor license. The isolated validation harness is in `unity-tools/` for replay after activation. No Unity import or rendering is verified.
- Tkinter controls and the real Rural desktop geometry-to-texture worker passed. Release suite passed **179 tests**. Seam welding rejected 1 Rural and 4 Urban candidates at half a dense voxel because the result failed the topology gate. The seam method remains an isolated experiment. Next validate the combined benchmark replay, capture topology probe results and finalize the requested reports.
- `faceb5a` — real-photo UV atlas, validated GLB/OBJ/MTL export, desktop job and textured Open3D viewer.
- `5b75d93` — benchmark replay, reserved stereo, topology and overlap diagnostics, and reproducible seam experiment.
- `b606878` — tested opt-in shared depth witness gate and measured benchmark documentation.
- The full Rural benchmark replay produced the identical textured GLB SHA-256 `2a37f8dd42987f4406908f1462f2faa2e414e6b51db1885ed60e1f43c39c2eff`. Both final meshes had intersections in all eight compact probe neighborhoods. The visibility-and-overlap experiment found no faces it could safely remove in either scene. Shared-witness depth retained the original point clouds and rejected 7,575 Rural and 8,333 Urban grid faces. It also found intersections in all neighborhoods and did not improve the Urban held-out 2% agreement. The benchmark assets remain `rural-export/` and `urban-export/`. The geometry target settings are 450,000 Rural and 900,000 Urban with 4096 atlases and stride-7 appearance holdout. The opt-in dense rule has real-scene and synthetic test coverage.

### Reproduction and continuation

```powershell
$p='outputs/airzoo-benchmark/major-surface-improvement'
.venv/Scripts/python.exe -B scripts/benchmark_surface_assets.py --scene rural --output "$p/replay-rural-next"
.venv/Scripts/python.exe -B scripts/benchmark_surface_assets.py --scene urban --output "$p/replay-urban-next"
.venv/Scripts/python.exe -B scripts/audit_surface_topology.py --mesh "$p/urban-geometry/mesh.ply" --output "$p/urban-topology-next.json"
```

Start further surface research from the original UAV depth inputs and the first-hit reserved stereo and intersection probes. Test any layer-aware fusion replacement on the same fixed inputs and require improved depth agreement and intersection evidence before promoting it. Unity is blocked by editor activation. After activation, replay `unity-tools/check_surface_import.ps1` under the output root with `-UnityEditor`, `-ExportDirectory "$p/rural-export"` and a fresh `-ProjectDirectory`. No sparse or dense benchmark needs to be repeated to recover the results already recorded here.

### Independent Blender continuation

Blender 5.3.0 is the active downstream import validator. Unity remains unverified. The original four bundles imported their correct UV atlases but lost 5 Rural and 10 Urban duplicate-position faces per format. Blender also returned 12 Rural and 24 Urban zero corner normals per format. Preserve this failed reference in `blender-baseline/`. Correction cleans geometry at export precision and splits shading at folded vertex fans. The weighted-winding exports in `rural-blender-ready/` and `urban-blender-ready/` passed all four strict imports. Their CPU Cycles replay was pixel-identical in all eight images.

`115f6d6` adds the Blender importer, geometry/material audit and rendered regression harness. Geometry-weighted winding lowered edge-connected components from 3,699 to 2,914 in Rural and 6,124 to 3,364 in Urban before export cleanup. Urban has 3,363 in Blender after exact-position cleanup. Rural and Urban P99 sampled displacement improved from 0.000717 to 0.000697 and from 0.000921 to 0.000844 SfM units. The same deviation thresholds were applied. Urban reserved-stereo agreement within 2% improved slightly on three of four pairs and fell slightly on the fourth.

The 40,000-face partial-visibility ablation raised photographic coverage from 32.47% to 36.55% Rural and from 21.44% to 23.65% Urban. It partitions the existing triangles and reruns the same per-child visibility tests. `blender-release/` passed four strict imports and all eight repeat renders were pixel-identical. All eight compact neighborhoods in each mesh still contain nonadjacent overlaps. A conservative camera-supported layer experiment found one removable Rural face and zero Urban faces, so it remains diagnostic only.

`8323fbd` adds geometry-weighted winding and export-precision and shading fixes. The preferred Blender-validated bundles for further texture work are `rural-visible-refinement/` and `urban-visible-refinement/`. Geometry-only comparisons should use `rural-weighted-geometry/` and `urban-weighted-geometry/` so tessellation changes do not confound topology counts. Reproduction commands, per-format reports and the latest held-out evidence are in `blender_continuation_report.md`. Do not repeat the Unity test for this milestone. For further overlap work, start with `scripts/experiment_surface_layer_pruning.py` on those geometry directories. Its negative result identifies the limitation of removing existing faces with current evidence. A new fusion candidate needs independent depth and visibility support before changing the selected assets.

`2c0a7c7` adds conforming visibility subdivision, the bounded layer-pruning experiment, and the measured benchmark documentation.

## MISSION III-B — GEOMETRIC ROOT-CAUSE RECOVERY

**IN PROGRESS.** Fixed Rural and Urban dense replays reproduced 950,775 and 1,623,418 fused points using the recorded sparse poses. The replays are in `outputs/airzoo-benchmark/geometric-root-cause-recovery/{rural,urban}-baseline`. Depth diagnostics preserve rectified rasters, sample-to-witness associations, and sample-to-fused-vertex indices. Six camera/projection tests passed.

The local PPK-aligned DSM audit located a large Rural elevation-error tail in unmodified stereo samples. Three mutually related native pairs (17→32, 31→32 and 32→31) have about 79–80% of accepted samples more than 10 m from the offset-aligned DSM. Their median effective disparities are 15.0–22.6 px. Native 19→16 has 99.8% above the same deviation threshold and a 14.0 px median effective disparity. Successful recentered maps have 11–12% above the threshold at median effective disparities of 214–225 px. This weakens the hypothesis that recentering caused the dominant measured elevation tail. The comparison uses a single baseline DSM offset because the DSM vertical datum is unverified. PPK is an alignment aid.

Next: replay both fixed scenes with a bounded disparity-conditioning check on both the sample and witness maps. Measure deviation, footprint, coverage by station, independent view consistency, and final mesh quality. Use fresh experiment directories. Leave this section in progress until geometry and coverage have been compared.

- `fa8afce` adds baseline depth instrumentation, numerical camera tests and the PPK/DSM audit. Both baseline dense PLYs match their historical SHA-256 hashes.
- Implemented opt-in `uav-conditioned` in CLI and the desktop Multi-view selector. At a fixed baseline voxel size it reduces Rural DSM P95 from 32.01 m to 11.93 m and Urban from 8.86 m to 8.30 m. Dense output is 953,171 Rural and 1,746,603 Urban vertices. Rural retains 93.17% of baseline half-meter XY cells and Urban 99.53%. The positive Rural deviation tail falls, while the fraction >10 m below the offset-aligned reference rises from 2.66% to 3.46%.
- The aligned Rural strip probe contains 74,411 baseline fused vertices and zero in either corrected run. Sampled accepted pixels trace to native 32→31, 31→32, 17→32 and 19→16, all from station 2. The Blender side/oblique renders in `rural-blender-geometry/` show the detached strip removed. Remaining gaps, residual height errors and sampled self-intersections need further investigation. Threshold 28 is opt-in. The original UAV profile stays available.
- Real native/recentered pair audit, fixed-voxel ablations and offset-aligned DSM reports are under `rural-rectification/`, `urban-rectification-final/`, `*-conditioned28-fixed/`, `*-conditioned-pairs28/` and `*-dsm-fixed/`. `docs/AIRZOO_GEOMETRIC_RECOVERY.md` contains replay instructions. Full test suite: 196 passed (14 existing deprecation warnings).
- Next: finish station isolation, expanded Rural replay, corrected Blender asset validation, and independent PyCOLMAP if the isolated wheel works. Record comparisons and unresolved limitations in the final report.

- `dc3020b` implements and validates the conditioned pair and witness profile.
- The first corrected Urban GLB/OBJ produced seven zero Blender corner normals in the same folded fan despite unit exported normals. Added geometry-neutral shading splits at shared edges with opposing face normals. Rural and Urban strict GLB/OBJ imports then all passed. The eight repeat renders are pixel-identical. See `final-blender-release/`, `final-blender-replay/` and `urban-normal-probe.log`. Blender compatibility does not establish Unity compatibility.
- Final regression after the shading fix passed 197 tests. Mesh probes still find intersections in all eight sampled neighborhoods per scene. Urban common reserved stereo agreement improved on three of four pairs. Rural has no viable held-out pair under the strict common-unused-camera rule.
- Station isolation and the 48-image replay are saved in `rural-stations/` and `rural48-*/`. PyCOLMAP CPU found 29/40 Rural and 40/40 Urban in the first comparison, and both exported pose sets drove the project MVS. A repeat fixes the half-pixel coordinate convention for the exact comparison. Optional CUDA COLMAP download is bounded and SHA-256 checked before execution.

- `7479fde` is the validated folded-normal export fix. It leaves the recovered triangle geometry unchanged.
- Both current default `uav` PLY outputs match the original hashes without diagnostics. Same-cell Rural DSM P95 changes only from 11.95 m to 11.90 m, so much of the improvement comes from discarding unsupported geometry. The 48-image case slightly worsens aggregate DSM P95 from 13.32 m to 13.35 m.
- Pixel-aligned COLMAP CPU models registered 29 Rural and 40 Urban. Median common-camera relative rotation differences are 0.68° and 1.38°. The CUDA PatchMatch replay of the original project poses removed the localized Rural strip too. It fused 69,139 Rural and 77,070 Urban points. The localized strip counts are 74,328 baseline and zero for conditioned project depth and for COLMAP on project poses. COLMAP MVS has substantially smaller footprint.
- `experiment_history.csv`, `final_report.md`, the per-scene JSON files and all figures are in the ignored mission output. Full COLMAP pose-plus-depth and the shared-offset B/D comparisons are complete. Incomplete engineering after that is persistent intersection removal, unreliable held-out Rural geometry and coverage beyond the validated scenes.

- `9806a69` adds the audited station, DSM and external reconstruction scripts. `866caa8` adds same-pose pixel-depth comparison. All four COLMAP/project pose-depth combinations completed in both scenes. Full COLMAP DSM P95 is 12.09 m Rural and 7.23 m Urban. The corrected project result on original poses is 11.93 m and 8.30 m. The full COLMAP Rural P99 is 80.06 m, so the tail remains problematic. The localized strip has 17,258 sampled accepted project depths and zero valid independent COLMAP depths at those pixels.
- Corrected raw and optimized meshes both show intersections in all eight sampled neighborhoods per scene. Raw winding conflicts are 16,131 Rural and 51,061 Urban. The optimized winding gate removes them, with components increasing from 397 to 2,647 and from 9 to 3,525. Visibility/coverage pruning removed zero faces in both revised meshes. The winding-before-decimation Rural ablation in `rural-winding-first/` measures the candidate against the original raw mesh.

- Final winding-first Rural ablation passed the raw-surface deviation bound but raised final fragmentation from 2,647 to 2,674 components. It was rejected for this objective. Save `rural-winding-first/metrics.json`. The project core, final Blender checks and all 197 tests are validated. All required reports are in `outputs/airzoo-benchmark/geometric-root-cause-recovery/`.

### Resume the remaining geometry work

**IN PROGRESS.** Retain `uav-conditioned` as an opt-in profile. Do not repeat the primary or external benchmark runs. Start with the remaining raw-mesh intersection neighborhoods in `*-corrected-raw-topology.json` and the verified independent geometric depth workspaces in `*-project-poses-colmap-dense/workspace/`. Extend the existing diagnostic probes to track which source pixels support both sides of the same intersecting layer. Use independent COLMAP depth agreement and true same-pose visibility to test this hypothesis before filtering or welding either layer. Compare using the same raw mesh bounds, survey footprint and Urban reserved pairs. Rural still has no strict common-unused-camera stereo pair and requires more cross-station geometric support for independent validation.

The scripts and command lines are in `docs/AIRZOO_GEOMETRIC_RECOVERY.md`. Final source and validation commits are `fa8afce`, `dc3020b`, `7479fde`, `9806a69`, and `866caa8`. The checkpoint-only commit is discoverable with `git log -1 -- Documentation/AGENT_PROGRESS.md`.

### Resumed validation

- `6c8c3ef` commits the verified recovered checkpoint. The original winding-first Rural PLY independently reloaded with 252,018 vertices and 445,002 faces. Its 2,674 optimized components remain worse than the retained 2,647, so the original selection stands.
- Saved same-pose COLMAP depths were reused for the intersection audit. Of 10,492 Rural and 15,051 Urban raw-mesh intersection pairs, 10,457 and 14,065 have neither face supported by two separated depth views. The controls show similarly low coherent depth coverage in Rural. Zero faces meet the conservative contradiction-and-supported-alternative test. No mesh faces were changed.
- The full-resolution fixed-grid ablation reproduced all original baseline hashes. Fixed Rural and Urban DSM P95 improve from 32.028 to 11.918 m and 8.867 to 8.306 m. Rural-48 goes from 13.323 to 13.349 m and retains 87.69% of baseline half-meter cells. Every one of its 1,397,878 surviving points exactly matches the baseline. Shared-cell P95 is unchanged at 12.134 m.
- Of 11,911 accepted depth pixels lost on unchanged Rural-48 rectified maps, 4,374 fail candidate disparity and 7,537 lose qualifying witness support. These groups have median absolute DSM residuals of 1.28 and 1.16 m against 4.09 m for retained pixels. The profile remains opt-in. Outputs are `*-resume-depth-layer-controls/`, `*-resume-grid-dsm/`, and `rural48-conditioning-loss.json` under the existing mission output.

- `8bf8786` adds an explicit fixed fusion grid origin, reported in metrics, and leaves the original origin default in place. The Rural-48 same-grid CLI replay reproduced the same PLY SHA-256.
- Relative tolerance sensitivity at 0.25%, 0.5%, and 1% found zero valid layer-removal candidates in either fixed scene. The validated new diagnostics use saved COLMAP outputs without rerunning COLMAP. Blender retained GLB/OBJ all passed, with eight pixel-identical renders in `resume-blender-replay/`. The winding-first PLY lost 17 imported faces compared with four in the selected PLY, equal to the duplicate triangle counts at float32 precision. The candidate remains rejected.
- Full suite after the resumed work passed 202 tests with 14 deprecation warnings. The final measurements and reproducible tools are in `docs/AIRZOO_GEOMETRIC_RECOVERY.md` and the output `resume_report.md`.

**IN PROGRESS.** The priority is the Rural-48 lost witness support. Read `rural48-conditioning-loss.json` and start with reference 16/source 0, which accounts for 6,697 of 7,537 pixels that lost a qualifying witness. Compare alternate source-view candidates only in the affected rectified regions, preserving the 28 px gate and existing 2% depth and 1.5 px return thresholds. Reuse the new fixed grid and quantify coverage on all three scenes. No raw mesh modification is justified by the current independent depth audit.

- `aef38fc` adds the controlled intersection and all-point DSM diagnostics, including the exact first-hit test. One bounded witness feasibility probe reused real Rural-48 inputs and retained the 28 px effective disparity, 2% depth, and 1.5 px return gates. On the largest loss map 16→0, three of eight alternate stereo pairs corroborated 3,340 of 6,697 lost pixels. Artifacts and the replay script are `rural48-alternate-witness-probe.json` and `probe_alternate_witnesses.py` in the mission output. This probe has not been fused. The next incomplete milestone is a bounded supplementary-witness fusion experiment with one vote per reference, fixed grid comparisons in all three scenes, and unchanged artifact/intersection gates. The current profile stays opt-in.

## MISSION III-C — RECONSTRUCTION RECOVERY AND MVP COMPLETION

**IN PROGRESS.** Starting HEAD `a69be3e`. Unrelated research deliverables and capture files are modified/untracked and preserved. Experiments use `outputs/airzoo-benchmark/mission-iii-c/`; all previous runs remain reference evidence. First implementation is bounded supplemental stereo for previously plausible candidate depths with insufficient conditioned witnesses. Alternate references must contribute new source views and pass the existing geometric gates. Run Rural-40, Urban-40 and Rural-48 with the saved fixed grids before selecting any assets. Continue with intersection provenance, larger real-image reconstruction and end-to-end integration.

### Validated alternate-view fusion

`depth_recovery.py` adds budgeted alternate stereo witnesses for pending primary pixels. Both sides retain the 28 px effective disparity gate, the 2% depth gate and 1.5 px round-trip check. Supplemental image pairs are disjoint from the primary pair and other counted votes at the pixel. Recovery uses one vote per camera reference, keeps the existing two-witness requirement for recentered targets and records source/reference/pixel/voxel provenance. Existing profiles keep their defaults.

Same-grid runs with a 16-pair budget recovered 8,915 Rural-40, 4,611 Urban-40 and 4,471 Rural-48 pixels. Fused point counts were 962,839, 1,750,100 and 1,402,248. Full-point DSM P95 went 11.918→11.904 m, 8.306→8.299 m, and 13.349→13.339 m. Rural-48 recovered 116 baseline half-meter cells and remains below original footprint. Outputs: `*-recovery/`, `*-dsm/` under Mission III-C. The first full recovery regression passed 203 tests.

Grid provenance traces 10,547/10,555 Rural and 14,700/14,739 Urban sampled raw intersections to different reference grids. Rural two-independent-normal consensus rejected no faces. Urban experiment and downstream validation are next. A separate overlap-aware 80-image run registered 47 cameras, with dense/texture export in progress. Fresh product CLI validation on the fixed Rural photos is also underway.

### Mesh evidence, scaling and product validation

- `661fe54` commits alternate-view fusion, recovered pixel provenance and exact source-grid tracing. The fixed Rural strip probe remains empty both in the fixed-grid candidate and the fresh production run with identical sparse poses.
- Grid normal consensus in `experiment_grid_normal_consensus.py` requires two disjoint supporting stereo pairs and disagreement at all three corners. Rural proposed zero grid faces. Urban flagged one input grid face with a surviving duplicate owner, so neither raw output changed. Urban reserved-view depth agreement was effectively unchanged after excluding successful supplemental views. Retain the previous Urban asset because components increase from 3,525 to 3,681 in the recovery candidate. Rural improves from 2,647 to 2,641 components and is selected. Both new fixed bundles passed four strict Blender imports and eight pixel-identical rerenders.
- Rural-80 keeps the 48 anchors, selected triggers 25–104, registered 47/80 over four PPK clusters (22/11/8/6), and verified 311 pairs. Sparse time 388.60 s, peak 610 MiB. The normal resumed product CLI exports 292,071 dense points, 397,979 optimized triangles and a 32.83% photographic texture. In its own fixed-grid comparison, footprint increases from 76,207 to 77,912 cells and DSM P95 changes 6.724→6.664 m. Its camera solution and alignment differ from Rural-48.
- The fresh `reconstruct-asset` command imported 40 genuine DJI photos and exported 896,742 points plus 437,682 optimized triangles in 143.06 s. Real fresh and expanded GLB/OBJ passed Blender and `verify_uav_workflow.py`. Desktop hidden startup, opt-in selector, 40 DJI inputs and the actual preview GLB loader passed. Interactive navigation was outside this check. Workflow manifest replacement had an intermittent Windows sharing failure in one test; implemented bounded atomic-replace retry and added a deterministic regression test.

- `eb0230a` commits the validated grid-normal and reserved-view checks. `a520746` commits bounded SfM scaling and reproducible overlap-based selection. The final release suite passed 208 tests with 14 deprecation warnings. The full report and local asset paths are under `mission-iii-c/reports/`; `docs/AIRZOO_MISSION_III_C.md` records the method and reproduction commands.
- **IN PROGRESS.** The current recovery accepts new evidence for estimated primary pixels only. It cannot replace a missing primary depth. The main geometric limitation remains crossing layers made by different image grids. The next experiment should select a geometrically compatible local depth sheet before tessellating overlaps, reusing exact source provenance, frozen spatial probes, DSM and common reserved views. Keep recovery opt-in and preserve the selected previous Urban asset.

- `878e9bd` commits the validated normal CLI and desktop integration, the quality verifier and launch documentation. Verified release assets, benchmark table and prior Urban selection are recorded in `mission-iii-c/reports/mission-results.json` and `mission-iii-c/reports/final_report.md`. No existing artifacts were overwritten and no COLMAP baseline was rerun.

## MISSION III-D — FINAL GEOMETRIC RECOVERY AND RELEASE

**IN PROGRESS.** Starting at `dc4f83a`. Existing changes to academic deliverables and capture files are preserved. Experiments use `outputs/airzoo-benchmark/mission-iii-d/`. Mission III-C reports, saved sparse poses, depth rasters and validated assets are the controls. No previous benchmark will be overwritten.

First compare surface generation on frozen Rural and Urban probes. Then advance coverage, desktop usability and final release validation independently if the mesh experiment fails. The current `uav-recovery` profile remains opt-in.

### Surface and coverage experiments

- Ball pivoting removed detected crossings on all 16 frozen Rural and Urban patches, with 28–68% triangle-area retention and more components in every patch. Rejected for production. The patch experiment and a crossing versus indexed shared-edge regression are retained.
- The optional dense-only `--recover-empty-depth` experiment compares alternate estimates at genuinely empty primary pixels. It requires two camera-disjoint conditioned round trips. Rural-48 gained 307 accepted pixels, 281 fused points and 27 half-meter cells with unchanged common-footprint DSM P95. Rural-40, Urban-40 and Rural-80 recovered no empty pixels. Existing mesh profiles remain unchanged. Saved commands and results are under `mission-iii-d/*-empty-depth*` and `rural48-hole-dsm-v2/`.
- The existing Urban quality-PnP hypothesis yielded the same 35/40 cameras, 6,066 landmarks and 0.731/0.948 px median/RMS reprojection error. Retained the prior selection and thresholds. `urban-quality-pnp-command.json` reproduces the ablation without repeating COLMAP.
- Focused geometry checks passed 28 tests and the product-targeted suite passed 60. No previous assets were overwritten.
