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
