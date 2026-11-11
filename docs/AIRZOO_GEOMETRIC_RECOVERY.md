# AirZoo dense geometry recovery

The fixed 40-image Rural and Urban comparisons reuse the same sparse poses, intrinsics and original image files. Dense replay matched the recorded point counts, 950,775 Rural and 1,623,418 Urban. Benchmarks and figures are in `outputs/airzoo-benchmark/geometric-root-cause-recovery/`.

## Source of the Rural strip

In the shared gravity-aligned frame, a spatial probe of the detached strip contained 74,411 baseline fused vertices. The raw and corroborated samples in this region trace primarily to four native stereo maps from capture station 2. All sampled accepted pixels in the probe came from pairs 32→31, 31→32, 17→32 and 19→16. Most of their samples had one witness, frequently another depth map from this group. The strip is visible before voxel fusion, and the same shape appears in the raw and optimized meshes. The recentered pairs did contribute elsewhere, but they did not contribute accepted samples in this probe.

Native and recentered projection matrices reconstructed known world points within numerical tolerance. Verified image matches in the weak native pairs still had several-pixel rectified epipolar tails. The audit establishes weakly conditioned and correlated depth estimates as a failure mechanism. It leaves the relative contributions of incorrect pixel matches, sparse pose error and approximate DJI intrinsics unresolved.

## Controlled comparison

The opt-in profile uses 28 px of effective disparity, defined as rectified focal length times stereo baseline divided by depth. The principal-point offset in measured disparity does not enter this calculation. A one-pixel disparity perturbation corresponds to approximately 3.6% first-order relative depth change at the gate. The policy skips pairs whose sparse overlap has a median below the gate and checks both candidate and witness depths. Existing reciprocal disparity, color and round-trip gates still apply. The ablation applied the same depth gate while keeping original pair selection. Fusion used the baseline voxel size per scene.

| Scene and variant | Dense vertices | Baseline XY cells retained | DSM absolute deviation P95 | Absolute deviation >10 m |
|---|---:|---:|---:|---:|
| Rural baseline | 950,775 | 100% | 32.01 m | 17.21% |
| Rural conditioning only | 851,197 | 92.82% | 11.98 m | 9.86% |
| Rural conditioned pairs | 953,171 | 93.17% | 11.93 m | 9.69% |
| Urban baseline | 1,623,418 | 100% | 8.86 m | 3.50% |
| Urban conditioning only | 1,574,189 | 99.50% | 8.35 m | 2.91% |
| Urban conditioned pairs | 1,746,603 | 99.53% | 8.30 m | 2.83% |

Both Rural corrections reduced the fixed strip probe to zero fused vertices. Conditioned source selection recovered additional supported coverage by choosing 31→13 in place of 31→32. Rural still loses some baseline footprint. The fraction of sampled Rural points more than 10 m below the centered DSM rose from 2.66% to 3.46% with conditioned pair selection, even though the large positive tail dropped. Treat this option as a partial correction.

PPK positions and low-roll gimbal observations align the reconstruction to the dataset's EPSG:4547 horizontal frame. A single baseline vertical offset is reused for every variant because the DSM vertical datum is unverified. This comparison measures residuals against the supplied DSM after approximate alignment. Terrain, structures and camera positions can differ from the reference for reasons unrelated to stereo disparity.

## Replay

```powershell
$p='outputs/airzoo-benchmark/geometric-root-cause-recovery'
$s='outputs/airzoo-benchmark/major-sfm-improvement/rural/preinit-bridges/reconstruction'
.venv/Scripts/python.exe -B -m vision3d.cli dense-sequence --sparse-dir $s --profile uav-conditioned --fusion-voxel-size 0.00145692970347358 --depth-diagnostics --mesh --glb --output "$p/rural-conditioned-replay"
```

Use `--profile uav` for the original behavior. Add `--min-effective-disparity 28` to that profile to apply only the pixel-level conditioning gate. Diagnostics are optional and retain bounded sample provenance and full rectified depth rasters. The desktop profile uses automatic voxel sizing unless the CLI comparison explicitly fixes it.

## Coverage and surface checks

On sampled half-meter cells shared by both Rural outputs, the P95 of cell-median absolute DSM residuals changes from 11.95 m to 11.90 m. Most of the larger whole-cloud improvement comes from removing unsupported geometry. The independent 48-image Rural run retains the same 31 registered cameras and uses the same voxel size, but DSM P95 changes from 13.32 m to 13.35 m. The conditioning gain does not generalize to every input.

Same-view Blender 5.3 renders show the detached Rural strip removed. Rural optimized mesh components drop from 2,914 to 2,647, while Urban rises from 3,364 to 3,525. All eight bounded intersection probes per corrected mesh still find intersections. On Urban, first-hit agreement with common reserved stereo improves on three of four tested pairs. Rural has no viable reserved pair under the common-unused-camera rule.

The first corrected Urban GLB and OBJ imported with seven zero corner normals at the same folded vertex fan. Export now splits shading at shared edges with opposing face normals while preserving triangle positions. Both final scenes passed Blender's GLB and OBJ import, atlas, UV, material, face and normal checks. All eight repeat renders were pixel-identical. Photographic area was 36.13% Rural and 22.77% Urban, down from 36.55% and 23.65%. Blender compatibility does not establish Unity compatibility.

## Independent and downstream reproduction

`scripts/audit_airzoo_rectification.py`, `scripts/trace_airzoo_region.py` and `scripts/replay_airzoo_stations.py` use the original sparse runs and project-local copies of their metadata. `scripts/compare_airzoo_dsm.py` uses optional pyproj for alignment to the provided DSM. `scripts/benchmark_airzoo_colmap.py` uses optional PyCOLMAP 4.2.1. It preserves the original camera poses when exporting our sparse model and accounts for the half-pixel principal-point convention in COLMAP. The independent SfM comparison uses the same decoded, resized primary photographs and fixed pinhole values. [COLMAP camera conventions](https://colmap.github.io/faq.html#using-calibration-from-opencv-kalibr-or-other-tools)

```powershell
.venv/Scripts/python.exe -B scripts/benchmark_airzoo_colmap.py --sparse $s --threads 6 --output "$p/rural-colmap-replay"
.venv/Scripts/python.exe -B scripts/benchmark_airzoo_colmap.py --sparse $s --export-project-model --output "$p/rural-colmap-model-replay"
.venv/Scripts/python.exe -B scripts/run_airzoo_colmap_dense.py --colmap "$p/colmap-cuda/bin/colmap.exe" --model "$p/rural-colmap-model-replay" --images "$p/rural-colmap-replay/images" --output "$p/rural-cross-backend-replay"
.venv/Scripts/python.exe -B -m vision3d.cli surface-asset --dense-dir "$p/rural-conditioned-replay" --geometry-weighted-winding --target-faces 450000 --output "$p/rural-surface-replay"
.venv/Scripts/python.exe -B -m vision3d.cli texture-asset --surface-dir "$p/rural-surface-replay" --holdout-stride 7 --refine-visible-faces 40000 --obj --output "$p/rural-texture-replay"
```

Use 900,000 target faces for the comparable Urban run. CUDA COLMAP is a benchmark dependency only. The project dense path remains CPU-based.

The pixel-aligned CPU COLMAP models register 29 Rural and 40 Urban cameras. Their median common-camera relative rotation differences from the project models are 0.68° and 1.38°. The nearly collinear Rural camera centers make a center-only rotation alignment unstable. COLMAP PatchMatch with the original project poses finds zero points in the localized Rural strip probe. It retains 65,692 half-meter XY cells in Rural and 59,352 in Urban, versus 94,887 and 126,128 for conditioned project MVS. This independent result supports the depth-stage diagnosis while showing the effect of different fusion coverage.

The full crossed comparison completed both scenes. The project MVS using COLMAP poses yielded DSM P95 residuals of 12.87 m Rural and 8.56 m Urban. Full COLMAP yielded 12.09 m and 7.23 m. The two MVS pipelines on the COLMAP Rural poses still have P99 residuals of 73.05 m and 80.06 m. Each pose set receives its own PPK/gimbal fit, and the comparison reuses a fixed per-scene DSM offset. Point sampling, footprint, and registration coverage differ across the backends.

`scripts/compare_airzoo_depth_backends.py` compares the sampled project depths directly with same-pose COLMAP depth at the projected pixel. None of 17,258 sampled project depths in the localized Rural strip had valid COLMAP depth at those pixels. The result establishes a lack of independent support in that probe. Across 13 common Rural references, the median per-view share within 5% of valid COLMAP depth was 94.09% before and 94.44% after.

## Raw-mesh intersections and expanded Rural diagnosis

The saved winding-first Rural candidate reloads in Open3D with 445,002 triangles. It passes the original raw-mesh deviation bound, but the optimized surface has 2,674 components instead of 2,647. Blender's direct PLY import also dropped 17 candidate faces and four retained PLY faces. These counts match the duplicate triangle positions at float32 precision. The original optimization order was retained. All four retained textured GLB/OBJ imports passed and all eight replay renders matched their references pixel for pixel.

The bounded raw-mesh audit compares four interior probes per face against saved same-pose COLMAP geometric depths. It checks coherent positive depth, projects using COLMAP's half-pixel convention and requires first-hit visibility for free-space evidence. Candidate removal requires contradiction from two views separated by at least 2°, no agreeing probe in any view, and a supported intersecting alternative. The depth tolerance is the greater of one fusion voxel or 0.5% of local depth. Controls are the nearest face centroids without a detected intersection within the same neighborhoods. These are bounded comparisons using another depth backend on the same source photographs. [COLMAP depth format](https://colmap.github.io/format.html#depth-and-normal-maps) · [Open3D ray casting](https://www.open3d.org/docs/release/tutorial/geometry/ray_casting.html)

| Evidence | Rural | Urban |
|---|---:|---:|
| Intersecting pairs in eight sampled neighborhoods | 10,492 | 15,051 |
| Pairs with neither face supported from two separated views | 10,457 | 14,065 |
| Intersecting faces with any coherent external depth | 3,036 / 11,148 | 13,397 / 14,264 |
| Nearby control faces with any coherent external depth | 1,850 / 7,059 | 7,645 / 8,232 |
| Faces that meet the conservative removal test | 0 | 0 |

The measurements did not support deleting either layer in the sampled pairs. Relative tolerance checks at 0.25% and 1%, using the same one-voxel floor, also yielded zero removal candidates for both scenes. Rural's low depth coverage also appears in nearby controls. Urban has greater depth coverage, but a smaller fraction of intersecting faces than controls have two separated views that agree and see the exact first-hit face. Both corrected raw meshes remain intersecting.

The all-point DSM comparison uses the original per-scene offset and evaluates every fused point. A fixed voxel size alone allows conditioning to move the grid because the default origin is the minimum retained sample coordinate. `--fusion-voxel-origin X Y Z` reuses the exact `voxel_origin` reported by a baseline run, together with its `--fusion-voxel-size`. The original profile and grid defaults were retained.

| Scene | Baseline DSM P95 | Conditioned DSM P95, same grid | Baseline half-meter cells retained |
|---|---:|---:|---:|
| Fixed Rural | 32.028 m | 11.918 m | 93.22% |
| Fixed Urban | 8.867 m | 8.306 m | 99.53% |
| Expanded Rural, 48 images | 13.323 m | 13.349 m | 87.69% |

All 1,397,878 remaining Rural-48 fused points are exact matches to the original output when the grid is fixed. Across the 68,516 common half-meter cells, the P95 of cell-median absolute residuals is 12.134 m in both runs. The retained footprint shrinks from 78,131 cells to 68,516. Comparing unchanged rectified rasters locates the loss upstream of fusion. Of the 11,911 removed accepted depth pixels, 4,374 are below the effective disparity threshold and 7,537 lose a qualifying witness. Their median offset-aligned absolute DSM residuals are 1.28 m and 1.16 m, compared with 4.09 m for the retained pixels. The slightly higher whole-cloud P95 reflects the removal of relatively well-aligned coverage. This comparison does not verify the DSM vertical datum or absolute survey accuracy.

The source evidence is in `*-resume-depth-layer-controls/`, `*-resume-grid-dsm/`, and `rural48-conditioning-loss.json`. Use `scripts/audit_intersecting_depth_layers.py` with the existing same-pose COLMAP workspace. Use `scripts/compare_airzoo_dsm.py --full-resolution` for all-point residuals and exact point membership. Use `scripts/audit_stereo_conditioning_loss.py` to classify lost accepted pixels on unchanged pairs. The new profile remains opt-in.

A bounded Rural-48 probe tested eight alternate stereo pairs for witness references 1, 2 and 47. These were the witnesses found in the sampled baseline provenance for reference 16/source 0, which lost 6,697 pixels through witness conditioning. Three recentered alternatives ran successfully. Together they corroborated 3,340 of the 6,697 lost pixels using the existing 28 px effective disparity, 2% depth and 1.5 px return gates. This is a pixel-level feasibility result on one reference pair. A supplementary-witness reconstruction and its coverage, DSM and intersection effects remain unverified. See `rural48-alternate-witness-probe.json` and its saved replay script in the mission output.
