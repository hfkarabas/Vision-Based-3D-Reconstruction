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
