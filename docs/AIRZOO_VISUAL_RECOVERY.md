# AirZoo visual recovery

The audited Rural GLB contains the expected triangles. The large gaps appear in the fused depth and again in the conversion from depth grids to triangles. The release export and the white preview background explain little of the missing structure. Mission III-E retains the release artifacts and keeps the new `uav-coverage` mode opt-in.

## Reproduce the diagnosis

Run from the repository root. The local release manifest supplies the frozen input and saved asset paths. Use a new output directory.

```powershell
$p = 'outputs/airzoo-benchmark/mission-iii-e'
$r = 'outputs/final-release/release-candidate'
$s = (Get-Content "$r/manifest.json" -Raw | ConvertFrom-Json).scenes.rural
.venv/Scripts/python.exe -B scripts/audit_reconstruction_visuals.py `
  --sparse $s.source_sparse_directory --dense $s.source_dense_directory `
  --raw "$($s.source_dense_directory)/mesh.ply" --surface "$r/rural/mesh.ply" `
  --glb "$r/rural/textured.glb" --size 480 --output "$p/rural-audit-replay"
```

The runner casts the same orthographic rays through the raw PLY, optimized PLY and the Open3D-loaded GLB. It undoes the GLB axis rotation and uses the gravity direction inferred from the registered low-roll gimbal observations. The frame contains the complete native dense cloud. The projected dense images retain one closest measured point per pixel. The first-hit GLB renders use either atlas UVs or vertex colors. Two-sided masks include all actual triangle hits. Geometric occupancy, first-hit photographic occupancy, near-background pixels and front-face occupancy are reported separately.

Use the matching `*-release-audit-command.json` and `*-crossed-mvs-command.json` arrays to replay the saved experiments for each scene. `scripts/render_blender_audit.py` provides independently imported GLB renders in the same frame. `scripts/audit_airzoo_capture.py` generates the original-photo sheet and verified overlap graph. `scripts/audit_depth_raster.py` plots representative reciprocal and corroborated depth, implied disparity and source-image reprojections from saved rasters.

## Use the bounded native retry

```powershell
.venv/Scripts/python.exe -m vision3d.cli reconstruct-asset `
  --input datasets/uav-photos --profile uav-coverage --obj --output outputs/uav-coverage-demo
```

In the desktop select **Multi-view → uav-coverage**, **Colored mesh** and **Photo GLB + OBJ**. The profile starts with the same settings as `uav-recovery`. If a stereo pair returns reciprocal samples for less than 5% of its reference canvas, selection continues within the same per-reference attempt budget. A recentered attempt can be scheduled for a weak native pair. If no candidate clears 5%, the best weak map remains eligible. All downstream geometry and corroboration gates still apply.

In the fixed Rural comparison, native pair 25→13 provided 289 reciprocal samples and 51 corroborated samples. The retry selected the recentered version of the same pair with 145,462 reciprocal samples. Overall fixed-grid coverage gained 3,877 half-meter cells with 99.998% baseline cell retention and DSM P95 11.904→11.836 m. Urban gained 21 cells with essentially unchanged P95. The retry is a bounded correction for weak rectified canvases. It does not fill broad missing terrain.

The no-cache Rural-40 CLI run completed with 28 cameras, 932,389 dense points and validated GLB/OBJ exports. The earlier no-cache control's pose file is byte-identical. At the frozen viewport, the largest top region grew from 3,384 to 3,766 pixels. Replay using the saved `rural-coverage-fresh-40-command.json` and a new output path.

## Reproduce the independent depth candidate

The independent replay uses saved project world-to-camera poses and COLMAP geometric depth rasters. It rejects pose or image identity mismatches and checks the resized pinhole model, including COLMAP's half-pixel coordinate convention. A proposed sample must pass two other camera checks with at most 2% relative depth error, a 1.5 px round trip and the existing color gate. At least one witness must have focal length times sine of parallax of at least 28 px. Project voxel fusion and image-grid meshing then create an open surface. The optional footprint-edge experiment lets adjacent, non-grazing triangles exceed the six-voxel limit only when all corners share two witnesses, up to a bounded twelve-voxel cap. It remains an experiment.

The external CLI needs the local COLMAP executable with CUDA and matching PyCOLMAP bindings. It is separate from the CPU reconstruction defaults. [COLMAP model and depth format](https://colmap.github.io/format.html#depth-and-normal-maps) · [COLMAP pixel convention](https://colmap.github.io/faq.html#using-calibration-from-opencv-kalibr-or-other-tools)

```powershell
$b = 'outputs/airzoo-benchmark'
$p = "$b/mission-iii-e"
$env:PYTHONPATH = (Resolve-Path "$b/geometric-root-cause-recovery/colmap-python").Path
.venv/Scripts/python.exe -B scripts/run_patchmatch_candidate.py `
  --sparse "$p/rural-block-fresh/sparse" --grid "$p/rural-block-fresh/dense" `
  --colmap "$b/geometric-root-cause-recovery/colmap-cuda/bin/colmap.exe" `
  --target-faces 980000 --output "$p/rural-block-patchmatch-reproduction"
```

The 28-photo block was selected before running SfM. PPK and gimbal directions scheduled candidate edges, and SIFT matches verified spatially distributed cross-station overlap. The selected photographs form a connected observation graph across six approximate camera stations. They cover a different part of Rural than the fixed 40-photo set. Repeat selection using `scripts/select_airzoo_overlap_block.py --images datasets/airzoo/rural/Scene2/query --count 28 --output <new-json>`. The exact source list and native command are saved in `rural-block-selection.json` and `rural-block-fresh-command.json`.

## Review the candidate

The independent-depth block has a substantially more connected visible core than native MVS on the same 23 cameras. Its offset-aligned DSM residuals are worse and it retains 62.83% of the native half-meter cells, while extending into other cells. These are offset-aligned comparisons in a PPK/gimbal frame with an unverified vertical datum. The result is **partial visual recovery** for a local block. It remains a review candidate.

The [local visual comparison](../outputs/airzoo-benchmark/mission-iii-e/rural-block-review/before-after.png) and [final report](../outputs/airzoo-benchmark/mission-iii-e/reports/final_report.md) identify the actual model and validation reports. Open the GLB through **Open UV GLB**. Keep the OBJ, MTL and atlas together. The native release and all earlier benchmark outputs remain available.
