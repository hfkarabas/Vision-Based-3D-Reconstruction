# UAV recovery and end-to-end validation

Mission artifacts are under `outputs/airzoo-benchmark/mission-iii-c/`. Previous AirZoo runs remain the references. The local DSM is an offset-aligned elevation reference with an unverified vertical datum. It does not enter stereo recovery.

## Alternate stereo witnesses

The opt-in `uav-recovery` profile starts with conditioned primary maps and spends at most 16 alternate pair attempts. The scheduler ranks witness references using sampled pending primary pixels with reciprocal stereo support. A pair must have sparse overlap and parallax, a feasible rectification and at least 28 px median effective disparity. Pixel-level admission checks both conditioned depths, a coherent bilinear footprint, the original relative depth and return-pixel limits and color agreement. Supplemental pairs must use distinct camera IDs and maintain at least 1° angular separation from the primary reference and within their own baseline. A camera can contribute once per pixel. Recentered candidates still require two witnesses. Qualified primary depth enters the existing voxel fusion, with exact alternate witness reference/source IDs and fused vertex IDs in optional diagnostics.

| Scene | Recovered pixels | Conditioned / recovery fused points | Original / conditioned / recovery DSM P95 | Conditioned / recovery baseline footprint retained |
|---|---:|---:|---:|---:|
| Rural-40 | 8,915 | 953,943 / 962,839 | 32.028 / 11.918 / 11.904 m | 93.22 / 93.22% |
| Urban-40 | 4,611 | 1,746,603 / 1,750,100 | 8.867 / 8.306 / 8.299 m | 99.53 / 99.53% |
| Rural-48 | 4,471 | 1,397,878 / 1,402,248 | 13.323 / 13.349 / 13.339 m | 87.69 / 87.84% |

Rural-48 recovers 116 baseline half-meter cells. Its retained footprint is still smaller than the original. Rural-40 preserves zero points in the previously traced strip probe. Across common cells, the large Rural-40 gain remains attributable mainly to removing unsupported geometry with conditioning. Recovery improves coverage selectively. It provides no guarantee about unobserved regions. Dense-only wall times for conditioning and recovery were 13.77/16.51 s Rural, 17.78/18.55 s Urban and 14.59/18.47 s Rural-48. Peak working sets were 642/649, 937/938 and 773/774 MiB, respectively. These are observed runs with diagnostics, on the same workstation.

Alternate depth is used for corroboration. The fused point remains the primary estimate, so pixels without any usable primary depth remain holes. Footprint counts describe reconstructed horizontal coverage. Total ground area visible to the cameras is unmeasured.

## Remaining intersections

Exact raw face provenance associates 10,547 of 10,555 Rural and 14,700 of 14,739 Urban sampled intersection pairs with different source grids. Of these, 6,856 Rural and 11,368 Urban pairs have unsigned normal angles over 45°. Distinct depth-grid tessellations are the measured source of most sampled intersections. Normal agreement at shared voxels was evaluated before accepting any face removal. Requiring two disjoint stereo pairs to agree on the normal and contradict all three corners of the candidate rejected zero output faces in both scenes. The rule stays an isolated experiment. The sample neighborhoods were ranked independently per mesh and do not establish an intersection-rate reduction over previous outputs.

The Rural recovered asset is selected. Its optimized component count is 2,641 versus 2,647 previously, with 36.17% photographic area. Urban's recovery candidate retains similar reserved-stereo agreement, but optimized components increased from 3,525 to 3,681. The previously validated Urban asset is retained for demonstration. All four candidate GLB/OBJ imports passed Blender. Recorded intersections and holes remain.

## 80-image and production workflows

`scripts/select_airzoo_window.py` kept all 48 prior Rural anchors and compared 33 contiguous windows using GPS step continuity and verified one- and two-frame overlap. It selected triggers 25–104 with 77 of 79 adjacent and 75 of 78 two-step pairs verified. The bounded sparse run at 1,200 px and 4,500 features registered 47 of 80 cameras across four PPK location clusters. It took 388.60 s with a 610 MiB peak working set and verified 311 image pairs. The resumed product workflow took another 38.28 s and exported 292,071 dense points and 397,979 optimized triangles. Its same-grid conditioned comparison had 284,061 points. Recovery increased half-meter cells from 76,207 to 77,912 and changed offset-aligned DSM P95 from 6.724 to 6.664 m. This 80-image comparison uses its own camera solution and alignment. Its P95 is not directly comparable with the fixed or 48-image metrics.

A fresh `reconstruct-asset` run on the original 40 Rural photos used the existing compatible feature and pair caches. It registered 28 cameras, exported 896,742 dense points and 437,682 optimized triangles, and completed in 143.06 s. The production poses match the saved Rural sparse gauge, and its strip probe remains empty. Its automatic voxel size differs from the fixed-grid comparison. `scripts/verify_uav_workflow.py` checked the verified camera graph, finite cloud, mesh topology, UV bounds and desktop preview loading. The fresh and expanded exports passed strict GLB/OBJ Blender imports. A hidden Tk check verified the profile selector, all 40 real DJI inputs and the GLB preview loader. Interactive mouse navigation was not tested.

The full suite passed 208 tests after adding bounded retry for an intermittent Windows manifest replacement failure. The two fixed recovery bundles also passed a repeat Blender check with eight pixel-identical renders.

## Commands

```powershell
# Fresh reconstruction and both asset formats
.venv/Scripts/python.exe -m vision3d.cli reconstruct-asset --input datasets/uav-photos --profile uav-recovery --obj --output outputs/uav-demo

# Resume after sparse reconstruction
.venv/Scripts/python.exe -m vision3d.cli reconstruct-asset --sparse-dir outputs/sparse-run --profile uav-recovery --obj --output outputs/uav-resume

# Validate saved geometry and the app's preview loader
.venv/Scripts/python.exe -B scripts/verify_uav_workflow.py --run outputs/uav-demo --output outputs/uav-demo-check.json

# Start the desktop
.venv/Scripts/python.exe -m vision3d.demo
```

For ablations, use `dense-sequence --profile uav-conditioned --recovery-pairs 16` and reuse the baseline `voxel_size` and `voxel_origin` via the fixed-grid flags. `--depth-diagnostics` writes recovered pixel provenance and, with `--mesh`, the grid faces used for intersection tracing. The saved 80-image and production sparse command arguments are in `rural80-sparse-command.json` and `production-smoke-command.json`. The 16-pair recovery profile remains optional, and city-scale coverage and intersecting layers remain open.
