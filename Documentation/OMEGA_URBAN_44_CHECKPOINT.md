# Urban LOD1 prototype checkpoint

The expanded Urban prototype contains photographic terrain and 44 white LOD1 volumes. Inspection of the saved GLB found 44 distinct building mesh IDs, one terrain mesh and an embedded texture. The saved Blender 5.3 reports record a GLB reimport and a separate BLEND reopen. Of the 44 volumes, 39 have reference DSM-derived height estimates and five use an explicitly marked 3 m display proxy. Two additional reviewed footprints have no volume. This is a reference-assisted product using AirZoo DOM and DSM, with no new UAV-only benchmark result.

## Local deliverables

All paths below are relative to `C:\projects\Vision-Based-3D-Reconstruction`.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `outputs/mission-omega/building-recovery-expanded/urban_white_lod1_expanded.glb` | 66433680 | `bd6b5cb9b30c30937171e00f94612c2255bd0355e0eb27535aa5599ae1f1d7fc` |
| `outputs/mission-omega/building-recovery-expanded/urban_white_lod1_expanded.blend` | 22855826 | `3816a14fdae5a1e4c2aa151baec22a473d667ce44dcae726f742cddf20590c7c` |

The two files remain under the existing `outputs/` ignore rule. **The Git commit and tag record their paths and hashes. Backing up or restoring the actual models requires a separate copy of the files.** This also applies to the ignored replay inputs below.

## Replay from saved inputs

The four files in `scripts/omega_urban_44/` are byte-identical copies of the existing expanded output's `source/` scripts. Their imports use the earlier building recovery helpers in `scripts/`. With the saved inputs and local software present, run from the repository root in PowerShell 7 and supply a fresh output directory.

```powershell
& .\scripts\omega_urban_44\run.ps1 -Project (Get-Location).Path -Output .\outputs\mission-omega\building-recovery-expanded-replay
```

The runner uses Python at `outputs/mission-omega/building-recovery/runtime/venv/Scripts/python.exe` and Blender 5.3. The saved package versions are in `outputs/mission-omega/building-recovery/reports/inference_environment.txt`. [The original recovery notes](OMEGA_URBAN_BUILDING_RECOVERY.md) explain how the 13-volume precursor used GeoAI, PDAL and GDAL. The expanded runner uses its saved inputs, so a Git checkout alone is insufficient for replay. The checkpoint did not rerun any reconstruction or inference.

The input paths in this table are relative to `outputs/mission-omega/building-recovery/`.

| Required input | SHA-256 |
|---|---|
| `delivery/inference_rgb.tif` | `5a87e12c6508672cc555d8fd51fd4f8f20282e4158fcfa85201ab5590d1ed70a` |
| `delivery/survey_dsm.tif` | `418695889f029f30333e45193428054c56918cc20900d4635178a5e80635a80b` |
| `delivery/building_footprints_projected.json` | `b39223943d33660f8952d6a59b30312077638b87866b122dca0acd0d9b3935c9` |
| `delivery/terrain_photo.png` | `ca62d87f74e46a3fc7fc25bc0b3f20bdd66fd43b56ddab0b94fe4dacf0684b67` |
| `delivery/grid.json` | `51ac95161f3653cc002e0b84360f7d3f06975be408338b4666544b436cc00d4f` |
| `delivery/inference.json` | `688191f855babcc0a4a08228bc57f9ccdd8ed06c184ae4f3ea62439df9d54ad2` |
| `delivery/urban_terrain_white_buildings.glb` | `d156c6d74fc227b7b3a363454ffb176ef44fc75a79cd2e58254f250c4674e636` |
| `survey-ground/blender_height_grid.npz` | `cefd3278b2a5e68368588a1527a0d27a7a29cfe3e0e27d0a47bc5e014aacce44` |
| `survey-ground/observed_ground.tif` | `8ee36ead2a409fddb9886946c47fb3cda5d5518d9d1c9a8b02172f64f3b1e678` |
| `reports/blender_buildings.json` | `9235854eef1d50a22106cce8cd6ae1e6de98f9694040005db8b7a1d201d94570` |

The saved expanded prototype reports are `outputs/mission-omega/building-recovery-expanded/reports/recovery.json`, `blender_glb_reimport.json` and `blender_reopen.json` in that report directory. They remain on disk.
