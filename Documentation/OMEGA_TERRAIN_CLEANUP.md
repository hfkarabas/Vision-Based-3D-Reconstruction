# Urban terrain cleanup and transfer validation

Research topic: **Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon**.

This work is an engineering repair to a reference-assisted product. It preserves the protected 44-volume release and the negative result of the adaptive experiment. No new budget-matched adaptive comparison or independent absolute elevation assessment was run.

## Reproduce the Urban product

From the repository in PowerShell 7, run:

```powershell
& .\scripts\run_omega_terrain_cleanup.ps1 -Output .\outputs\mission-omega\terrain-cleanup-replay
```

The target must be a new directory. The runner uses the saved 44-volume footprint data, its native 1 m grid, the aligned reference DSM/DOM, PDAL 2.10.2, the existing GIS/GeoAI Python environment and Blender 5.3. The original GeoAI model and camera reconstruction remain available in the previous output. The main runner does not recompute either.

For another survey, use `omega_terrain_cleanup.py classify` on its bounded north-up metre-based DSM, then `clean` with its current DTM or matching height grid and the resulting classification directory. Supply reviewed polygon masks and an explicit projected CRS when regional processing is appropriate. Leave masks out for a generic shape-screened run. Select the input role explicitly and keep reference products out of scientific geometry. Use `omega_terrain_blender.py terrain-only` with the aligned RGB and projected-origin manifest. The `compare` command fixes the cameras and lighting for two asset imports. The raw Urban input paths, inherited baseline hashes and generation chain are recorded in `reports/input_provenance.json` inside the output.

## Implementation

- `scripts/omega_terrain_cleanup.py` builds bounded PDAL SMRF and CSF recipes. It uses Rasterio/GDAL FillNodata on their common ground, retaining masks for source support and uncertainty. A distance-transform-weighted transition applies only to the selected local edit fringe. Proposals that worsen steep edges in the surrounding ROI are rejected. The court uses scikit-learn RANSAC regression on a reviewed interior with separate held-out cells. The script also prepares footprints and cross-matches clipped GeoAI candidates.
- `scripts/omega_terrain_blender.py` reuses the established BMesh terrain and building export. Its foundation adjustment uses native subdivision and Blender BVH ray checks. Unsupported trials retain the existing flat foundation. The script also renders matched views and validates terrain-only exports.
- `configs/omega_terrain_urban_regions.json` holds the reviewed image regions, one small roof and the region-specific taper widths. `configs/omega_terrain_urban_views.json` stores inspection views. The core tools accept other projected grids and masks.
- `scripts/run_omega_terrain_cleanup.ps1` replays the Urban path. `tests/unit/test_omega_terrain_cleanup.py` checks supported sloped planes, abstention and the hard fill radius.

## Measured outcome

The cleaned 485 × 574 m Urban asset retains 556,042 valid terrain triangles, the original textured extent and all 44 original building IDs. All 39 originally supported roof elevations remain fixed. An image-reviewed frontage roof with an estimated 3.21 m DSM-to-ground height adds one volume. The total is 40 estimated and five provisional, with two older footprints still lacking heights. Terrain-following foundations passed on 43 of 45 volumes. The worst sampled burial decreased from 11.62 to 6.43 m. The 6.43 m case and one other retain flat foundations. Both require review if shallow foundation fidelity is important.

The raised blue court was already near 27 m in the reference DSM. The independently reconstructed surface returns have a 26.38 m median, in their own unverified vertical reference. The erroneous mound is present in the expanded input height grid. A classification support gap and subsequent filling bridged lower surroundings across the court. RANSAC fitted a 2.07% slope with P95 error of 0.44 m on 48 held-out DSM cells, versus 9.00 m for the old DTM at those same cells. This is held-out agreement within one source DSM. The platform elevation remains raised and the terrain texture stretches across its steep perimeter.

At the east road ROI, adjacent jumps over 1 m/m fell from 480 to 318 and maximum from 9.47 to 6.28 m/m. The west road went from 78 to 52 jumps with the maximum unchanged. Some road spikes remain. For the selected vegetation patches, the maximum gradient decreased from 8.01 to 5.13 m/m while the count over 1 m/m moved from 475 to 476. The forest canopy can still obscure the ground. The first direct replacement introduced sharp seams. A strict edge veto removed all road and vegetation changes. The selected bounded taper substantially reduced the measured maxima while preserving the separate slope control.

Of 279,186 valid Urban cells, 262,757 were unchanged exactly. All original NoData cells and a 1.91% graded terrain control were preserved. Numerical reductions measure geometry on the frozen grid, with no absolute accuracy conclusion. The Urban scientific strict surface is a separate intersection of two ground classifications on the cached COLMAP-derived surface. It uses no AirZoo reference elevation.

## Verification

Clean Blender processes imported the Urban, Rural and independent survey GLBs. Urban GLB reimport retained the 45 white closed solids and embedded orthophoto. The saved Urban BLEND was reopened and its oblique PNG matched the original pixel-for-pixel. The fixed-frame before/after render series includes the entire Urban extent, the court, roads, vegetation, shops and boundary areas. Human review is pending. The project regression suite passed 221 tests. Three optional GIS tests skipped there and passed in the installed GIS environment. All outputs remain ignored and must be backed up separately.

Rural retained its original NoData extent and reduced adjacent jumps above 1 m/m from 3,703 to 2,079. The same generic settings ran on two open GeoNadir surveys. The small Morgan City sample reduced the jump count from 223 to 112, but its false-color texture and limited exposed ground leave the surface interpretation uncertain. In the independent Argyle2 forest survey, the proposal failed the transition gate and the baseline stayed intact. These runs establish the ability to process different datasets and also show why a lower roughness statistic alone cannot certify good terrain. Dataset attribution and replay inputs are recorded in the output's `reports/cross_dataset_validation.md`.

## Generated asset hashes

Paths are relative to the repository root. The commit does not carry the asset binaries.

| Artifact | SHA-256 |
|---|---|
| `outputs/mission-omega/terrain-cleanup/asset/urban_terrain_cleaned_white_buildings.glb` | `ec4915e312973a004fba1c66938332086faf6a8edba20c56f82d8440578a5d6e` |
| `outputs/mission-omega/terrain-cleanup/asset/urban_terrain_cleaned_white_buildings.blend` | `649a6cb259a24def98aa91aa2d5b53b5fb75d41be3e1e576a334f8fd824f0bba` |
