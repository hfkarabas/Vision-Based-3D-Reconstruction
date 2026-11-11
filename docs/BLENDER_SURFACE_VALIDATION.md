# Blender validation for AirZoo surface assets

Blender is the active downstream compatibility check. It exercises an independent importer and renderer. Blender results provide no Unity import or runtime evidence.

Run from the repository root with the local Blender executable. The reference directory contains accepted Blender reports and renders produced with the same version, settings and image size. Omit `--reference` when creating a new reference. Keep reference renders outside Git with the benchmark outputs.

```powershell
$p='outputs/airzoo-benchmark/major-surface-improvement'
$blender='C:/Program Files/Blender Foundation/Blender 5.3/blender.exe'
.venv/Scripts/python.exe -B scripts/validate_blender_assets.py --blender $blender --rural "$p/rural-visible-refinement" --urban "$p/urban-visible-refinement" --reference "$p/blender-refined" --output "$p/blender-replay-new"
```

The driver launches Blender in background with factory settings once per format and scene. The `bpy` inspector verifies triangle counts, photographed surface area, UV bounds and nondegeneracy, imported corner normals, and material bindings. It hashes the image bytes that Blender actually loaded and compares them with the source atlas. For topology inspection, it collapses exact-position vertex splits in an analysis array. This avoids counting UV or shading seams as separate geometry. It also runs bounded nonadjacent BVH overlap probes in eight compact neighborhoods.

The inspector writes top and oblique CPU Cycles renders with transparent backgrounds. It connects each imported base-color source to an emission shader for the regression views so lighting cannot hide a material or UV defect. Cross-format silhouettes must have at least 0.995 intersection over union. Same-format replay requires 0.999 silhouette overlap, foreground mean absolute RGB difference of at most 0.25 intensity units, and at most 0.5% of shared foreground pixels changing by more than four units. The original fallback colors differ between GLB and OBJ, so cross-format color differences are recorded for diagnosis.

Original assets lost five Rural and ten Urban duplicate-position faces in each format and produced 12 and 24 invalid Blender corner normals, respectively. The corrected weighted and visibility-refined exports passed strict Blender 5.3.0 import checks. These checks accept open boundaries. The per-asset reports and local overlap measurements still identify unresolved intersecting geometry.

## Fixed-subset results

| Metric | Rural previous → current | Urban previous → current |
|---|---:|---:|
| Edge-connected components measured in Blender after exact-position grouping | 3,699 → 2,914 | 6,123 → 3,363 |
| Imported triangles | 434,640 → 627,303 | 848,136 → 1,067,951 |
| Invalid corner normals per format | 12 → 0 | 24 → 0 |
| Photographic area | 32.62% → 36.55% | 21.73% → 23.65% |
| Textured GLB size | 22.95 → 43.05 MB | 37.06 → 69.90 MB |

The two formats had identical top and oblique silhouettes per scene. All eight same-format repeat images matched pixel-for-pixel. The full suite passed 186 tests. Weighted winding reduced P99 input-to-output sample displacement to 0.000697 and 0.000844 SfM units using the existing bounds. The visibility-refined exports add triangles by partitioning the original planes. Their overlap pair counts depend on that tessellation. All eight compact neighborhoods in each scene still show overlaps. The camera-supported pruning experiment found one safely removable face in Rural and none in Urban, so pruning remains diagnostic.

To reconstruct the measured assets, run `surface-asset` on each Mission II dense directory with `--geometry-weighted-winding` and the existing Rural 450,000 or Urban 900,000 face target. Then run `texture-asset` on each output with `--holdout-stride 7 --refine-visible-faces 40000 --obj`. The archived results and exact commands are under `outputs/airzoo-benchmark/major-surface-improvement/`.
