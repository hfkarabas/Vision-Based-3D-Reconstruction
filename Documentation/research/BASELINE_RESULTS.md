# Independent baseline benchmark checkpoint

Measured 12 November 2026. Full camera sets, 900-pixel dense pilot. No native poses imported.
User visual acceptance remains pending. This is not a completed Mission 0 verdict.

| Measure | Rural | Urban |
|---|---:|---:|
| Registered / original | 190 / 190 | 161 / 161 |
| Sparse points | 63234 | 38215 |
| Mean reprojection error (original px) | 1.339 | 1.3069 |
| Fused points | 1577923 | 1292342 |
| Median per-map valid depth fraction | 0.5407 | 0.5207 |
| Median distinct supporting images | 5.0 | 5.0 |
| Occupied 0.5 m XY cell area (not valid completeness) | 214578.75 | 165865.75 |
| Delaunay faces / components | 473578 / 15 | 494205 / 69 |
| Poisson faces / components | 404968 / 12 | 620132 / 37 |
| Poisson largest component fraction | 0.99944 | 0.93869 |
| Poisson boundary / nonmanifold edges | 7578 / 0 | 22062 / 0 |
| Poisson positive local intersection probes / 8 | 4 | 3 |
| Poisson nominal photographic UV area fraction | 0.98966 | 0.96674 |
| Fused-cloud centred DSM absolute median / P95 (m) | 0.178 / 1.688 | 0.276 / 25.834 |
| Fused-cloud centred DSM RMSE (m) | 6.593 | 11.903 |
| 1 m upper-envelope DSM P95, all occupied cells (m) | 2.555 | 6.386 |
| Stereo native log minutes (shared GPU) | 43.753 | 40.83 |

## Interpretation and limits

Rural Poisson visibly recovers fields, roads and low buildings; Urban recovers roads, gardens and high-rise structures but retains important holes and disconnected fragments. Raw Delaunay is rejected as a usable whole asset because of peripheral stretched/extrapolated faces. Poisson is the preferred review candidate, not a certified watertight or intersection-free environment. Both raw methods are preserved.

Poisson has positive bounded self-intersection probes (4 Rural, 3 Urban); zero nonmanifold edges does not prove intersection freedom. Each probe examines at most 12000 faces in one of eight dense neighborhoods. Negative probes are not global certificates. Missing stored PLY normals are reported separately from invalid computed normals.

Nominal photo UV assignment is not independent texture correctness. Total mesh area is not supported reconstructed area; the latter remains unmeasured. Occupied XY cells can include outliers and overlapping vertical surfaces. No 3D precision/completeness/F-score or absolute geospatial accuracy is established.

DSM figures use all fused points inside a conservative DOM mask, a similarity fitted to camera RTK positions and a fitted median vertical offset. They are height-reference agreement only. Urban has a serious tail and Rural has rare extreme outliers; do not hide these behind the median. DSM lineage, vertical datum and the Urban DOM CRS conflict remain unresolved. Historical native DSM results use a different mask/input set and cannot establish superiority.

A facade can correctly lie below a rooftop DSM. The separately reported fixed 1 m upper-envelope diagnostic reduces Urban P95 from 25.834 to 6.386 m over 269220 occupied cells, consistent with a representation effect as well as outliers; it does not prove the entire tail was erroneous geometry. Minimum-three-point cells have P95 4.142 m but cover only 154291 cells, so this must not replace the complete result. Rural upper-envelope P95 is 2.555 m over 393868 cells (1.468 m over 167431 minimum-three-point cells). Each diagnostic fits its own offset and is sensitive to high outliers. No metric is absolute accuracy.

The fusion cache_size argument is 2 GB but installed use_cache defaults false; this is not a hard memory cap. Measured per-process peak RSS and whole-device GPU samples are in JSON. Stereo ran concurrently; downstream Urban was queued for memory and resumed. No observer remains suspended.

COLMAP 4.2.1 produced SfM, depth, fusion, both surfaces and photographic textures. The project adapter only serializes existing vertices/faces/UVs to OBJ, omitting unreferenced vertices. Blender 5.3 independently imported OBJ, exported/reimported GLB and verified bounds, faces, UV and image binding. All four native validation invocations returned 0; initial failed import-bounds evidence is retained separately.

## Remaining gates

- Explicit user inspection/acceptance of both full montages.
- Matched-input native/backend experiment and common-footprint comparison.
- OpenMVS comparison (unavailable locally; not performed).
- Supported-area and spatial-hole validation with independent reference geometry.
- Literature expansion: 44 screened records, not 200–300; stronger novelty claim prohibited.
- Local commit: .git is read-only in this session; git add failed creating index.lock. No commit or push occurred.

See Documentation/RESEARCH_FOUNDATION.md and the five companion research documents for decisions and replay.
