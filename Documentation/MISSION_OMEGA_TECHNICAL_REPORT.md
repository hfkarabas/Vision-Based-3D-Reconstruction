# Mission Omega technical report — verified partial delivery

**Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon**

## Result and engineering reuse

The principal usable output is `outputs/mission-omega/asset/rural_terrain_textured.glb`, a real photo-textured, height-derived terrain. It is 17,631,284 bytes, contains 499,044 triangles and two embedded textures, and has a metric bounding size of [631.0, 398.00006103515625, 15.158441543579102]. It was reimported into a clean Blender scene, rendered from top/oblique/low angle and inspected by the agent. Human visual acceptance remains PENDING. This is not project completion.

COLMAP supplies the previously computed independent SfM/MVS, new bounded depth maps/fusion, and photo texturing. PDAL supplies ELM/statistical outlier filtering, SMRF/CSF ground classification, MAD screening and rasterization. GDAL/rasterio supply bounded interpolation, CRS/raster adapters and fixed-grid merging. QGIS/Qgis2threejs supplies the existing height-field mesher/UV/scene generator. Three.js and Blender supply official file export and rendering. New code consists of command scheduling, file/CRS adapters, quality measurements, manifests and QA. No feature matcher, MVS, DTM filtering engine, triangulator, UV algorithm or renderer was implemented. See dependency and audit reports.

ODM did not run: its Docker runtime is absent and WSL reports not installed. `reports/stages/official-odm-start-blocked.json` records an actual blocked command. QGIS native/embedded export APIs were incompatible; upstream standalone Qgis2threejs scene export plus unmodified Three.js GLTFExporter worked. Native Blender OBJ image paths required a small output-path normalization adapter and were independently reimported afterward.

## Data lineage and geometry limits

Rural/Urban originals are 190/161 genuine DJI RGB JPGs, 4032×3024, independently rehashed. Frozen SfM cameras and 900-pixel dense clouds were reused. No supplied reference DSM/DOM, external DEM/LiDAR or OSM footprint contributes geometry. Photo texturing uses only those original-image-derived undistorted photos. A geospatial orthographic emission render of that textured DTM is draped in QGIS; this is **not ODM orthophoto or DSM building-correct true ortho**. Occluded ground texture is a projection/inference, not observed bare-earth appearance.

EPSG:4547 is confirmed by supplied metadata, with explicit easting/northing metre order. WGS84 XMP camera positions were projected and a historical camera-RTK in-sample similarity was reused. The realization between RTK WGS84 and CGCS2000 and the vertical datum remain unresolved. MRK Ellh/XMP RtkAlt are not silently converted to orthometric height. Blender uses X=east,Y=north,Z=up and origin [403922.0, 3130096.5, 0]; GLB is Y-up. Exact matrices and source camera parameters are in geospatial provenance and inherited SfM metrics. Z scale is exactly 1. Float64 matrix composition avoids large-coordinate float32 cancellation.

The selected initial ROI is the camera-position envelope plus a fixed 50 m buffer, bounds [403606, 3129897, 404238, 3130296]. It was not selected from an attractive rendered crop. Strict DTM has 235.496 supported 1 m cells (93,3885% of 252.168 m²). A separate production mask rejects 138 local quality outliers and GDAL fills at most 8 m: 15.285 interpolated cells, 250.643 total surface cells, 1.525 remaining missing cells. Interpolation and rejection masks persist. Those counts do not certify true ground beneath vegetation/buildings. The production height range is [6.837867736816406, 21.996334075927734]; the maximum adjacent-row jump remains 8.126 m and requires inspection/uncertainty, rather than being hidden by height exaggeration.

Topology: one connected Rural surface after analysis-only seam welding at 0,1 mm, zero degenerate triangles, zero nonmanifold edges. 399 shared seam samples have maximum height disagreement 0.0 m. The open outer and NoData boundaries are intentional; watertightness is not expected. Original deliverable geometry is not welded/repaired by this QA.

## Larger coverage and Urban failure evidence

Expanded Rural uses the same 190 images and cached dense cloud with a predeclared camera-envelope +200 m ROI: [403456, 3129747, 404388, 3130446], 651.468 m². Strict raster support is 495.808 m²; the separate 8 m-fill product has 575.992 m² valid/interpolated cells and 75.476 m² missing. It exports to 40,444,748 bytes / 1,145,919 triangles. There are 8 components, with 99.5615% of surface area in the main component; seam deviation is 0.0 m. Peripheral texture/geometry holes remain visible in all full-extent previews, so larger/global coverage is only PARTIAL. This is not a hidden replacement of the better first asset.

Urban SMRF, CSF and two bounded production interpolation variants were actually exercised. CSF strict support is 117.437 / 279.450 m². The 64 m production variant rejects 150 quality cells, adds 121.654 interpolated cells and reaches 238.941 cells; its range is [11.696341514587402, 32.498714447021484]. The roughly 44% interpolated footprint and remaining approximately 12 m abrupt transitions make it unsuitable for claiming measured urban bare earth. Full Blender views were inspected and the Urban model is **NOT ACCEPTED**. The diagnostic is preserved, with no invented buildings. A public Overpass request was blocked before transmission by automatic approval review; separate consent is pending. No local attributable footprint dataset was found. Matched footprints, measured heights, estimated heights and white building objects are all zero.

Complete input footprint is not independently certified: oblique images see beyond camera positions and the delivered ROI. Raw cloud bounds, camera extents, strict masks and all missing areas are retained. No claim of full-scene completeness is made.

## Controlled real experiment and negative result

The frozen initial 900-pixel reconstruction is a common starting state, not a zero-cost reconstruction. All arms use the same full 252.168 m² ROI, 1 m raster grid, same calibration/CRS and cache policy. Shared 1200-pixel undistortion took 76,916 s; its cost is separate. Each arm actually produced eight full 1200×900 geometric depth maps, with two passes, three iterations, ten samples and three source images per reference: 8.640.000 allocated reference pixels / 51.840.000 allocated pixel-iterations. Complete image work is charged; only the merge is tile-restricted. Wall time is **measured, not equal by definition**. This is matched work, not a strict equal-seconds trial.

Tiles are a fixed 4×4 grid. Score = 0,7 NoData fraction + 0,3 min(P95 adjacent DTM difference / 2 m,1). Sparse-track support supplies eligible images/regions, not evaluation reference. At least 100 sparse points/four registered views are required. Existing high gradients can be real terrain, so this score is a proxy with known confounding. No aesthetic low-color score or test DSM is used. New ground rasters fill holes only through rasterio/GDAL merge; existing valid heights do not change. Thus zero common-support height difference is guaranteed by policy, not a precision improvement.

| Arm | Actual full maps | Additional supported cells (m²) | Final support % | Measured incremental stage s | Peak process-tree RSS MB |
|---|---:|---:|---:|---:|---:|
| B0_uniform | 8 | 62 | 93.413121 | 112.014 | 885.26 |
| B1_random_seed42 | 8 | 3 | 93.389724 | 106.835 | 884.94 |
| B2_feedback | 8 | 0 | 93.388535 | 105.227 | 886.57 |
| B2_feedback_revisit | 8 | 0 | 93.388535 | 111.416 | 884.84 |
| B3_support_proxy | 8 | 0 | 93.388535 | 103.100 | 886.85 |

Uniform is the strongest tested control. Feedback added zero cells and did not beat it. Uniform also increases the >3 m row-jump count from 57 to 63, so extra support alone is not verified quality gain. The original no-revisit loop recomputes feedback but is rank-equivalent to a frozen score on disjoint unused tiles; this limitation is disclosed. An exploratory second candidate permits revisiting the highest-current-defect tile with the next four unused support-ranked views; it also adds zero cells. B3 is only a sparse-support proxy inspired by preliminary geometry/view reasoning, **not exact Mostegel confidence or RQF reproduction**.

**H1 NOT SUPPORTED by this pilot.** This is not proof that all quality-feedback methods fail. One scene, two actions/arm, one random seed, proxies rather than independent bare-earth truth, and interactive desktop hardware prevent a general accuracy/variance claim. Selector overhead and manual preparation are not fully timed; per-stage costs and common costs are separated. No absolute height RMSE/MAE is reported: a DTM-to-DSM comparison would confound buildings/trees, and supplied reference lineage/datum is unresolved. No publishable novelty is demonstrated.

## Validation and acceptance

Unity 6000.6.1f1 with official glTFast 6.21.0 actually imported/instantiated the model: 499,044 triangles, two textures, size {'x': 631.0, 'y': 15.158515930175781, 'z': 398.00006103515625}. Import took 31,513 s, sampled peak process-tree RSS 2.935.427.072 bytes. This does not verify standalone runtime/player or Unity rendered material appearance. Existing desktop GLB loader and native Blender OBJ/MTL/image reimport both succeeded; no GUI rewrite was needed.

| Gate | Status | Evidence/limit |
|---|---|---|
| G0 | PASS | Real image hashes/tools/hardware/metadata checked; vertical datum explicitly unresolved |
| G1 | PARTIAL | Actual UAV-derived DTM/DSM and georeferenced photo-DTM projection exist; ODM unavailable, DSM outliers and independent vertical accuracy unresolved; not an ODM true orthophoto |
| G2 | PASS | Rural only: native Blender import, three inspected full-extent views, one connected surface, embedded textures; user acceptance PENDING |
| G3 | BLOCKED | No footprint data obtained or invented; GPS-bound Overpass egress rejected by auto-review; explicit separate consent pending; Urban ground remains uncertain |
| G4 | PARTIAL | 651468 m² larger ROI exported/inspected; 575992 m² supported+interpolated; peripheral holes, texture defects and 8 components remain; whole input footprint not certified |
| G5 | PASS | Five real measured fixed-work arms, 8 complete 1200x900 maps each; H1 NOT SUPPORTED in this pilot; no independent accuracy/generalization claim |
| G6 | PASS | Portable GLB and relative-path OBJ/MTL/images verified in Blender; actual Unity Editor glTFast import passed; standalone runtime/player and Unity rendered appearance NOT VERIFIED |

Machine: Intel Core Ultra 5 245KF (registry verified), 14 logical CPUs, 16.782.508.032 bytes RAM, RTX 5070 12.227 MiB VRAM, driver 617.42. Current free resources and exact software build/archives are in environment evidence. Jobs use two CPU threads and bounded caches. Per-stage RSS is sampled process-tree RAM, not peak GPU VRAM; no unmeasured VRAM or full end-to-end time is invented. Git was not written; unrelated historical changes remain intact. The local ZIP is a review package, not public redistribution authorization.
