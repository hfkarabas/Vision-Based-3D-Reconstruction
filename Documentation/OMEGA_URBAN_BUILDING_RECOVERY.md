# Urban automatic building reconstruction

The graduation research topic remains **Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon**. The Urban LOD1 scene is a separate product experiment. The previous UAV-only pipeline, G0–G6 findings and the negative result for the adaptive hypothesis retain their original inputs and evaluation.

## Evidence and input roles

The Urban Scene1 DOM is the photographic aerial orthophoto distributed with AirZoo-Real. The companion DSM represents surface height. These are **reference products** for the UAV-only benchmark. This delivery uses both as explicitly declared product inputs. Neither contributes new input or scores to the scientific baseline. The prior COLMAP-derived photo projection onto the DTM has displacement and voids around tall roofs. It was inspected as a diagnostic and excluded from segmentation and validation of roof boundaries.

The DOM TIFF labels its CRS EPSG:4326, while the DSM has no CRS tag. The two rasters have the same 28,349 × 18,324 grid and 0.05270 m pixel spacing. Their world files and pixel coordinates agree, including the south-to-north row direction. Projected coordinate magnitudes, the supplied XML and the official dataset metadata support EPSG:4547. Rasterio/GDAL applies that CRS in memory and resamples the overlapping crop north-up at approximately 0.30 m. The input files retain their original tags. Model inference, the polygon overlay and the terrain texture use that same crop and affine. No independent horizontal control points or absolute vertical datum were established.

## Why the first test yielded one candidate

All trials reuse the same 93 vector candidates. The inference inputs in the height trials were checked for identical decoded RGB, dimensions, affine and CRS.

| First gate failed | Initial reconstructed DTM + reference DSM | Repeated with observed-ground pairing | Reconstructed DTM + reconstructed DSM | Reference-derived terrain and DSM |
|---|---:|---:|---:|---:|
| Insufficient terrain support within footprint | 38 | 38 | 38 | 1 |
| Unstable ground or inconsistent local height offset | 16 | 14 | 9 | 21 |
| Insufficient positive roof relief | 31 | 32 | 34 | 33 |
| Insufficient roof/ground samples or paired ground | 0 | 1 | 3 | 0 |
| Footprint clipped by crop | 7 | 7 | 7 | 7 |
| Eligible for mesh checks | 1 | 1 | 2 | 31 |
| Passed Blender solid/foundation checks | Initial curve failed | 1 | 0 | 17 |
| Released after image review | — | Diagnostic only | — | 13 |

The repeated first trial requires paired annulus ground to come from the original support mask as well as the valid DTM. It changed which rejection surfaced first for three candidates. The one stable result was Building_0090, an estimated 3.14 m. The two candidates from reconstructed elevations failed actual mesh tests. Building_0016 exceeded the 4 m foundation burial allowance. Building_0018 had inadequate interior support or clearance. That experiment still used reference DOM imagery, so it is ineligible for a UAV-only benchmark.

Terrain coverage is a demonstrated failure mechanism, independent of which DSM supplied roof estimates. The original production DTM covers 79.84% of the 1 m grid, with gaps around some tall buildings. Geometric agreement of the source grids supports the CRS correction. The median elevation difference between classified reference ground and the original DTM on 153,250 overlapping cells was +0.764 m, with P10 and P90 of −0.249 and +1.162 m. This is a ground-cell comparison with uncertain vertical datum. Roof and ground inference used paired local samples and a local offset. A global datum shift cannot resolve gaps or restore roof relief. Several roof detections actually correspond to parking or painted ground, where the near-zero DSM relief is expected. Height and foundation rejection thresholds were retained.

## Product construction and quality

PDAL 2.10.2 SMRF classified the 1 m reference DSM. Ground candidates within 1 m of the detected roofs were excluded. GDAL FillNodata interpolated at most 12 m from classified ground. Of 279,450 cells, 175,745 are classified ground samples, 103,422 are interpolated and 283 remain NoData. The terrain grid is a new product artifact. Areas under buildings are interpolated where supported. Some trees and roof remnants remain visible in the DSM-derived terrain.

GeoAI orthogonalizes the detected outlines. A 2 cm topology-preserving simplification removes numerical and collinear detail. GEOS constrained triangulation retains interior holes. Blender extrudes the resulting planar face regions with its native BMesh operation. The preliminary native Curve extrusion had zero-area faces on actual raster-derived outlines. The triangulated-face implementation passed closed-solid and area checks on the accepted footprints.

Blender checks every candidate's projected foundation against the actual rendered terrain using BVH rays. The release's 13 solids passed 1,091 boundary and 850 interior tests. Maximum sampled burial at the boundary is 3.93 m and minimum sampled roof clearance is 1.51 m. The flat base extends 0.10 m below the lowest sampled boundary point. This base is a terrain-contact approximation on slopes. Projected area derived from each native prism's volume and height agrees with its input footprint to within 0.00022% relative. The delivered geometry measures 485 × 574 m across the terrain grid, with 555,958 terrain triangles and a packed 1,620 × 1,917 orthophoto. DSM-derived height estimates run from 2.06 to 53.24 m. These are LOD1 estimates without independently surveyed building heights.

The 17 geometry-passing outlines all overlap visible roof surfaces in the image review. Four high-rise roof fragments were excluded from the GLB after close inspection. The main parking lot supplied clear false positives Building_0051, Building_0053 and Building_0056. Painted play surfaces supplied Building_0085 and Building_0086. Some central and southern cross-shaped towers were missed or incompletely traced. All 93 vectors and their disposition remain in the detection layer. The visible false-positive examples are a lower bound. There is no independent building inventory from which to calculate precision or recall. The delivered volumes approximate roof projections and include a greenhouse-like structure.

## Model and software

- [GeoAI 0.43.1](https://pypi.org/project/geoai-py/0.43.1/) and its [official WHU example](https://opengeoai.org/examples/building_detection_whu/) supplied tiled inference and orthogonalization. GeoAI source is MIT licensed. The local inference environment uses Python 3.12.14, PyTorch 2.10.0+cu128, CUDA 12.8, TorchVision 0.25.0+cu128, segmentation-models-pytorch 0.6.0, timm 1.0.31, Rasterio 1.5.2, GeoPandas 1.2.0, Shapely 2.2.0 and PyProj 3.8.0.
- The [EfficientNet-B4 / UNet++ checkpoint](https://huggingface.co/giswqs/whu-building-unetplusplus-efficientnet-b4/tree/09df9efd323bbd3d56b98b4857129b5baa2d3) is Apache-2.0 per its published model card. Revision is `09df9efd323bbd3d56b98b4857129eb9b5baa2d3`. The SHA-256 of `model.pth` is `922af7c96c0dc44256ab8b4d1a071f2151e0a921c997af80b55bb766bcc30dc6`. The official WHU sample crop yielded 27 polygons in 21.54 s. Urban yielded 93 candidates in 22.12 s, with 425 MiB peak allocated CUDA memory on an RTX 5070. These runtimes include polygon generation.
- [AirZoo-Real](https://huggingface.co/datasets/RingoWRW97/AirZoo-Real) supplied the already local aerial rasters. Inference used a revision-pinned local cache and was offline. No footprint service or external inference endpoint was used.
- [PDAL SMRF](https://pdal.io/en/stable/stages/filters.smrf.html), GDAL and Blender 5.3 supplied ground classification, height-grid interpolation, native extrusion, GLB export and reload. No prior QGIS 4 headless run was repeated.

## Reproduce or change datasets

Run from the repository root with PowerShell 7. Set the Blender user resources folder to a writable path if needed.

```powershell
& .\scripts\run_omega_building_recovery.ps1 -Config .\configs\omega_urban_building_recovery.json -Output .\outputs\mission-omega\building-recovery\replay-urban
```

The output directory must be new. The config pins explicit roles for the three geospatial inputs. The run verifies each cached RGB block against the actual target grid before reusing segmentation. For new imagery, remove `reuse_inference` and set `hub_cache` to the local Hub cache with the verified snapshot. Use `existing_terrain` with a compatible terrain GLB and its CRS/origin manifest. Set `require_reconstruction_only` for runs that must refuse reference products. Survey-assisted mode must be selected explicitly and writes a separate terrain and interpolation-distance rasters.

Inputs are an 8-bit RGB orthophoto, a surface model and a metre-based projected DTM with a ground-support mask. Use verified co-registration and adequate resolution to resolve roof boundaries. The tested inference scale is approximately 0.30 m per pixel. This stage consumes orthophotos and heights produced upstream. It has no independently established minimum raw-image overlap or camera calibration tolerance. Ground and roof support, stable local offsets and valid terrain rays are required. The reusable workflow refuses to export building geometry when there are no valid candidates. Optional manual exclusions affect the released volumes and are recorded with their reasons.

The installation used a separate Python 3.12 environment. Use the recorded package freeze and the official PyTorch CUDA 12.8 wheel index to reproduce its dependencies. The released GeoAI package imports Lightning, Leafmap and rioxarray while loading the two APIs used here. The offline Hub cache has a `refs/main` entry containing the pinned revision and a `snapshots/<revision>/` directory containing the matching `model.pth` and `config.json`. Verify the checkpoint hash before reuse. A CUDA-capable Blackwell GPU ran the sample. A CPU mode exists, with no measured CPU throughput for this dataset.

## Verification and remaining limitations

- A fresh wrapper run reused the saved mask after exact RGB-grid comparison. It exported 13 independent building mesh nodes. Blender 5.3 reopened the GLB and checked white matte materials, closed geometry, elevation bounds and the embedded photographic texture. A separate Blender process opened the published BLEND and reproduced the 1,400 × 1,120 oblique render with identical decoded RGB pixels.
- The full reconstruction regression suite passed 221 tests, with 14 dependency deprecation warnings. The first attempt could not create pytest's normal temporary directory because Windows interpreted mode 0700 as a restrictive ACL. The passing run used inherited workspace permissions for pytest's dedicated scratch directory. No test assertions or project defaults changed.
- The same pretrained model ran on a separate Rural crop. It produced 36 candidates. Five passed raster elevation checks and two passed Blender geometry and foundation checks. Most rejected candidates had insufficient relief. Rural is a secondary transfer check with no published replacement for the previous Rural assets.
- Unity 6000.6.1f1 with the existing glTFast 6.21.0 import script was attempted in an isolated project. The initial attempt failed to connect to the Package Manager IPC. The second connected and exited with code 198 because an active Editor license was unavailable. No Urban Unity import, Unity render or runtime compatibility was verified.
- The final scope is a reference-assisted Urban review asset. Near-ground surface reconstruction retains some tree and roof remnants, the area has incomplete building coverage, and absolute height accuracy was not evaluated. The original Rural GLB and the adaptive metrics match their pre-run SHA-256 hashes. Overall delivery status is PARTIAL.
