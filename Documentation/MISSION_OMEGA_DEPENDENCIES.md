# Mission Omega dependencies

Exact package versions, build identifiers, channels, checksums and code licenses are in `outputs/mission-omega/reports/environment_verified.json`. Software is local to this workspace except previously installed Blender, Chrome, Unity and COLMAP.

| Reused/installed software | Actual version | Invocation and role | License / official origin |
|---|---|---|---|
| COLMAP CUDA, reused | 4.2.1 bd1fcf6 | cached independent SfM/MVS; mesh_texturer; bounded patch_match_stereo/stereo_fusion | BSD-3-Clause for COLMAP, dependency terms separate; https://github.com/colmap/colmap |
| PDAL, installed | 2.10.2 | `tools/gis/Library/bin/pdal.exe pipeline ...`; ELM, SMRF, MAD, raster writers | BSD; https://pdal.io/ |
| GDAL, installed | 3.13.3 | GDAL FillNodata/Translate through rasterio/osgeo; raster merge/reprojection | MIT/X; https://gdal.org/ |
| rasterio / pyproj | 1.5.2 / 3.8.0 | raster measurement, fixed-grid merge, CRS conversion | BSD / MIT; https://rasterio.readthedocs.io/ and https://pyproj4.github.io/pyproj/ |
| QGIS, installed | 4.2.3 | `scripts/mission_omega_run_qgis.py ... --web`; relative-path QGZ and DEM surface | GPL-2.0-or-later; https://qgis.org/ |
| Qgis2threejs, source build | 3.3; commit 922f1663e4767361fc98205d005ac63c8cfaa447 | upstream DEM NoData-aware meshing, UV and scene export | Repository/license files retained; GPL family, individual SPDX headers vary; https://github.com/minorua/Qgis2threejs |
| Three.js | 0.184.0 | unmodified GLTFExporter on upstream QGIS scene in isolated Chrome | MIT; https://github.com/mrdoob/three.js |
| Blender, reused | 5.3.0 02f0899d78e3 | background native GLB/OBJ/PLY I/O, inspection and full-extent renders | GPL; https://blender.org/ |
| plyfile / trimesh | 1.1.5 / 5.2.0 | standard textured PLY parser and OBJ/MTL exporter, no custom parser | BSD / MIT; https://github.com/dranjan/python-plyfile and https://github.com/mikedh/trimesh |
| SciPy / NumPy | GIS 1.18.1 / 2.5.4 | existing median, labels, percentiles, matrix transforms; quality measurements | BSD; https://scipy.org/ and https://numpy.org/ |
| Playwright | 1.65.1 | isolated local Chrome, official exporter invocation; no website uploads | Apache-2.0; https://playwright.dev/python/ |
| Unity, reused | 6000.6.1f1 | real import smoke test, distinct from Blender; result JSON controls status | Proprietary editor license; https://unity.com/ |
| Unity glTFast, planned smoke dependency | 6.21.0, official registry verified | `com.unity.cloud.gltfast`; actual test status recorded separately | Apache-2.0; https://github.com/atteneder/glTFast |
| ODM, NOT installed/run | none | official documentation checked; Docker absent, WSL not installed | AGPL-3.0; https://github.com/OpenDroneMap/ODM and https://docs.opendronemap.org/ |

The first `tools/qgis` environment is a failed compatibility attempt and is not the active exporter. `tools/qgis4` is the clean active environment. Qgis2threejs embedded WebEngine/native 3D export attempts failed; the successful route uses upstream standalone scene export plus Three.js GLTFExporter. These attempts are not counted as an ODM result.

AirZoo data and its derived textures are separate from code licenses. Dataset card: https://huggingface.co/datasets/RingoWRW97/AirZoo-Real ; no public redistribution permission is inferred from download availability. The local review ZIP is not a public release. Original images are not bundled. No OSM data was obtained without the separately requested location-egress consent; if obtained later, © OpenStreetMap contributors / ODbL must accompany the footprint output. Reference DOM/DSM are excluded from generation.

No 3dfier, OpenMVS, CloudCompare or external DEM/LiDAR was silently used. Portable installs do not change Windows optional features or security settings. Full environment manifests and upstream archive hashes permit replay without relying on moving branch names.
