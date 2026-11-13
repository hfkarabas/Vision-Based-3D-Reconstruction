# Mission Omega reproducibility

Research title: **Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon**

All commands run from `C:\projects\Vision-Based-3D-Reconstruction`. Originals and historical Mission 0 caches are required for reconstruction replay; the local ZIP intentionally contains neither. They are not required to view GLB/BLEND/OBJ. Every source hash is in `00_inventory.json`; source-image manifests and affine matrices are under `raw/`. Command arrays, actual start/end/seconds/return codes and sampled RSS are under `reports/stages/`; `processing_costs.csv` summarizes measured stages. Some direct exploratory UI/export commands were not independently timed; their times are NOT substituted with estimates. Do not sum overlapping stage durations into alleged end-to-end time.

## Open verified output
```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.3\blender.exe' 'C:\projects\Vision-Based-3D-Reconstruction\outputs\mission-omega\asset\rural_terrain_textured.blend'
```
The existing demo is `python -m vision3d.demo`; select the saved GLB using its existing picker. The loader itself was exercised; interactive picker/camera gestures were not retested. No native engine/GUI changes are required.

## Reuse source orientation and reproduce strict rasters
```powershell
.venv/Scripts/python.exe scripts/mission_omega_pipeline.py rural
.venv/Scripts/python.exe -c "import sys;sys.path.insert(0,'scripts');import mission_omega_pipeline as p;p.pipeline('rural','v3-mad')"
```
Success markers skip completed expensive jobs. The source is frozen `outputs/research-baseline/{scene}/dense-900/workspace/fused.ply`, with the independent camera solution from `sfm-2000/models/0`. To change settings, use a NEW variant/output namespace and stage identifiers; never reuse a success marker for changed parameters. `v4-csf` is a documented alternative, not a silent replacement. `rural/expanded200` reuses the full v3 classified cloud, and raster writer argument `200` sets the camera-envelope buffer. It does not use a reference DEM.

The GIS interpreter is `outputs/mission-omega/tools/gis/python.exe` with `Library/bin` on PATH and matching PROJ_DATA/GDAL_DATA. `mission_omega_pipeline.env()` sets these supported conda variables. QGIS has a separate clean prefix `tools/qgis4`; `mission_omega_run_qgis.py` supplies official QGIS/Qt paths. Exact conda names/builds/checksums and archive digests are in `environment_verified.json`; the failed first QGIS prefix is not a supported runtime.

## Production terrain and official export
For the initial scene, run `mission_omega_prepare_surface.py rural` with the GIS interpreter through the stage wrapper. It invokes existing SciPy quality measurements and GDAL FillNodata (8 m), preserving strict/rejected/interpolated/support masks. No manual edit of elevations is part of replay.
```powershell
.venv/Scripts/python.exe scripts/mission_omega_run_qgis.py rural production8 --web
.venv/Scripts/python.exe scripts/mission_omega_browser_export.py rural production8
& 'C:\Program Files\Blender Foundation\Blender 5.3\blender.exe' -b -t 2 --python-exit-code 1 --python scripts/mission_omega_blender.py -- rural production8 --prepare
```
The prepared `terrain_sfm_precise.ply` is in the cached SfM frame using float64-composed transforms. Invoke the exact installed COLMAP mesh_texturer argument array from `reports/stages/rural-production8-photo-texture-precise.json`. Its output is `photo-texture-precise/mesh.ply` and texture.png. Then:
```powershell
.venv/Scripts/python.exe scripts/mission_omega_texture_adapter.py rural production8
& 'C:\Program Files\Blender Foundation\Blender 5.3\blender.exe' -b -t 2 --python-exit-code 1 --python scripts/mission_omega_blender.py -- rural production8 --obj C:/projects/Vision-Based-3D-Reconstruction/outputs/mission-omega/asset/rural-production8/photographic-obj/terrain.obj
```
Use the GIS interpreter with `mission_omega_georef_mosaic.py rural` to make the exact-bounds GeoTIFF. Run QGIS/Chrome export again for `production8-ortho`; this is the draped, smaller final GLB. Run native Blender QA, then `mission_omega_fix_obj_paths.py` and `mission_omega_verify_obj.py` to verify portable OBJ references. `--python-exit-code 1` makes Blender script failures explicit. Do not treat exit status alone as texture success; validation JSON/images must exist and match source triangle counts.

Qgis2threejs source snapshot is commit `922f1663e4767361fc98205d005ac63c8cfaa447`; download that exact official GitHub archive, not moving master. Its ordinary npm lock/esbuild build is retained. Only upstream meshing/UV/Three.js export is invoked. Embedded QtWebEngine/native bindings were unsuccessful; the successful `--web` path loads the resulting local scene in isolated installed Chrome. No data is uploaded. Browser process launching may require this environment's execution approval.

Expanded Rural repeats the same steps for `expanded200-production8` and then `expanded200-production8-ortho`, with texture scale 0.65 to bound memory. It does not rerun SfM/MVS. Full ROI, peripheral failures, and all masks remain visible. Urban `production64-csf` is an explicitly rejected diagnostic: CSF source, lower-tail quality abstention, maximum 64 m interpolation, no measured building heights. Never pass that variant off as a successful city model.

## Real fixed-work experiment and single-region restart
```powershell
.venv/Scripts/python.exe scripts/mission_omega_research.py rural prepare
.venv/Scripts/python.exe scripts/mission_omega_research.py rural run
.venv/Scripts/python.exe scripts/mission_omega_research.py rural revisit
.venv/Scripts/python.exe scripts/mission_omega_research.py rural confidence
```
To rerun ONLY one selected region in a new namespace:
```powershell
.venv/Scripts/python.exe scripts/mission_omega_research.py rural one-action --tile r0c3 --arm replay_r0c3 --index 0
```
This computes four whole reference maps, fuses them with existing COLMAP, invokes PDAL classification, and merges only the tile's holes. It is NOT a cropped MVS engine. Full images and cost are charged. `decision.json` freezes selected filenames and tile bounds; successful stages are resumable. Hardlinks are read-only reuse of workspace image/sparse files; source files are not changed. Actual maps and dimensions are counted from outputs. The common 1200-pixel undistortion took 76.916 s. Tested 8-map arm stage sums were approximately 103–112 s, with measured process-tree RSS below 0.89 GB; these are observations on this machine, not a runtime promise. The previous full 900px MVS costs are historical/common inputs and must not be counted as zero-cost end-to-end reconstruction.

## Actual Unity smoke test
The isolated project `outputs/mission-omega/unity-smoke` pins `com.unity.cloud.gltfast=6.21.0`. `Packages/packages-lock.json` records actually resolved dependencies. Exact executed command is in `reports/stages/unity-real-editor-import.json`. Its Editor script imported a real GLB, checked mesh/texture counts/dimensions and saved `Assets/MissionOmegaReview.unity`. It exited successfully after 31.513 s, peak sampled process-tree RAM 2.935 GB. Standalone player/runtime and Unity rendered appearance remain untested. The full Unity cache/editor are not bundled.

## Blockers and exact next commands
ODM cannot start: `docker.exe run --rm opendronemap/odm --help` is actually blocked because docker.exe does not exist. `wsl --status` reports WSL not installed. If the user elects to enable that platform, the official system-level step is `wsl --install -d Ubuntu` in an Administrator terminal, with reboot if requested by Windows. It is not run by these scripts; no security feature was disabled. A usable local established-tool fallback already exists.

Building query, ONLY AFTER explicit approval of GPS-derived bounds egress:
```powershell
.venv/Scripts/python.exe scripts/mission_omega_fetch_osm.py
```
Automatic approval review rejected that specific transfer; no query was transmitted. Consent alone does not prove footprint coverage or alignment: inspect actual returned data before any extrusion, attribute ODbL, and distinguish measured/tagged/estimated heights. No fabricated LOD1 output is provided.

Human visual review of the complete models is still required. Data authors' redistribution terms and vertical datum/independent bare-earth checkpoints require external clarification; there is no honest local command that supplies missing survey truth. No commit was fabricated: Git is read-only. The delivery manifest lists new glue/source hashes and package contents; unrelated pre-existing user changes remain intact.
