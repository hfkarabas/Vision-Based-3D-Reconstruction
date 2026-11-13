"""Persist exact local dependency/command provenance, without secrets."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.metadata,json,os,platform,shutil,subprocess,zipfile
import psutil
import winreg
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mission-omega';reports=OUT/'reports';reports.mkdir(exist_ok=True)
packages={}
for name in ['gis','qgis','qgis4']:
    entries=[]
    for p in (OUT/'tools'/name/'conda-meta').glob('*.json'):
        d=json.loads(p.read_text());entries.append({k:d.get(k) for k in ['name','version','build','channel','url','sha256','md5','license']})
    packages[name]=entries
extras={n:importlib.metadata.version(n) for n in ['numpy','scipy','Pillow','psutil','plyfile','trimesh','playwright']}
archives={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
    for p in (OUT/'tools').glob('*.zip')}
data=dict(measured_at=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),cpu=platform.processor(),logical_cpus=os.cpu_count(),
    ram=psutil.virtual_memory()._asdict(),disk=shutil.disk_usage(ROOT)._asdict(),conda_packages=packages,python_packages=extras,archives=archives,
    qgis2threejs_commit=zipfile.ZipFile(OUT/'tools/Qgis2threejs.zip').comment.decode(),
    blender='5.3.0 / 02f0899d78e3',colmap='4.2.1 / bd1fcf6 CUDA',unity_editor='6000.6.1f1',
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv'],text=True).strip())
with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
    data['cpu_model_verified_registry']=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
(reports/'environment_verified.json').write_text(json.dumps(data,indent=2))
inventory=json.loads((OUT/'00_inventory.json').read_text());inventory['subsequent_environment_evidence']=str(reports/'environment_verified.json')
(OUT/'00_inventory.json').write_text(json.dumps(inventory,indent=2,ensure_ascii=False))
versions={name:{p['name']:p['version'] for p in value} for name,value in packages.items()}
text=f'''# Mission Omega dependencies

Exact package versions, build identifiers, channels, checksums and code licenses are in `outputs/mission-omega/reports/environment_verified.json`. Software is local to this workspace except previously installed Blender, Chrome, Unity and COLMAP.

| Reused/installed software | Actual version | Invocation and role | License / official origin |
|---|---|---|---|
| COLMAP CUDA, reused | 4.2.1 bd1fcf6 | cached independent SfM/MVS; mesh_texturer; bounded patch_match_stereo/stereo_fusion | BSD-3-Clause for COLMAP, dependency terms separate; https://github.com/colmap/colmap |
| PDAL, installed | {versions['gis'].get('pdal')} | `tools/gis/Library/bin/pdal.exe pipeline ...`; ELM, SMRF, MAD, raster writers | BSD; https://pdal.io/ |
| GDAL, installed | {versions['gis'].get('gdal')} | GDAL FillNodata/Translate through rasterio/osgeo; raster merge/reprojection | MIT/X; https://gdal.org/ |
| rasterio / pyproj | {versions['gis'].get('rasterio')} / {versions['gis'].get('pyproj')} | raster measurement, fixed-grid merge, CRS conversion | BSD / MIT; https://rasterio.readthedocs.io/ and https://pyproj4.github.io/pyproj/ |
| QGIS, installed | {versions['qgis4'].get('qgis')} | `scripts/mission_omega_run_qgis.py ... --web`; relative-path QGZ and DEM surface | GPL-2.0-or-later; https://qgis.org/ |
| Qgis2threejs, source build | 3.3; commit {data['qgis2threejs_commit']} | upstream DEM NoData-aware meshing, UV and scene export | Repository/license files retained; GPL family, individual SPDX headers vary; https://github.com/minorua/Qgis2threejs |
| Three.js | 0.184.0 | unmodified GLTFExporter on upstream QGIS scene in isolated Chrome | MIT; https://github.com/mrdoob/three.js |
| Blender, reused | 5.3.0 02f0899d78e3 | background native GLB/OBJ/PLY I/O, inspection and full-extent renders | GPL; https://blender.org/ |
| plyfile / trimesh | {extras['plyfile']} / {extras['trimesh']} | standard textured PLY parser and OBJ/MTL exporter, no custom parser | BSD / MIT; https://github.com/dranjan/python-plyfile and https://github.com/mikedh/trimesh |
| SciPy / NumPy | GIS {versions['gis'].get('scipy')} / {versions['gis'].get('numpy')} | existing median, labels, percentiles, matrix transforms; quality measurements | BSD; https://scipy.org/ and https://numpy.org/ |
| Playwright | {extras['playwright']} | isolated local Chrome, official exporter invocation; no website uploads | Apache-2.0; https://playwright.dev/python/ |
| Unity, reused | 6000.6.1f1 | real import smoke test, distinct from Blender; result JSON controls status | Proprietary editor license; https://unity.com/ |
| Unity glTFast, planned smoke dependency | 6.21.0, official registry verified | `com.unity.cloud.gltfast`; actual test status recorded separately | Apache-2.0; https://github.com/atteneder/glTFast |
| ODM, NOT installed/run | none | official documentation checked; Docker absent, WSL not installed | AGPL-3.0; https://github.com/OpenDroneMap/ODM and https://docs.opendronemap.org/ |

The first `tools/qgis` environment is a failed compatibility attempt and is not the active exporter. `tools/qgis4` is the clean active environment. Qgis2threejs embedded WebEngine/native 3D export attempts failed; the successful route uses upstream standalone scene export plus Three.js GLTFExporter. These attempts are not counted as an ODM result.

AirZoo data and its derived textures are separate from code licenses. Dataset card: https://huggingface.co/datasets/RingoWRW97/AirZoo-Real ; no public redistribution permission is inferred from download availability. The local review ZIP is not a public release. Original images are not bundled. No OSM data was obtained without the separately requested location-egress consent; if obtained later, © OpenStreetMap contributors / ODbL must accompany the footprint output. Reference DOM/DSM are excluded from generation.

No 3dfier, OpenMVS, CloudCompare or external DEM/LiDAR was silently used. Portable installs do not change Windows optional features or security settings. Full environment manifests and upstream archive hashes permit replay without relying on moving branch names.
'''
(ROOT/'Documentation/MISSION_OMEGA_DEPENDENCIES.md').write_text(text,encoding='utf-8')
print('Recorded dependency manifests')
