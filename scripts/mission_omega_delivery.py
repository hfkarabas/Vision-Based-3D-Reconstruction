"""Assemble measured local review delivery and Turkish documentation."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,shutil,time,zipfile
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mission-omega';DOC=ROOT/'Documentation'
started=time.perf_counter()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def text(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(x,encoding='utf-8')
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
base=OUT/'asset/rural-production8-ortho';large=OUT/'asset/rural-expanded200-production8-ortho'
bv=read(base/'blender_validation.json');lv=read(large/'blender_validation.json')
bt=read(base/'topology_validation.json');lt=read(large/'topology_validation.json')
bq=read(OUT/'raw/rural/production8-ortho/raster_qa.json');lq=read(OUT/'raw/rural/expanded200-production8-ortho/raster_qa.json')
uq=read(OUT/'raw/urban/production64-csf/raster_qa.json');uv=read(OUT/'asset/urban-production64-csf/blender_validation.json')
unity=read(OUT/'reports/unity_validation.json');demo=read(OUT/'reports/demo_loader_validation.json')
assert unity['success'] and demo['existing_demo_loader_success']
assert bv['zero_area_triangles']==0 and bt['connected_surface_components']==1 and bt['nonmanifold_edges']==0
assert read(base/'obj_validation.json')['success'] and read(large/'obj_validation.json')['success']
assert bv['sha256']==digest(base/'rural_terrain.glb')
copies={base/'rural_terrain.glb':OUT/'asset/rural_terrain_textured.glb',
        base/'rural_terrain.blend':OUT/'asset/rural_terrain_textured.blend',
        large/'rural_terrain.glb':OUT/'asset/rural_expanded_terrain_textured.glb',
        OUT/'asset/urban-production64-csf/urban_terrain_photographic.glb':OUT/'asset/urban_DIAGNOSTIC_NOT_ACCEPTED.glb'}
for src,dst in copies.items():shutil.copy2(src,dst)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
def caption(src,dst,lines):
    im=Image.open(src).convert('RGB');canvas=Image.new('RGB',(im.width,im.height+86),'#151a20');canvas.paste(im,(0,0));draw=ImageDraw.Draw(canvas)
    for i,line in enumerate(lines):draw.text((14,im.height+8+i*25),line,fill='white',font=font)
    canvas.save(dst)
for prefix,source,qa,label in [
    ('rural','rural_production8-ortho',bq,'RURAL — ilk açılacak model; insan görsel onayı BEKLİYOR'),
    ('rural_expanded','rural_expanded200-production8-ortho',lq,'GENİŞ RURAL — çevre eksikleri/ufak parçalar dahil tam alan'),
    ('urban_diagnostic','urban_production64-csf_photographic',uq,'URBAN TANI — KABUL EDİLMEDİ; beyaz LOD1 bina YOK')]:
    stats=qa['production'];bounds=qa['roi_bounds']
    for angle in ['top','oblique','low_angle']:
        caption(OUT/'preview'/f'{source}_{angle}.png',OUT/'preview'/f'{prefix}_{angle}.png',[
            label,f'EPSG:4547 sınır [Emin,Nmin,Emax,Nmax]={bounds}; Z=1:1; Blender EEVEE; tam alan',
            f'Raster desteği + interpolasyon: {stats["production_valid_cells"]:,} m²; interpolasyon: {stats["interpolated_cells"]:,} m²; görünmeyen alanlar saklanmadı'])
    images=[Image.open(OUT/'preview'/f'{prefix}_{a}.png').convert('RGB') for a in ['top','oblique','low_angle']]
    montage=Image.new('RGB',(images[0].width*3,images[0].height))
    for i,im in enumerate(images):montage.paste(im,(i*im.width,0))
    montage.save(OUT/'preview'/f'{prefix}_montage.png')
gates={
    'G0':dict(status='PASS',evidence='00_inventory.json; reports/environment_verified.json',detail='Real image hashes/tools/hardware/metadata checked; vertical datum explicitly unresolved'),
    'G1':dict(status='PARTIAL',evidence='raw/{rural,urban}/*/raster_qa.json',detail='Actual UAV-derived DTM/DSM and georeferenced photo-DTM projection exist; ODM unavailable, DSM outliers and independent vertical accuracy unresolved; not an ODM true orthophoto'),
    'G2':dict(status='PASS',evidence='asset/rural-production8-ortho/{blender_validation,topology_validation}.json; preview/rural_*.png',detail='Rural only: native Blender import, three inspected full-extent views, one connected surface, embedded textures; user acceptance PENDING'),
    'G3':dict(status='BLOCKED',evidence='reports/building_gate.json',detail='No footprint data obtained or invented; GPS-bound Overpass egress rejected by auto-review; explicit separate consent pending; Urban ground remains uncertain'),
    'G4':dict(status='PARTIAL',evidence='asset/rural_expanded_terrain_textured.glb; preview/rural_expanded_*.png',detail='651468 m² larger ROI exported/inspected; 575992 m² supported+interpolated; peripheral holes, texture defects and 8 components remain; whole input footprint not certified'),
    'G5':dict(status='PASS',evidence='research/rural/{protocol.json,metrics.csv,comparisons.png}',detail='Five real measured fixed-work arms, 8 complete 1200x900 maps each; H1 NOT SUPPORTED in this pilot; no independent accuracy/generalization claim'),
    'G6':dict(status='PASS',evidence='reports/unity_validation.json; asset/rural-production8-ortho/obj_validation.json; reports/demo_loader_validation.json',detail='Portable GLB and relative-path OBJ/MTL/images verified in Blender; actual Unity Editor glTFast import passed; standalone runtime/player and Unity rendered appearance NOT VERIFIED')}
save(OUT/'reports/building_gate.json',dict(status='BLOCKED',footprints_obtained=0,matched_footprints=0,measured_building_heights=0,estimated_building_heights=0,
    white_building_objects=0,external_GIS_geometry_used=False,query_transmitted=False,
    rejection='Automatic approval review rejected transmission of GPS-derived UAV footprint bounds to public Overpass without separate explicit consent.',
    pending_question='Approve GPS-derived Urban bounding-box egress to Overpass/OSM?',
    next_command_after_explicit_consent='.venv/Scripts/python.exe scripts/mission_omega_fetch_osm.py'))
save(OUT/'reports/asset_validation.json',dict(overall='PARTIAL DELIVERY; not project completion',gates=gates,
    best_asset=str(OUT/'asset/rural_terrain_textured.glb'),blender=bv,topology=bt,unity_editor=unity,
    human_visual_acceptance='PENDING',urban_status='NOT ACCEPTED: extensive inferred ground, holes and steep artifacts; no buildings',research_conclusion='H1 NOT SUPPORTED by tested proxy evidence'))
origin=bv['geospatial_origin']
geo=dict(title='Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon',
    branch='UAV RGB geometry + camera RTK georegistration; no external DEM/LiDAR/OSM geometry',
    imagery={'rural':190,'urban':161},pixel_dimensions=[4032,3024],camera_calibration='Independent COLMAP shared SIMPLE_RADIAL self-calibration; see inherited SfM metrics and source manifest',
    horizontal_crs='EPSG:4547 CGCS2000 / 3-degree Gauss-Kruger CM 114E; ALWAYS easting/northing metres',
    camera_input='XMP WGS84 longitude/latitude projected with pyproj always_xy; realization between WGS84 RTK and CGCS2000 unresolved',
    vertical_datum='UNRESOLVED: MRK Ellh and XMP RtkAlt; no geoid correction or independent vertical certification',
    camera_fit='Historical in-sample Sim3 to RTK camera centers; not independent ground/checkpoint accuracy',
    blender_local_origin_projected_m=origin,
    blender_world_to_projected=[[1,0,0,origin[0]],[0,1,0,origin[1]],[0,0,1,origin[2]],[0,0,0,1]],
    global_glb_Yup_to_projected=[[1,0,0,origin[0]],[0,0,-1,origin[1]],[0,1,0,origin[2]],[0,0,0,1]],
    unity='glTFast performs handedness conversion; Editor size verified; compass orientation/runtime render not separately inspected',
    Z_scale=1.,terrain_is_inferred_bare_earth=True,reference_rasters_used_as_geometry=False,
    texture='COLMAP mesh_texturer on DTM, Blender orthographic emission render, GDAL bounds/CRS, QGIS drape; not DSM building-correct true ortho',
    north_up=True,raster_row_step_m=-.5,
    scenes={'rural_best':bq,'rural_expanded':lq,'urban_diagnostic':uq},
    full_input_footprint='NOT CERTIFIED: camera footprints and complete raw cloud bounds are reported separately; oblique images include peripheral backgrounds beyond delivered ROI',
    raw_cloud_stats_evidence=['reports/stages/rural-v3-mad-stats.log','reports/stages/urban-v3-mad-stats.log'],
    caveats=['Bare earth beneath foliage/buildings is not directly observed','Supported raster cells are not independent accuracy certificates','All masks/NoData/interpolation retained','Urban diagnostic is not a validated urban asset'])
save(OUT/'reports/geospatial_provenance.json',geo)
stages=[]
for p in (OUT/'reports/stages').glob('*.json'):
    d=read(p);stages.append(dict(stage=p.stem,started_at=d.get('started_at'),ended_at=d.get('ended_at'),seconds=d.get('seconds'),
        peak_process_tree_rss_bytes=d.get('peak_process_tree_rss_bytes'),sampled_cpu_seconds=d.get('sampled_cpu_seconds'),
        returncode=d.get('returncode'),command=json.dumps(d.get('command'),ensure_ascii=False)))
with (OUT/'reports/processing_costs.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(stages[0]));writer.writeheader();writer.writerows(sorted(stages,key=lambda s:s['started_at'] or ''))
results=read(OUT/'research/rural/metrics.json')
rows='\n'.join(f'| {r["arm"]} | {r.get("reference_depth_maps",0)} | {r.get("additional_cells",0)} | {r["valid_fraction"]*100:.6f} | {r["incremental_seconds"]:.3f} | {r["peak_rss_bytes"]/1e6:.2f} |' for r in results[1:])
gate_table='\n'.join(f'| {k} | {v["status"]} | {v["detail"]} |' for k,v in gates.items())
readme=fr'''# ÖNCE BUNU AÇIN — Mission Omega, doğrulanmış kısmi teslim

**İlk dosya:** `asset/rural_terrain_textured.blend` (hazır kadraj ve paketli dokular). Taşınabilir model: `asset/rural_terrain_textured.glb`.

Bu bilgisayarda tek satır:
```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.3\blender.exe' '{OUT / 'asset/rural_terrain_textured.blend'}'
```
Veya `OPEN_RURAL_IN_BLENDER.cmd` dosyasına çift tıklayın. Blender içinde farenin orta tuşuyla döndürün, tekerlekle yakınlaşın; doku için Material Preview kullanın. Ölçek 1:1'dir. Referans DSM geometrisi, uydurma bina veya eski parçalı Poisson yüzeyi kullanılmadı.

Beklenen görünüm: gerçek fotoğraflarla dokulanmış tarla/yol/bitki örtüsü arazisi; ince fakat gerçek 15,16 m düşey aralık. Bu sahne doğal olarak az engebelidir; yükseklik abartılmadı. Binalar ayrı beyaz hacimler olarak henüz yoktur. Orman/kenar boşluklarını ve bazı çukur/örtülme belirsizliklerini özellikle inceleyin.

* İlk model: {bv['bytes']:,} bayt, {bv['triangles']:,} üçgen; 252.168 m² ROI; 235.496 m² ilk raster desteği. Üretim yüzeyinde 250.643 m² hücre var; 15.285 m² interpolasyon içeriyor. Bu alanlar bağımsız doğrulanmış yer ölçümü değildir.
* Daha geniş deneme: `asset/rural_expanded_terrain_textured.glb`, {lv['bytes']:,} bayt; 651.468 m² ROI, 575.992 m² destek/interpolasyon. Çevre kusurları saklanmadı. İlk modelin daha iyi görünen alanı bu geniş görünüm içinde ayrıca görülebilir.
* Urban: `asset/urban_DIAGNOSTIC_NOT_ACCEPTED.glb` yalnız çalışma alanında tanı dosyasıdır, başarılı şehir modeli değildir ve ZIP'e dahil edilmez. Beyaz LOD1 bina sayısı **0**. Yeni Urban arazisinin 121.654 m²'si en çok 64 m GIS interpolasyonuna dayanır; dik geçişler ve eksikler nedeniyle kabul edilmedi.

Önce `preview/rural_montage.png`, sonra ayrı `rural_top.png`, `rural_oblique.png`, `rural_low_angle.png` görüntülerine bakın. Hepsi gerçek GLB'nin temiz Blender sahnesinde EEVEE ile, aynı tam alan ve Z=1:1 kullanılarak üretildi; alt yazılar sınırları belirtir. `rural_expanded_montage.png` ve `urban_diagnostic_montage.png` tüm kapsam kusurlarını gösterir. Kırpılarak saklanan başarısız bölge yoktur.

Blender GLB/OBJ ve mevcut demo yükleyicisi doğrulandı. Unity {unity['editorVersion']} / glTFast 6.21.0 **gerçek Editor içe aktarımı** başarılı: {unity['triangles']:,} üçgen, {unity['textures']} doku. Standalone oyuncu/runtime ve Unity'nin render edilmiş görüntüsü test edilmedi. Unity sahnesi çalışma alanında `unity-smoke/Assets/MissionOmegaReview.unity`; GLB'yi kendi projenize resmi glTFast paketini kurarak Assets'e alın.

Araştırma sonucu: **H1 bu pilotta desteklenmedi.** Eşit iş bütçesinde uniform +62, rastgele +3, kalite geri beslemesi +0 m² ek raster desteği verdi; tekrar ziyaret ve destek güveni kontrolü de +0 verdi. Bunlar doğruluk kazanımı değildir. Ayrıntılar `research/rural/metrics.csv` ve `comparisons.png` içinde. Yeni/özgün yöntem kanıtlanmadı.

İnsan görsel kabulü **BEKLİYOR**. Düşey datum ve bağımsız zemin doğruluğu çözülmedi. DSM/ortofoto tam kabul kapısı kısmi; ODM Docker/WSL önkoşulu yok. Urban OSM sorgusu için GPS sınırlarının dışarı gönderilmesi otomatik onay incelemesince reddedildi; ayrıca açık izin sorusu bekliyor. İzin verilirse sonraki komut: `.venv/Scripts/python.exe scripts/mission_omega_fetch_osm.py`.

`reports/asset_validation.json`, `geospatial_provenance.json`, `delivery_manifest.json` ve `Documentation/` kesin kanıttır. AirZoo kaynak lisansı belirsiz olduğundan bu **yerel inceleme paketi** kamuya yeniden dağıtım izni değildir; orijinal fotoğraflar ve dev önbellekler ZIP'te yoktur.
'''
text(OUT/'README_OPEN_FIRST.md',readme)
text(OUT/'OPEN_RURAL_IN_BLENDER.cmd','@echo off\n"C:\\Program Files\\Blender Foundation\\Blender 5.3\\blender.exe" "%~dp0asset\\rural_terrain_textured.blend"\n')
technical=f'''# Mission Omega technical report — verified partial delivery

**Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon**

## Result and engineering reuse

The principal usable output is `outputs/mission-omega/asset/rural_terrain_textured.glb`, a real photo-textured, height-derived terrain. It is {bv['bytes']:,} bytes, contains {bv['triangles']:,} triangles and two embedded textures, and has a metric bounding size of {bv['dimensions_m']}. It was reimported into a clean Blender scene, rendered from top/oblique/low angle and inspected by the agent. Human visual acceptance remains PENDING. This is not project completion.

COLMAP supplies the previously computed independent SfM/MVS, new bounded depth maps/fusion, and photo texturing. PDAL supplies ELM/statistical outlier filtering, SMRF/CSF ground classification, MAD screening and rasterization. GDAL/rasterio supply bounded interpolation, CRS/raster adapters and fixed-grid merging. QGIS/Qgis2threejs supplies the existing height-field mesher/UV/scene generator. Three.js and Blender supply official file export and rendering. New code consists of command scheduling, file/CRS adapters, quality measurements, manifests and QA. No feature matcher, MVS, DTM filtering engine, triangulator, UV algorithm or renderer was implemented. See dependency and audit reports.

ODM did not run: its Docker runtime is absent and WSL reports not installed. `reports/stages/official-odm-start-blocked.json` records an actual blocked command. QGIS native/embedded export APIs were incompatible; upstream standalone Qgis2threejs scene export plus unmodified Three.js GLTFExporter worked. Native Blender OBJ image paths required a small output-path normalization adapter and were independently reimported afterward.

## Data lineage and geometry limits

Rural/Urban originals are 190/161 genuine DJI RGB JPGs, 4032×3024, independently rehashed. Frozen SfM cameras and 900-pixel dense clouds were reused. No supplied reference DSM/DOM, external DEM/LiDAR or OSM footprint contributes geometry. Photo texturing uses only those original-image-derived undistorted photos. A geospatial orthographic emission render of that textured DTM is draped in QGIS; this is **not ODM orthophoto or DSM building-correct true ortho**. Occluded ground texture is a projection/inference, not observed bare-earth appearance.

EPSG:4547 is confirmed by supplied metadata, with explicit easting/northing metre order. WGS84 XMP camera positions were projected and a historical camera-RTK in-sample similarity was reused. The realization between RTK WGS84 and CGCS2000 and the vertical datum remain unresolved. MRK Ellh/XMP RtkAlt are not silently converted to orthometric height. Blender uses X=east,Y=north,Z=up and origin {origin}; GLB is Y-up. Exact matrices and source camera parameters are in geospatial provenance and inherited SfM metrics. Z scale is exactly 1. Float64 matrix composition avoids large-coordinate float32 cancellation.

The selected initial ROI is the camera-position envelope plus a fixed 50 m buffer, bounds {bq['roi_bounds']}. It was not selected from an attractive rendered crop. Strict DTM has 235.496 supported 1 m cells (93,3885% of 252.168 m²). A separate production mask rejects 138 local quality outliers and GDAL fills at most 8 m: 15.285 interpolated cells, 250.643 total surface cells, 1.525 remaining missing cells. Interpolation and rejection masks persist. Those counts do not certify true ground beneath vegetation/buildings. The production height range is {bq['production']['production_vertical_range_m']}; the maximum adjacent-row jump remains {bq['production']['vertical_edge_abs_percentiles_m'][-1]:.3f} m and requires inspection/uncertainty, rather than being hidden by height exaggeration.

Topology: one connected Rural surface after analysis-only seam welding at 0,1 mm, zero degenerate triangles, zero nonmanifold edges. {bt['shared_horizontal_samples']} shared seam samples have maximum height disagreement {bt['max_shared_XY_height_deviation_m']} m. The open outer and NoData boundaries are intentional; watertightness is not expected. Original deliverable geometry is not welded/repaired by this QA.

## Larger coverage and Urban failure evidence

Expanded Rural uses the same 190 images and cached dense cloud with a predeclared camera-envelope +200 m ROI: {lq['roi_bounds']}, 651.468 m². Strict raster support is 495.808 m²; the separate 8 m-fill product has 575.992 m² valid/interpolated cells and 75.476 m² missing. It exports to {lv['bytes']:,} bytes / {lv['triangles']:,} triangles. There are {lt['connected_surface_components']} components, with {lt['largest_component_area_fraction']*100:.4f}% of surface area in the main component; seam deviation is {lt['max_shared_XY_height_deviation_m']} m. Peripheral texture/geometry holes remain visible in all full-extent previews, so larger/global coverage is only PARTIAL. This is not a hidden replacement of the better first asset.

Urban SMRF, CSF and two bounded production interpolation variants were actually exercised. CSF strict support is 117.437 / 279.450 m². The 64 m production variant rejects 150 quality cells, adds 121.654 interpolated cells and reaches 238.941 cells; its range is {uq['production']['production_vertical_range_m']}. The roughly 44% interpolated footprint and remaining approximately 12 m abrupt transitions make it unsuitable for claiming measured urban bare earth. Full Blender views were inspected and the Urban model is **NOT ACCEPTED**. The diagnostic is preserved, with no invented buildings. A public Overpass request was blocked before transmission by automatic approval review; separate consent is pending. No local attributable footprint dataset was found. Matched footprints, measured heights, estimated heights and white building objects are all zero.

Complete input footprint is not independently certified: oblique images see beyond camera positions and the delivered ROI. Raw cloud bounds, camera extents, strict masks and all missing areas are retained. No claim of full-scene completeness is made.

## Controlled real experiment and negative result

The frozen initial 900-pixel reconstruction is a common starting state, not a zero-cost reconstruction. All arms use the same full 252.168 m² ROI, 1 m raster grid, same calibration/CRS and cache policy. Shared 1200-pixel undistortion took 76,916 s; its cost is separate. Each arm actually produced eight full 1200×900 geometric depth maps, with two passes, three iterations, ten samples and three source images per reference: 8.640.000 allocated reference pixels / 51.840.000 allocated pixel-iterations. Complete image work is charged; only the merge is tile-restricted. Wall time is **measured, not equal by definition**. This is matched work, not a strict equal-seconds trial.

Tiles are a fixed 4×4 grid. Score = 0,7 NoData fraction + 0,3 min(P95 adjacent DTM difference / 2 m,1). Sparse-track support supplies eligible images/regions, not evaluation reference. At least 100 sparse points/four registered views are required. Existing high gradients can be real terrain, so this score is a proxy with known confounding. No aesthetic low-color score or test DSM is used. New ground rasters fill holes only through rasterio/GDAL merge; existing valid heights do not change. Thus zero common-support height difference is guaranteed by policy, not a precision improvement.

| Arm | Actual full maps | Additional supported cells (m²) | Final support % | Measured incremental stage s | Peak process-tree RSS MB |
|---|---:|---:|---:|---:|---:|
{rows}

Uniform is the strongest tested control. Feedback added zero cells and did not beat it. Uniform also increases the >3 m row-jump count from 57 to 63, so extra support alone is not verified quality gain. The original no-revisit loop recomputes feedback but is rank-equivalent to a frozen score on disjoint unused tiles; this limitation is disclosed. An exploratory second candidate permits revisiting the highest-current-defect tile with the next four unused support-ranked views; it also adds zero cells. B3 is only a sparse-support proxy inspired by preliminary geometry/view reasoning, **not exact Mostegel confidence or RQF reproduction**.

**H1 NOT SUPPORTED by this pilot.** This is not proof that all quality-feedback methods fail. One scene, two actions/arm, one random seed, proxies rather than independent bare-earth truth, and interactive desktop hardware prevent a general accuracy/variance claim. Selector overhead and manual preparation are not fully timed; per-stage costs and common costs are separated. No absolute height RMSE/MAE is reported: a DTM-to-DSM comparison would confound buildings/trees, and supplied reference lineage/datum is unresolved. No publishable novelty is demonstrated.

## Validation and acceptance

Unity {unity['editorVersion']} with official glTFast 6.21.0 actually imported/instantiated the model: {unity['triangles']:,} triangles, two textures, size {unity['sizeMetres']}. Import took 31,513 s, sampled peak process-tree RSS 2.935.427.072 bytes. This does not verify standalone runtime/player or Unity rendered material appearance. Existing desktop GLB loader and native Blender OBJ/MTL/image reimport both succeeded; no GUI rewrite was needed.

| Gate | Status | Evidence/limit |
|---|---|---|
{gate_table}

Machine: Intel Core Ultra 5 245KF (registry verified), 14 logical CPUs, 16.782.508.032 bytes RAM, RTX 5070 12.227 MiB VRAM, driver 617.42. Current free resources and exact software build/archives are in environment evidence. Jobs use two CPU threads and bounded caches. Per-stage RSS is sampled process-tree RAM, not peak GPU VRAM; no unmeasured VRAM or full end-to-end time is invented. Git was not written; unrelated historical changes remain intact. The local ZIP is a review package, not public redistribution authorization.
'''
text(DOC/'MISSION_OMEGA_TECHNICAL_REPORT.md',technical)
repro=r'''# Mission Omega reproducibility

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
'''
text(DOC/'MISSION_OMEGA_REPRODUCIBILITY.md',repro)
(OUT/'Documentation').mkdir(exist_ok=True)
for name in ['MISSION_OMEGA_AUDIT.md','MISSION_OMEGA_DEPENDENCIES.md','MISSION_OMEGA_PRIOR_WORK_AND_NOVELTY.md','MISSION_OMEGA_TECHNICAL_REPORT.md','MISSION_OMEGA_REPRODUCIBILITY.md']:
    shutil.copy2(DOC/name,OUT/'Documentation'/name)
text(OUT/'ATTRIBUTION.md','''# Attribution and local-use scope

AirZoo-Real imagery: Xiaoya Cheng, Rouwan Wu, Xinyi Liu, Zeyu Cui, Yan Liu, Na Zhao, Yu Liu, Maojun Zhang, Shen Yan, **AirZoo: A Unified Large-Scale Dataset for Grounding Aerial Geometric 3D Vision**, ECCV 2026 acceptance as reported by the authors. Dataset/card: https://huggingface.co/datasets/RingoWRW97/AirZoo-Real . Licensing terms are not yet supplied by the authors; no public redistribution permission is assumed for photos or derived textures. This is a local review bundle, not a public release.

Geometry/textures use COLMAP (BSD-3-Clause; dependency terms separate), PDAL (BSD), GDAL (MIT/X), QGIS (GPL), Qgis2threejs (GPL family; per-file SPDX/license retained in the workspace), Three.js (MIT), Blender (GPL), SciPy/NumPy/rasterio (BSD), pyproj (MIT), plyfile (BSD), trimesh (MIT). See exact versions, notices and official links in Documentation/MISSION_OMEGA_DEPENDENCIES.md and reports/environment_verified.json. Their existing algorithms generated the product; the mission adds orchestration/CRS/file adapters, measurements and QA only.

Unity Editor is separately licensed. Official Unity glTFast 6.21.0 is Apache-2.0. Playwright is Apache-2.0. No software binary/tool cache is redistributed in this review ZIP. No OSM/Overture building data was obtained, so no building source/height accuracy is claimed. No reference DOM/DSM contributes geometry.
''')
sources=list((ROOT/'scripts').glob('mission_omega_*.py'))
source_manifest=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p),lines=len(p.read_text(encoding='utf-8').splitlines())) for p in sources]
save(OUT/'reports/integration_source_manifest.json',source_manifest)
(OUT/'scripts').mkdir(exist_ok=True)
for p in sources:shutil.copy2(p,OUT/'scripts'/p.name)
shutil.copy2(OUT/'tools/Qgis2threejs-master/package-lock.json',OUT/'reports/qgis2threejs-package-lock.json')
files=[OUT/'README_OPEN_FIRST.md',OUT/'OPEN_RURAL_IN_BLENDER.cmd',OUT/'ATTRIBUTION.md',OUT/'asset/rural_terrain_textured.glb',OUT/'asset/rural_terrain_textured.blend',OUT/'asset/rural_expanded_terrain_textured.glb']
files+=list((base/'obj').glob('*'))+list((large/'obj').glob('*'))
files+=list((OUT/'Documentation').glob('*.md'))+list((OUT/'scripts').glob('*.py'))
files+=[OUT/'preview'/f'{prefix}_{angle}.png' for prefix in ['rural','rural_expanded','urban_diagnostic'] for angle in ['top','oblique','low_angle','montage']]
files+=[OUT/'preview'/n for n in ['rural_terrain_quality_map.png','rural_expanded200-production8_terrain_quality_map.png','urban_production64-csf_terrain_quality_map.png']]
files+=[OUT/'reports'/n for n in ['asset_validation.json','geospatial_provenance.json','processing_costs.csv','environment_verified.json','integration_source_manifest.json','building_gate.json','unity_validation.json','demo_loader_validation.json','qgis2threejs-package-lock.json']]
files+=[OUT/'00_inventory.json',base/'blender_validation.json',base/'topology_validation.json',base/'obj_validation.json',large/'blender_validation.json',large/'topology_validation.json',large/'obj_validation.json']
for variant in ['production8-ortho','expanded200-production8-ortho']:
    files+=list((OUT/'gis').glob(f'rural-{variant}.*'))
    files+=[OUT/'raw/rural'/variant/n for n in ['dtm.tif','orthographic_photo_mosaic.tif','raster_qa.json','geospatial_transform.json','source_manifest.json']]
for variant in ['production8','expanded200-production8']:
    files+=[OUT/'raw/rural'/variant/n for n in ['strict_support.tif','quality_rejected.tif','interpolated.tif','production_support.tif']]
files+=[p for p in (OUT/'research/rural').rglob('*') if p.is_file() and (p.name in ['metrics.csv','metrics.json','protocol.json','feedback_v2_protocol.json','B3_support_proxy_protocol.json','comparisons.png','tile_state.json','decision.json','merge_metrics.json','before_after_difference.tif'])]
files+=list((OUT/'reports/stages').glob('*.json'))
files+=list((OUT/'unity-smoke/Packages').glob('*.json'))+list((OUT/'unity-smoke/Assets/Editor').glob('*.cs'))
files=sorted(set(files));assert all(p.is_file() for p in files)
manifest=[dict(path=str(p.relative_to(OUT)).replace('\\','/'),bytes=p.stat().st_size,sha256=digest(p)) for p in files]
save(OUT/'reports/delivery_manifest.json',dict(created_at=datetime.now(timezone.utc).isoformat(),files=manifest,
    exclusions='Original images, reference DSM/DOM, tools/binaries, huge caches, rejected Urban GLB, Unity Library and potentially account-specific full Editor log',
    license='Local review only; AirZoo public redistribution rights unresolved; see dependency report'))
files.append(OUT/'reports/delivery_manifest.json')
archive=OUT/'MISSION_OMEGA_REVIEW.zip'
if archive.exists():archive=OUT/('MISSION_OMEGA_REVIEW_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
    for p in files:z.write(p,str(p.relative_to(OUT)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names=z.namelist();assert len(names)==len(set(names))
save(OUT/'reports/package_validation.json',dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),files=len(files),CRC_test='PASS',
    original_images_bundled=False,huge_caches_bundled=False))
status=read(OUT/'mission_status.json');status.update(last_updated_at=datetime.now(timezone.utc).isoformat(),phase='P6_verified_partial_delivery',outcome='partial',
    active_tasks=['Awaiting explicit location-egress consent and human visual acceptance'],gates=gates,
    blockers=['Urban footprint request blocked by automatic approval review pending separate explicit GPS-egress consent','Urban inferred terrain not accepted; no white LOD1 buildings','ODM runtime absent; independent vertical datum/reference accuracy unresolved'],
    paths=dict(best_asset=str(OUT/'asset/rural_terrain_textured.glb'),larger_asset=str(OUT/'asset/rural_expanded_terrain_textured.glb'),readme=str(OUT/'README_OPEN_FIRST.md'),package=str(archive)))
save(OUT/'mission_status.json',status)
with (OUT/'progress.log').open('a',encoding='utf-8') as f:f.write(f'{status["last_updated_at"]} scripts/mission_omega_delivery.py: {time.perf_counter()-started:.3f}s; local package CRC verified; gates G0 PASS/G1 PARTIAL/G2 PASS/G3 BLOCKED/G4 PARTIAL/G5 PASS/G6 PASS; human acceptance PENDING\n')
print(json.dumps(dict(best_asset=status['paths']['best_asset'],package=str(archive),package_bytes=archive.stat().st_size,files=len(files),gates={k:v['status'] for k,v in gates.items()}),ensure_ascii=False))
