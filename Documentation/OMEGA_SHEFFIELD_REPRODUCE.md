# Sheffield Cross yeniden çalıştırma komutları

Komutları depo kökünden PowerShell 7 ile çalıştırın. `-replay` yolları yeni çıktı dizinleridir. Resmî ODM kurulumu, Qgis2threejs 3.3 kaynak ağacı ve Node bağımlılıkları, PDAL/GDAL, Blender, mevcut GIS Python ortamı ve GeoAI ağırlıkları yerelde bulunmalıdır.

```powershell
$root = 'outputs/mission-omega/reusable-uav/sheffield-full-production'
$legacy = 'outputs/mission-omega/building-recovery'
$colmap = 'outputs/airzoo-benchmark/geometric-root-cause-recovery'
$python = "$legacy/runtime/venv/Scripts/python.exe"
$blender = 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe'
$pdal = 'outputs/mission-omega/tools/gis/Library/bin/pdal.exe'
$plugin = 'outputs/mission-omega/tools/Qgis2threejs-master'
$settings = 'configs/omega_reusable_uav.json'
$images = 'datasets/omega-terrain-validation/sheffield/full'
$historical = "$legacy/terrain-validation/sheffield-full"
$workspace = "$legacy/terrain-validation/sheffield-full-colmap"
$env:PYTHONPATH = "scripts;outputs/mission-omega/reusable-uav/gis-extra;$legacy/terrain-validation/alternate-python"
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$colmapExtent = @(333031, 3102867, 333282, 3103066)
$odmExtent = @(333027, 3102858, 333284, 3103067)
```

COLMAP önbelleğini doğrulayın. Aynı komuttan `--verify-only` çıkarılırsa kamera ortofoto/DSM aşaması yeni dizinde çalışır. `--classifier csf` veya `--classifier smrf` eklenirse ilgili PDAL varyantı da çalışır.

```powershell
& $python scripts/omega_uav_cached_products.py --verify-only --images $images --workspace $workspace --output "$root/colmap-replay" --settings $settings --pdal $pdal --colmap "$colmap/colmap-cuda/bin/colmap.exe" --blender $blender --crs EPSG:32617 --extent @colmapExtent
```

Bağımsız ODM karşılaştırması için orijinal fotoğrafları değiştirmeden yeni çalışma dizinine kopyalayın. Windows sanal alanı OpenSfM işçilerine izin vermediğinde komutu normal bir yerel PowerShell oturumunda çalıştırın. İlk tamamlanan koşu, aynı parametrelerle `--rerun-from opensfm` kullandı.

```powershell
$odmRoot = Join-Path (Resolve-Path $root).Path 'odm-replay'
$destImages = Join-Path $odmRoot 'sheffield/images'
New-Item -ItemType Directory -Path $destImages -ErrorAction Stop | Out-Null
Get-ChildItem -LiteralPath $images -File -Filter '*.JPG' | Copy-Item -Destination $destImages
$env:ODMDONTSETPYCACHE = '1'
$env:PYTHONPYCACHEPREFIX = Join-Path (Resolve-Path $root).Path 'tools/odm-cache'
& "$root/tools/odm/run.bat" --project-path $odmRoot sheffield --dsm --dtm --dem-resolution 50 --orthophoto-resolution 20 --max-concurrency 4 --pc-quality low --feature-quality medium --skip-3dmodel --skip-report
```

Aşağıdaki devam komutları tamamlanmış ODM rasterlarını kullanır. `$odm` yolu tekrar çalıştırılmış proje klasörüyle değiştirilebilir. Çalışma kutusu yoğun LAZ sınırından belirlenir. İnceleme poligonları bu komutlara girdi değildir.

```powershell
$odm = "$root/odm/sheffield"
$observed = "$root/observations-replay"
$products = "$root/products-replay"
$alternate = "$root/alternate-replay"
$foundation = "$root/foundation-replay"
$mesh = "$root/qgis-replay"
$delivery = "$root/delivery-replay"
& $python scripts/omega_odm_products.py observed --cloud "$odm/odm_georeferencing/odm_georeferenced_model.laz" --pdal $pdal --settings $settings --output $observed --extent @odmExtent --crs EPSG:32617
& $python scripts/omega_odm_products.py transfer --rgb "$odm/odm_orthophoto/odm_orthophoto.tif" --dsm "$odm/odm_dem/dsm.tif" --dtm "$odm/odm_dem/dtm.tif" --observed "$observed/classified_ground.tif" --settings $settings --output $products --extent @odmExtent --crs EPSG:32617
& $python scripts/omega_building_review.py alternate --rgb "$products/orthophoto/orthophoto.tif" --checkpoint "$legacy/terrain-validation/checkpoints/building_footprints_usa.pth" --output $alternate
# Template contains the same projection and texture mapping as the ODM raster export.
$template = "$products/template"
New-Item -ItemType Directory -Path $template -ErrorAction Stop | Out-Null
Copy-Item "$products/orthophoto/supported_photo.png" "$template/terrain_photo.png"
$ground = Get-Content "$products/ground/complete.json" -Raw | ConvertFrom-Json
@{crs='EPSG:32617';bounds=$odmExtent;origin_enu=$ground.origin;texture_label=$ground.texture_label} | ConvertTo-Json | Set-Content -Encoding utf8NoBOM "$template/grid.json"
& $python scripts/omega_uav_foundation_trial.py --ground "$products/ground" --dense "$observed/dense_count.tif" --polygons "$alternate/geoai_polygons.gpkg" --inference "$alternate/inference.json" --orthophoto "$products/orthophoto" --asset-template $template --pdal $pdal --blender $blender --radius 8 --surrounded --preserve-observed --output $foundation
& $python scripts/omega_qgis_dem.py --dtm "$foundation/ground/dtm.tif" --plugin $plugin --output $mesh
New-Item -ItemType Directory -Path $delivery -ErrorAction Stop | Out-Null
foreach($file in 'grid.json','terrain_photo.png','building_footprints_projected.json') { Copy-Item "$foundation/buildings/$file" $delivery }
& $blender -b -t 4 --python-exit-code 1 --python scripts/omega_building_blender.py -- --output $delivery --dem-mesh "$mesh/terrain.ply" --dem-manifest "$mesh/grid.json" --name sheffield_cross_full --allow-terrain-only
& $python scripts/omega_building_review.py foundations --rgb "$products/orthophoto/orthophoto.tif" --projected-assessment "$delivery/building_footprints_projected.json" --validation "$delivery/reports/blender_buildings.json" --output "$delivery/building_detection_overlay.png"
& $blender -b -t 4 --python-exit-code 1 --python scripts/omega_building_reopen.py -- --scene "$delivery/sheffield_cross_full.blend" --validation "$delivery/reports/blender_glb_reimport.json" --render "$delivery/preview/reopened.png" --report "$delivery/reports/blender_native_reopen.json"
```

Teslim edilen varlıklar için aynı ölçümleri ve eşleştirilmiş görselleri yeniden üretmek üzere:

```powershell
& $python scripts/omega_site_coverage.py --config "$root/comparison-odm.json" --colmap $workspace --pycolmap "$colmap/colmap-python" --blender $blender --output "$root/coverage-common-replay"
& $python scripts/omega_site_coverage.py --config "$root/comparison-odm-full.json" --colmap $workspace --pycolmap "$colmap/colmap-python" --blender $blender --output "$root/coverage-full-replay"
& $blender -b -t 4 --python-exit-code 1 --python scripts/omega_terrain_blender.py -- compare --before "$historical/asset/sheffield_terrain.glb" --before-grid "$historical/asset/grid.json" --after "$root/odm-delivery/sheffield_cross_full.glb" --after-grid "$root/odm-delivery/grid.json" --views "$root/views.json" --output "$root/matched-replay"
```
