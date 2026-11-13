# UAV Blender masaüstü alpha

## Kullanım

Windows’ta depo kökündeki `run_demo.bat` dosyasını açın. Varsayılan **UAV Blender scene** profilinde **Select Folder → Start Reconstruction → Open Blend File** yolunu izleyin. Girdi, birbirini örten orijinal JPEG/PNG fotoğrafların doğrudan bulunduğu klasördür. En az üç farklı konumda geçerli EXIF GPS gerekir. Jeoreferans ve zemin desteği ODM ürünlerinde ayrıca kontrol edilir. Önceki Open3D ve Castle akışlarına dönmek için **Multi-view** listesinden **Sparse only** seçin.

Uygulama arka planda bir iş başlatır ve aşamaları gösterir. Aşama hatasında neden ve günlük yolu görünür. Her çalıştırma varsayılan olarak `outputs/demo-uav/run_*/` altında ayrı dizin alır. `run.json` girdi özetlerini, araç bilgilerini, aşamaları ve doğrulanmış dosyanın yolunu saklar. **Open Blend File** yalnızca seçili girdiye ait etkin koşu için doğrulama tamamlanınca kullanılabilir. Yerel dosyanın SHA-256 değeri açılırken tekrar kontrol edilir. Geçerli araziye rağmen bina hacmi üretilemezse arayüz **PARTIAL** yazar ve doğrulanmış araziyi incelemeye açar.

## Yerel araçlar ve tekrar üretim

Depodaki mevcut GIS/GeoAI Python ortamı, OpenDroneMap Windows kurulumu, PDAL/GDAL, Node ve Qgis2threejs derleme bağımlılıkları ile Blender gerekir. Kullanılan sürümler ODM 3.6.2, GeoAI 0.43.1, Qgis2threejs 3.3 ve Blender 5.3.0. RGB Mask R-CNN checkpoint’i `aa2b25d27b2c78f0e92c7b6a4813a571f9bc2288` ve SHA-256 `3aea5d0da7803be31ff4e2e1e5ac747f15cab2a13d6bb1a9319b6a6ba8bc7bca`. Gerektiğinde yerel kurulum yollarını `VISION3D_GIS_PYTHON`, `VISION3D_ODM` (kurulum dizini), `VISION3D_PDAL`, `VISION3D_QGIS_DEM` (eklenti kaynak dizini), `VISION3D_NODE`, `VISION3D_BLENDER` ve `VISION3D_BUILDING_MODEL` ile verin. [ODM seçenekleri](https://docs.opendronemap.org/arguments/) · [GeoAI modeli ve çıkarıcı](https://opengeoai.org/examples/building_footprints_usa/) · [Qgis2threejs](https://github.com/minorua/Qgis2threejs)

```powershell
$recovery = 'outputs/mission-omega/building-recovery'
$env:PYTHONPATH = "outputs/mission-omega/reusable-uav/gis-extra;$recovery/terrain-validation/alternate-python"
& "$recovery/runtime/venv/Scripts/python.exe" scripts/omega_desktop.py '<photo folder>' --output 'outputs/demo-uav/cli-replay'
```

Önbellek, fotoğraf adları ve SHA-256 değerleri, ODM’nin etkin seçenekleri, yerel sürüm ve bileşen özetleri ile üretilen dosya özetleri eşleştiğinde ODM ürünlerini kullanır. Yeni koşu raster hazırlığını, yerel GeoAI tespitini, Qgis2threejs mesh’i ve Blender doğrulamasını yeniden yapar. Önceki üretim kayıtlarını içe almak için aynı komuta `--import-odm <project>`, `--producer-log <log>` ve `--producer-command <command JSON>` verin ve yeni bir `--output` seçin. Bu işlem önceki dosyaları değiştirmez. Sheffield testi, önceki tam saha koşusunun `odm/sheffield` dizinini, `odm/odm_retry.log` ve `odm/retry-command.json` ile içe aldı. Yerelde kurulmuş üreticiye ait sürüm kaydı, etkin seçenekler ve kaynak fotoğraf özetleri doğrulandı.

Ortak ODM profili dört işçi kullanır. Önceki ürünler yoksa çalışma başlangıcında en az 2 GiB boş RAM ve girdi büyüklüğüne göre disk alanı kontrol edilir. Mevcut 0,50 m arazi ve 0,20 m fotoğraf rasterı için boyut sınırları ortak ayarlardan gelir. CRS raster ve LAZ üzerinden, çalışma sınırı LAZ sınırlarından türetilir. Daha büyük sahalar için otomatik bölme doğrulanmadı. Tahmin edilen zemin, doğrudan zemin noktası desteği ve NoData ayrı rasterlarda saklanır.

## Sheffield doğrulaması

Bu alpha, `0a16d8c` tam saha çalışması üzerine `codex/omega-run-demo-alpha` dalında geliştirildi. Yalnızca ODM ara ürünleri yeniden kullanıldı. Geçerli kaynak 172/172 kamera, yerel GeoAI sonucu 25 aday ve Blender sonucu 8 ayrı LOD1 hacimdi. Çalışma kutusunun en doğu sınırı, önceki yaklaşık kutudan farklı olarak PDAL’ın bildirdiği LAZ sınırından geldi. Aynı model farklı raster boyutunda çalıştı. Bu yüzden 25 aday, önceki 28 adaylık tam saha deneyinden farklıdır.

Blender 5.3, yeni sahneyi ayrı süreçte yeniden açtı. Native dosyada paketli fotoğraf dokusu ve ayrı `Photographic terrain` ile `Buildings - detected roof LOD1` koleksiyonları doğrulandı. GLB yeniden içe alındı. Kabul sayısı geometrik kontrollerden geçen hacim sayısıdır. Bina envanteri veya ölçülmüş mutlak kot doğruluğu ifade etmez. İlk masaüstü denemesinde GeoAI dağıtımının yanlış paket adıyla sorgulanması girdi ve araç doğrulamasını durdurdu. Arayüz günlük yolunu gösterdi ve Blender düğmesi kapalı kaldı. Sorgu `geoai-py` paket adıyla düzeltildi.

Son sürüm `run_demo.bat` üzerinden çalıştırıldı. Klasör seçici 172 fotoğraf gösterdi. **Start Reconstruction** çalışırken ikinci başlatma kapalı kaldı. Koşu 69,8 saniyede tamamlandı. **Open Blend File** düğmesinden açılan Blender pencere başlığı `outputs/demo-uav/run_20261113T154849169068Z_8d2a1c3c/scene/terrain_white_buildings.blend` yolunu gösterdi. Blender’da kortlar, otopark, akarsu koridoru ve ev sıraları görüldü. 246,5 × 200,0 m sahnede 364.752 arazi üçgeni ve 1280 × 1045 paketli doku doğrulandı. Genel görünüm ve bina yakın planı aynı koşunun `scene/preview/` klasöründe. `scene/reports/` yeniden açma, GLB yeniden içe alma, temel ve zemin kanıtını içeriyor.

Ardından gerçek klasör seçicide fotoğrafsız bir klasöre geçildi. Eski yol temizlendi ve **Open Blend File** anında kapandı. Başlatma, girdi doğrulamasında “At least 3 JPEG/PNG photos are required. Found 0.” hatası verdi. Arayüz `outputs/demo-uav/run_20261113T155137070573Z_c64c4f20/input.log` yolunu gösterdi ve düğme kapalı kaldı.

Tam test paketi 224 geçti ve 9 isteğe bağlı test atlandı. GIS ortamında önbellek ile aktarım kontrollerinin dördü geçti. Bu oturumda ODM yeniden hesaplanmadan doğrulanmış ürünler kullanıldı. Önbelleksiz masaüstü koşusu ve ikinci veri seti deneyi açık.

Temel hazırlığı doğrudan zemin hücrelerini korudu. Değişen 4.020 DTM hücresinin 1.908’i reddedilen aday tamponlarıyla kesişti. Bu 477 m², kabul edilen adaylarla paylaşılabilen tamponları da içerir. Değişim raporu yerel etkileri gösterir. Reddedilen adayları temel işleminden ayıran kalite düzeltmesi ve yeni veri seti deneyi sonraki çalışmaya bırakıldı.

Tez konusu “Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon” olarak korundu. Alpha sonuçları ürün doğrulamasıdır. G0–G6 ve önceki olumsuz adaptif rekonstrüksiyon bulgusu ayrı deney kayıtlarında duruyor.
