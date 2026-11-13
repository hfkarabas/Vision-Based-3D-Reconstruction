# Sheffield Cross tam saha rekonstrüksiyonu

## Girdi ve karşılaştırma

Sheffield Cross kaynak arşivi `143f556ea3128835ab46cd694714f8f78319699b` revizyonundan gelir. Arşivin lisansı Unlicense. 4000 × 3000 çözünürlükteki 172 orijinal JPEG’in SHA-256 değerleri mevcut COLMAP önbelleğine ve OpenDroneMap çalışma kopyasına karşı doğrulandı. COLMAP’ta 172 kamera, 112.888 seyrek ve 1.395.810 yoğun nokta var. Yoğun eşleşmenin önceki çözünürlüğü 640 px. Bu çalışma o COLMAP aşamasını tekrar çalıştırmadan kamera dokulu Delaunay yüzeyinden yeni ortofoto ve DSM üretti. GCP tablosu mevcuttu. İki rekonstrüksiyon da kontrol noktalarını çözüm girdisi olarak kullanmadı. Düşey datum için bağımsız doğrulama bulunmuyor. [Veri kaynağı](https://github.com/pierotofy/drone_dataset_sheffield_cross/tree/143f556ea3128835ab46cd694714f8f78319699b) · [COLMAP](https://github.com/colmap/colmap)

- Önceki tam saha `outputs/mission-omega/building-recovery/terrain-validation/sheffield-full/` altında. Dokusu yoğun nokta renklerinin yukarıdan izdüşümüydü. 28 karelik pilot `outputs/mission-omega/reusable-uav/sheffield-pilot/` altında.
- Yeni çalışma `outputs/mission-omega/reusable-uav/sheffield-full-production/` altında. Girdi envanteri doğrulaması `cache_verification.json` içinde. Kapsam kutusu EPSG:32617 cinsinden `[333031, 3102867, 333282, 3103066]`. PDAL sayımında 1.395.810 yoğun noktanın tamamı bu kutuda.
- Bölge etiketleri `Documentation/evaluation/sheffield_full_regions.json` içinde. Ölçüm, ortak 0,25 m merkez örnekleri ve Blender BVH düşey ışınları kullanıyor. Bağlantı 4-komşulukla ölçülüyor. Önceden bildirilen kort yüzdelerinin poligonları bulunmadığından aşağıdaki eski/yeni değerler aynı yeniden işaretlenmiş bölgelerden hesaplandı.

## COLMAP tabanlı aday

| Bölge | Eski DSM / DTM / mesh (%) | Yeni DSM / DTM / mesh (%) | Eski → yeni en büyük mesh parçası (%) | Yeni dokulu mesh (%) |
|---|---:|---:|---:|---:|
| Tenis | 100,00 / 70,25 / 45,56 | 99,17 / 99,07 / 98,30 | 33,99 → 98,30 | 98,30 |
| Basketbol | 100,00 / 78,66 / 65,41 | 95,11 / 95,37 / 91,16 | 64,51 → 91,16 | 90,77 |
| Park otoparkı | 97,86 / 72,26 / 57,65 | 83,65 / 84,91 / 75,40 | 57,65 → 73,17 | 74,10 |
| Akarsu koridoru | 100,00 / 86,03 / 79,75 | 97,79 / 97,69 / 95,38 | 41,59 → 95,38 | 95,29 |
| Ev sırası | 99,94 / 8,31 / 5,08 | 96,70 / 98,14 / 96,19 | 3,32 → 96,18 | 94,73 |
| Doğu sınırı | 64,59 / 47,70 / 29,35 | 36,16 / 37,17 / 22,00 | 13,65 → 11,16 | 21,05 |
| 49.949 m² çalışma kutusu | 81,83 / 44,24 / 36,79 | 71,88 / 72,58 / 65,95 | 15,13 → 64,52 | 65,21 |

Eski DSM 1 m nokta rasteriydi. Yeni 0,20 m DSM kamera dokusunun geçerli olduğu ve 0,50 m hücrede yoğun nokta bulunan piksellerle sınırlandı. Önceki DSM oranlarının daha yüksek olması tek başına daha fazla fotoğraf veya ölçülmüş yüzey anlamına gelmez. Doğu sınırındaki mesh kapsamı yine de geriledi.

Kortlarda yoğun nokta desteği %100. Basketbol kortunda ilk yeni eksik kamera dokulu yüzeyde oluşuyor. Otoparkın %7,4’ü yoğun nokta aşamasında eksik. Yoğun nokta olup kamera dokulu yüzey çıkmayan oran ayrıca yaklaşık %8,9. Mesh aşamasında boşluklar genişliyor. Doğu sınırında yoğun destek yalnızca %41,5. Sınırda ham kamera renderının yaklaşık %38’i yoğun destekten yoksun olduğu için son rasterdan çıkarıldı. Akarsu çevresinde çizgisel doku izlenebiliyor ve kesitte DTM ile Blender yüzeyi birbirini takip ediyor. Bağımsız su tabanı kotu bulunmuyor.

### Zemin ve mesh aktarımı

PDAL CSF ve SMRF aynı yoğun bulutu, aynı 0,50 m hücreleri ve aynı kamera DSM maskesini kullandı. Tenis ve basketbol alanlarında iki sınıflandırıcı da hücrelerin tamamında sınıf 2 nokta verdi. Ev sırası inceleme alanında SMRF %83,24, CSF %25,75 doğrudan zemin işaretledi. Ortak 143.590 hücrede SMRF eksi CSF medyanı 0 m, 95. yüzdeliği +3,18 m. Mutlak fark hücrelerin %11,86’sında 1 m’yi aştı. Görüntülerdeki çatılar ve daha düşük konut çevresi kotları, CSF’yi bu aday için daha tutarlı kıldı. Mutlak doğruluk ölçümü yapılmadı. [PDAL CSF](https://pdal.io/en/stable/stages/filters.csf.html) · [PDAL SMRF](https://pdal.io/en/stable/stages/filters.smrf.html)

Temel işlemi öncesindeki **aynı CSF DTM** kullanıldığında:

| Bölge | Blender grid DTM kaybı (m²) | Qgis2threejs DTM kaybı (m²) | Qgis2threejs mesh–DTM farkı medyan / p95 (m) |
|---|---:|---:|---:|
| Tenis | 2,25 | 1,88 | 0,036 / 0,144 |
| Basketbol | 18,19 | 15,94 | 0,021 / 0,098 |
| Park otoparkı | 94,19 | 81,00 | 0,030 / 0,148 |
| Akarsu koridoru | 9,50 | 7,38 | 0,020 / 0,085 |
| Tüm çalışma kutusu | 4.176,25 | 3.540,06 | 0,021 / 0,124 |

Her iki yüzeyde NoData örneklerine düşen mesh alanı 0 m². Kot farkları en yakın DTM hücresine göre hesaplandığından hücre içi interpolasyonu da içerir. Qgis2threejs 3.3’ün kendi `GridGeometry.loadData` işlemi kullanıldı. Node adaptörü Float32 rasterı aktarır ve çıkan geometriyi PLY olarak yazar. Blender bu geometriyi dokular ve mevcut temel kapılarını uygular. Windows QGIS başlatıcısındaki DLL sorunu nedeniyle eklentinin dağıtılan TypeScript geometri modülü Node’da bağımsız çalıştırıldı. [Qgis2threejs](https://github.com/minorua/Qgis2threejs) · [GDAL FillNodata](https://gdal.org/en/stable/programs/gdal_fillnodata.html)

### Adaylar ve doğrulama

GeoAI 0.43.1 WHU EfficientNet-B4 + UNet++ `09df9efd323bbd3d56b98b4857129eb9b5baa2d3` kamera ortofotosunda 20 poligon verdi. Yedisi raster kapısını geçti ve yedisi de ilk Blender temel kontrolünde kaldı. GeoAI RGB Mask R-CNN ResNet50 `aa2b25d27b2c78f0e92c7b6a4813a571f9bc2288` 40 aday verdi. Sınıflandırılmış çevre zemini ve GDAL’ın en fazla 8 m temel interpolasyonu ile 24 aday raster kapısını, 22 aday Blender temel ve çatı açıklığı kapılarını geçti. Temel DTM’sinde önceki rasterda boş olan 1.415 hücreye sınırlı temel tahmini yazıldı. Çatı yükseklikleri 2,57–4,39 m aralığında, medyan 3,41 m. Çatı izi taban izi yaklaşımıdır. Bazı çatılar kısmi veya eksik. Kort, yol, havuz ve araç üzerinde reddedilen tespitler var. 22 sayısı geometri kapısını geçen hacim sayısıdır. Etiketlenmiş bağımsız bina envanteri bulunmuyor. [WHU modeli](https://huggingface.co/giswqs/whu-building-unetplusplus-efficientnet-b4) · [GeoAI RGB çıkarıcısı](https://opengeoai.org/examples/building_footprints_usa/)

Görüntü üzerindeki kontrol, adayların bir kısmının gözlenen zemin üstünde olduğunu gösterdi. Genel `--preserve-observed` seçeneğiyle doğrudan desteklenen DTM hücreleri herhangi bir tespit tamponu içinde de korundu. Aynı 40 adaydan 22’si raster kapısını ve 19’u Blender kapısını geçti. Bu sürüm aynı mesh kapsamına sahip. Basketbol alanında bina adayının tamponuna bağlı tahmin payı %21,41’den %0,43’e indi. 19 hacimli COLMAP sürümü `delivery-keep-ground/` altında, 22 hacimli önceki varyant `delivery/` altında. İkisi de korundu. Sayılar eşsiz bina sayısını veya tespit doğruluğunu ölçmez.

COLMAP adayı Blender 5.3’te yeniden açıldı. 261.938 arazi üçgeni, 22 kapalı mat beyaz hacim ve paketli 1255 × 995 piksel doku doğrulandı. GLB bağımsız Blender sürecinde yeniden içe alındı. 799.184 düşey ışının destek maskesi ve isabet kotları BLEND ile GLB’de eşleşti. Arazi sınırı yaklaşık 248,5 × 196,0 m. 172 kameranın −69 m yerel referans düzlemine geometrik izdüşümü 51.956,71 m². Mesh bunun içinde 32.940,0 m² kapsıyor. Fotoğraf izdüşümü görünürlük, örtüşme veya yeniden yapılandırılabilir zemin garantisi içermeyen bir üst sınırdır.

Kamera ortofoto ara işlemleri Delaunay 68,17 s ve 542 MiB tepe süreç belleği, COLMAP doku 7,42 s ve 471 MiB, Blender ortofoto renderı 4,25 s. PDAL SMRF 2,05 s, CSF 3,33 s. GeoAI WHU 24,79 s ve 425 MiB tahsisli CUDA belleği. Alternatif model 22,64 s. Çalışma alanında `*.json` ve `*.log` yan dosyaları gerçek komutları saklıyor.

## Teslim edilen OpenDroneMap sürümü

İlk OpenDroneMap denemesinde Windows korumalı çalıştırma OpenSfM işçilerine erişim vermedi. Aynı girdiyle `opensfm` aşamasından devam eden yerel ODM 3.6.2 çalışması tamamlandı. OpenSfM 172/172 kamerayı tek bileşende kaydetti, 43.100 seyrek nokta çıkardı ve OpenMVS/ODM 6.107.108 filtrelenmiş yoğun nokta üretti. Ortofoto, DSM ve DTM gerçekten yazıldı. Dört işçiyle OpenSfM 209,44 s, OpenMVS 690,06 s ve ölçülen aşamaların toplamı yaklaşık 1.024 s. Birleşik tepe RAM ölçülmedi. Bölme ve birleştirmeye gerek kalmadı. Windows kurulum varlığı SHA-256 `c9dbfabf0d066529a7b6c753fd45bf3366a0314266160cd677373bab4caff294`. [ODM 3.6.2](https://github.com/OpenDroneMap/ODM/releases/tag/v3.6.2) · [ODM çıktı kılavuzu](https://docs.opendronemap.org/outputs/)

| Bölge | ODM DSM / DTM / mesh (%) | En büyük mesh parçası (%) | Dokulu mesh (%) | DTM, aynı hücrede zemin noktası olmadan (%) |
|---|---:|---:|---:|---:|
| Tenis | 100 / 100 / 100 | 100 | 100 | 0,00 |
| Basketbol | 100 / 100 / 100 | 100 | 100 | 0,00 |
| Park otoparkı | 100 / 100 / 100 | 100 | 100 | 17,95 |
| Akarsu koridoru | 100 / 100 / 100 | 100 | 100 | 0,74 |
| Ev sırası | 100 / 100 / 100 | 100 | 100 | 61,89 |
| Doğu sınırı | 91,52 / 91,37 / 90,23 | 90,23 | 90,23 | 38,89 |
| Öncekiyle ortak 49.949 m² | 90,25 / 90,22 / 89,88 | 89,88 | 89,87 | 21,89 |
| ODM tam 53.713 m² kutusu | 85,33 / 85,30 / 84,89 | 84,89 | 84,89 | 20,94 |

Otoparkta doğrudan yoğun kapsam %92,90. Kalan %7,10’da DSM’ye çevresel destekten değer geliyor. Akarsu inceleme bölgesinde yoğun destek %100 ve kesit boyunca Blender DTM’yi izliyor. Kesitte su yüzeyi, kıyı ve örtü geometrisi birbirine karışabilir. Modelden su tabanı derinliği çıkarılmadı. Doğu sınırında ODM yoğun desteği %67,22. DSM’nin %27,33’ünde aynı örnek hücrede yoğun nokta yok. Bu yüzden alan kazanımının tümü yeni ölçülmüş geometriden gelmiyor. Yerel ortofoto daha okunaklı kort çizgileri, araç ve kanal dokusu sağladı. Ev sırasındaki eksik ve bölünmüş çatı tespitleri ile doğu yolunun kesilen kenarı hâlâ görünüyor.

ODM’nin varsayılan üç adımlı DEM boşluk doldurması ve yerel yüzey interpolasyonu çalıştı. Yeşil harita hücreleri en az bir sınıf 2 nokta ile doğrudan desteklenir. Turuncu hücrelerde son DTM var, aynı 0,50 m hücrede sınıf 2 nokta yok. Gri hücreler NoData. Bu ayrım, turuncu hücrede başka bir yakın ölçünün kullanılmadığı anlamına gelmez. Mesh, ortak kutuda DTM’nin 171,75 m²’sini kaybetti. Mesh–DTM medyan farkı 0,005 m ve p95 0,042 m. 0,25 m ışın örneklerinde DTM NoData üzerinde yüzey bulunmadı. COLMAP ve ODM DTM’lerinin ortak kutuda medyan farkı +0,95 m. Doğu sınırında +2,37 m. Her ikisi de EXIF tabanlı olduğundan bu farklar bağımsız kot doğruluğu ölçüsü olarak kullanılmadı.

Son işlem alanı ODM yoğun bulutunun yuvarlanmış `[333027, 3102858, 333284, 3103067]` sınırından geliyor. Blender arazi sınırı EPSG:32617 cinsinden yaklaşık `[333032.25, 3102861.25, 333278.75, 3103061.25]`, 246,5 × 200,0 m. 51.956,71 m² düzlem fotoğraf izdüşümünün 45.596,38 m²’si Blender yüzeyiyle kesişiyor. Yaklaşık 6.360,33 m² izdüşüm alanı dışında kalıyor. Harita ölçümü beklenen geometrik görünümün üst sınırıdır.

Ana sahne, doğrudan zemin hücrelerini koruyan ODM + Qgis2threejs + GeoAI Mask R-CNN varyantıdır. 28 otomatik adaydan 11’i raster kapısını, 8’i Blender temel ve çatı açıklığı kontrollerini geçti. WHU alternatifinde 19 adaydan 9 hacim geçti ve o varyant ayrı tutuldu. Ana sahne, bina sayısı tek başına seçim ölçüsü olarak kullanılmadan, doğrudan zemin hücrelerinin korunması ve yeni temel interpolasyonunun izlenebilirliği için seçildi. İncelemede bazı tespitler çatı parçasına karşılık geliyor ve aynı binanın birden fazla parçası bulunabiliyor. Bağımsız envanterle doğrulanmış bina sayısı ve doğruluk yüzdesi yok.

Blender 5.3 native sahneyi yeniden açıp render etti. Ayrı Blender süreçleri GLB’yi yeniden içe aldı. İki dosyada 859.408 ortak tam saha ışınının destek maskesi ve ölçülen kotları eşleşti. 364.742 arazi üçgeni, paketli 1285 × 1045 fotoğraf dokusu, 8 kapalı mat beyaz LOD1, 0 dejenere arazi üçgeni doğrulandı. **Teslim PARTIAL.** Doğu uçuş sınırında yoğun nokta ve doku eksikleri, ev sırasında çatı tespiti eksikleri sürüyor. Önceki pilot ve eski tam çıktı yerinde korundu.

## Dosyalar ve tekrar çalıştırma

Çıktı kökü `outputs/mission-omega/reusable-uav/sheffield-full-production/`.

- Ana BLEND ve GLB `odm-delivery/sheffield_cross_full.{blend,glb}`. Üstten ve eğik renderlar `odm-delivery/preview/{top,oblique}.png`. Temel, UV, doku ve yeniden içe alma raporları `odm-delivery/reports/`.
- Sabit kamera karşılaştırmaları `odm-matched-views/`. Akarsu kesiti `profiles/odm_stream_cross_section.{png,csv}`. Bina kanıtı `odm-delivery/building_detection_overlay.png` ve `building_footprints.geojson`.
- Harita `odm-coverage-full-final/flight_vs_mesh.png`, gözlem ve interpolasyon `odm-coverage-full-final/dtm_evidence.png`, ölçümler `odm-coverage-common-final/coverage.json` ve `odm-coverage-full-final/coverage.json`.
- SHA-256 BLEND `c4aaa9b829313b5d8b5a82bc6c5cecc5144b796dd2ab584d3a4accf3cf13977e`. GLB `135004bc16b118360cf361b66b4d202501d093fa6acf2e4f8375df9ada4a03bf`. Bu dosyalar Git dışında.

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe' 'C:/projects/Vision-Based-3D-Reconstruction/outputs/mission-omega/reusable-uav/sheffield-full-production/odm-delivery/sheffield_cross_full.blend'
```

`Documentation/OMEGA_SHEFFIELD_REPRODUCE.md` komutları açıklar. Gerçek çalıştırma seçenekleri `odm/retry-command.json`, `odm/sheffield/options.json`, `comparison-odm.json`, `comparison-odm-full.json` ve işlem yan günlüklerinde bulunur.
