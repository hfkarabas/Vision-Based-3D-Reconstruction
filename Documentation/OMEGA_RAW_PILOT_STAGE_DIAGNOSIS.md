# Sheffield temel zinciri ve Urban derinlik kaybı

Başlangıç `16f0326b95255aec1265241a3b4f0f1b42703803`, dal `codex/omega-reusable-uav-pipeline`. Her iki pilotun COLMAP, ortofoto ve DSM çıktıları yeniden kullanıldı. G0–G6 ve olumsuz adaptif rekonstrüksiyon sonucu korundu. Araştırma konusu **Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon**. Tam veri seti çalıştırılmadı.

## Sheffield

İlk çalıştırmada PDAL boş `returns` argümanını reddetti. Argüman çıkarıldı. Fotogrametrik PLY'de LiDAR dönüş alanları olmadığı için PDAL tüm noktaları işledi. Başlangıç, ara ve son sürüm özetleri `outputs/mission-omega/reusable-uav/stage-diagnostics/` altında `sheffield-executed-version.json`, `sheffield-pdal-interface-fix.json` ve `executed-source-and-assets.json` dosyalarındadır.

### Zemin sürekliliği

Ölçüm, önceki denetimin aynı 0,20 m piksellerini kullanır. DTM gözlemi PDAL zemin sınıfına, tahmin GDAL FillNodata ile tamamlanan hücreye karşılık gelir. Son sütun düşey Blender BVH isabetlerinin en büyük 4-komşulu bileşeninin ilgili etiket alanına oranıdır. Ayrı etiketlerin bir arada ölçülmesi birleşim metriğini sınırlar.

| Bölge | DSM | Gözlenen DTM | Tahmin DTM | Toplam DTM | Blender mesh | En büyük parça |
|---|---:|---:|---:|---:|---:|---:|
| Görünür zemin | %95,85 | %95,25 | %0,37 | %95,62 | %90,29 | %78,19 |
| Açık zemin | %95,99 | %95,65 | %0,09 | %95,74 | %90,62 | %79,10 |
| Avlu geçişi, 9,80 m² | %83,67 | %60,82 | %24,49 | %85,31 | %61,22 | %31,84 |

Temel denemesi ve SMRF başlangıcı görünür zeminde aynı DTM ve mesh kapsamını verdi. Avlu geçişinin %16,33'ü DSM'de yoktu. Sınıflandırma ve kısmi geri kazanım sonrasında **24,08 yüzde puan** DTM desteği mesh'te kayboldu. Mevcut Blender grid birleştiricisi NoData köşesini ve ona bağlı yüzleri çıkarıyor. Açık zeminde aynı kayıp 5,11 puan. Yerel süreklilik sorunu çözülmedi. Ayrı Blender süreçlerinde BLEND ve GLB üzerindeki 119.025 ışının maskesi eşleşti, kotlar 1 mm toleransı içinde kaldı.

### Çatı, yükseklik ve temel

WHU üç poligon verdi. Kuzeydeki gri çatıyla 0,663 IoU elde etti ancak poligon kapsam sınırında kesilmişti. Batı ve orta çatıda kısmi örtüşme vardı (IoU 0,264 ve 0,377). Doğudaki kahverengi çatı kaçırıldı. IoU değerleri değerlendirme için çizilen çatı içlerine göredir. SMRF tabanında çatı altındaki DTM yüksek kaldı ve içteki iki adayın çatı–zemin farkı 0,26 ve 0,21 m idi. CSF ile 3,83 ve 3,93 m elde edildi. İki adayın temel raster kapsamı %98 eşiğinin altında kaldı.

| İşlem | Hazır işlem | Proje kararı |
|---|---|---|
| Zemin ve DTM | PDAL SMRF/CSF ve GDAL FillNodata | Sınıf 2, grid, dolgu sınırı ve destek maskesi |
| Çatı kotu ve yükseklik | GeoAI vektörleme, GDAL DSM−DTM, rasterstats zonal medyan, GEOS üçgenleme | Çatı iç tamponu, çevre örnek sayısı, kapsam, rölyef ve kot yayılımı eşikleri |
| Temel ve hacim | GDAL sınıflandırılmış çevre zemini interpolasyonu, GEOS dışbükey zarf, Blender BVH ve BMesh extrusion | Çatı çevresini ankrajdan çıkarma, 8 m arama sınırı, düz taban için en düşük çevre kotuna 0,10 m gömülme ve mevcut Blender temas/açıklık kapıları |

Yalnız yoğun hücrelere izin veren deneme ve bir/iki hücrelik morfolojik kapatma, temel çevresindeki mesh boşluğunu gideremedi. Son opt-in deneme, GDAL interpolasyonunu yakın, sınıflandırılmış zemin ankrajlarının GEOS dışbükey zarfıyla sınırladı. 53 yoğun noktasız hücre çevre ankrajlarından tahmin edildi. Bunlar doğrudan zemin gözlemi sayılmadı. Üretim varsayılanları korunuyor.

GeoAI RGB Mask R-CNN alternatifi iki poligon üretti. Kuzey gri çatı adayının inceleme alanını kapsaması %99,60, IoU değeri **0,730**. Aday raster kapısından, 51 çevre ve 33 iç Blender kontrolünden geçti. **1 doğrulanmış LOD1**. Medyan yükseklik **3,99 m**, yerel çatı kotu **−64,64 m**, düz temel kotu **−69,16 m**, en düşük örneklenen çatı açıklığı **2,08 m**, en yüksek çevre gömülmesi **2,45 m**. Düşey datum doğrulanmadı. Çevredeki boşluklar yakın planda görülebilir. İkinci alternatif adayın çevresinde yalnızca üç gözlenen zemin hücresi vardı ve aday raster desteği kapısında kaldı.

WHU `09df9efd323bbd3d56b98b4857129eb9b5baa2d3`, RGB Mask R-CNN `aa2b25d27b2c78f0e92c7b6a4813a571f9bc2288`. PDAL 2.10.2, GDAL CLI 3.13.3, rasterstats 0.20.0, GeoAI 0.43.1 ve Blender 5.3.0 kullanıldı. [PDAL SMRF](https://pdal.org/en/stable/stages/filters.smrf.html) · [GDAL FillNodata](https://gdal.org/en/stable/programs/gdal_fillnodata.html) · [GDAL raster calc](https://gdal.org/en/stable/programs/gdal_raster_calc.html) · [rasterstats](https://pythonhosted.org/rasterstats/manual.html) · [GeoAI alternatifi](https://opengeoai.org/examples/building_footprints_usa/)

## Urban

23 kayıtlı COLMAP 4.2.1 kamerası için pycolmap izdüşümü, ardından fotometrik/geometrik derinlik haritalarının geri izdüşümü kullanıldı. Derinlik örnekleri 0,50 m XY hücrelerinde ve değerlendirme yüzeyinin ±1,5 m çevresinde sınandı. Yüzey maskesiz kamera mesh'inden geliyor. Eksik kotlar için bölge medyanı kullanıldı (ana yolun %30,2'si). Görüş hacmi optik görünürlük için üst sınırdır. Örtülme, poz ve yüzey hatası ayrı ayrı doğrulanmadı. Hücrede birden fazla derinlik haritası bulunması COLMAP füzyon tutarlılığını garanti etmez.

| Bölge | En az 3 kamera görüş hacmi | Fotometrik ≥1 | Geometrik ≥1 | Geometrik ≥3 | Füzyon 3 | Füzyon 2 |
|---|---:|---:|---:|---:|---:|---:|
| Ana yol | %100 | %99,95 | %67,92 | %35,96 | %29,16 | %34,17 |
| Güney kaldırım | %100 | %95,41 | %82,46 | %49,62 | %40,50 | %48,16 |
| Doğu açık zemin | %100 | %14,71 | %0,00 | %0,00 | %0,00 | %0,46 |
| Görünür zemin | %100 | %82,33 | %57,76 | %31,86 | %25,89 | %30,58 |

Doğuda ilk ölçülebilir kayıp fotometrik derinlikte başladı. Geometrik haritalarda aynı kotu destekleyen örnek kalmadı. Yol ve kaldırımda önce geometrik tutarlılık sonrasında, sonra füzyonda ek kayıp var. Yalnızca `StereoFusion.min_num_pixels=3 → 2` değiştirildi. Kameralar, derinlik haritaları ve 0,50 m sayım reçetesi aynı. 2,63 sn süren deney 148.492 yerine 362.113 nokta üretti. Görünür zemin desteği 4,69 yüzde puan arttı. %85 hedefinin altında kaldı. Doğruluk kazancı ölçülmedi. Üretim ayarı 3. [COLMAP füzyon uygulaması](https://github.com/colmap/colmap/blob/main/src/colmap/mvs/fusion.cc)

## Yeniden çalıştırma

Komutlar ve çıktı yolları `outputs/mission-omega/reusable-uav/stage-diagnostics/` altındaki JSON günlüklerindedir. Doğrulanmış Sheffield çıktısı `sheffield-verified/`, Urban karşılaştırması `urban-depth-baseline/` ve `urban-depth-min2/` altındadır.

```powershell
$o = 'outputs/mission-omega/reusable-uav'
$a = "$o/stage-diagnostics"
$p = "$o/sheffield-pilot"
$r = 'outputs/mission-omega/building-recovery'
$env:PYTHONPATH = (Resolve-Path "$o/gis-extra").Path + ';' + (Resolve-Path "$r/terrain-validation/alternate-python").Path
& "$r/runtime/venv/Scripts/python.exe" scripts/omega_uav_foundation_trial.py --ground "$a/sheffield-csf/ground" --dense "$p/orthophoto/dense_count.tif" --polygons "$a/sheffield-alternate/geoai_polygons.gpkg" --inference "$a/sheffield-alternate/inference.json" --orthophoto "$p/orthophoto" --asset-template "$p/buildings" --pdal 'outputs/mission-omega/tools/gis/Library/bin/pdal.exe' --blender 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe' --radius 8 --surrounded --output "$a/sheffield-foundation-replay"
```

Otomatik testlerde 221 test geçti ve beş isteğe bağlı GIS testi atlandı. Bu oturumun gerçek GIS ve Blender işlemleri ayrıca çıktı üzerinde kontrol edildi.

**PARTIAL.** Bir Sheffield binası örneklenen temas ve yükseklik kapılarını geçti. Avlu geçişi ile görünür zemin sürekliliği ve Urban yoğun zemin kapsamı eksik kaldı.
