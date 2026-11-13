# Sheffield alpha kalite doğrulaması

Bu doğrulama `codex/omega-run-demo-alpha` dalındaki `3e55b9c` masaüstü koşusunu temel alır. Tez konusu “Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon” olarak kalır. Bu çalışma ürün doğrulamasıdır. G0–G6 bulguları ve olumsuz adaptif rekonstrüksiyon sonucu önceki kayıtlardadır.

## Gerçek masaüstü koşusu

- Önceki koşu `outputs/demo-uav/run_20261113T154849169068Z_8d2a1c3c/`.
- Doğrulanan koşu `outputs/demo-uav/run_20261113T161219050033Z_ab3fa5f5/`. Windows'ta `run_demo.bat` üzerinden Sheffield'ın 172 orijinal fotoğrafı seçildi. **Start Reconstruction** tamamlandı ve **Open Blend File** bu koşunun dosyasını Blender 5.3.0'da açtı. Blender pencere başlığındaki yol, koşu kaydıyla eşleşti. Toplam süre arayüzde 56,8 saniye, kayıtlı kamera sayısı 172/172.
- ODM 3.6.2 rekonstrüksiyonu, mevcut 6.107.108 noktalı LAZ, DSM, DTM ve ortofoto doğrulanan önbellekten geldi. İki koşunun işlenen ortofoto dosyalarının SHA-256 değeri aynı (`3cd3212cb7658522fe2c8dc1c97a663b40cdc1bbd5b0e51495b1a3fadb2ec1bf`). Karşılaştırma aynı EPSG:32617 gridinde yapıldı. Kotlar aynı rekonstrüksiyon referansına aittir.
- Yeni `.blend` Blender arka planında yeniden açıldı. 1280 × 1045 fotoğraf dokusu paketli, arazi ve beyaz LOD1 nesneleri ayrı koleksiyonlarda. GLB yeniden içe aktarımında 1 arazi ve 13 kapalı bina katısı doğrulandı. 364.826 arazi üçgeni ölçüldü. Eşleştirilmiş GLB karşılaştırmasında iki modelin arazi yüzeyinde de dejenere üçgen saptanmadı. `scene/reports/blender_native_reopen.json` ve `scene/reports/blender_glb_reimport.json` ölçümleri içerir.

## 1. Doğu yolu ve reddedilen temeller

Eski akış, tüm otomatik adayların 1,5 m tamponunu GDAL FillNodata temel denemesine katıyor ve bina daha sonra elense bile bu DTM'yi Blender'a gönderiyordu. Eski `Building_0025` asfalt/peyzaj üzerinde yanlış tespitti. Temel raster kapsamı %96,78 olduğu için mevcut %98 koşulundan geçemedi. Yeni geçici deneme mevcut kotları koruyunca aday raster koşulunu geçti, ardından Blender yüzey sürekliliği/çatı açıklığı kontrolünde elendi. Eski temel maskesi ölçülen iki hücrede de 1, yeni maske 0.

| Doğu yolu, EPSG:32617 | Başlangıç DTM | Eski DTM | Yeni DTM | Eski Blender | Yeni Blender |
|---|---:|---:|---:|---:|---:|
| 333259,75 E / 3102905,25 N | −66,647 m | −62,329 m | −66,647 m | Yüzey boşluğu | −66,647 m |
| 333263,25 E / 3102905,25 N, tampon içi en büyük artış | −66,774 m | −62,329 m | −66,774 m | −62,329 m | −66,774 m |

Tamponda eski DTM'nin değiştirdiği 334 hücre vardı. Yeni DTM'de sayı 0. Tüm sahada yalnız reddedilen adayların tamponlarına düşen değişiklik 1.908 hücreden 0'a indi (477 m² → 0 m²). Doğrudan gözlenen 138.274 zemin hücresi korundu. Kesitteki 121 yol örneğinde Blender yüzey isabeti 93'ten 121'e çıktı. Eski ve yeni renderlarda aynı kamera ve kadraj kullanıldı. Eski Blender modelindeki sivri yüzey kesit ve renderla doğrulandı. Kullanıcının işaret ettiği üçüncü Sheffield ekran görüntüsünün özgün dosyası bu doğrulamada bulunamadığı için o görüntünün kamerasıyla bire bir eşleştirme yapılmadı.

Üretim zinciri her denemede aynı başlangıç DTM'sinden başlar. GDAL FillNodata ve mevcut gözlenen zemin maskesi geçici temel rasterını hazırlar. Rasterstats ve GDAL çatı–zemin koşullarını, Qgis2threejs ve Blender ise gerçek yüzeyde temel ve çatı açıklığını sınar. Reddedilen adaylar çıkarıldıktan sonra kalanlarla başlangıç DTM'sinden yeniden üretilir. Kalan adayların tamamı geçene kadar işlem sürer. Sheffield'da iki tur yeterli oldu. Son mesh yeniden kontrol edilir. Son DTM'de onaylı tamponların dışındaki veya gözlenen zemin üzerindeki değişiklik koşuyu hataya düşürür. Orijinal ürünler, eşikler ve görüntüler korunmuştur.

## 2. Tek katlı evler ve kontrol alanları

GeoAI 0.43.1 ile aynı ODM ortofotosunda Mask R-CNN 25, WHU EfficientNet-B4 + UNet++ 19 aday verdi. WHU, mevcut ana adaylarla iki yönde %20'den fazla örtüşmeyen 12 ek aday sağladı. Üretimde toplam 37 aday değerlendirildi. Blender geometri kontrolünden geçen 13 hacmin 8'i önceki modelden, 5'i WHU'dan geldi. Bu sayı arazi teması ve çatı açıklığını geçen hacimleri gösterir. Aşağıdaki görsel inceleme yaklaşık çatı alanlarını kapsar. %80 ve üzeri tam, %5 üzeri ve %80 altı kısmi, %5 ve altı tespit yok olarak sınıflandırıldı.

| Ortofotoda incelenen çatı bölgesi | Mask R-CNN | WHU | Son tespit | İlk kayıp ve son geometri |
|---|---:|---:|---:|---|
| Kuzeybatı uzun çatı | %20 | %98 | %20, kısmi | Ana model kısmi. WHU tam öneri mevcut hacimle örtüştüğü için eklenmedi. Eski kısmi hacim 0010 duruyor. |
| Kuzeyde alçak açık renk çatı | %0 | %71 | %71, kısmi | İlk kayıp Mask R-CNN tespiti. WHU 0028'in eşleşen çatı ve temel örnek kapsamı %100 ve çevre zemin desteği 953 hücre. Çatı–DTM farkı 0,94 m kaldığından rasterda elendi. |
| Kuzeydoğu ara çatı bölgesi | %0 | %32 | %32, kısmi | İlk kayıp tespit. 0030 ve 0033 iki ayrı parça olarak geçti. Tam çatı birleşimi yok. |
| Kuzeydoğu uçtaki kahverengi ev | %0 | %84 | %84, tam | İlk kayıp tespit. WHU 0031 geçti. 459 çevre zemin hücresi, %100 çatı ve temel örnek kapsamı, raster farkı 3,72 m. Blender tabanı −67,19 m, çatı −62,88 m. |
| Doğudaki alçak çatı | %0 | %47 | %47, kısmi | İlk kayıp tespit. 0034 rasterdan geçti (2,72 m, 712 çevre zemin hücresi). Blender temel rölyefi/çatı açıklığı kontrolünde elendi. |
| Doğudaki gri ev | %0 | %0 | %0 | İki model de kaçırdı. Görünür bölgenin DSM medyanı −63,52 m, ilk DTM medyanı −66,87 m. |
| Doğu alt sıra | %70 | %0 | %70, kısmi | Mask R-CNN'in eski iki kısmi hacmi 0015 ve 0016 korundu. |

Yedi örnek bölge içinde eski tespit sayımı **0 tam, 2 kısmi, 5 tespit yok**, yeni sayım **1 tam, 5 kısmi, 1 tespit yok**. Geometriyi geçen hacimler eski koşuda 2, yeni koşuda 4 çatı bölgesinde görülüyor. Yeni hacim 0036 ortofotodaki park sundurmasıyla örtüşüyor. 0035 doğudaki çatı kenarı ve araç girişinin birleşiminde kalıyor. Bağımsız bina olarak görsel teyidi yoktur. Önceden var olan 0019 ve 0023 aynı çatı üzerinde örtüşen iki hacimdir. Bunlar ve örnek bölge sınıfları tüm saha için kesin bina sayısı veya doğruluk oranı üretmez.

Yanlış tespit örnekleri mavi kort, basketbol kortunun kenarı, iki havuz, otoparkın kenarı ve doğu yolundaki 0025. İncelenen kontrollerde bu tespitlerden hacim çıkmadı. İki havuz, kortlar, otopark ve akarsudaki 8.002 Blender örneğinin tamamında iki koşuda da yüzey bulundu. Her kontrolün eşleşen noktalardaki mutlak kot değişimi medyanı 0 m. Önceki DTM'de bir havuzda 60, diğerinde 1 ve otoparkta 23 değişen hücre vardı. Yeni koşuda bu üç bölgede fark 0.

**Zemin sınıflandırma karşılaştırması.** Aynı ODM LAZ üzerinde PDAL 2.10.2 ile hazır CSF çalıştırıldı (0,5 m, rigidness 3, 500 iterasyon, eşik 0,5). Sınıf 2, aynı 0,5 m hücrelere GDAL ile minimum kot kullanılarak aktarıldı. Kuzey alçak çatısının WHU iç bölgesinde zemin olarak etiketlenen hücreler mevcut SMRF'de 292/670, CSF'de 656/670 idi. Kuzeybatı uzun çatının iç bölgesinde 7/989 → 986/989 artışı da çatı kotunu zemine taşırdı. CSF doğudaki alçak çatının iç bölgesinde 15/207 → 1/207 iyileşme gösterdi. Mevcut GDAL temel denemesi bu doğu adayını raster aşamasından geçirdi ve Blender'da temel/çatı kontrolü geçmedi. CSF değişimi genelleştirilmedi. Mevcut sınıflandırma ve temel eşikleri korundu.

## Tekrar çalıştırma ve kanıt

Depo kökünde `run_demo.bat` açıp `datasets/omega-terrain-validation/sheffield/full` klasörünü seçin. Aşağıdaki komutlar aynı koşuları tekrar üretmeden ölçüm ve eşleştirilmiş renderları üretir. CSF çalıştırma komutu `outputs/demo-uav/sheffield-quality-review/csf.json`, PDAL yapılandırması yanındaki `csf_pipeline.json` dosyasında kayıtlıdır. Yeni çıktı dizini seçerek var olan raporları koruyun.

```powershell
$old = 'outputs/demo-uav/run_20261113T154849169068Z_8d2a1c3c'
$new = 'outputs/demo-uav/run_20261113T161219050033Z_ab3fa5f5'
$qa = 'outputs/demo-uav/sheffield-quality-review'
$repeat = "$qa/reproduction"
$blender = 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe'
$env:PYTHONPATH = (Resolve-Path '.venv/Lib/site-packages').Path
$env:MPLCONFIGDIR = (Resolve-Path $qa).Path
& 'outputs/mission-omega/building-recovery/runtime/venv/Scripts/python.exe' scripts/evaluate_sheffield_alpha_quality.py --before $old --after $new --output $repeat --regions Documentation/evaluation/sheffield_alpha_quality.json --blender $blender --csf "$qa/csf_classified.tif"
& $blender -b -t 4 --python-exit-code 1 --python scripts/omega_terrain_blender.py -- compare --before "$old/scene/terrain_white_buildings.glb" --after "$new/scene/terrain_white_buildings.glb" --before-grid "$old/scene/grid.json" --after-grid "$new/scene/grid.json" --views "$repeat/views.json" --local-only --output "$repeat/matched-views"
```

- Koşu kaydı `run.json` model revizyonları, SHA-256, araçlar ve çalıştırılan komutları içerir. [GeoAI WHU](https://opengeoai.org/examples/building_detection_whu/) · [GeoAI Mask R-CNN](https://opengeoai.org/examples/building_footprints_usa/) · [PDAL CSF](https://pdal.org/en/stable/stages/filters.csf.html) · [GDAL FillNodata](https://gdal.org/en/stable/programs/gdal_fillnodata.html)
- Değerlendirme `outputs/demo-uav/sheffield-quality-review/evaluation.json`, `east_road_section.csv`, `east_road_section.png`, `models_north.png` ve `models_east.png`. `matched-views` aynı kamera parametreleri ve altı Blender renderını içerir.
- Sahne dosyası `outputs/demo-uav/run_20261113T161219050033Z_ab3fa5f5/scene/terrain_white_buildings.blend`. Aynı dizinde GLB, fotogrametrik ortofoto üzerinde aday konturları ve ayrı doğrulama kayıtları bulunur.
- Tam otomatik test paketi, GIS bağımlılıkları yüklü ortamda 235 başarılı. Yeni testler, geçici bir turda başarılı olup sonraki raster turunda elenen adayı ve yayınlanan DTM'de yetkisiz veya doğrudan gözlenen hücre değişikliklerini kapsar.
- Arazi düzeltmesi doğrulandı. Ev kapsamı kısmen düzeldi. Eksik ve kısmi çatılar, bölünmüş ve örtüşen hacimler ile 0035'in görsel belirsizliği sürüyor.
