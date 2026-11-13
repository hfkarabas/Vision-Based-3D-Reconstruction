# Mimari (plan adımı 3)

Tarih: 2026-11-13. Bu belge Codex dönemindeki `PROPOSED_ARCHITECTURE.md` belgesinin yerine geçer.
Gerekçe: literatür taraması (Claude Doc) ve `research/DATASET_PROTOCOL.md`.

## 1. İlke

Her aşamayı kanıtlanmış bir açık kaynak araç yapar. Bizim kodumuz yalnızca dört şey içerir:
1. Orkestrasyon: araçları çağırır, aşamaları sırayla çalıştırır, sonucu kaydeder.
2. Ayar türetme: sabit sayı kullanmaz, değerleri verinin kendisinden hesaplar (GSD, görüntü sayısı, RAM).
3. Kalite geri besleme zamanlayıcısı: araştırmanın özgün katkısı, plan adımı 5.
4. Değerlendirme: bağımsız referansa göre doğruluk ve tamamlanmışlık ölçümü.

Yeni paketin adı `src/uav3d`. Eski `src/vision3d` legacy olarak kalır ve yeni pakete import edilmez.

## 2. Araç zinciri

| Aşama | Araç | Sürüm | Kaynak |
|---|---|---|---|
| SfM, yoğun MVS, DSM/DTM, ortofoto | ODM (OpenSfM, OpenMVS, PDAL SMRF, GDAL dahil) | 3.6.2 | `ODM_Setup_3.6.2.exe` (GitHub release) |
| Nokta bulutu, raster işlemleri | ODM içindeki `pdal.exe` ve `gdal*.exe` | ODM'ye bağlı | ayrı kurulum gerekmez |
| LoD1.2/LoD2.2 bina | Roofer | 1.0.0 | `roofer-windows-x86_64-v1.0.0.zip` |
| Bina taban izi | OSM Overpass; yoksa isteğe bağlı ML tespiti | — | `gis` / `buildings-ml` ek grupları |
| Python tarafı | rasterio, geopandas, shapely, pyproj, laspy, cjio | `pyproject.toml` → `gis` | PyPI tekerlekleri |
| Görselleştirme | Blender (isteğe bağlı), glTF | 5.x | kullanıcı kurulumu |

Bu seçimin sonuçları:
- Codex'in ayrı conda GIS ortamına (4,1 GB) gerek kalmıyor, çünkü PDAL ve GDAL ODM'nin içinde geliyor.
- Qgis2threejs zincirine de gerek kalmıyor; arazi mesh'i ODM'nin `dem2mesh` aracından ya da DTM'den glTF'e üretiliyor.

## 3. Kurulum

Tek komut: `powershell -File setup.ps1`

1. `uv sync --extra gis --extra dev` komutuyla Python ortamını `.venv` olarak kurar.
2. Araçları `UAV3D_TOOLS` dizinine kurar. Varsayılan `<repo>/.tools/`, `.gitignore` kapsamında.
   - ODM: kurulum dosyası indirilir ve SHA-256 değeri `tools.lock.json` ile karşılaştırılır. Ardından `/VERYSILENT /CURRENTUSER /DIR=.tools\odm` ile sessiz kurulur.
   - Roofer: zip indirilir, SHA-256 kontrol edilir, açılır.
3. `uav3d doctor` komutu her aracın sürümünü ve yolunu doğrular. Eksik araç varsa nedenini söyleyerek hata verir.

Hiçbir yol `outputs/` altına bağlı değildir. Her araç önce ortam değişkenine (`UAV3D_ODM`, `UAV3D_ROOFER`, `UAV3D_BLENDER`), sonra `UAV3D_TOOLS` dizinine bakar.

## 4. Akış

```
uav3d run <görüntü klasörü> --out <çıktı> [--budget-ram-gb N] [--budget-minutes M]
```

1. **ingest:** EXIF'ten görüntü sayısını, kamerayı ve GPS bilgisini okur. GPS yoksa yerel koordinatla devam eder ve bunu kayda yazar.
2. **plan:** RAM bütçesi ve görüntü sayısından ODM ayarlarını türetir. Gerekirse `--split` ve `--max-concurrency` belirlenir. Kurallar bölüm 5'te.
3. **sfm:** ODM'yi `--end-with opensfm` ile çalıştırır. `reconstruction.json` dosyasından gerçek GSD hesaplanır: her çekim için odak uzaklığı × görüntü genişliği ve zemine göre yükseklik kullanılır.
4. **dense + dem + ortho:** ODM `--rerun-from openmvs` ile devam eder. Çözünürlükler GSD'nin katı olarak verilir.
5. **buildings:** taban izleri (OSM ya da tespit) ODM nokta bulutuyla Roofer'a verilir. Çıktı LoD2.2 CityJSON; başarısız binalar için LoD1.2'ye geri düşülür.
6. **export:** arazi mesh'i ve binalar glTF ile CityJSON olarak yazılır. Blender sahnesi isteğe bağlı.
7. **record:** `run.json` dosyasına araç sürümleri, türetilen tüm ayarlar, aşama süreleri, tepe RAM ve hata nedenleri yazılır.

Plan adımı 5'teki zamanlayıcı 4. aşamanın yerine geçen bir döngü olacak. Döngü karolar üzerinde çalışır ve aynı kayıt biçimini kullanır.

## 5. Ayar türetme kuralları (dataset bağımsız)

| Ayar | Kural | Eski sabit (Codex) |
|---|---|---|
| Yoğun görüntü çözünürlüğü | `--pc-quality` RAM bütçesinden seçilir. Varsayılan `high`, yani yarım çözünürlük | 800 px sabit |
| DSM/DTM hücresi | 2 × GSD | 0,50 m sabit |
| Ortofoto hücresi | 1 × GSD | 0,20 m sabit |
| Bölme (`--split`) | Görüntü sayısı, ölçülen görüntü başına bellek ile bütçeden hesaplanır | yok |
| Zemin filtresi | ODM SMRF, yayımlanmış varsayılanlarıyla. Değiştirilecekse yalnızca ayar setlerinde | Sheffield'da ayarlanmış |
| Bina en küçük alan | (8 × GSD)² ile 10 m² arasındaki büyük olan | 16 m² sabit |

Kural katsayıları yalnızca **ayar** setlerinde belirlenir. Belirlendikten sonra `configs/defaults.json` dosyasında dondurulur. Bu dosyada datasete özel hiçbir değer yoktur.

## 6. Kod düzeni

```
src/uav3d/
  cli.py          # run, doctor, eval
  tools.py        # araç bulma ve sürüm doğrulama
  ingest.py       # EXIF, GPS, kamera
  plan.py         # ayar türetme (bölüm 5)
  odm.py          # ODM çağrısı, GSD hesabı
  buildings.py    # taban izi + Roofer
  export.py       # glTF / CityJSON
  record.py       # run.json, süre ve RAM ölçümü
  evaluate/       # referansa göre P/R/F (eylem seçmeyen ayrı modül)
configs/defaults.json
setup.ps1, tools.lock.json
```

`evaluate/` modülü referans veriyi okuyan tek yerdir. `plan.py` ve zamanlayıcı onu import edemez; bu kural bir testle denetlenir (protokol bölüm 5.4).
