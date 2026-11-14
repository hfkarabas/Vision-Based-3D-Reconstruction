# CLAUDE.md — Vision-Based-3D-Reconstruction

Bu dosya, projede çalışan Claude oturumları için kalıcı bağlamdır. Her oturumun başında okunur.

## Proje hedefi

İHA (UAV) görüntülerinden 3B arazi (terrain) ve bina rekonstrüksiyonu.

Araştırma konusu (tez başlığı, değiştirilmez):
**"Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon"**

## Kurallar (kullanıcının koyduğu, her kararın çerçevesi)

1. **Önce literatür.** Geçmiş makaleler okunur; mümkün olduğunca açık kaynak yayınlanmış ve başarısı kanıtlanmış projeler/araçlar kullanılır. Kendi kodumuz yalnızca gerekli noktalarda yazılır (entegrasyon, orkestrasyon, araştırmanın özgün katkısı, değerlendirme).
2. **Araştırma konusu yukarıdaki başlıktır.** Ürün doğrulaması (bir sahnede çalışan demo) araştırma sonucu değildir; ikisi ayrı raporlanır.
3. **Genellik.** Program yalnızca seçili dataset için değil, bilgisayarda olan ve olmayan, ileride test/validation için kullanılacak tüm datasetlerde çalışmalıdır. Datasete özel eşik, yol, bölge dosyası veya sabit ayar üretim yoluna girmez. Ayar setleri ile hold-out doğrulama setleri baştan ayrılır.

## Teslim

**30 Kasım 2026: 40–50 sayfalık proje raporu (Türkçe) + 4–5 sayfalık konferans bildirisi** (tez değil). LaTeX,
Overleaf'te düzenlenecek; kaynak `paper/` (`report.tex`, `conference.tex`, ortak `refs.bib`; bkz. `paper/README.md`).
Biçim alandaki yayın geleneklerine göre (IEEE konferans; rapor A4/12 pt/Times/yazar-yıl). Ürün cilası ikincil;
araştırma katkısı, deney ve yazım öncelikli.

## Çalışma şekli

- Kullanıcı: Furkan. Türkçe, doğrudan ve kısa iletişim tercih eder; dolgu yok. Gece koşularında fan sesi istemez → uzun koşular `--threads 10` ile.
- İzin modu Manual; Windows'ta sandbox yok, komutlar kullanıcı hesabıyla çalışır. Silme yerine arşivleme tercih edilir.
- Proje 2026-11 başında Codex ile geliştirildi; 2026-11-13'ten itibaren Claude ile devam ediliyor. Codex raporlarındaki iddialar doğrulanmadan kabul edilmez.
- Makine: Core Ultra 5 245KF (14 çekirdek), 15,6 GB RAM, RTX 5070, Docker/WSL yok. Oturumda `NoDefaultCurrentDirectoryInExePath=1` → `.bat` başlatıcılar yanındaki dosyaları bulamaz.

## Mevcut durum (2026-11-14)

Çalışma dalı: `claude/tree-building` (push edilmedi). Plan adımları 1–4 büyük ölçüde tamam.

### Yeni akış: `src/uav3d` (üretim yolu)

`Documentation/ARCHITECTURE.md`. Kurulum: `powershell -File setup.ps1` (uv + `.tools/` altına SHA-256 doğrulamalı ODM 3.6.2 ve Roofer 1.0.0; `tools.lock.json`). Kontrol: `python -m uav3d doctor`.

```
python -m uav3d run <görüntü klasörü> --out <çıktı> [--threads N] [--budget-ram-gb G] [--gcp f] [--reuse-odm] [--skip-buildings] [--skip-export]
```

Aşamalar: ingest (EXIF, GCP otomatik) → plan (RAM/görüntü yükünden pc-quality, split) → ODM iki faz (SfM → GSD hesabı → dense/DEM/orto 1×/2×GSD) → bina (OSM + nokta bulutundan taban izi → Roofer LoD1.2/2.2 → bina başına kalite: yükseklik, RMSE) → export (`scene/scene.glb`, `buildings.city.json`, `scene.json`, `scene.blend`, `preview.png`). Her şey `run.json`'a (süre, tepe RAM, hata aşaması) yazılır. Katsayılar `configs/defaults.json` (geçici; yalnız ayar setlerinde belirlenir).

Doğrulanan: Brighton Beach (18 foto, 2,7 dk, 2,8 GB), Sheffield (172 foto, 36 dk, 7,9 GB, 25 bina: 22 ok / 3 düşük güven). 241 test geçiyor.

Önemli bulgular:
- ODM'yi `run.bat` ile değil kendi Python'u ile çağır (`.bat` her zaman exit 0 döner); başarı = "ODM app finished" log satırı.
- Kopyalanan ODM kurulumu mutlak yollar içerir → `setup.ps1` `Repair-OdmPaths` ile düzeltir.
- ODM 3.6 nokta bulutu ASPRS sınıflı (2 zemin, 3 bitki, 6 bina) → Roofer varsayılanları doğru.
- OpenMVS RTX 5070'i kullanmıyor (GPU %1) → muhtemelen eski CUDA derlemesi; hızlanma fırsatı, incelenmedi.
- "Sheffield" seti aslında ABD/Florida (28,04K −82,70B).
- Bina taban izi (dal `claude/tree-building`): nokta düzlemselliği filtresi (yüzey değişimi ≤0,03 @ ~1 m, aday içi ≥%60 düzlemsel) banliyö/ağaçlıkta iyi (Sheffield 29 bina, Zoo adayları 1351→324). AirZoo (ayar setleri) başarısız: urban'da yüksek bloklar hiç bulunmuyor (zemin sınıfı 112 m'ye çıkıyor, SMRF+tepelik arazi), rural'da teraslı tarlalar/yamaçlar "bina". Kök neden: ODM OpenPointClass sınıflandırması eğitim alanı dışında güvenilmez; yalnız geometriyle çözülemiyor → görüntü tabanlı bina tespiti + nDSM birleşimi gerekli (karar bekliyor).
- Arazi temizliği (`uav3d/terrain.py`, ODM sonrası "terrain" aşaması): ELM + istatistiksel aykırı + "alçak yüzey" testi (5 m hücre medyanı, 25 m pencere, eğime göre tolerans; su yansımalarının oluşturduğu sanal yüzeyleri siler) → SMRF (ODM parametreleri, ODM sınıfları korunur) → kaba DTM (10×GSD) + doldurma + medyan → ODM ızgarasına. AirZoo rural: çukur %3,4→%1,2, en derin −354→−64 m, iç bölge referans DSM ile uyumlu. Kalan büyük hatalar kenarlarda ve düşük örtüşmede (2 görüntü: medyan |hata| 2,3 m; ≥8: 0,5 m) → `opensfm/stats/overlap.tif` referanssız güvenilirlik sinyali, `run.json`'a yazılıyor. AirZoo "query" görüntüleri haritalama uçuşu değil (en çok 8 örtüşme). Orman altında SMRF alt dalları zemin sayıyor (Brighton), kaba ölçüde kabul.
- Kullanıcı gereksinimi: binada yalnız varlık + kaba genişlik/yükseklik yeterli (LoD1 kutu kabul). Ağaçlar da araziden ayrı kaba nesneler olacak (CHM + tek ağaç tespiti / sık alanda blok).
- Roofer RMSE (LoD2.2) referanssız bina kalite sinyali: gerçek çatılar 0,2–0,5 m, ağaç karışımı >1 m. Araştırmadaki kalite geri beslemesi için aday.

### Eski akışlar (legacy)

- `src/vision3d` + `scripts/omega_*` (21 dosya) + `run_demo.bat`: Codex demo yolu, hâlâ çalışır durumda ama `outputs/mission-omega/` altındaki araçlara bağlı. Yeni akış tamamen yerini alınca legacy'ye ayrılacak.
- Kaliteyi bozan eski ayar: `configs/omega_reusable_uav.json` → `dense_max_image_size: 800`, sabit metrik hücreler.

### Araştırma ve veri

- Literatür (plan 1): Claude Doc https://claude.ai/artifact/PPwpvVYfQJiud2e7Uvox8b. Özgün katkı: sabit RAM/süre bütçesinde karo bazlı, kalite geri beslemeli yeniden yoğunlaştırma zamanlayıcısı; H1–H3 belgede. Açık soru: tepe RAM bütçesi (belgede yorum).
- Protokol (plan 2): `Documentation/research/DATASET_PROTOCOL.md`, `configs/dataset_registry.json`. Yerel eski setler (Sheffield, AirZoo) = ayar. Genellik: `datasets/odm/` (brighton_beach, aukerman, toledo, helenenschacht, waterbury, zoo; ziegeleipark.zip 29,8 GB MD5 doğrulandı, açılmadı). Hold-out: UseGeo (`datasets/usegeo/`, iniyor; sunucu sertifikası süresi dolmuş, kullanıcı onayıyla doğrulamasız indirildi), JB3D (kısmi), UrbanScene3D/ISPRS Zollern (indirilmedi).
- Temizlik (plan 4): `Documentation/CLEANUP_PLAN.md`; arşiv `C:\projects\_archive\vision3d-2026-11-13\`.

### 2026-11-14 ilerleme

- **SAM 3 görüntü kanıtı** (`uav3d/detect.py`, "detect" aşaması, ayrı venv `setup.ps1 -WithML`): ortofoto karolarında "building" → `detect/building_score.tif`. Kullanım: (a) bina noktaları SMRF'den saklanır (arazi), (b) görüntü taban izleri + eğilme toleranslı kapsama, (c) kabul edilen izlerde DTM+1 m üstü noktalar sınıf 6'ya (Roofer girdisi). Bina F1 (ayar, IoU≥0,3): Sheffield 0,93→0,95, urban 0,03→0,13, rural 0,00 (rekonstrüksiyon DOM'a göre ~40 m çarpık; etiket uygulanamaz). Ayrıntı `Documentation/research/BUILDING_EVAL.md`. Smoke G1–G4 6/6 geçti.
- **Disk politikası:** her koşudan sonra ODM ara dosyaları silinir (`odm.prune`); yalnız bölgesel yeniden yoğunlaştırma için `--keep-odm-intermediates`. Proje ~300 GB → ~140 GB boş.
- **Geometri değerlendirmesi** (`uav3d/evaluate/geometry.py`, Tanks&Temples P/R/F, düz zeminde dikey ofset hizalı): UseGeo-1 (ayar) F = 0,40/0,64/0,90 @ 10/20/50 cm, dikey sapma +0,35 m (ODM öz-kalibrasyonu odak uzaklığını %2,2 büyük buluyor). `Documentation/research/GEOMETRY_EVAL.md`.
- UseGeo bölünmesi: dataset-1 = ayar (`outputs/tune/usegeo_1`), dataset-2/3 = hold-out (`outputs/holdout/`). `bench/prepare_usegeo.py` geo.txt üretir.
- UseGeo-2 (327 foto) planlayıcı RAM modelinde bütçeyi aşıyor → split (249/alt model) yolunu ilk kez sınıyor.

## Takvim (30 Kasım makale)

- 15–16 Kas: ürünü dondur. 17–21 Kas: adaptif zamanlayıcı (sabit RAM/süre bütçesi, düşük kaliteli ilk geçiş, referanssız kalite sinyalleri → zayıf bölgeleri yeniden yoğunlaştır; önce fizibilite: OpenMVS ROI/alt sahne veya karo bazlı görüntü alt kümesi). 22–25 Kas: UseGeo-2/3 hold-out, %25/50/75 bütçe; tekdüze düşük / tekdüze yüksek / statik seçiciye karşı. 26–29 Kas: makale. 30 Kas: teslim.

## Sıradaki işler

1. **UseGeo-2/3 gece koşusu** (`bench/run_usegeo_night.ps1 -Threads 10`, 2026-11-14 18:46 başladı) → `outputs/holdout/usegeo_{2,3}/eval_geometry.json`. Split yolu hatasızsa adaptif deney için ara dosyalar duruyor (boş disk ≥90 GB ise).
2. Adaptif zamanlayıcı (araştırma katkısı): tasarım `Documentation/research/ADAPTIVE_SCHEDULER.md`. Fizibilite
   tamam: OpenMVS ROI kırpması görüntü atmıyor (maliyet düşmez); çalışan yol OpenSfM `export_openmvs` görüntü listesiyle
   karo başına alt sahne (`bench/feasibility/tile_export.py`; 8 görüntü, 150 s, 0,3 GB). Sıradaki: `uav3d/refine.py`.
3. İsteğe bağlı: UseGeo sabit kalibrasyonunu ODM'ye vererek +0,35 m sapmayı gider.
4. Ağaç katmanı (CHM + tek ağaç tespiti) — zaman kalırsa.
5. Büyük ölçek: Ziegeleipark diske sığmıyor (zip korunuyor, onaysız silinmez); alternatif UseGeo 3 şeridin birleşimi.
6. OpenMVS GPU kullanımı; `src/vision3d` legacy ayrımı — düşük öncelik.
