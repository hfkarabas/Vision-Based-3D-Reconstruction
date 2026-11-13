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

## Çalışma şekli

- Kullanıcı: Furkan. Türkçe, doğrudan ve kısa iletişim tercih eder; dolgu yok.
- İzin modu Manual; Windows'ta sandbox yok, komutlar kullanıcı hesabıyla çalışır. Silme yerine arşivleme tercih edilir.
- Proje 2026-11 başından itibaren Codex ile geliştirildi; sonuçlar yetersiz bulunduğu için 2026-11-13'ten itibaren Claude ile devam ediliyor. Codex'in ürettiği raporlardaki iddialar doğrulanmadan kabul edilmez.

## Mevcut durum (2026-11-13 analizi)

### İki akış var

1. **`src/vision3d` (kendi pipeline, ~400 KB):** SIFT/ORB → eşleştirme → artımlı SfM → sınırlı BA (SciPy) → StereoSGBM → voksel füzyon → grid mesh → UV doku → GLB/OBJ; Tkinter demo + Open3D viewer + CLI.
   - `Documentation/PIPELINE_AUDIT.md`: hiçbir bileşen özgün avantaj göstermedi; yoğun derinlik ve mesh en zayıf halka.
   - **Karar önerisi:** Kural 1 gereği üretim yolundan çıkarılıp arşive (legacy/karşılaştırma) ayrılacak. Silinmeyecek. Yararlı parçalar: provenance, tanı/denetim betikleri, export doğrulama, masaüstü kabuğu.
2. **Omega UAV akışı (`scripts/omega_*`, dal `codex/omega-run-demo-alpha`):** ODM 3.6.2 → PDAL/GDAL (SMRF zemin, DTM/DSM) → GeoAI (Mask R-CNN + WHU) bina tespiti → Qgis2threejs mesh → Blender 5.3 LOD1 sahne. `run_demo.bat` → `vision3d.demo --uav` → `scripts/omega_desktop.py`.
   - Sheffield (172 foto) üzerinde çalıştı; 13 bina hacmi geçti, eksik/kısmi çatılar sürüyor (`Documentation/OMEGA_SHEFFIELD_ALPHA_QUALITY.md`).

### Bilinen sorunlar

- **Taşınabilirlik:** GIS Python ortamı ve PYTHONPATH eklemeleri `.gitignore`'daki `outputs/` altından yükleniyor (`outputs/mission-omega/building-recovery/runtime/venv`, `outputs/mission-omega/reusable-uav/gis-extra`). geopandas, rasterio, psutil, yaml vb. `pyproject.toml`'da yok. Temiz klonda UAV profili çalışmaz.
- **Genellik (Kural 3 ihlalleri):** sabit ODM bayrakları (`--pc-quality low` vb.), Sheffield'da ayarlanmış eşikler, datasete özel config'ler (`configs/omega_terrain_urban_regions.json` vb.), EXIF GPS zorunlu, otomatik bölme (split-merge) büyük sahada doğrulanmamış, bina detektörü tek sahada sınanmış.
- **Araştırma:** `Documentation/RESEARCH_HYPOTHESIS.md` geniş fikrin literatürde büyük ölçüde var olduğunu söylüyor; önceki adaptif deneme olumsuz. "Büyük ölçekli" ve "kaynak kısıtlı" kısmı hiç test edilmedi. Özgün katkı henüz daraltılmadı.
- **Düzen:** kökte 15 `pytest-cache-files-*`, Open3D ekran görüntüleri, Codex prompt/çalıştırıcıları (`MISSION*_PROMPT.txt`, `RUN_MISSION_*.ps1`); `outputs/` içinde 35 geçici pytest dizini ve 12 MB `codex-night/mission0.log`; belgeler `docs/` ve `Documentation/` arasında bölünmüş, dil karışık; `pyproject.toml` açıklaması/sürümü eski; `scripts/` (~720 KB) paketin iki katı ve betikler arası import ile bağlı.

### Mevcut varlıklar

- Literatür kaydı: `Documentation/research/literature-register.{csv,json}` (L1–L42), `Documentation/LITERATURE_REVIEW.md`, `PRIOR_WORK_COMPARISON.md`, `MISSION_OMEGA_SOURCE_RESEARCH/`, PDF'ler `deliverables/kaynakca/`.
- Datasetler (`datasets/`): airzoo (rural/urban), omega-terrain-validation (sheffield, jb3d), sceaux, temple, tanks_and_temples/caterpillar, middlebury-motorcycle.
- Bağımsız COLMAP baseline: Rural 190/190, Urban 161/161 (`Documentation/research/BASELINE_RESULTS.md`).

### 2026-11-13 ilerleme

- Literatür taraması (plan 1) tamam: Claude Doc https://claude.ai/artifact/PPwpvVYfQJiud2e7Uvox8b. Karar: ODM/OpenMVS omurga, Roofer LoD2.2, PDAL SMRF; özgün katkı = bütçe altında karo bazlı kalite geri beslemeli yeniden yoğunlaştırma.
- Dataset protokolü (plan 2) tamam: `Documentation/research/DATASET_PROTOCOL.md`, `configs/dataset_registry.json`. Yerel tüm setler "ayar" rolünde (kirli). ODMdata 6 set `datasets/odm/`; Ziegeleipark ve UseGeo (`datasets/usegeo/`, sertifikası süresi dolmuş sunucu, kullanıcı onayıyla) indirildi/iniyor.
- Temizlik (plan 4) `claude/cleanup` dalında yapıldı: `Documentation/CLEANUP_PLAN.md`; arşiv `C:\projects\_archive\vision3d-2026-11-13\`. Üretimde kalan: 21 `scripts/omega_*` + `src/vision3d`.
- Kaliteyi bozan bulunan ayar: `configs/omega_reusable_uav.json` → `dense_max_image_size: 800`, sabit metrik hücre boyutları (GSD'den bağımsız). Yeni mimaride düzeltilecek.
- Bu makine: 15,6 GB RAM.

## Plan (öncelik sırası)

1. **Sistematik literatür taraması** (Docs olarak):
   - (a) büyük ölçekli / kaynak kısıtlı UAV fotogrametri: ODM split-merge, COLMAP/GLOMAP, OpenMVS, karo tabanlı işleme;
   - (b) kalite geri beslemeli / adaptif rekonstrüksiyon: görüntü seçimi, belirsizlik, bölgesel yeniden işleme;
   - (c) arazi/bina ayrımı ve LOD bina: zemin filtreleme, çatı segmentasyonu, PolyFit/City3D benzeri.
   - Çıktı: kaynak tablosu, her aşama için "hangi açık kaynak araç ve neden" kararı, savunulabilir daraltılmış araştırma sorusu.
2. **Dataset envanteri ve test protokolü:** farklı ülke/kamera/GSD/ölçekte açık İHA setleri; ayar ve hold-out ayrımı; ölçülebilir "genel çalışır" kriterleri.
3. **Mimari:** ince orkestrasyon katmanı + açık kaynak araçlar, tek komutla tekrarlanabilir kurulum (hiçbir şey `outputs/`'a bağlı değil), datasetten bağımsız ayarlar, `pyproject.toml`'da `gis` bağımlılık grubu.
4. **Temizlik:** kök ve `outputs/` geçici dosyaları arşive; `src/vision3d` legacy olarak ayrılır; belgeler tek dizinde birleşir.
5. **Hipotez deneyi:** dondurulmuş ODM/COLMAP referansına karşı statik seçici vs. adaptif zamanlayıcı, önceden ilan edilmiş bütçelerde.
