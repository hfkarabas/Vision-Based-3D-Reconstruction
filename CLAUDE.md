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

- Kullanıcı: Furkan. Türkçe, doğrudan ve kısa iletişim tercih eder; dolgu yok. Gece koşularında fan sesi istemez → uzun koşular `--threads 10` ile.
- İzin modu Manual; Windows'ta sandbox yok, komutlar kullanıcı hesabıyla çalışır. Silme yerine arşivleme tercih edilir.
- Proje 2026-11 başında Codex ile geliştirildi; 2026-11-13'ten itibaren Claude ile devam ediliyor. Codex raporlarındaki iddialar doğrulanmadan kabul edilmez.
- Makine: Core Ultra 5 245KF (14 çekirdek), 15,6 GB RAM, RTX 5070, Docker/WSL yok. Oturumda `NoDefaultCurrentDirectoryInExePath=1` → `.bat` başlatıcılar yanındaki dosyaları bulamaz.

## Mevcut durum (2026-11-14)

Çalışma dalı: `claude/cleanup` (push edilmedi). Plan adımları 1–4 büyük ölçüde tamam.

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
- Roofer RMSE (LoD2.2) referanssız bina kalite sinyali: gerçek çatılar 0,2–0,5 m, ağaç karışımı >1 m. Araştırmadaki kalite geri beslemesi için aday.

### Eski akışlar (legacy)

- `src/vision3d` + `scripts/omega_*` (21 dosya) + `run_demo.bat`: Codex demo yolu, hâlâ çalışır durumda ama `outputs/mission-omega/` altındaki araçlara bağlı. Yeni akış tamamen yerini alınca legacy'ye ayrılacak.
- Kaliteyi bozan eski ayar: `configs/omega_reusable_uav.json` → `dense_max_image_size: 800`, sabit metrik hücreler.

### Araştırma ve veri

- Literatür (plan 1): Claude Doc https://claude.ai/artifact/PPwpvVYfQJiud2e7Uvox8b. Özgün katkı: sabit RAM/süre bütçesinde karo bazlı, kalite geri beslemeli yeniden yoğunlaştırma zamanlayıcısı; H1–H3 belgede. Açık soru: tepe RAM bütçesi (belgede yorum).
- Protokol (plan 2): `Documentation/research/DATASET_PROTOCOL.md`, `configs/dataset_registry.json`. Yerel eski setler (Sheffield, AirZoo) = ayar. Genellik: `datasets/odm/` (brighton_beach, aukerman, toledo, helenenschacht, waterbury, zoo; ziegeleipark.zip 29,8 GB MD5 doğrulandı, açılmadı). Hold-out: UseGeo (`datasets/usegeo/`, iniyor; sunucu sertifikası süresi dolmuş, kullanıcı onayıyla doğrulamasız indirildi), JB3D (kısmi), UrbanScene3D/ISPRS Zollern (indirilmedi).
- Temizlik (plan 4): `Documentation/CLEANUP_PLAN.md`; arşiv `C:\projects\_archive\vision3d-2026-11-13\`.

## Sıradaki işler

1. **Genellik koşusu sonucu:** `bench/run_smoke.ps1 -Threads 10` (2026-11-14 gecesi başlatıldı) → `outputs/smoke/summary.md` (G1–G4). Hataları datasete özel değil, genel kuralla düzelt.
2. Bina taban izi iyileştirme (ayar setlerinde): ağaç yapışması, tırtıklı sınırlar, birkaç yanlış pozitif.
3. Ziegeleipark (7 169 foto): split yolunu ilk kez sınar; gece, `--threads 10`.
4. OpenMVS GPU kullanımı (RTX 5070 / CUDA sürümü).
5. Değerlendirme modülü `uav3d/evaluate` (UseGeo LiDAR'a göre P/R/F); sonra hipotez deneyi (plan 5).
6. `src/vision3d` ve eski demo yolunu legacy'ye ayır; `docs/` + `Documentation/` birleştir.
