# Temizlik planı (2026-11-13)

Durum: **uygulandı (2026-11-13), `claude/cleanup` dalı.** Sonuç:
- 27,5 GB silindi.
- 54,2 GB `C:\projects\_archive\vision3d-2026-11-13\` klasörüne taşındı.
- `outputs/` 91,8 GB'tan 10,7 GB'a indi.
- 77 ölü betik/çalıştırıcı, 4 ölü test ve 11 config arşivlendi.
- Testler: 216 geçti, 9 atlandı. Atlananlar GIS çalışma ortamı gerektiriyor; o ortamda pytest olmadığı için bu testler doğrulanmadı.

Plandan sapmalar:
- ODM kurulumu ve bina modeli, arşivlenen Sheffield/terrain-validation klasörlerinin içindeydi. İkisi geri alındı: `outputs/mission-omega/tools/odm` ve `building-recovery/terrain-validation/checkpoints`.
- `tests/unit/test_airzoo_diagnostics.py` betikleri adıyla dinamik yüklediği için grafikte görünmedi. Yalnızca ölü kodu test ettiği için o da arşivlendi.
- `configs/sceaux.json` kaldı, çünkü eski demo onu okuyor.

## 1. Bulgular

### Kod: üretim yolu küçük, ölü kod büyük

Grafik, `run_demo.bat`'tan başlayan erişilebilirlik analiziyle çıkarıldı. Kenarlar şunlardan oluşuyor: Python importları, `-m` modül çağrıları ve alt süreçle çağrılan dosya adları. Betik: scratchpad'deki `reach.py`; bu sonuçları üretti.

| Grup | `scripts/` | `src/` | Kök |
|---|---|---|---|
| Üretim yolunda (demo → UAV profili + eski profiller) | 21 | 44 | 1 (`run_demo.bat`) |
| Yalnızca testlerden çağrılıyor | 14 | 3 | — |
| Hiçbir yerden çağrılmıyor | **73** | 1 (`cli/__main__.py`) | 3 (`RUN_MISSION_*.ps1`, `run-mission0.ps1`) |

`scripts/` içindeki 108 dosyanın 87'si üretimde kullanılmıyor. 73 ölü betiğin 49'u git'te izleniyor, 24'ü hiç commit edilmemiş.

**Ölü kod algoritmayı doğrudan bozmuyor**, çünkü hiç çalışmıyor. Asıl zarar dolaylı:
- Hangi sürümün çalıştığını izlemek zorlaşıyor. Örneğin `mission_omega_*` ile `omega_*` aynı işin iki kuşağı.
- Üretim betikleri birbirini import ediyor. `omega_terrain_blender.py`, `omega_urban_44.build`'i çağırıyor.
- Codex yeni deneme açtıkça eskisini kapatmamış.

**Kaliteyi doğrudan bozan somut ayarlar** üretim yolundaki `configs/omega_reusable_uav.json` dosyasında:
- `dense_max_image_size: 800`. 4000 px görüntüler yoğun eşleştirmede 800 px'e indiriliyor. Eksik ve kısmi çatıların muhtemel ana nedeni bu.
- `ortho_cell_m: 0.20` ve `terrain_cell_m: 0.50` sabit metrik değerler. GSD'ye göre ölçeklenmiyor, bu da Kural 3'ü ihlal ediyor.

Bunlar temizlikle değil, yeni mimariyle (plan adımı 3) düzelecek.

### Disk: 144 GB'ın dağılımı

| Yer | GB | Dosya | Not |
|---|---|---|---|
| `outputs/` | 91,8 | 413 073 | Codex deney çıktıları + araç kurulumları |
| `datasets/` | 52 → ~88 | 2 639 | İndirmeler bitince |
| `.venv`, `.uv-cache`, `.python` | 1,5 | 40 241 | Çalışma ortamı |

`outputs/mission-omega/tools` (25,5 GB) içinden üretimin gerçekten kullandıkları:
- `gis` (PDAL, 4,1 GB)
- `Qgis2threejs-master` (0,1 GB)

Ek olarak `building-recovery/runtime/venv` (5,2 GB) ve `reusable-uav/gis-extra` de kullanılıyor. ODM ise `C:/ODM` altında kurulu.

## 2. Önerilen işlemler

Arşiv konumu proje dışında: `C:\projects\_archive\vision3d-2026-11-13\`. Aynı disk olduğu için taşıma anlık. Proje klasörü küçülür, hiçbir şey kaybolmaz.

### A. Kalıcı silme (yeniden indirilebilir/üretilebilir) — yaklaşık 27,5 GB

| Öğe | GB |
|---|---|
| `outputs/mission-omega/tools/mamba` (paket önbelleği / kurulum) | 9,07 |
| `outputs/mission-omega/tools/qgis` ve `qgis4` (kullanılmayan iki tam QGIS kurulumu) | 12,02 |
| `outputs/mission-omega/building-recovery/runtime/uv-cache`, `reusable-uav/uv-cache`, `.uv-cache` | ~5,8 |
| `outputs/mission-omega/tools/npm-cache`, `outputs/matplotlib-cache` | 0,14 |
| `outputs/pytest-*`, kökteki 15 `pytest-cache-files-*`, tüm `__pycache__` | ~0,5 |

### B. Arşive taşıma — yaklaşık 55 GB

- `outputs/` içindeki tüm deney ve rapor çıktıları. Örnekler: `airzoo-benchmark` (17,5 GB), `research-baseline` (8,5 GB), `building-recovery/terrain-validation` (5,2 GB), `demo*`, `final-release`, `mv-*`, `completion-*`, `sceaux-*`, `motorcycle-*`, `codex-night`.
- **Geçici olarak yerinde kalır:** `tools/gis`, `tools/Qgis2threejs-master`, `runtime/venv`, `gis-extra`. Mevcut demo bunlarla çalışıyor. Yeni mimari kurulunca bunlar da arşive gider.

### C. Kod

- 73 ölü betik, kökteki 3 Codex çalıştırıcısı ve `src/vision3d/cli/__main__.py` arşive gider. Git'te izlenenler tek bir commit'le kaldırılır, bu yüzden geçmişte durmaya devam eder.
- Yalnızca testten çağrılan 14 betik, test ettikleri dosyalarla birlikte arşive gider.
- Üretim yolundaki 21 betik ve `src/vision3d` (44 dosya) şimdilik kalır. Plan adımı 3'te yeni orkestrasyon katmanı bunların yerini alınca `src/vision3d` legacy olarak ayrılır.

### D. Kök ve config karmaşası

- Arşive taşınacak kök dosyalar:
  - Open3D ekran görüntüleri ve kamera JSON'ları
  - `MISSION*_PROMPT.txt`
  - `RUN_MISSION_*.ps1`
  - `run-mission0.ps1`
- Datasete özel config'ler arşive gider:
  - `omega_terrain_urban_*`
  - `omega_urban_*`
  - `omega_terrain_validation_*`
  - `motorcycle.json`, `sceaux.json`, `recovery_experiments.json`, `sequence_quality.json`, `benchmark_suite.json`, `research_baseline.json`
  - `configs/omega_urban_44/`, `configs/unity/`
- Kalanlar: `omega_reusable_uav.json` (demo hâlâ okuyor) ve `dataset_registry.json`.

### E. Git

Mevcut dalda (`codex/omega-run-demo-alpha`) commit edilmemiş değişiklikler var: `deliverables/` silmeleri, `omega_desktop.py` değişikliği ve çok sayıda izlenmeyen belge. Temizlik `claude/cleanup` adlı yeni bir dalda yapılır:
1. Önce mevcut durum olduğu gibi bir "Codex son durum" commit'iyle kaydedilir.
2. Ardından taşımalar ayrı commit'ler hâlinde yapılır.

## 3. Sonra (bu planın dışında)

- `docs/` ile `Documentation/` birleştirilecek.
- `datasets/`'in proje dışına (ör. `C:\data\vision3d`) taşınması ve bir ortam değişkeniyle gösterilmesi düşünülebilir. Bu, Kural 3'ün "yol sabitlenmez" şartını da güçlendirir.
