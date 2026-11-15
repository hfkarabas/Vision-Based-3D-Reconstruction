# Bina etiketleri (ayar setleri)

Amaç: bina tespitinin eşiklerini ayar setlerinde ölçerek seçmek (`Documentation/research/DATASET_PROTOCOL.md`).
Kit `python bench/make_label_kit.py` ile üretildi (sabit tohum 20261114; örnek kareler rastgele, elle seçilmedi).

| Sahne | Dosya | Etiketlenecek alan | Arka plan |
|---|---|---|---|
| sheffield | `sheffield.gpkg` | tüm uçuş alanı | bizim ortofoto |

Arka plan görüntüleri `_backgrounds/` altında (git'e girmez; kit betiği yeniden üretir).

## QGIS'te çizim

1. QGIS'i aç, yeni proje.
2. Sürükle-bırak: `_backgrounds/<sahne>.tif` ve `<sahne>.gpkg` (iki katmanı da seç: `area`, `buildings`).
3. `area` katmanına sağ tık → Properties → Symbology → dolguyu "No brush" yap, kenar kırmızı. Karelerin içini etiketleyeceksin.
4. `buildings` katmanını seç → kalem simgesi (Toggle Editing).
5. View → Toolbars → **Shape Digitizing Toolbar** aç → **Add Rectangle from 3 Points** seç.
   İki tıkla binanın bir kenarı, üçüncü tıkla genişliği. Çıkan formda `uncertain` boş bırak (emin değilsen 1 yaz).
6. Bitirince Save Layer Edits (disket simgesi). Sonra sıradaki sahne.

## Kurallar

- Karenin içindeki **tüm** binaları çiz; kareden taşan binaları da tam çiz.
- Kaba dikdörtgen yeter; L şekli → iki dikdörtgen. 3 × 3 m'den küçükleri atla.
- Sera, baraka, çadır dahil: duvarı ve çatısı olan her şey bina.
- Çatı görünen yere göre çiz.
- Emin değilsen çiz ve `uncertain = 1` yaz; hiç çizmemekten iyidir.
