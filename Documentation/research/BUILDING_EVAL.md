# Bina tespiti değerlendirmesi (ayar seti)

Etiketler: `labels/buildings/` (elle çizildi, 2026-11-14; kaba dikdörtgenler, birbirine değen parçalar tek bina).
Ölçüm: `python -m uav3d.evaluate.buildings`. Eşleşme: IoU ≥ 0,3, birebir. Yalnızca etiketli alan içinde sayılır.
Tahmin olarak "ok" ve "low_confidence" durumundaki binalar kullanılır.

Etiketli tek set Sheffield'dır (dik çekim, banliyö). Eğik çekimli setler kapsam dışıdır (2026-11-15); daha önce
ayar setleri arasında bulunan iki eğik çekim seti ve onlara ait etiketler projeden kaldırıldı.

## Başlangıç çizgisi (2026-11-14; OSM + nokta bulutu taban izi + düzlemsellik + arazi temizliği)

| Sahne | Etiket | Tahmin | TP | FP | FN | Kesinlik | Bulma | F1 | Medyan IoU |
|---|---|---|---|---|---|---|---|---|---|
| Sheffield (ABD, banliyö) | 29 | 27 | 26 | 1 | 3 | 0,96 | 0,90 | 0,93 | 0,74 |

## Ablasyon (2026-11-14), F1

| Varyant | Sheffield |
|---|---|
| A: arazi temizliği yok, düzlemsellik yok | 0,74 |
| B: yalnız düzlemsellik | 0,93 |
| C: yalnız arazi temizliği | 0,89 |
| D: ikisi | 0,93 |

## SAM 3 ve sonrası (2026-11-14 öğleden sonra), F1

| Adım | Sheffield |
|---|---|
| Başlangıç (D) | 0,93 |
| + SAM 3 görüntü kanıtı, görüntü taban izleri, Roofer için yeniden etiketleme | 0,97 |
| + bina maskeli SMRF, yükseklikle orantılı görüntü toleransı (mevcut varsayılan) | 0,95 |
| nDSM kaynaklı taban izi (kapalı) | 0,68 |
| + parça birleştirme (kapalı) | 0,91 |

## Genellik setleri, yeni akış (2026-11-14 akşam)

G1–G4 hepsi sağlandı (6/6). Bina sayıları düştü (Toledo 74 → 6, Helenenschacht 10 → 0, Zoo 1366 → 247):
görsel kontrolde düşüş yanlış pozitiflerin (ağaçlar) elenmesi; Toledo'daki 5–6 yapı bulunuyor,
Helenenschacht'ta gerçekten bina yok.
