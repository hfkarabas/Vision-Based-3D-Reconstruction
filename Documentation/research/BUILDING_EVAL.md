# Bina tespiti değerlendirmesi (ayar setleri)

Etiketler: `labels/buildings/` (Furkan, 2026-11-14; kaba dikdörtgenler, birbirine değen parçalar tek bina).
Ölçüm: `python -m uav3d.evaluate.buildings`. Eşleşme: IoU ≥ 0,3, birebir. Yalnızca örnek alanlar içinde sayılır.
Tahmin olarak "ok" ve "low_confidence" durumundaki binalar kullanılır.

## Başlangıç çizgisi (2026-11-14, `claude/tree-building`, yöntem: OSM + nokta bulutu taban izi + düzlemsellik + arazi temizliği)

| Sahne | Etiket | Tahmin | TP | FP | FN | Kesinlik | Bulma | F1 | Medyan IoU |
|---|---|---|---|---|---|---|---|---|---|
| Sheffield (ABD, banliyö) | 29 | 27 | 26 | 1 | 3 | 0,96 | 0,90 | 0,93 | 0,74 |
| AirZoo urban (Çin, yüksek blok) | 14 | 58 | 1 | 57 | 13 | 0,02 | 0,07 | 0,03 | 0,56 |
| AirZoo rural (Çin, kırsal) | 11 | 24 | 0 | 24 | 11 | 0,00 | 0,00 | 0,00 | — |

Yorum: yöntem ODM'nin nokta sınıflandırmasının iyi olduğu sahnede iyi, Çin sahnelerinde çöküyor
(bkz. CLAUDE.md "Önemli bulgular"). Görüntü tabanlı kanıt eklenmeden genellenemez.
