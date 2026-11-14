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

## Ablasyon (2026-11-14): Sheffield'da seçilen adımlar AirZoo'yu kötüleştirdi mi?

Aynı ODM çıktıları, `--skip-terrain` / `--config` ile; F1:

| Varyant | Sheffield | AirZoo urban | AirZoo rural |
|---|---|---|---|
| A: arazi temizliği yok, düzlemsellik yok | 0,74 | 0,00 | 0,00 |
| B: yalnız düzlemsellik | 0,93 | 0,03 | 0,00 |
| C: yalnız arazi temizliği | 0,89 | 0,07 | 0,06 |
| D: ikisi (mevcut) | 0,93 | 0,03 | 0,00 |

Sonuç: hiçbir adım AirZoo'yu A'ya göre kötüleştirmedi; AirZoo her varyantta başarısız.
Kök nedenler (bina tespitinden bağımsız):
- **Urban:** SMRF penceresi 18 m (ODM varsayılanı), kule çatıları 40–50 m → büyük düz çatılar
  zemin sınıfına düşüyor, kuleler DTM'ye gömülüyor ve tespit edilemiyor. Hizalama doğru
  (SIFT+RANSAC: 111 inlier, 0,3 m).
- **Rural:** bizim rekonstrüksiyon referans DOM'a göre bükülmüş (etiketli evler ~40 m kayık,
  SIFT+RANSAC yalnız 25/72 inlier). Görüntüler haritalama uçuşu değil (örtüşme ≤8, RTK
  uygulanmamış). Bu sahnede F1 tespit başarısını değil hizalama hatasını ölçüyor; DOM'a
  çizilmiş etiketler bu rekonstrüksiyon için geçersiz.
- Ayrıca: sabah yapılan faz korelasyonu hizalama kontrolü (zayıf tepe 0,018) güvenilir değildi.

SAM 3 ilk deneme (kendi ortofotomuz, 15 cm): urban kare 2'de kuleler, rural kare 4'te ev/sera/
barakalar doğru yerde; tarla ve ağaç yok. GPU 5 GB, karo başına 0,3 s.

## SAM 3 ve sonrası (2026-11-14 öğleden sonra), F1

| Adım | Sheffield | AirZoo urban | AirZoo rural |
|---|---|---|---|
| Başlangıç (D) | 0,93 | 0,03 | 0,00 |
| + SAM 3 görüntü kanıtı, görüntü taban izleri, Roofer için yeniden etiketleme | 0,97 | 0,20 | 0,00 (FP 24 → 1) |
| + bina maskeli SMRF, yükseklikle orantılı görüntü toleransı (mevcut varsayılan) | 0,95 | 0,13 | 0,00 |
| nDSM kaynaklı taban izi (kapalı) | 0,68 | 0,17 | — |
| + parça birleştirme (kapalı) | 0,91 | 0,14 | — |

Urban kuleleri: noktalar doğru yerde ve doğru yükseklikte (Roofer 56–88 m) ama nokta bulutu adayları
çatının %1–20'si büyüklüğünde parçalar (çatıda az, cephede çok nokta; görüntüler haritalama uçuşu değil).
Görüntü maskesi ise ortofotoda yatık görünen kuleye oturuyor (gerçek ortofoto değil). Kalan seçenekler:
gerçek ortofoto (DSM'e görünürlüklü izdüşüm) ya da görüntü taban izinin yatma vektörüyle düzeltilmesi.

## Genellik setleri, yeni akış (2026-11-14 akşam)

G1–G4 hepsi sağlandı (6/6). Bina sayıları düştü (Toledo 74 → 6, Helenenschacht 10 → 0, Zoo 1366 → 247):
görsel kontrolde düşüş yanlış pozitiflerin (ağaçlar) elenmesi; Toledo'daki 5–6 yapı bulunuyor,
Helenenschacht'ta gerçekten bina yok.
