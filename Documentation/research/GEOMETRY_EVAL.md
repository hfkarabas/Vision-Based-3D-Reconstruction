# Nokta bulutu doğruluğu (LiDAR referansı)

Ölçüm: `python -m uav3d.evaluate.geometry <run> <lidar.las>`; Tanks and Temples tanımı, ortak alan, voksel 5 cm.
Düşey sapma düz çıplak zeminde ölçülüp ayrıca raporlanır; "hizalı" skorlar bu sapma çıkarılarak hesaplanır.

## UseGeo-1 (ayar seti; 224 foto, 42 MP, GSD 1,73 cm; ODM 62 dk, tepe RAM 9,1 GB)

Düşey sapma: +0,352 m (25 064 düz zemin çifti, MAD 0,056 m). ODM kamera merkezleri geo.txt ile ≤1 cm uyumlu;
ODM'nin öz-kalibre odak uzaklığı UseGeo kalibrasyonundan %2,2 büyük (4738 px vs 4637 px) → olası kaynak.

| Bulut | F@10 cm | F@20 cm | F@50 cm (P / R) |
|---|---|---|---|
| ODM ham | 0,197 | 0,423 | 0,887 (0,968 / 0,818) |
| ODM hizalı | 0,395 | 0,639 | 0,902 (0,960 / 0,850) |
| Temizlenmiş hizalı | 0,395 | 0,633 | 0,894 (0,971 / 0,827) |

Arazi temizliği bu sahnede nötr (yansıma yok, çukur oranı zaten %0,17).

## Hold-out: UseGeo-2 ve UseGeo-3 (ayardan sonra bir kez; ikisi de split, 2 alt model)

| Şerit | Foto | GSD | ODM süre / tepe RAM | Düşey sapma (çift, MAD) |
|---|---|---|---|---|
| UseGeo-2 | 327 | 1,87 cm | 109 dk / 9,9 GB | +0,216 m (11 315, 0,044) |
| UseGeo-3 | 277 | 1,85 cm | 109 dk / 10,2 GB | +0,013 m (18 722, 0,047) |

GSD: geo.txt kamera yüksekliği − DTM medyanı, ODM odak uzaklığıyla (UseGeo-1'de aynı yöntem 1,72 cm verir;
UseGeo-3 alt model SfM'lerinden 1,82–1,84 cm).

| Bulut | F@10 cm | F@20 cm | F@50 cm (P / R) |
|---|---|---|---|
| UseGeo-2 ODM ham | 0,254 | 0,515 | 0,897 (0,964 / 0,839) |
| UseGeo-2 ODM hizalı | 0,443 | 0,718 | 0,919 (0,974 / 0,870) |
| UseGeo-2 temizlenmiş hizalı | 0,441 | 0,710 | 0,908 (0,983 / 0,843) |
| UseGeo-3 ODM ham | 0,462 | 0,699 | 0,914 (0,975 / 0,861) |
| UseGeo-3 ODM hizalı | 0,471 | 0,701 | 0,914 (0,974 / 0,861) |
| UseGeo-3 temizlenmiş hizalı | 0,469 | 0,690 | 0,902 (0,983 / 0,833) |

Düşey sapma şeritten şeride değişiyor (+0,35 / +0,22 / +0,01 m) → sabit düzeltme değil, sahneye özgü
öz-kalibrasyon etkisi. Temizlik her iki hold-out'ta precision'ı artırıp recall'u biraz düşürüyor (F nötr).
