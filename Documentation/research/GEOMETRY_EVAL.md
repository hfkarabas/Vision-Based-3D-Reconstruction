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
