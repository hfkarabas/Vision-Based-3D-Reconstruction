# Dataset envanteri ve test protokolü

Tarih: 2026-11-13. Makine tarafından okunabilir kayıt: [`configs/dataset_registry.json`](../../configs/dataset_registry.json).
Araştırma sorusu ve hipotezler: literatür taraması belgesi (Claude Docs).

## 1. Temel kural

Datasetler üç role ayrılır ve bir datasetin rolü sonradan değişmez:

| Rol | Amaç | Ne yapılabilir | Ne yapılamaz |
|---|---|---|---|
| **Ayar (tune)** | Eşik, bütçe ve parametre seçimi | Her türlü deneme, hata ayıklama | Sonuç olarak raporlanmaz |
| **Genellik (smoke)** | "Her datasette çalışır" kanıtı | Ürün yolunu değiştirmeden çalıştırma | Datasete özel ayar, hata sonrası yeniden ayar |
| **Hold-out (test)** | Araştırma hipotezinin (H1–H3) sınanması | Dondurulmuş sürümle tek resmi koşu | Sonuç görüldükten sonra eşik değiştirmek |

Daha önce Codex oturumlarında üzerinde çalışılmış her yerel dataset kirlenmiş sayılır ve yalnızca **ayar** rolü alır.

## 2. Envanter

### 2.1 Yerelde olanlar

| Kimlik | Kaynak | Ülke | Platform / kamera | Görüntü | Referans | Lisans | Rol |
|---|---|---|---|---|---|---|---|
| `sheffield_cross` | [pierotofy/drone_dataset_sheffield_cross](https://github.com/pierotofy/drone_dataset_sheffield_cross) | ABD (Florida; adı yanıltıcı) | DJI Phantom 3 (FC300S), 4000×3000 | 172 | GCP dosyası | Public domain (Unlicense) | Ayar |
| `airzoo_rural_s2` | [AirZoo-Real](https://huggingface.co/datasets/RingoWRW97/AirZoo-Real) | Çin (EPSG:4547) | DJI Matrice 4T, 4032×3024, RTK/PPK | 193 dosya (190 kayıtlı) | DOM + DSM | Doğrulanmalı | Ayar |
| `airzoo_urban_s1` | AirZoo-Real | Çin | DJI Matrice 4T, RTK/PPK | 164 dosya (161 kayıtlı) | DOM + DSM | Doğrulanmalı | Ayar |
| `jb3d` (kısmi) | [JB3D](https://github.com/JB3D-Dataset/Multi-View-Stereo) | Belirtilmemiş | İHA video 3840×2160 @30 fps | Kısmi indirme | LiDAR (5 bulut), ECEF/ENU | CC BY 4.0 | Hold-out adayı (bkz. 3) |
| `tnt_caterpillar` | Tanks and Temples | — | El kamerası video | — | Lazer (yerelde yok) | T&T şartları | Kapsam dışı (İHA değil) |
| `temple`, `sceaux`, `middlebury` | Middlebury / OpenMVG örnekleri | — | Masaüstü nesne | 6–30 | — | — | Kapsam dışı (eski pipeline birim testi) |

JB3D yerelde Codex tarafından kısmen indirildi; yalnızca Day 1 LiDAR ve tek bir uçuş arşivi var. Kamera konum düzeltmesinin uygulanıp uygulanmadığı doğrulanmadan kullanılmayacak.

### 2.2 İndirilecek açık setler

**Referanslı (geometrik değerlendirme):**

| Kimlik | Kaynak | Ülke | İçerik | Referans | Lisans | Rol |
|---|---|---|---|---|---|---|
| `usegeo_1` | [UseGeo](https://github.com/3DOM-FBK/usegeo) (Nex vd. 2024) | Belirtilmemiş (FBK/ITC) | 224 görüntü, 7952×5304, GSD 1,7 cm, 80 m | LiDAR ~50 nk/m², derinlik haritaları | CC BY-NC-SA 4.0 | Ayar (referanslı) |
| `usegeo_2` | UseGeo | aynı | 328 görüntü, GSD 1,8 cm | aynı | aynı | Hold-out |
| `usegeo_3` | UseGeo | aynı | 277 görüntü, GSD 1,9 cm | aynı | aynı | Hold-out |
| `jb3d_bench` | JB3D | Belirtilmemiş | 14 video dizisi, 36 752 kare (+5 000 RGB foto) | 5 LiDAR bulutu | CC BY 4.0 | Hold-out (farklı ülke/kamera/çekim tipi) |
| `urbanscene3d_real` | [UrbanScene3D](https://vcc.tech/UrbanScene3D) (Lin vd. 2022) | Çin | Gerçek sahneler, toplam 128 k+ görüntü (16 sahne, sentetik dahil) | Trimble X7 lazer | Doğrulanmalı | Hold-out, büyük ölçek |
| `h3d` | [Hessigheim 3D](https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/default.aspx) (Kölle vd. 2021) | Almanya | İHA LiDAR + dokulu mesh, eğik görüntü GSD 2–3 cm | İHA LiDAR ~800 nk/m² | CC BY 4.0 / BY-NC-SA çelişkili | Aday: ham görüntülerin yayımlanıp yayımlanmadığı doğrulanmalı |
| `isprs_zollern` | [ISPRS/EuroSDR multi-platform](https://www2.isprs.org/commissions/comm2/icwg-2-1a/benchmark_main/) (Nex vd. 2015) | Almanya | İHA + yer görüntüsü, 1000+ görüntü | GNSS/total station hedefleri | Atıf şartı | Hold-out (yalnızca hedef noktası doğruluğu) |

UseGeo'nun üç şeridi aynı 1100×650 m alanda, aynı kamerayla çekildi. Bu yüzden UseGeo-2/3 üzerindeki sonuç "aynı sahne ailesinde hold-out" olarak raporlanır. Genelleme iddiası JB3D, UrbanScene3D ve ISPRS Zollern'den gelmelidir.

**Referanssız genellik seti** ([ODMdata](https://github.com/OpenDroneMap/ODMdata) listesinden):

| Kimlik | Görüntü | Neden seçildi |
|---|---|---|
| `brighton_beach` | 18 | Çok küçük set; alt sınır davranışı |
| `aukerman` | 77 | ODM'nin standart test seti |
| `toledo` | 87 | Kentsel |
| `helenenschacht` | 176 | Farklı üretici (Autel Evo II Pro RTK) + GCP |
| `waterbury` | 248 | Orta boy |
| `shitan_tw` | 493 | Tayvan; farklı coğrafya |
| `zoo` | 524 | Orta-büyük |
| `ziegeleipark` | 7 169 | **Büyük ölçek ve kaynak kısıtı testi** |

Bu setlerin lisansları tek tek indirilirken kayda geçirilecek. Lisansı belirsiz olan set yayına girmez, yalnızca yerel testte kullanılır.

## 3. "Genel çalışır" ölçütleri

Bu ölçütler genellik ve hold-out setlerinde, **datasete özel hiçbir ayar olmadan**, aynı komutla ölçülür.

| Kod | Ölçüt | Eşik (dondurulmuş) |
|---|---|---|
| G1 | Akış hatasız biter ve zorunlu çıktıları üretir: DSM, DTM, ortofoto, nokta bulutu, arazi mesh'i, bina CityJSON'u ya da gerekçeli "bina yok" raporu | Genellik setlerinin ≥ %90'ı |
| G2 | Başarısızlık sessiz değildir: hata kodu ve hangi aşamada durduğu raporlanır | %100 |
| G3 | Tepe RAM, ilan edilen bütçenin altında kalır (ölçülür, tahmin edilmez) | %100 |
| G4 | Kayıtlı görüntü oranı | ≥ %95, ya da bağımsız COLMAP koşusundan en fazla 2 puan düşük |
| G5 | GCP veya RTK varsa kontrol noktası RMSE değeri | Yatayda ≤ 3×GSD, düşeyde ≤ 5×GSD |
| G6 | Koordinat sistemi: EXIF GPS yoksa akış durmaz, yerel koordinatla devam eder ve bunu raporlar | Testte en az bir GPS'siz set |

G5 eşikleri bu belgeyle ilan edildi. Ayar setlerinde gerçekleşmezse eşik değil, yöntem değiştirilir.

## 4. Araştırma ölçütleri (hold-out)

- **Geometri:** Bağımsız referansa göre kesinlik (precision), tamamlanmışlık (recall) ve F-skoru, τ ∈ {10, 20, 50} cm. τ değerleri UseGeo LiDAR aralığından (~14 cm) türetildi ve sonuç görülmeden önce donduruldu.
- **Maliyet:** Duvar saati ve tepe RAM. Teşhis, zamanlama, birleştirme ve dışa aktarma dahil, uçtan uca ölçülür.
- **Bütçe–kalite eğrisi:** Tam kalite tek geçişin %25, %50 ve %75'i.
- **Bina:** Çatı noktalarında LiDAR'a göre RMSE, geçerli LoD2.2 üreten bina oranı, varsa taban izi IoU değeri.
- **İstatistik:** Sahne veya karo bloğu bazında eşleştirilmiş güven aralığı. Milyonlarca nokta bağımsız örnek sayılmaz. Rastgelelik içeren her koşu en az 3 tohumla tekrarlanır.

## 5. Dondurma prosedürü

1. Ayar setlerinde çalışma bittiğinde tüm eşik ve bütçeler tek bir yapılandırma dosyasında toplanır, commit edilir ve `holdout-freeze-N` etiketiyle işaretlenir.
2. Hold-out koşuları yalnızca bu etiketten yapılır. Koşu kaydında commit hash'i, araç sürümleri (ODM, OpenMVS, PDAL, Roofer) ve donanım bulunur.
3. Hold-out sonucundan sonra yapılan her değişiklik yeni bir etiket gerektirir. Önceki sonuç silinmez, "N. donma" olarak raporlanır.
4. Referans veri (LiDAR, DSM) yalnızca değerlendirmede okunur. Zamanlayıcı ve eşik seçimi referansı hiçbir zaman görmez; bu, kod düzeyinde ayrı modüllerle zorlanır.

## 6. Açık işler

- [ ] UseGeo indir (3 şerit) ve lisans/konum bilgisini kayda ekle
- [ ] JB3D: kamera düzeltmesini doğrula, bir benchmark dizisini tamamla
- [ ] UrbanScene3D gerçek sahne listesi ve lisans
- [ ] H3D: ham görüntü erişimi var mı?
- [ ] ODMdata genellik setlerini indir; her birinin lisansını kaydet
- [ ] AirZoo-Real lisansını doğrula
