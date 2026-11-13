# Ham İHA pilotları — yoğun destek ve görüntü–DSM kapsaması

Araştırma konusu **Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon**. G0–G6 ve olumsuz adaptif rekonstrüksiyon sonucu bu ölçümün kapsamı dışındadır.

## Girdiler ve işlem sınırı

Her iki pilot 28 ham fotoğraf ve 69 × 69 m bir değerlendirme alanı kullandı. Görüntü listeleri, CRS ve sınırlar `outputs/mission-omega/reusable-uav/{urban,sheffield}-selection.json` dosyalarındadır. Daha önce tamamlanan COLMAP kamera, PatchMatch ve füzyon işlemleri yeniden çalıştırılmadı. Urban görüntü–DSM çıktısı da yeniden üretilmedi. Sheffield’da mevcut füzyon ve derinlik çalışma alanından PDAL nokta sayımı, COLMAP Delaunay yüzeyi ve kamera dokulaması, Blender ortografik RGB ve yükseklik geçişleri çalıştırıldı. Zemin sınıflandırma, DTM, bina veya tam veri seti işlemi çalıştırılmadı.

| Pilot | Kayıt | Doğrulanmış çift | Füzyon noktaları | Önceki COLMAP toplamı | Görüntü–DSM aşamalarının toplamı |
|---|---:|---:|---:|---:|---:|
| Urban | 23/28 | 145 | 148.492 | 274,23 sn | 13,41 sn |
| Sheffield | 28/28 | 378 | 412.593 | 277,78 sn | 26,02 sn |

Görüntü–DSM süresi, dört aşamanın kayıtlı süre toplamıdır. Dosya yazma ve Python hazırlığı bu toplama dahil değildir. Sheffield Delaunay aşamasında örneklenen süreç RSS zirvesi 297,5 MiB idi. PDAL ve Blender başlamadan önce sırasıyla 5,96 ve 5,99 GiB kullanılabilir sistem RAM’i kaydedildi. `colmap/dense/fused.ply` dosyalarının SHA-256 değerleri Urban için `e147ef80fdb0a1e0414a617dcf9f9e03b2d2051412e17a8c61c97c0b221f863d`, Sheffield için `9aa26f33329400d3ab69c82e80b78a92c7d66631570036430a43b3bde211860c`.

## Aynı ölçüm yöntemi

`pilot-comparison/review_regions.json` içinde görülebilir yol veya yaya yolu, açık zemin ve belirgin çatı içleri etiketlendi. Etiketler orijinal kareler ve maskesiz RGB görünüm üzerinden çizildi. Urban’ın kayıtlı AirZoo DOM görüntüsü yalnızca konumların görsel kontrolünde kullanıldı. Üretim girdileri ile hiçbir maskeye eklenmedi. Urban’ın yükseltilmiş kortu ayrı kontrol alanıdır. Sheffield’ın ana yolu pilot sınırı dışında kaldığı için 9,80 m²'lik avlu geçişi ölçüldü. Bu küçük geçiş bir şehir yolu genellemesini desteklemez. Alanlar incelenen örneklerdir, pilot içindeki tüm görülebilir sınıf yüzeylerinin sayımı yapılmadı.

1. **Yoğun destek.** PDAL `dense_count.tif` üzerinde 0,50 m hücrede en az bir füzyon noktası. Maske, üretimle aynı en yakın komşu örneklemesiyle 0,20 m görüntü ızgarasına taşındı. Bir dolu hücre doğru yükseklik veya metrekaresinin her noktasında bağımsız ölçüm anlamına gelmez.
2. **Maske öncesi görüntü–DSM.** Blender RGB alfa değeri en az 252, yüzey alfası en az 0,985 ve yükseklik sonlu.
3. **Nihai görüntü–DSM.** GeoTIFF RGB ve DSM geçerlilik maskelerinin kesişimi. Denetim her pikselde bu maskenin ilk iki ölçümün kesişimiyle aynı olduğunu doğruladı.

Tüm oranlarda payda aynı satırın 0,20 m pikselleridir. Tüm pilot için ayrıca 0,50 m yerel PDAL ızgarasındaki dolu hücre oranı Urban’da %30,60 ve Sheffield’da %90,91’dir. Aşağıdaki %30,61 ve %90,94 değerleri aynı desteğin görüntü ızgarasına taşınmış ölçümüdür. Görüntü üzerindeki çokgenler sadece denetim etiketleridir.

| Pilot ve kapsam | Alan (m²) | Yoğun destek | Maske öncesi RGB+DSM | Nihai RGB+DSM |
|---|---:|---:|---:|---:|
| Urban — tüm pilot | 4.761,00 | %30,61 | %90,33 | %29,40 |
| Urban — incelenen görünür zemin | 268,68 | %25,89 | %81,58 | %24,86 |
| Urban — ana yol | 158,72 | %29,16 | %69,71 | %28,18 |
| Urban — açık zemin | 109,96 | %21,17 | %98,73 | %20,08 |
| Urban — üç çatı | 277,96 | %87,62 | %94,82 | %83,08 |
| Urban — yükseltilmiş kort | 234,60 | %34,53 | %100,00 | %34,53 |
| Sheffield — tüm pilot | 4.761,00 | %90,94 | %92,54 | %85,65 |
| Sheffield — incelenen görünür zemin | 850,24 | %99,96 | %95,87 | %95,85 |
| Sheffield — avlu geçişi | 9,80 | %98,78 | %83,67 | %83,67 |
| Sheffield — açık zemin | 840,44 | %99,98 | %96,02 | %95,99 |
| Sheffield — dört çatı | 377,04 | %99,92 | %94,40 | %94,31 |

## Kaybın ilk görüldüğü yer

**Urban.** İncelenen görünür zeminin %74,11’inde 0,50 m yoğun nokta desteği yok. Bunun %56,72 puanında maske öncesi geçerli RGB ve DSM olmasına rağmen noktaya dayalı maske pikseli çıkardı. %17,39 puanında ne yoğun destek ne de geçerli render vardı. Yoğun destek varken geçerli render bulunmayan alan %1,03 puan. Doğu açık zemin örneğinde maskesiz RGB+DSM %100, yoğun destek %0. Bu bölgede mesh renderının yüzey göstermesi, dayanak yoğun noktaların varlığını kanıtlamıyor. Maskeyi kaldırmak ölçülmemiş yüzeyi geçerli sayardı. İlk kanıt yoğun füzyon desteği eksikliği. Bu denetim, eksik desteğin görüntü örtüşmesinden, açıdan veya PatchMatch ve füzyon kabulünden hangi oranda kaynaklandığını ayırmadı.

**Sheffield.** İncelenen görünür zeminde yoğun nokta bulunmayan kısım yalnızca %0,04. Maske, maskesiz RGB+DSM’den %0,02 puan çıkardı. Buna karşılık %4,11 puan yoğun destekli olduğu halde render çiftinde yoktu. Aynı ayrım avlu geçişinde %15,10 puan ve çatılarda %5,60 puan. Buradaki yerel boşluklar maske öncesinde mevcut. Bu denetim, Delaunay yüzeyi, kamera dokusu ataması ve alfa eşiğinin kendi paylarını ayırmadı. Tüm Sheffield pilotunda maskenin çıkardığı %6,90 puan, ölçülen görünür zeminde görülen kayıpla karıştırılmamalı.

Her iki pilot için önceden belirlenen %85 görünür zemin yoğun destek ve görüntü–DSM eşiklerini Sheffield’ın incelenen zemin birleşimi geçti, Urban’ınki geçemedi. Sheffield’ın küçük avlu geçişi ayrı değerlendirildiğinde nihai kapsam %83,67. Henüz DTM sürekliliği veya LOD1 kabulü ölçülmedi.

## Dosyalar ve devam

İşlem dizini `outputs/mission-omega/reusable-uav/`. `urban-pilot/orthophoto/` ve `sheffield-pilot/orthophoto/` klasörleri maskesiz RGB PNG, maskeli ortofoto GeoTIFF, DSM GeoTIFF, 0,50 m yoğun nokta sayımı ve aşama günlüklerini içerir. `pilot-comparison/` altında iki karşılaştırma PNG’si, ham fotoğraf seçkileri, alan tanımları ve piksel sayılarıyla `coverage.csv` ile `coverage.json` bulunur.

Mevcut Sheffield önbelleğinden yalnızca ortofoto/DSM aşamasına dönmek için:

```powershell
& 'C:\projects\Vision-Based-3D-Reconstruction\scripts\run_omega_uav_pilot_orthophoto.ps1' -PilotOutput 'C:\projects\Vision-Based-3D-Reconstruction\outputs\mission-omega\reusable-uav\sheffield-pilot'
```

Komut mevcut manifesti ve tamamlanmış COLMAP çıktısını ister. Bu oturumun sonunda Sheffield ortofoto aşaması zaten tamamlanmış olduğu için çağrı doğrulanmış önbelleği kullanır. İki pilotun denetimini tekrarlamak için depo kökünden:

```powershell
$root = 'outputs/mission-omega/reusable-uav'
& 'outputs/mission-omega/building-recovery/runtime/venv/Scripts/python.exe' scripts/omega_uav_coverage_audit.py --root $root --review 'Documentation/evaluation/omega_raw_pilot_review_regions.json' --output "$root/pilot-comparison"
```

Bu ara kontrolün commit’i raporu, değerlendirme poligonlarını ve kapsama denetimini içerir. Devam komutu ve uçtan uca üretim entegrasyonu mevcut çalışma ağacındaki yerel dosyalara bağlıdır. DTM ve bina aşamalarının doğrulanması sonraki oturuma kalmıştır.
