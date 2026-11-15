# Kalite geri beslemeli bölgesel yeniden yoğunlaştırma — tasarım

Durum: taslak (2026-11-14). Araştırma katkısının tasarımı; literatür gerekçesi Claude Doc'ta
(https://claude.ai/artifact/PPwpvVYfQJiud2e7Uvox8b, "Boşluk analizi"). Hipotezler H1–H3 orada önceden ilan edildi
ve burada değiştirilmez.

## 1. Problem

Sabit bir hesaplama bütçesi `B` (duvar saati) ve tepe RAM sınırı `M` verildiğinde, SfM bir kez çözülmüş bir İHA
bloğunda yoğun eşlemeyi (MVS) hangi bölgelerde hangi çözünürlükte yapmalıyız ki bağımsız referansa göre
geçerli tamamlanmışlık (τ eşiğinde recall, precision korunarak) en yüksek olsun?

Kısıtlar:
- SfM ve kamera modeli bütün yöntemlerde aynıdır (bir kez çalışır, dondurulur).
- Bölge seçimi yalnız **referanssız** sinyallerle yapılır; referans (LiDAR) yalnız değerlendirmede kullanılır.
- Ayarlar (eşik, ağırlık, karo boyu) yalnız ayar setinde (UseGeo-1) belirlenir; UseGeo-2/3 hold-out.

## 2. Bölge birimi: dünya ızgarasında karo, karo başına görüntü alt kümesi

ODM'nin `--split` alt modelleri bölge birimi olarak kaba (yüzlerce görüntü) ve her biri SfM'i yeniden çalıştırıyor;
bu, "aynı kamera modeli" kısıtını bozar. Bunun yerine bölge = dünya XY ızgarasında kare karo (kenar `T`).

Fizibilite (2026-11-14, ODM 3.6.2'nin OpenMVS/OpenSfM ikilileri, Sheffield sahnesi, 2 iş parçacığı):

- **OpenMVS ROI kırpması bölgesel maliyet düşürmez.** `--import-roi-file tile.txt --crop-to-roi 1` sahneye ROI
  koyar ama görüntüleri atmaz; ardından yoğunlaştırma 172 görüntünün hepsine derinlik haritası hesapladı (50 m karo için
  ~20 s/görüntü). ROI yalnız füzyonu sınırlar. (`--estimate-roi 1 --export-roi-file` sahne OBB'sini yazar: 3×3 dönme,
  merkez, yarı-uzunluklar, topocentric çerçeve.) `--archive-type 3` ile kaydedilen kırpılmış sahne bozuk çıkıyor.
- **Çalışan yol: karo başına görüntü alt kümesi.** OpenSfM'in `export_openmvs` eylemi bir görüntü listesi alıyor
  (`export_only`); çıktı yolunu ana sahneyi ezmeyecek klasöre yönlendirip yalnız karodaki seyrek noktaları ≥k kez gören
  çekimler dışa aktarılır. Aynı SfM, aynı kamera modeli, aynı OpenMVS bayrakları.
  Ölçüm: 30×30 m karo, k=400 → 8/172 görüntü; `DensifyPointCloud` düzey 3: **150 s, tepe RAM 0,3 GB**, 31 MB PLY.
- Maliyet görüntü sayısıyla orantılı ve OpenMVS her görüntünün **tam** derinlik haritasını hesaplıyor: karoyu kısmen gören
  her görüntü tam maliyet. Karo kenarı `T`, görüntü izi `F` için ek yük ≈ (T+F)²/T². Sheffield'da (küçük sahne, büyük iz)
  50 m karo 138/172 görüntü gördü. Sonuç: `T` ≥ 2–3·F seçilir ve seçilen bitişik karolar tek sahnede birleştirilir
  (görüntü birleşimi bir kez hesaplanır); ek yük deneyde ölçülüp raporlanır.
- Bu OpenMVS derlemesinde GPU yolu RTX 5070'de (sm_120) çöküyor (0xC0000005, CUDA çekirdeği yok); `--cuda-device -2`
  (CPU) zorunlu. ODM'nin GPU'yu "kullanmaması" bundan: hata sonrası CPU'ya düşüyor.
- **ODM'nin istediği düzey ≠ kullanılan düzey.** ODM `--resolution-level` ile birlikte `--min-resolution 640` verir;
  OpenMVS kenar 640 px'in altına düşerse düzeyi düşürür. 42 MP UseGeo görüntülerinde "medium" = düzey 4 istenir,
  düzey 3 (994×663 px) kullanılır (`.dmap` başlığından doğrulandı). İlk "ince" geçiş (düzey 3) bu yüzden kaba geçişle
  aynı çıktı (nokta sayısı 28,95 M / 29,06 M; karo kazancı ≈ 0) ve geçersiz sayıldı
  (`outputs/tune/usegeo_1/*_l3_same_as_coarse*`). İnce geçiş = düzey 2 (1988 px). `refine.effective_size` ve
  `bench/fine_pass.py` bu durumu artık reddediyor.
- **Karo başına görüntü (UseGeo-1, 25 m karo, `bench/tile_shots.py`):** k=50 → 260 karo, medyan 17 görüntü (p10 4,
  p90 44); k=200 → 226 karo, medyan 11 (p10 3, p90 31,5); k=400 → yalnız 36 karo. Toplam 224 görüntü.
- **Alt sahne kapsaması k'ya bağlı (UseGeo-1, karo 85, düzey 3 = kaba geçişle aynı düzey).** Tam sahne karoda
  123 k nokta üretiyor. Alt sahneyle: k=200 → 7 görüntü, 38,7 k nokta (%31); k=50 → 24 görüntü, 97,1 k (%79);
  k=20 → 26 görüntü, 97,3 k (doyum). Eksik %21, alt sahnede bulunmayan görüntülerin derinlik haritalarından
  gelen katkı. Varsayılan k=50. Replay'in "alt sahne karoyu tam sahne gibi yoğunlaştırır" varsayımı bu
  nedenle iyimser; gerçek alt sahne koşusuyla (`bench/validate_subsets.py`) P/R farkı ölçülecek.
- Karo bulutu ODM'nin topocentric çerçevesinde; birleştirme için ODM'nin georeferans dönüşümü (coords.txt ofseti) uygulanır.
- Betik: `bench/feasibility/tile_export.py` (ODM Python'u ile, `ODM_HOME` ortam değişkeniyle) → `uav3d/refine.py`'ye
  taşınacak.

## 2a. Ayar sonuçları (UseGeo-1, 2026-11-15)

- Derinlik haritası süresi: kaba (düzey 3) 1145 s, ince (düzey 2) 4590 s, 224 görüntü → oran 4,0 = 4^(L_c−L_f).
  İnce füzyonun tepe RAM'i 10,9 GB (en büyük alt sahne), derinlik haritaları 6,7 GB.
- Karo F20 kazancı medyan +0,030 (p10 −0,053, p90 +0,098); karoların %29'unda negatif. Negatif karolar kaba geçişte
  de zayıf (P 0,39, R 0,30), yoğunlukları düşük (medyan 0,77 vs 1,61); ince düzey ikisini de düşürüp 3,2× nokta
  ekliyor. Tam ince F20 0,637 < oracle %75 0,661.
- Spearman (gain_f20): yoğunluk +0,33, örtüşme +0,18, boşluk −0,17, rölyef +0,16, pürüzlülük +0,14 (hepsi p<0,05).
  Yön: kazanç yoğun/çok örtüşmeli karolarda → "düşük örtüşme önce" statik kural ters çalışıyor.
- Model (`configs/gain_model.json`, commit f1036b4, hold-out görülmeden donduruldu): yoğunluk 0,0020, rölyef 0,0006,
  bias −0,0012; tahmin–gerçek katkı Spearman 0,68. Seçim yalnız tahmini katkı > 0 olan karolar (oracle gibi).
- Replay F20 (β 25/50/75): adaptive 62,6/63,4/64,6; static 59,2/58,4/60,6; random ort. 60,5/60,4/61,9;
  oracle 63,1/64,6/66,1; kaba 61,6; tam ince 63,7.
- Gerçek alt sahne koşusu (β=0,5, `outputs/tune/usegeo_1/validate/adaptive_0.5`): F10/F20/F50 41,1/64,2/86,6;
  replay 40,0/63,4/88,6; kaba 38,0/61,6/87,5. P20 79,0 (replay 75,0), R20 54,1 (55,0). 116 görüntü (tahmin 111),
  derinlik haritası 2705 s (tahmin 2377, +%14), toplam 3062 s, tepe 10,4 GB.
  İlk skor (F20 65,0) geçersizdi: `refine.merge`'deki PDAL `filters.crop` (multipolygon, outside=true) kaba noktaları
  hiç atmıyordu; düzeltme `refine.drop_tiles` (bd6d045). Hatalı bulut `merged_overlap_bug.laz` olarak saklandı.

## 3. Döngü

1. **Geçiş 0 — SfM** (ODM `--end-with opensfm`), bir kez. GSD hesaplanır (mevcut akış).
2. **Geçiş 1 — kaba tekdüze yoğunlaştırma**, düzey `L_c` (ör. `--resolution-level` 3). Ürünler: nokta bulutu, DSM,
   ODM `overlap.tif`, derinlik haritası güvenleri. Maliyeti ölçülür: `C_1`.
3. **Sinyaller** (karo başına, referanssız):
   - `s_ov`: medyan görüntü örtüşmesi (ODM `opensfm/stats/overlap.tif`) — düşük örtüşme ↔ büyük hata
     (UseGeo 1/2/3, LiDAR'a göre kaba F20 ile Spearman 0,62 / 0,61 / 0,46; <5 görüntü medyan F20 0,23–0,46,
     10–20 görüntü 0,81–0,84).
   - `s_void`: DSM boşluk oranı (geçiş 1 bulutundan, ince ızgarada).
   - `s_dens`: nokta yoğunluğu / beklenen yoğunluk (GSD ve düzeyden).
   - `s_rough`: yerel pürüzlülük (DSM'in düzlem artığı) — yapı/bitki kenarları ince düzeyden en çok kazanır.
   - `s_conf`: derinlik haritası güven ortalaması (OpenMVS `.dmap` güven kanalı) — uygulanabilirse.
   - Bina varsa Roofer RMSE (> 1 m düşük güven) yalnız ikincil sinyal.
4. **Beklenen kazanç** `g(t) = w · φ(s(t))`; ağırlıklar ayar setinde, referansla ölçülen karo kazancına doğrusal/monoton
   regresyonla uydurulur (yalnız UseGeo-1).
5. **Maliyet** `c(t) = n_img(t) · κ(L_f)`; `n_img` dışa aktarılacak görüntü sayısından (ucuz), `κ` geçiş 1'deki görüntü
   başı süreden `4^(L_c−L_f)` ölçeklemesiyle, ilk ince karolar bitince çevrimiçi düzeltilir.
6. **Seçim**: kalan bütçe `B − C_1` içinde `g/c` oranına göre açgözlü sırt çantası; her adımda gerçek süre ölçülür,
   bütçe aşılacaksa durulur (anytime). RAM: karo kenarı `T`, en kötü karo ince düzeyde `M`'i aşmayacak şekilde seçilir.
7. **Birleştirme**: seçilen karoların içinde geçiş 1 noktaları silinir, ince noktalar eklenir (PDAL, karo poligonu);
   ardından mevcut `terrain` aşaması (temizlik, DTM/DSM) aynen çalışır.

## 4. Karşılaştırma (deney, 22–25 Kas)

Bütçe `B` ∈ {%25, %50, %75} × (tam ince tekdüze geçişin süresi). Hepsi aynı SfM'den:
- **U-low**: yalnız geçiş 1 (bütçenin geri kalanı kullanılmaz).
- **U-mid**: bütçeye sığan en ince tekdüze düzey.
- **Static**: aynı bütçe, karo sırası yalnız a priori sinyalden (örtüşme `s_ov`), geri besleme yok (H2 kontrolü;
  Mostegel vd. 2018'in sabit foto setinde statik sıralamasına karşılık).
- **Random**: aynı bütçe, rastgele karo sırası (3 tohum).
- **Adaptive**: §3.
- **Oracle** (yalnız analiz): karo kazancı referanstan; üst sınır.

Ölçütler: τ = 10/20/50 cm'de precision/recall/F (mevcut `uav3d.evaluate.geometry`, dikey ofset hizalı), duvar
saati, tepe RSS, seçilen karo oranı. Çürütme ölçütleri literatür belgesindeki gibi.

## 4a. Hold-out sonuçları (UseGeo-2/3, 2026-11-15)

`bench/holdout_analysis.py` → `outputs/holdout/hypothesis.json`, şekil `paper/figures/butce_kalite.pdf`.
Uzamsal blok bootstrap (4×4 karo = 100 m süper blok, 2000 tekrar), τ = 20 cm. Uygulanan U-low ve tam ince;
U-mid ve split-merge karşılaştırması yapılmadı (raporda sınırlılık).

| F20 % | U2 β25/50/75 | U3 β25/50/75 |
|---|---|---|
| Static | 68,4 / 70,1 / 72,6 | 63,1 / 64,1 / 66,2 |
| Random | 69,8 / 70,7 / 73,2 | 64,1 / 65,2 / 65,3 |
| Adaptive | 72,4 / 73,6 / 75,3 | 64,4 / 66,2 / 68,9 |
| Oracle | 72,5 / 74,1 / 75,5 | 67,2 / 68,7 / 71,5 |
| kaba / tam ince | 70,1 / 77,3 | 64,9 / 70,8 |

- Toplam maliyet c = (T_kaba + β T_ince) / T_ince ≈ 0,25 + β (üç şeritte kaba/ince 0,25–0,26).
- Spearman(tahmin, gerçek katkı) 0,66 / 0,69 (ayar 0,68). Zararlı karo payı %7 / %10 (ayar %29).
- H1 reddedildi: recall artışı c=0,5'te +1,7 / +1,1, β=0,5'te +2,8 / +2,4 puan (eşik +5).
- H2: Adaptive − Static F, β=0,5: U2 +3,5 [1,6; 5,6], U3 +2,0 [−0,2; 3,8]; c=0,5'te ikisi de anlamlı.
- H3: yön genellendi, büyüklük genellenmedi (tam ince kazancın β=0,5'te geri kazanılan payı %89 / %48 / %22).
- Ek yük: c≈1'de tam ince, hold-out şeritlerde Adaptive'den 2,0 / 1,9 puan iyi; ayarda Adaptive 0,9 puan önde.
  Seçici inceltme yalnız c<1'de anlamlı. Tepe bellek avantajı yok (alt sahneler de 10–11 GB).

## 5. Açık riskler

- Karo sınırında dikiş: komşu karoları gören görüntüler farklı düzeyde → füzyonu ROI ile karo + kenar payına sınırla.
- Küçük sahnede (iz/karo oranı büyük) ek yük kazancı yiyebilir; bu bir bulgu olarak raporlanır (büyük blokta oran küçülür).
- Hold-out UseGeo-2/3 ODM'de split yoluna düştü (RAM modeli). Deney tek SfM ister: SfM'i split'siz çalıştırmak
  (UseGeo-1'de SfM tepe RAM'i 6,6 GB, bütçe altında) ve yoğunlaştırmayı karo döngüsüne bırakmak bölgesel yöntemin ana
  argümanıdır: bellek sınırını alt modelde değil karoda çözer.
- ODM split-merge Windows'ta submodel yeniden çalıştırmasında `python3 run` çağırıyor (yok/Store kısayolu);
  `tools.Odm.split_shims()` ile düzeltildi (2026-11-14).
