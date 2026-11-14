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

## 3. Döngü

1. **Geçiş 0 — SfM** (ODM `--end-with opensfm`), bir kez. GSD hesaplanır (mevcut akış).
2. **Geçiş 1 — kaba tekdüze yoğunlaştırma**, düzey `L_c` (ör. `--resolution-level` 3). Ürünler: nokta bulutu, DSM,
   ODM `overlap.tif`, derinlik haritası güvenleri. Maliyeti ölçülür: `C_1`.
3. **Sinyaller** (karo başına, referanssız):
   - `s_ov`: medyan görüntü örtüşmesi (ODM `opensfm/stats/overlap.tif`) — düşük örtüşme ↔ büyük hata
     (AirZoo rural, referans DSM'e göre: 2 görüntü medyan |hata| 2,3 m, ≥8 görüntü 0,5 m; UseGeo-1'de LiDAR'la
     tekrarlanacak).
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

## 5. Açık riskler

- Karo sınırında dikiş: komşu karoları gören görüntüler farklı düzeyde → füzyonu ROI ile karo + kenar payına sınırla.
- Küçük sahnede (iz/karo oranı büyük) ek yük kazancı yiyebilir; bu bir bulgu olarak raporlanır (büyük blokta oran küçülür).
- Hold-out UseGeo-2/3 ODM'de split yoluna düştü (RAM modeli). Deney tek SfM ister: SfM'i split'siz çalıştırmak
  (UseGeo-1'de SfM tepe RAM'i 6,6 GB, bütçe altında) ve yoğunlaştırmayı karo döngüsüne bırakmak bölgesel yöntemin ana
  argümanıdır: bellek sınırını alt modelde değil karoda çözer.
- ODM split-merge Windows'ta submodel yeniden çalıştırmasında `python3 run` çağırıyor (yok/Store kısayolu);
  `tools.Odm.split_shims()` ile düzeltildi (2026-11-14).
