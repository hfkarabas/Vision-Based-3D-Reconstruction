# UAV Görüntülerinden Kullanılabilir 3D Asset: Kaynak Taraması ve Kesin Teknik Karar

**Tarih:** 12 Kasım 2026. **Hedef:** *Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon.*

**Önemli kanıt sınırı:** Bu çalışma 147 farklı akademik kaynağın başlık/bibliyografya düzeyinde ön elemesini, en ilgili yayınların erişilebilir özet/yöntemlerini, resmi yazılım belgelerini ve daha önce kullanıcı tarafından sağlanan AirZoo deney raporlarını bir araya getirir. **147 ayrı tam metin okuması, sistematik tarama veya yeni deney değildir.** Proje Windows kaynak kodunu ve orijinal AirZoo dosyalarını bu oturumda doğrudan çalıştırıp denetlemedim.

## 1. Karar (yeni SfM/MVS/mesher yazılmayacak)

İlk çalışır ve görsel açıdan incelenebilir sürüm:

1. **OpenDroneMap (ODM):** UAV RGB fotoğrafları veya desteklenen video girdilerinden SfM/MVS, georeferanslı nokta bulutu, **DTM**, **DSM** ve gerçek **ortofoto** oluştur. Varsayılan Poisson/özgür 3D mesh'i nihai arazi yüzeyi diye kabul etme.
2. **QGIS 3D Map View:** DTM'yi terrain elevation olarak, ODM ortofotosunu terrain texture olarak kullan. Böylece arazinin tek değerli yükseklikle temsil edilebilen kısmı tutarlı bir 2.5D yüzey olur. İki rasterın aynı CRS, aynı yükseklik datumu, hizalı hücre ağı ve yeterli doğruluğa sahip olması kontrol edilir.
3. **Bina ayak izleri:** QGIS **QuickOSM/Overpass** ile OpenStreetMap `building` poligonlarını getir; yoksa **Overture Maps** GeoParquet/GeoJSON'dan uygun alanın bina poligonlarını al. Verinin kapsamı ve konum doğruluğu ayrıca kontrol edilir. İki kaynağı körlemesine birleştirme.
4. **Bina yükseklikleri:** Öncelik bina yükseklik etiketleri ve desteklenen ODM DSM–DTM farkı; QGIS **Zonal statistics** üzerinden uygun çatı ve çevre zemin örneklerini ayır. Eksik yükseklik varsa `building:levels` veya yaklaşık değer etiketiyle belirsizliği kaydet; asla tahmini metreyi kesin ölçüm gibi sunma.
5. **LOD1 bina mesh'i:** QGIS 3D sembolojisinde bina poligonlarına `Extrusion` uygula. Beyaz, fotoğrafik dokusuz mat materyal kullan. DTM'nin üzerindeki bina yükseltisini iki kere uygulama.
6. **Export:** QGIS **Export 3D Scene → OBJ + textures** birincil yol; **Qgis2threejs → glTF** ikinci yol. Blender ile OBJ'yi açıp **GLB** dönüştür ve orijin/scale/UV kontrol et. Unity tarafında hazır **glTFast** veya uygun ithalat yolu kullan; sahne gerçekten açılmadan Unity compatible deme.
7. **Üretim doğrulaması:** Önce örtüşmesi güçlü ve coğrafi bakımdan tutarlı **tek küçük bölge**; sonra Rural/Urban daha büyük alanlar. İlk gerçek kabul kapısı: Blender'da doğru arazinin gerçekten görünmesi. Ardından coğrafi kapsam, DSM/DTM hata analizi, işlem süresi, bellek ve Unity import doğrulaması.

**İleri seçenekler, zorunlu ilk yol DEĞİL:** TU Delft'in `3dfier` projesi nokta bulutu (LAS/LAZ) ve semantik 2D poligonları birlikte kullanarak yol/zemin/LOD1 bina için birbirine dikilmiş şehir yüzeyleri oluşturur. QGIS ekleminde problemler çıkarsa kullan. OpenMVS, COLMAP, AliceVision, PDAL, CloudCompare/CSF, BlenderGIS, OSM2World, Urban Export ise ihtiyaca bağlı alternatiftir. Hepsini aynı anda kurma.

## 2. Araştırma konusu ve sınırlar

**Değişmez araştırma başlığı:** *Hesaplama Kaynakları Kısıtlı Büyük Ölçekli İHA Görüntülerinde Kalite Geri Beslemeli Bölgesel Adaptif 3B Rekonstrüksiyon*.

- **Araştırma girdisi:** İHA RGB görüntüleri, bunlardan kurulmuş ilk rekonstrüksiyon ve sınırlı işlem bütçesi.
- **Araştırma çıktısı:** Görüntülerin gözlediği sınırlar içinde daha yüksek doğruluk/bütünlükte 3D asset veya aynı kaliteyi daha az sürede/bellekte üretme.
- **Araçlar:** ODM, QGIS, GIS filtreler, WebODM/NodeODM, Qgis2threejs, 3dfier, Blender; sıfırdan SfM/MVS, heightmap ve geometri algoritması yok.
- **Muhtemel *en küçük* katkı:** ODM'yi standart şekilde çalıştır; raster `NoData`, yüksek belirsizlik, eksik ortofoto, yetersiz bina/görüntü desteği ve işlem maliyetini ölçen **bölgesel kalite haritası** oluştur; WebODM/ODM'nin mevcut split/overlap ve kalite parametreleriyle problemli bölgeleri tekrar işle, yalnızca doğrulanmış fayda varsa yeni ürüne al. Bunun için QGIS Processing Modeler, Raster Calculator, Zonal Statistics ve mevcut optimizasyon kütüphaneleri kullanılabilir; uzun bir custom Python motoru yazmaya gerek yok.
- **Özgünlük uyarısı:** Bölgesel kalite önceliği ve adaptif görüntü seçimi zaten literatürde. Çalışmanın özgün yanı ancak **sabit işlem bütçesi altında, önceden çekilmiş UAV görüntüleri ve çıktı raster/mesh kalitesiyle kapalı döngü yeniden işleme** kombinasyonunun mevcut yöntemlere göre gerçekten farklı ve iyileştirici olduğu kanıtlanırsa ortaya çıkar. Araştırma fikri şu an bir *hipotez*, kanıtlanmış literatür boşluğu değil.
- **Dış veri ayrımı:** Haritadan/Urban Export/LiDAR'dan doğrudan alınan arazi/bina geometrisi pratik ürünün tamamlayıcısı olabilir, ancak RGB-only araştırmanın sonucunu değerlendirmek için saklı kaynak olarak kullanılamaz. `UAV-only`, `UAV+external-OSM`, `UAV+external-DEM/LiDAR` farklı deney kolları olarak etiketlenir. Üretimde dış LiDAR mümkün olsa bile RGB görüntüden LiDAR ölçümü üretildiği iddia edilemez.

## 3. Teknoloji karşılaştırma matrisi

| Teknoloji | Yaptığı hazır iş | Girdi/çıktı | Karar / lisans | Kaynak |
|---|---|---|---|---|
| **OpenDroneMap** | UAV fotogrametri, DTM, DSM, ortofoto, nokta bulutu, texturing | Image/video -> TIFF, LAZ, OBJ/GLB | **Zorunlu ilk test**; AGPL-3.0 | https://github.com/OpenDroneMap/ODM |
| **QGIS** | DEM 3D yüzeyi, coğrafi raster texture, bina extrude, OBJ export | GeoTIFF, GeoJSON -> OBJ | **Zorunlu**; GPL | https://docs.qgis.org/testing/en/docs/user_manual/map_views/3d_map_view.html |
| **QuickOSM** | Bina tabanı alma | OSM -> QGIS vector | **Bina için birinci seçenek**; OSM verisi ODbL | https://plugins.qgis.org/plugins/QuickOSM/ |
| **Qgis2threejs** | DEM + vektör -> glTF | QGIS scene -> glTF | **GLB/glTF alternatif**; QGIS sürüm uyumuna bak | https://plugins.qgis.org/plugins/Qgis2threejs/ |
| **3dfier** | LAS/LAZ + semantik poligon -> bitişik LOD1 şehir modeli | LAZ + shape -> OBJ, CityJSON | **Yedek/2. aşama**; GPL-3.0 | https://github.com/tudelft3d/3dfier |
| **OpenMVS** | Dense, meshing, refine, texture | Camera + images -> pointcloud, mesh | **Gerekirse**; AGPL-3.0 | https://github.com/cdcseacave/openMVS |
| **COLMAP** | SfM + MVS, fusion, mesh | Images -> camera model/cloud/mesh | **Bağımsız referans**; BSD | https://github.com/colmap/colmap |
| **WebODM/NodeODM** | GUI, job organization, split/large-area processing | Image batches -> ODM artifacts | **Sabit bütçe araştırmasında** | https://github.com/OpenDroneMap/WebODM |
| **PDAL** | Nokta bulutu sınıflandırma/filter | LAS/LAZ -> LAS/LAZ | **Gerekirse** | https://pdal.io |
| **CloudCompare + CSF** | Geometrik inceleme, ground filtering | PLY/LAS/LAZ -> analyzed cloud | **Gerekirse** | https://www.cloudcompare.org |
| **Overture Maps** | Açık bina geometrisi | Geographic parquet -> polygons | **OSM yedeği**; kaynak özelinde lisans | https://docs.overturemaps.org/getting-data/ |
| **OSM2World** | OSM semantiğinden 3D şehir | OSM -> OBJ/glTF etc. | **Karşılaştırma/yedek** | https://osm2world.org |
| **Blender** | GLB export, asset QA | OBJ/glTF -> GLB | **Zorunlu asset QA** | https://www.blender.org |
| **Urban Export** | Dış harita/DEM verilerinden şehir asset'i | OSM/DEM -> exported model | **Kıyas/yasal veri**; kaynak kod açık doğrulanmadı | https://urban-export.ildar.space/ |

## 4. Hazır birinci pilotun teknik reçetesi

### A. Veri/georeferanslama

- Orijinal gerçek UAV RGB fotoğraflarını temin et; önce birbirine gerçekten örtüşen 30–60 fotoğraf seç (sayının kendisi kesin değil). AirZoo Rural ve Urban için bütün 190/161 fotoğrafta COLMAP camera registration daha önce raporlandı; fakat bu görüntülerin meshing/DTM başarısını garanti etmez.
- Görüntülerin EXIF geotag yoksa kamera merkezlerinin gerçek koordinatlarıyla ODM `geo.txt` tanımını hazırla; başlıkta doğru EPSG, sonraki satırlarda dosya adı ve `X Y Z`. ODM formatı: https://docs.opendronemap.org/geo/. PPK / yükseklik datumu denetle. Sahne dünya koordinatları belli değilse OSM binalarını üzerine otomatik yerleştiremezsin.
- Veri setinde örnek referans DOM/DSM olabilir. Onları `UAV-only` üretim hattının DTM girdisi yapma ve sonra aynı rasterı bağımsız ground truth diye kullanma.

### B. ODM kontrollü pilot

- Desteklenen Docker örneği; `X:/odm-data` pilot kökünün Windows'taki örnek konumudur, **gerçek dosya yolu değildir**. `/datasets/rural-pilot/images/` içine resimler konur.

```powershell
# Once: docker pull opendronemap/odm
# Image klasoru: X:\odm-data\rural-pilot\images\
docker run --rm -it -v "X:/odm-data:/datasets" opendronemap/odm --project-path /datasets rural-pilot --dtm --dsm --dem-gapfill-steps 0 --dem-euclidean-map
```

- Açıklama: `--dem-gapfill-steps 0` NoData'yı interpolasyonla kapatmayı kapatır; bu ilk bilimsel testte **kasıtlıdır**. ODM default interpolasyonu 3 adım yapar. Referans: https://docs.opendronemap.org/arguments/.
- Çıktı kontrolü: `odm_dem/dtm.tif`, `odm_dem/dsm.tif`, `odm_orthophoto/odm_orthophoto.tif`, nokta bulutu, rapor. Dosyaların gerçek varlığını kontrol et.
- **Geçme kriteri:** Tek bölgenin DTM'sinde anlamlı yükseklikler ve ortofoto ile aynı ayak izi. DSM ve DTM arasındaki farkın mantıklı olması. NoData alanlarını saklamamak.

### C. QGIS ile yüzey/bina

- QGIS projesinin CRS'sini gerçek metre kullanan projected CRS'ye al; AirZoo referansının EPSG:4547 olduğu raporlandı, ama fiilî DTM'nin CRS'sini doğrula.
- `dtm.tif` = 3D terrain elevation; ortofoto = texture olarak 3D Map View. Uygun arazi yükseklik ölçeğini **1:1** ayarla; modele olduğundan dik tepeler ekleyen vertical exaggeration kullanma.
- QuickOSM ile `building` vektör tabanı çek, CRS/datum dönüştür, raster ile bindir ve gözle kontrol et. Poligon bulunmazsa Overture ya da ölçülen ortofoto üzerinde **etiketli manuel digitizing** (tek seferlik hazırlık) ile pilota geç; yüzlerce bina için otomatik olmayan iş maliyetini açıkça kaydet.
- Eğer DSM+DTM iyi ise bina çatı değerini güvenilir DSM piksellerinden, bina çevresi zeminini DTM'den tahmin et; QGIS Zonal Statistics kullan. Çatısı hiç görülmeyen bina için kesin yükseklik iddia etme.
- Polygon extrusion -> beyaz LOD1 bina, texture yok. Görünmeyen çatı biçimini gerçeğiymiş gibi üretme.
- Export 3D Scene -> OBJ; texture MTL ve görselleriyle Blender'da import. İkinci yol: Qgis2threejs glTF (QGIS3 ve QGIS4 sürüm uyumu kritik). Blender -> GLB.
- **Geçme kriteri:** QGIS ve Blender'da aynı arazinin tek, tanınabilir surface olması; bina yüksekliğinin zeminle uyumu; renk/texture ve world scale; Unity asset içe aktarımı ayrı test.

### D. Kademeli genişletme

- Pilot başarılıysa tüm Rural ve sonra Urban; yüksek bellek basıncında WebODM/ODM split/overlap imkânı; her coğrafi tile kendi quality/coverage raporuyla birleştirilir.
- Bir tile başarısız olduğunda bütün sahneyi yeniden çalıştırmadan yalnızca o bölgeyi işlemek araştırma odağına uygundur. Ancak bu yöntemin önceki yayınlardan farkı ve işlem bütçesine göre avantajı gösterilmeli.

## 5. Bilimsel araştırma: en yakın rakipler ve doğrudan öğrendiklerimiz

| Çalışma | Bulgusu / kullandığı mekanizma | Bizim işimize yarayan ve özgünlükte risk yaratan yön |
|---|---|---|
| Mostegel, Fraundorfer & Bischof (2018), *Prioritized Multi-view Stereo Depth Map Generation Using Confidence Prediction*, https://doi.org/10.1016/j.isprsjprs.2018.03.022 | MVS depth hesaplamalarını görüntüden çıkarılan confidence ve geometri üzerinden önceliklendiriyor. Bazı deneylerinde yalnızca %5 key view ile erişilebilir kalitenin %70'ini sağlıyor. | **Önceliklendirme + hesaplama bütçesi tek başına yeni değil.** Bizim fikir sadece bir greedy image selector olamaz. |
| Lim et al. (2021), *An Optimal Image–Selection Algorithm for Large-Scale Stereoscopic Mapping of UAV Images*, https://doi.org/10.3390/rs13112118 | Belirli ROI'yi örten doğru stereopairs bulup gerekli görüntü sayısını azaltıyor. | İlk görüntü seçimi zaten yapılmış; kapalı döngü mesh/DTM geri beslemesi ayrımını netleştirmeliyiz. |
| Neumann et al. (2025), *Point Cloud Quality Metrics for Incremental Image-Based 3D Reconstruction*, https://doi.org/10.1007/s11042-025-20596-6 | Yerel nokta bulutu kalite metrikleri ve RQF geliştirip hangi alanların ek görünümlere gereksinimi olduğunu belirliyor. Çalışmanın kodu ve benchmarkı açık. | **Quality feedback de tek başına yeni değil.** RQF en güçlü karşılaştırma adayımız. |
| Zhang, Liu & Haala (2024), *Guided by model quality: UAV path planning for complete and precise 3D reconstruction of complex buildings*, https://doi.org/10.1016/j.jag.2024.103667 | Mevcut kaba modele bakıp görüntü eksiklerini belirliyor, UAV'yi yeniden fotoğraf çekmeye yönlendiriyor. | Bizim deneyde **yeniden uçuş yok**; önceden çekilmiş görüntülerin yeniden işlenmesi ve bölgesel bütçe kararları ayrım olabilir. |
| Lou et al. (2025 preprint / 2026 journal), *On-the-Fly Feedback SfM*, https://github.com/IRIS-LAB-whu/OntheflySfMFeedback | Kademeli SfM, mesh-quality indicator, yeni flight-path planning. Açık kod mevcut. | Ölçütleri/çok bileşenli kalite göstergesini yeniden yazmadan kullanma ve prior-work kıyas. Online yeni görüntü planlamasıyla offline yeniden işleme ayrımını test et. |
| Ledoux et al. (2021), *3dfier: automatic reconstruction of 3D city models*, https://doi.org/10.21105/joss.02866 | LAS/LAZ + GIS poligonlarından topoloji uyumlu LOD1 şehir, terrain, yollar. Açık kod GPL. | **Hibrit city model fikri yeni değil; doğrudan kullan.** |
| Stathopoulou et al. (2024), *Depth Estimation and 3D Reconstruction from UAV-borne Imagery: Evaluation on the UseGeo Dataset*, https://doi.org/10.1016/j.ophoto.2024.100065 | COLMAP, OpenMVS, ACMMP ve diğer yöntemler için ortak ölçüm kodu yayımlıyor; farklı aşamalarda farklı motorlar öne çıkıyor. | Tek bir algoritmanın her sahnede iyi olduğunu iddia etme; hazır değerlendirme kodu kullan. |
| *Sensitivity Analysis of Parameters and Contrasting Performance of Ground Filtering Algorithms with UAV Photogrammetry-based and LiDAR Point Clouds* (2020), https://doi.org/10.1080/17538947.2020.1791267 | Fotogrammetrik nokta bulutundan DTM türetme hataları ve LiDAR farkı, eğim/bitki örtüsü bağımlılığı. | DTM filtering de ölçülmeli; özellikle orman/bitki örtüsü altında RGB ile çıplak zemin görülmüyorsa garanti edilemez. |

## 6. Özgünlük için denenecek en küçük bilimsel hipotez

> **H1:** Önceden toplanmış UAV fotoğraflarıyla üretilen ilk DTM/ortofoto ve 3D asset üzerinde otomatik tespit edilen *geçerli geometri boşlukları, lokal yükselti belirsizliği, görüş yetersizliği ve doku sorunları* temel alınarak, ODM/WebODM'nin mevcut bölgesel bölme ve yüksek kalite işleme seçenekleri yalnızca gerekli yerde uygulanırsa, aynı RAM/zaman bütçesinde standarda göre daha yüksek **referansla doğrulanmış** arazi/asset bütünlüğü elde edilir.

**Bu henüz kanıtlanmış akademik yenilik değil.** Metot şu koşullarda kabul edilir: (a) yakın rakiplerin makaleleri incelenmiş; (b) standart ODM, random/equal-budget, image/ROI selection, Mostegel-benzeri öncelik ve RQF-benzeri kalite göstergesi baselines; (c) tüm yöntemler eşit işlem süresi veya belirlenmiş bütçede; (d) referans DSM/DTM/LiDAR *girdi olarak kullanılmaz*, yalnızca değerlendirme; (e) alt bölge/tile değiştiren kararların gerçek iyileşme sağladığı ölçülür.

Özgün mekanizma oluşturmadan yalnızca araçları birleştirmenin de bitirme için faydalı yazılım uygulaması olabileceğini, ancak akademik yayınlarda *yeni yöntem* diye sunulamayacağını kabul et.

## 7. Sıralı teslim kapıları

| Kapı | Kanıt | Geçmeden sonraki işe geçilmez |
|---|---|---|
| G0 | ODM ve QGIS, bilgisayarda kurulmuş, giriş görüntüleri doğrulanmış; raster CRS kontrol edilmiş | Gerçek araçların çalıştığı kanıt |
| G1 | ODM bir tutarlı küçük bölgeden gerçek DTM, DSM, ortofoto ve NoData maskesi üretmiş | Photo->GIS baseline |
| G2 | QGIS terrain+ortho tek, tanınabilir, uygun ölçekli arazi olarak incelenmiş; Blender OBJ açılmış | İlk işe yarar 3D asset |
| G3 | OSM/Overture bina tabanları georeferansla hizalanmış; beyaz extruded binalar ve GLB/OBJ başarılı | İlk hibrit asset |
| G4 | Rural/Urban için büyük alanlar, tile mekanizması, Blender ve gerçek Unity import doğrulanmış | Ürün doğrulaması |
| G5 | Aynı kaynak ve süre/bellek bütçesi altında adaptive scheduler ile standard pipeline/öncelik baseline kıyası + ablations | Akademik hipotez testi |

**Başarısızlık durumunda öncelik:** G1 kötü DTM/ortofoto -> EXIF/PPK/GCP, görüntü örtüşmesi, alternatif ODM workflow; G2 kötü mesh -> QGIS DEM rasterını doğrudan yükselti yüzeyi, serbest Poisson mesh'e geri dönme; G3 coğrafi veri yok -> dış bina vektörü kapsaması veya etiketli eksik bina; G4 performans -> WebODM split/LOD; G5 katkı yok -> açıkça olumsuz deney sonucu ve daha dar hipotez.

## 8. Dış kaynak ve lisans notları

- ODM: AGPL-3.0; 3dfier: GPL-3.0; OSM/Overture veri kaynağına özgü açık veri lisansları ve attribution. Açık erişim = telifsiz/koşulsuz kopyala-yapıştır değil.
- Urban Export çevrimiçi hizmeti açık harita verisi kullanıyor olabilir ama sitenin kendisini açık kaynak proje saymak için doğrulanmış depo yok; yalnızca izin verilen yöntemle çıktı al veya kamuya açık veri kaynaklarını direkt edin.
- Bina konumu/yükseklik dış GIS verisinden geliyorsa çıktıdaki UAV katkısı ve dış veri katkısı açıkça raporlanır.
- Geometrisi dış DEM/LiDAR'a dayanan 3D asset de kullanışlı olabilir; ama **sadece drone RGB'den rekonstrüksiyon** hipotezinin başarısı diye raporlanamaz.
- Kullanıcının gerçek Windows klasörüne bu çalışmada erişim yoktur; hiçbir program çalıştırılmış gibi söylenmez.

## 9. Kaynakların doğrudan bağlantıları

- ODM kaynak ve resmi Docker örneği: https://github.com/OpenDroneMap/ODM
- ODM dem gap filling ve varsayılanlar: https://docs.opendronemap.org/arguments/
- ODM PPK image geo file: https://docs.opendronemap.org/geo/
- QGIS 3D terrain, raster texture ve OBJ: https://docs.qgis.org/testing/en/docs/user_manual/map_views/3d_map_view.html
- Qgis2threejs glTF: https://plugins.qgis.org/plugins/Qgis2threejs/
- QuickOSM: https://plugins.qgis.org/plugins/QuickOSM/
- 3dfier: https://github.com/tudelft3d/3dfier
- OSM ODbL: https://www.openstreetmap.org/copyright
- 2018 prioritized MVS: https://doi.org/10.1016/j.isprsjprs.2018.03.022
- 2021 image selection: https://doi.org/10.3390/rs13112118
- 2025 RQF and project code: https://kai-neumann.github.io/PointCloudMetrics/
- 2024 model quality: https://doi.org/10.1016/j.jag.2024.103667
- 2026 feedback SfM repo: https://github.com/IRIS-LAB-whu/OntheflySfMFeedback
- 2024 UseGeo evaluation: https://github.com/UseGeoEvaluation/DepthEstimationAnd3DReconstruction
- 2020 UAV versus LiDAR ground filtering: https://doi.org/10.1080/17538947.2020.1791267
- 2021 3dfier paper: https://doi.org/10.21105/joss.02866
- 147-study bibliographic preliminary register: companion `uav_literature_screening.md`
