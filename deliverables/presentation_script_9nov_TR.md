# 9 Kasım Bitirme Projesi Ara Sunumu

Ekran yönergelerini sesli okumayın. PDF’yi iki sayfası hazır olacak şekilde, VS Code’u da gezinme kartındaki sekme sırasıyla açın. Kesin değerler Tablo 1’de. Köşeli parantez içindeki yönergeler prova içindir.

**Prova hesabı** — 710 sözcük. Dakikada 95–110 sözcükle konuşma yaklaşık 6:27–7:28. Duraklama ve ekrana bakma payı 40 saniye, pencere ve satır geçişleri 65 saniye. Toplam yaklaşık 8:12–9:13. Zaman çizelgesi 08:50’i hedefler. Sesli prova yapılmadı.

## 00:00–00:36 · Problem ve amaç
**Ekran** — `research_paper_interim_TR.pdf`, s. 1, başlık, Özet ve §1 Giriş, 1–2. paragraflar. Önce başlığı, sonra Giriş’in ilk paragrafını gösterin. [Sorudan sonra kısa duraklayın.]

Hocam merhaba. Bir binanın farklı açılardan çekilmiş fotoğraflarını düşünelim. Bu fotoğraflardan hem kameraların nerede durduğunu hem de binanın üç boyutlu yapısını çıkarabilir miyiz? Projemde bu soruya bakıyorum. Deneylerde kamera kalibrasyonu elimde. Kameraların çekim sırasındaki pozlarını ise fotoğraflardan hesaplıyorum. İlk zorluk şu. İki fotoğrafta birbirine benzeyen pencereler, binanın farklı yerlerinde olabilir.

## 00:36–01:19 · Artımlı yaklaşım ve literatür
**Ekran** — `research_paper_interim_TR.pdf`, s. 1, §2 İlgili Çalışmalar, tek paragraf (sol alttan sağ sütuna devam eder), atıflar [1]–[4]. İlk iki cümleden sağ sütundaki devamına geçin.

Structure-from-Motion, yani SfM, kamera pozlarını ve sahnedeki noktaları birlikte bulmamı sağlıyor. Önce iki fotoğrafla küçük bir harita kuruyorum. Sonra bu haritaya yeni kameralar ve noktalar ekliyorum. Incremental SfM dediğim akış bu. Makaledeki yöntemler literatürde yerleşik. SIFT için Lowe’a, görüntü geometrisi için Hartley ve Zisserman’a dayanıyorum. Artımlı akış için Schönberger ve Frahm’ı, Bundle Adjustment için Triggs ve arkadaşlarını kaynak gösterdim.

## 01:19–02:01 · Betimleyiciden eşleşmeye
**Ekran** — VS Code, `src/vision3d/features/matching.py`, `ratio_filter()` 8–11 ve `match()` 32–51. Önce 43–46’daki komşu aramasını, sonra 48–51’deki ters yön denetimini gösterin.

İlk iş, aynı ayrıntıyı iki fotoğrafta bulmak. SIFT, Feature Extraction aşamasında her ayrıntının çevresini özetleyen bir descriptor üretiyor. Buradaki match fonksiyonu bu descriptor’ları karşılaştırıyor. Her nokta için en yakın iki adaya bakıyorum. İki aday birbirine çok yakın puan alıyorsa seçim belirsiz demektir. Bu eşleşmeyi eliyorum. Eşleşmenin ters yönde de geçerli olmasını istiyorum.

## 02:01–02:28 · Geometrik doğrulamanın yeri
**Ekran** — `src/vision3d/features/matching.py`, `match()` 52–58. Koşullu doğrulamayı gösterin. Diziye özgü çağrı daha sonra `src/vision3d/reconstruction/sequence.py`, `match_sequence()` 42–61’de gösterilecek.

Görünüşün benzemesi tek başına yeterli değil. RANSAC ile eşleşmelerin iki görüntü arasındaki geometriye uyup uymadığına bakıyorum. Bu dosyada doğrulama bir seçenek. Dizi akışında ise sequence dosyasında, kalibrasyona göre düzeltilmiş noktalarla yapıyorum. Geometriden geçen çiftler bir sonraki aşamaya gidiyor.

## 02:28–03:11 · Essential Matrix ve göreli poz
**Ekran** — `src/vision3d/geometry/two_view.py`, `reconstruct()` 83–106 ve 118–146. `findEssentialMat`, ardından `recoverPose` ve öteleme normalizasyonunu işaret edin. [Recover pose satırında iki saniye bekleyin.]

Şimdi iki kameranın birbirine göre nasıl durduğunu bulmam gerekiyor. Ortak noktaları Camera Intrinsics ile normalize ediyorum. Ardından OpenCV, bu eşleşmelerden Essential Matrix’i hesaplıyor. Recover pose çağrısı bana ikinci kameranın dönüşünü ve öteleme yönünü veriyor. İlk kamerayı referans alıyorum. İki kamera arasındaki uzaklığı bir birim kabul ediyorum. Bu yüzden ölçüler metre cinsinden değil.

## 03:11–03:45 · İlk üç boyutlu noktalar
**Ekran** — `src/vision3d/geometry/two_view.py`, `triangulate()` 57–80 ve çağrısı `reconstruct()` 147–161. Üçgenleme çağrısından `valid` maskesine ilerleyin. [İki bakış ışınını tarif ettikten sonra kısa duraklayın.]

İki kameranın pozu belli. Aynı pencere köşesine iki kameradan birer bakış ışını çizdiğimizi düşünelim. Triangulation, bu iki gözlemden köşenin üç boyutlu yerini tahmin ediyor. Nokta iki kameranın da önünde olmalı. Işınlar birbirine fazla yakınsa derinlik tahmini zayıflıyor. Kod böyle adayları eliyor. Kalan noktalar ilk haritayı oluşturuyor.

## 03:45–04:30 · Yeni kameranın kaydı
**Ekran** — `src/vision3d/reconstruction/incremental.py`, `pnp_correspondences()` 169–179, `estimate_pnp()` 182–211. İzden piksel seçimini, PnP ve LM çağrılarını gösterin.

Sırada yeni bir fotoğraf var. Bu fotoğraftaki bazı ayrıntıların haritada üç boyutlu karşılığı zaten var. Bir tarafta haritadaki nokta, diğer tarafta fotoğraftaki piksel duruyor. PnP bu eşleşmelerden yeni Camera Pose’u hesaplıyor. RANSAC tutarsız eşleşmelerin etkisini azaltıyor. Levenberg–Marquardt ile pozu bir kez daha düzeltiyorum. Yeterli sayıda güvenilir eşleşme varsa ve bunlar görüntüye yayılıyorsa kamera kayda aday oluyor.

## 04:30–05:01 · Haritanın büyümesi
**Ekran** — `src/vision3d/reconstruction/incremental.py`, `register()` 216–231 ve `extend_landmarks()` 244–294. Önce pozun kaydını, sonra 265–291’deki çift sıralaması ve çoklu görünüm kontrolünü gösterin.

Yeni kamerayı kaydedince daha önce yerini bulamadığım bazı ayrıntılara ikinci bir açıdan bakabiliyorum. Kod, yeterli açı farkı olan görüntü çiftlerinden yeni noktalar hesaplıyor. Sonra bu noktaları diğer kayıtlı görüntülerde de kontrol ediyor. Böylece yeni kamera hem mevcut noktalara gözlem ekliyor hem de haritanın büyümesini sağlıyor.

## 05:01–05:48 · Yerel Bundle Adjustment
**Ekran** — `src/vision3d/reconstruction/bundle.py`, `local_bundle_adjust()` 29–58, iç `residual()` 96–104, çözüm ve kabul 113–147. Önce sabit kameraları, sonra piksel farkı ve kabul koşullarını gösterin. Denklem karşılığı PDF s. 1, §3.3, Denklem (1) ve “Burada” paragrafı. [Piksel farkını gösterip iki saniye bekleyin.]

Haritadaki bir noktayı fotoğrafa geri yansıttığımda, tahmin ettiğim piksel ile ölçtüğüm piksel tam üst üste gelmeyebilir. Aradaki uzaklığa Reprojection Error diyorum. Bundle Adjustment ile seçili kamera pozlarını ve noktaları birlikte ayarlıyorum. Amaç, bu farkları küçültmek. Buradaki çözüm yerel. Haritanın ölçeğini ve referansını korumak için iki sınır kamerasını sabit tutuyorum. Yeni çözüm daha fazla geçersiz gözlem oluşturursa önceki haritayı koruyorum.

## 05:48–06:23 · Dizi akışı
**Ekran** — `src/vision3d/reconstruction/sequence.py`, `reconstruct_sequence()` 89–112, 151–170, 185–231 ve `match_sequence()` 38–65, 78–86. Önce giriş ve ön işleme, sonra 187–195 ve 208–230’daki çağrılar.

Bu dosya bütün pipeline’ı yönetiyor. Önce fotoğrafları hazırlıyor. Sonra seçili görüntü çiftlerini eşleştiriyor ve geometrik doğrulamadan geçiriyor. Aynı ayrıntıya ait gözlemleri bir izde topluyor. Ardından başlangıç çiftini seçiyor. Her yeni kamera kaydından sonra haritayı genişletiyor. Yerel Bundle Adjustment açıksa onu da bu döngüde çağırıyor.

## 06:23–06:53 · Deneyde değiştirilen adım
**Ekran** — `src/vision3d/reconstruction/sequence.py`, `reconstruct_sequence()` 237–253 ve 265–278. `refine_landmarks` çağrısını, önce-sonra ölçümünü ve `sparse.ply` çıktısını gösterin. Yardımcı uygulama `src/vision3d/reconstruction/refinement.py`, `refine_landmarks()` 14–75.

Deneyde bir de ayrı son adım var. Burada kameraları sabit tutuyorum. En az üç fotoğrafta görülen noktaların yerini tek tek iyileştiriyorum. Tablo bu adımın öncesini ve sonrasını karşılaştırıyor. Pipeline sonunda renkli Sparse Point Cloud’u bir PLY dosyasına yazıyor. Şimdi makaledeki tabloya dönelim.

## 06:53–07:39 · Ölçülen sonuçlar
**Ekran** — `research_paper_interim_TR.pdf`, s. 2, §4.1 Deney düzeni ve bulgular, ilk paragraf ve Tablo 1. Önce kamera ve nokta satırlarını, sonra RMS, medyan ve süre satırlarını gösterin. [RMS ve medyan satırlarını gösterirken kısa duraklayın.]

Castle’da altı kameranın tamamı kaydoldu. Yaklaşık üç bin dokuz yüz nokta elde ettim. Temple’da otuz kameranın on sekizi kaydoldu. Orada yaklaşık iki bin yedi yüz nokta var. Son iyileştirmeyle RMS hata Castle’da yaklaşık yüzde on dokuz, Temple’da yaklaşık yüzde otuz iki düştü. Ama medyan hata iki dizide de arttı. İşlem de daha uzun sürdü. Örneğin Castle’da yaklaşık dört saniye eklendi.

## 07:39–08:13 · Görsel ve yorum
**Ekran** — `research_paper_interim_TR.pdf`, s. 2, §4.1 Şekil 1(a)–(b) ve §4.2 Bulguların yorumu, tek paragraf (sol alttan sağ üste devam eder). Fotoğraftaki cepheyi ve buluttaki boşlukları karşılaştırın. [Şekle bakılması için iki saniye bekleyin.]

Solda Castle fotoğrafı, sağda altı görüntüden ürettiğim bulut var. Cephenin bazı hatları seçiliyor. Aralarda boşluklar da görünüyor. Son adımda kamera pozları ve nokta sayıları aynı kaldı. Düşen RMS, sabit kameralarla görüntüde daha iyi bir toplu uyum gösteriyor. Noktaların gerçek yüzeye daha yakın olduğunu söylemek için ayrı bir ölçüm gerekiyor.

## 08:13–08:50 · Sınırlılıklar ve kapanış
**Ekran** — `research_paper_interim_TR.pdf`, s. 2, §5 Sınırlılıklar ve Planlanan Değerlendirme, 1–2. paragraflar. Temple kayıt kapsamını ve son değerlendirme hedefini gösterin.

Bugünkü çıktı, kalibre edilmiş bir diziden üretilen renkli bir Sparse Point Cloud. Temple’daki eksik kamera kayıtları hâlâ çözülmesi gereken bir konu. Serbest çekimlerle sistemi henüz uçtan uca doğrulamadım. Yoğun rekonstrüksiyon, mesh ve doku eşleme de sonraki aşamalar. Final değerlendirmesinde kamera kayıt oranını ve piksel hatasını ayrı vereceğim. Uygun referans varsa üç boyutlu geometriyi de ayrıca karşılaştıracağım. Teşekkür ederim.

## Kaynak kontrolü · Sesli okunmayacak

Birincil kaynak `deliverables/research_paper_interim_TR.docx` ve aynı metnin iki sayfalık PDF’si. PDF s. 1’de başlık, Özet, Anahtar Kelimeler, §1’in iki paragrafı, §2’nin tek paragrafı, §3.1 ve §3.2’nin tek paragrafları ve §3.3’te Denklem (1) bulunur. §3.3’ün son paragrafı s. 2’ye taşar. S. 2’de §4.1’in deney paragrafı, Tablo 1, Şekil 1 ve açıklaması, §4.2’nin iki sütuna yayılan tek paragrafı, §5’in iki paragrafı ve Kaynaklar [1]–[6] bulunur. [5] Castle fotoğraflarının, [6] TempleRing dizisinin kaynağıdır.

Yardımcı çağrılar — Özellik çıkarımı `src/vision3d/features/__init__.py`, `extract()`, 23. satırdan başlar. Temel matris doğrulaması `src/vision3d/geometry/__init__.py`, `verify()`, 6. satırdan başlar. Sabit kameralı nokta iyileştirmesi `src/vision3d/reconstruction/refinement.py`, `refine_landmarks()`, 14. satırdan başlar. Tablo 1’in sayıları `outputs/reconstruction-quality/comparison.csv` içindeki dört Castle ve Temple karşılaştırma satırıyla, makaledeki yuvarlama basamaklarında eşleşir. Makaleyle kod arasında sunumu etkileyen bir çelişki bulunmadı.
