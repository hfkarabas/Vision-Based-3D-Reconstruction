# 9 Kasım Bitirme Projesi Ara Sunumu

Ekran yönergelerini sesli okumayın. PDF’yi iki sayfası hazır olacak şekilde, VS Code’u da gezinme kartındaki sekme sırasıyla açın. Ondalık değerleri konuşma metninde yazıldığı gibi okuyun.

**Prova hesabı** — 1.057 sözcük (boşlukla ayrılan birimler, yalnızca konuşma). Dakikada 122–130 sözcükle yaklaşık 8:08–8:40 konuşma ve 45–55 saniye ekran geçişi, toplam yaklaşık 8:53–9:35. Aşağıdaki akış 9:05’i hedefler.

## 00:00–00:45 · Problem ve amaç
**Ekran** — `research_paper_interim_TR.pdf`, s. 1, başlık, Özet ve §1 Giriş, 1–2. paragraflar. Önce başlığı, sonra Giriş’in ilk paragrafını gösterin.

Hocam merhaba. Projemin adı “Çoklu Görüntülerden Artımlı Hareketten Yapı Çıkarımıyla Seyrek Üç Boyutlu Rekonstrüksiyon”. Amacım, aynı durağan sahnenin farklı açılardan çekilmiş fotoğraflarından kamera pozlarını ve renkli üç boyutlu noktaları çıkarmak. Sisteme kameraların çekim sırasındaki konumlarını vermiyorum. Deneylerde kamera iç parametrelerini kullanıyorum. Temel zorluk, iki görüntüde gördüğüm benzer ayrıntıların gerçekten aynı fiziksel noktaya ait olup olmadığını anlamak. Özellikle tekrarlayan cephe desenleri yanlış eşleşme üretebiliyor. Bakış açıları birbirine çok yakınsa doğru eşleşme bulsam bile derinliği güvenilir biçimde kestirmek zorlaşıyor.

## 00:45–01:35 · Artımlı yaklaşım ve literatür
**Ekran** — Aynı PDF, s. 1, §2 İlgili Çalışmalar, tek paragraf (sol alttan sağ sütuna devam eder), atıflar [1]–[4]. İlk iki cümleden sağ sütundaki devamına geçin.

Hareketten yapı çıkarımı, yani SfM, bu iki problemi birbirine bağlar. Eşleşmelerden kamera pozlarını, pozlar bilindiğinde de yeni sahne noktalarını kestiriyorum. Artımlı yaklaşımda önce geometrisi yeterince güçlü iki görüntüyle başlıyorum. Diğer görüntüler, kurulan haritaya destekleri oldukça ekleniyor. Makalede Lowe’u özellik betimleme ve oran testi, Hartley ile Zisserman’ı epipolar geometri ve üçgenleme için kaynak gösterdim. Schönberger ve Frahm artımlı akışa, Triggs ve arkadaşları ise demet dengelemeye dayanak sağlıyor. Benim bu aşamadaki çalışmam, bu yöntemleri denetlenebilir bir işlem hattında birleştirmek ve sabit kameralı bir son iyileştirmenin etkisini ölçmek. Şimdi ilk adımı kodda göstereyim.

## 01:35–02:20 · Betimleyiciden eşleşmeye
**Ekran** — VS Code, `src/vision3d/features/matching.py`, `ratio_filter()` 8–11 ve `match()` 32–51. Önce 43–46’daki komşu aramasını, sonra 48–51’deki ters yön denetimini gösterin.

Özellik çıkarımı ayrı bir modülde, features paketinin extract fonksiyonunda yapılıyor. Buradaki match fonksiyonu hazır betimleyicileri alıyor. Deneylerde SIFT kullandım ve kod bu betimleyicileri Öklid uzaklığıyla karşılaştırıyor. Her özellik için en yakın iki adayı buluyorum. Birinci aday ikinciye göre yeterince yakın değilse eşleşmeyi eliyorum. Örneğin bir cephede birbirine çok benzeyen iki pencere varsa, ikisinden biri az farkla öne çıkmış olabilir. Oran testi böyle belirsiz adayları azaltıyor. Dizi akışında ters yönde de aynı testi yapıyorum ve yalnızca iki yönde tutarlı bulunan eşleşmeleri koruyorum.

## 02:20–02:45 · Geometrik doğrulamanın yeri
**Ekran** — Aynı dosya, `match()` 52–58. Koşullu doğrulamayı gösterin. Diziye özgü çağrı daha sonra `sequence.py`, `match_sequence()` 42–61’de gösterilecek.

Bu dosyada geometrik doğrulama seçeneği var. Fakat dizi çalıştırmasında bu seçeneği kapatıyorum ve doğrulamayı bir üst aşamada yapıyorum. Orada kameraların iç parametreleriyle bozulmayı giderip noktaları ortak sanal piksel ölçeğine taşıyorum. Ardından temel matrisi RANSAC ile kestiren verify yardımcısı devreye giriyor. Böylece görünüşü benzeyen adaylardan, iki görüntünün geometrisiyle tutarlı olanları seçiyorum. İlk kamera çiftini kurmak için şimdi kalibre edilmiş geometriye geçiyorum.

## 02:45–03:25 · Esas matris ve göreli poz
**Ekran** — `src/vision3d/geometry/two_view.py`, `reconstruct()` 83–106 ve 118–146. `findEssentialMat`, ardından `recoverPose` ve öteleme normalizasyonunu işaret edin.

Burada girdi, doğrulanmış piksel çiftleri. Her kameranın kendi iç parametreleriyle noktaları normalize ediyorum. Esas matris için RANSAC eşiğini de bu koordinatlara uygun ölçeğe çeviriyorum. Homografinin veya salt dönmenin baskın olduğu adayları başlangıç için reddediyorum. Çünkü böyle bir çift, sahne derinliğini ayırmak için yeterli bilgi vermeyebilir. Recover pose adımı ikinci kameranın göreli dönüşünü ve öteleme yönünü çıkarıyor. İlk kamera referans oluyor ve öteleme uzunluğunu birim alıyorum. Bu nedenle elde ettiğim koordinatları doğrudan metre cinsinden yorumlayamıyorum.

## 03:25–03:55 · İlk üç boyutlu noktalar
**Ekran** — Aynı dosya, `triangulate()` 57–80 ve çağrısı `reconstruct()` 147–161. Üçgenleme çağrısından `valid` maskesine ilerleyin.

İki kameranın göreli pozu belli olunca aynı noktaya karşılık gelen iki gözlemden bir üç boyutlu aday hesaplıyorum. Buradaki valid maskesi hangi adayları kullanabileceğimi belirliyor. Nokta her iki kameranın önünde olmalı, iki görüntüye geri yansıtıldığında ölçülen piksellere yakın düşmeli ve bakış ışınları arasındaki açı yeterli olmalı. Bu kontrollerden çok az nokta geçerse bütün başlangıç çifti reddediliyor. Sonraki adımda bu geçerli noktaları yeni kameraları kaydetmek için kullanıyorum.

## 03:55–04:40 · Yeni kameranın kaydı
**Ekran** — `src/vision3d/reconstruction/incremental.py`, `pnp_correspondences()` 169–179, `estimate_pnp()` 182–211. İzden piksel seçimini, PnP ve LM çağrılarını gösterin.

Artık elimde haritada konumu bilinen noktalar var. Yeni görüntüde aynı izlere ait özellikleri bulup bir tarafta üç boyutlu koordinatı, diğer tarafta ölçülen pikseli eşliyorum. Burada iki boyutlu eşleşmelerden yeni bir esas matris çıkarmıyorum. Bilinen üç boyutlu noktalara göre yeni kameranın pozunu çözüyorum. Kod, OpenCV’nin RANSAC tabanlı PnP çağrısını kullanıyor. Sonra ilk çözümün iç noktalarıyla Levenberg–Marquardt iyileştirmesi yapıyor. Pozu kabul etmeden önce pozitif derinliği ve piksel hatasını yeniden ölçüyorum. İç noktaların sayısı, oranı ve görüntüde kapladıkları alan da yeterli olmalı.

## 04:40–05:10 · Haritanın büyümesi
**Ekran** — Aynı dosya, `register()` 216–231 ve `extend_landmarks()` 244–294. Önce pozun kaydını, sonra 265–291’deki çift sıralaması ve çoklu görünüm kontrolünü gösterin.

Kamera kabul edilirse haritaya ekleniyor. Mevcut noktalara tutarlı yeni gözlemler bağlanıyor. Henüz noktası olmayan izlerde ise kayıtlı görüntü çiftlerini ışın açısına göre sıralıyorum. Kod en geniş açıdan başlıyor ve gerektiğinde ilk üç çifti deniyor. Üçgenlenen noktayı ortak koordinat sistemine taşıyıp diğer kayıtlı gözlemlerde de denetliyorum. Böylece aynı iz için birden fazla nokta oluşturmadan haritayı büyütüyorum.

## 05:10–06:10 · Yerel demet dengeleme
**Ekran** — `src/vision3d/reconstruction/bundle.py`, `local_bundle_adjust()` 29–58, iç `residual()` 96–104, çözüm ve kabul 113–147. Önce sabit kameraları, sonra piksel farkı ve kabul koşullarını gösterin. Denklem karşılığı PDF s. 1, §3.3, Denklem (1) ve “Burada” paragrafı.

Demet dengelemede temel ölçüm şu. Bir üç boyutlu noktayı tahmin edilen kameradan görüntüye yansıtıyorum ve ölçülen pikselden farkını hesaplıyorum. Makaledeki bir numaralı denklem bu iki bileşenli farkın uzunluğunu gösteriyor. Çözücüye ise yatay ve düşey farkları ayrı ayrı veriyorum. Yerel çözümde son eklenen kameralardan bir bölümünün pozlarını ve seçili noktaları birlikte değiştiriyorum. İç parametreleri ve iki sınır kamerasını sabit tutuyorum. Bu iki kamera mevcut referans çerçevesini ve ölçeği koruyor. Çözüm, seyrek bir türev yapısı ve büyük hataların etkisini sınırlayan soft L bir kaybı kullanıyor. Sonucu önce bir aday durumda değerlendiriyorum. Harita hatası artarsa, yeni geçersiz gözlemler oluşursa veya kamera hareketi sınırı aşarsa önceki durumu koruyorum. Bu işlem bütün haritayı serbest bırakan küresel bir optimizasyon değil.

## 06:10–06:45 · Dizi akışı
**Ekran** — `src/vision3d/reconstruction/sequence.py`, `reconstruct_sequence()` 89–112, 151–170, 185–231 ve `match_sequence()` 38–65, 78–86. Önce giriş ve ön işleme, sonra 187–195 ve 208–230’daki çağrılar.

Bu parçaların bir araya geldiği giriş noktası reconstruct sequence. Görüntüyü hazırlıyor, özellikleri çıkarıyor ve yeniden boyutlandırmaya göre kamera iç parametrelerini ölçekliyor. Match sequence, seçili çiftleri eşleştirip geometrik olarak doğruluyor ve eşleşmelerden gözlem izlerini kurduruyor. Ardından başlangıç çifti seçiliyor. Döngüde iki boyutlu ve üç boyutlu desteği daha güçlü görüntülere öncelik veriliyor. Bir kamera eklenince yeni noktalar oluşturuluyor. Yerel demet dengeleme açıksa bu adımın ardından çağrılıyor.

## 06:45–07:15 · Deneyde değiştirilen adım
**Ekran** — Aynı dosya, `reconstruct_sequence()` 237–253 ve 265–278. `refine_landmarks` çağrısını, önce-sonra ölçümünü ve `sparse.ply` çıktısını gösterin. Yardımcı uygulama `src/vision3d/reconstruction/refinement.py`, `refine_landmarks()` 14–75.

Tablodaki önce ve sonra ayrımı, az önce gösterdiğim yerel demet dengelemenin açılıp kapatılması değil. Burada ayrı bir son adım var. Kameralar sabitken en az üç görünümde gözlenen noktaların koordinatları tek tek iyileştiriliyor. İyileştirme hata ve geometri sınırlarını geçerse noktayı güncelliyorum. Bu koşullar sağlanmazsa önceki koordinatı kullanıyorum. Kod önce ve sonra harita ölçümlerini kaydediyor. Sonunda renkli seyrek bulutu PLY dosyasına yazıyor. Şimdi bu çıktının makaledeki karşılığına döneyim.

## 07:15–08:05 · Ölçülen sonuçlar
**Ekran** — `research_paper_interim_TR.pdf`, s. 2, §4.1 Deney düzeni ve bulgular, ilk paragraf ve Tablo 1. Önce kamera ve nokta satırlarını, sonra RMS, medyan ve süre satırlarını gösterin.

Castle’da altı kameranın altısını kaydettim ve üç bin dokuz yüz on beş renkli nokta elde ettim. Temple dizisinde otuz kameranın on sekizi kaydedildi ve iki bin altı yüz altmış dört nokta elde edildi. Aynı kabul edilmiş gözlemler üzerinden karşılaştırdığımda RMS hata Castle’da sıfır virgül beş yüz seksen yediden sıfır virgül dört yüz yetmiş yedi piksele düştü. Temple’da sıfır virgül altı yüz elli dörtten sıfır virgül dört yüz kırk sekize indi. Medyan ise iki dizide de yükseldi. Dolayısıyla tablo, her gözlemin iyileştiğini göstermiyor. Son adım ek süre de getiriyor. Örneğin Castle çalıştırmasında süre on altı virgül yetmiş dört saniyeden yirmi virgül yetmiş beş saniyeye çıktı.

## 08:05–08:35 · Görsel ve yorum
**Ekran** — Aynı PDF, s. 2, §4.1 Şekil 1(a)–(b) ve §4.2 Bulguların yorumu, tek paragraf (sol alttan sağ üste devam eder). Fotoğraftaki cepheyi ve buluttaki boşlukları karşılaştırın.

Solda Castle’ın giriş fotoğraflarından biri var. Sağdaki bulut ise altı görüntüden üretilen gerçek çıktı. Cephenin bazı ayrıntıları seçiliyor, ama aralarında boşluklar var. Son iyileştirme kameraları ve nokta sayısını değiştirmedi. Temple’da referansa göre ölçtüğüm kamera dönüşü ve öteleme yönü hataları da aynı kaldı. RMS’deki düşüşü, sabit kameralar altında görüntülerle daha iyi toplu uyum olarak yorumluyorum. Bu ölçüm tek başına gerçek yüzeye olan üç boyutlu uzaklığı vermiyor.

## 08:35–09:05 · Sınırlılıklar ve kapanış
**Ekran** — Aynı PDF, s. 2, §5 Sınırlılıklar ve Planlanan Değerlendirme, 1–2. paragraflar. Temple kayıt kapsamını ve son değerlendirme hedefini gösterin.

Şu anda kalibre edilmiş görüntü dizisinden kamera pozlarını ve dışa aktarılabilir seyrek bir nokta bulutunu üretebiliyorum. Temple’da on iki görünümün kaydedilememesi, örtüşme ve güvenilir eşleşme ihtiyacının sürdüğünü gösteriyor. Serbest çekimlerle uçtan uca doğrulama, yoğun rekonstrüksiyon, ağ üretimi ve doku eşleme henüz tamamlanmadı. Sonraki değerlendirmede kaç kameranın kaydedildiğini, görüntü düzlemindeki uyumu ve uygun referans varsa üç boyutlu geometriyle uyumu ayrı ayrı inceleyeceğim. Teşekkür ederim.

## Kaynak kontrolü · Sesli okunmayacak

Birincil kaynak `deliverables/research_paper_interim_TR.docx` ve aynı metnin iki sayfalık PDF’si. PDF s. 1’de başlık, Özet, Anahtar Kelimeler, §1’in iki paragrafı, §2’nin tek paragrafı, §3.1 ve §3.2’nin tek paragrafları ve §3.3’te Denklem (1) bulunur. §3.3’ün son paragrafı s. 2’ye taşar. S. 2’de §4.1’in deney paragrafı, Tablo 1, Şekil 1 ve açıklaması, §4.2’nin iki sütuna yayılan tek paragrafı, §5’in iki paragrafı ve Kaynaklar [1]–[6] bulunur. [5] Castle fotoğraflarının, [6] TempleRing dizisinin kaynağıdır.

Yardımcı çağrılar — Özellik çıkarımı `src/vision3d/features/__init__.py`, `extract()`, 23. satırdan başlar. Temel matris doğrulaması `src/vision3d/geometry/__init__.py`, `verify()`, 6. satırdan başlar. Sabit kameralı nokta iyileştirmesi `src/vision3d/reconstruction/refinement.py`, `refine_landmarks()`, 14. satırdan başlar. Tablo 1’in sayıları `outputs/reconstruction-quality/comparison.csv` içindeki dört Castle ve Temple karşılaştırma satırıyla, makaledeki yuvarlama basamaklarında eşleşir. Makaleyle kod arasında sunumu etkileyen bir çelişki bulunmadı.
