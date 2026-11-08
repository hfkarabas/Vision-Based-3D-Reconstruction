# 9 Kasım Sunum Gezinme Kartı

**Hedef 08:50.** PDF = `deliverables/research_paper_interim_TR.pdf`. Kod yollarının kökü `src/vision3d/`. VS Code’da sekmeleri aşağıdaki sırayla açın. Satıra gitmek için Ctrl+G kullanın.

| Süre | Ekran ve konum | İşaret edilecek nokta |
|---|---|---|
| 00:00–01:19 | PDF s. 1, başlık, Özet, §1 par. 1–2, §2 tek paragraf | Problem, artımlı akış ve [1]–[4] dayanakları |
| 01:19–02:28 | `features/matching.py` · `ratio_filter` 8–11, `match` 32–58 | KNN, oran, ters yön ve koşullu doğrulama |
| 02:28–03:45 | `geometry/two_view.py` · `reconstruct` 83–106, 118–161, `triangulate` 57–80 | Essential Matrix → göreli poz → geçerli ilk noktalar |
| 03:45–05:01 | `reconstruction/incremental.py` · `pnp_correspondences` 169–179, `estimate_pnp` 182–211, `register` 216–231, `extend_landmarks` 244–294 | Mevcut noktalarla yeni poz; yeni izlerle haritanın büyümesi |
| 05:01–05:48 | `reconstruction/bundle.py` · `local_bundle_adjust` 29–58, `residual` 96–104, 113–147 | Piksel farkı, iki sabit kamera ve yerel çözüm |
| 05:48–06:53 | `reconstruction/sequence.py` · `match_sequence` 38–65, 78–86, `reconstruct_sequence` 89–112, 151–170, 185–253, 265–278 | Eşleşme ve kayıt akışı; ayrı sabit kameralı iyileştirme ve PLY |
| 06:53–08:13 | PDF s. 2, §4.1 ilk paragraf, Tablo 1, Şekil 1(a)–(b), §4.2 tek paragraf | Kamera kapsamı, RMS–medyan farkı ve gerçek bulut |
| 08:13–08:50 | PDF s. 2, §5 par. 1–2 | Kapsam, sınırlılıklar ve sonraki değerlendirme |

**Hazır konumlar** — VS Code’da sırasıyla 43, 99, 190, 96 ve 187. satırları açın. PDF’ye dönerken doğrudan ikinci sayfadaki Tablo 1’e geçin.
