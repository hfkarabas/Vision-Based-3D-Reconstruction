# Interim paper format and reference audit

## A. Formatting comparison

The two-page source was `C:/Users/furka/OneDrive/Desktop/document kopyası.pdf`. Both pages were rendered and inspected beside both revisions at the same scale. The sample supplied layout evidence only. No sample research text or citations were transferred.

Measurements are in PDF points from the upper-left corner. **Measured** means a PDF page box, glyph extent, reported font size or baseline difference. **Inferred** means a layout setting reconstructed from repeated text positions. Glyph extents do not establish the exact original TeX margin, skip or font-size command. Revision font names and paragraph settings were checked in the editable Word files, with PDF positions checked separately.

| Property | Reference PDF | Turkish paper | English paper | Status |
|---|---|---|---|---|
| Page box | 595.28 × 841.89 pt, measured | 595.32 × 841.92 pt | 595.32 × 841.92 pt | A4. Word export rounding under 0.05 pt |
| Side margins | 48.19 pt inferred from recurring left edge and right endpoint near 547.09 | 48.19 pt configured, text begins at x≈48.24 | Same | Matched to Word rounding |
| Vertical margins | Page 2 first glyph top ≈51 pt, lower body glyphs end near y=787 pt on both pages. Frame approximately y=51–789, inferred | 45 pt top, 53 pt bottom configured. Glyph positions differ from frame | Same | Near match. Source margin commands unavailable |
| Columns | Two. Starts x=48.19 and 306.85. Nominal width ≈240.24 and gap ≈18.42, inferred. Occasional punctuation overhangs | Two, starts x≈48.24 and 306.89, width ≈240.3, gap 18.4 | Same | Near match |
| Title | LMRoman12-Bold, 14.35 pt measured. Centered across both columns, first glyph top y≈43.30, line interval ≈17.94 | LM Roman 12 bold, 14.5 pt configured, first PDF text extent top ≈48.5, 18 pt interval | Same | Same face and alignment. Title sits about 5 pt lower |
| Author and affiliation | Centered. LMRoman10-Regular 9.96 pt and LMRoman9-Regular 8.97 pt. Author, affiliation and supervisor top ≈93.66, 108.41, 122.36 | LM Roman 10 at 10 pt and LM Roman 9 at 9 pt. Tops ≈97.1, 112.7, 127.7 | Same | Same hierarchy, up to about 5 pt lower |
| Abstract and keywords | Standalone left-aligned 14.35 pt bold heading at y≈159.34. Body 9.96 pt. Inline 9.96 pt bold keyword label | 14.5 pt bold heading at y≈157.8, body and keyword label 10 pt | Same | Near match |
| Main headings | LMRoman12-Bold 14.35 pt, left aligned, Arabic numbering and text offset ≈24 pt | LM Roman 12 bold 14.5 pt, left aligned, 24 pt hanging heading offset | Same | Same hierarchy |
| Subheadings | LMRoman12-Bold 11.96 pt, left aligned with decimal numbering | LM Roman 12 bold 12 pt, 30 pt tab from column edge | Same | Near match. Wrapping varies with wording |
| Body and paragraphs | LMRoman10-Regular 9.96 pt, justified. Repeated baseline interval ≈11.95 pt. Continuous paragraphs usually indent ≈9.96 pt, with no measurable extra vertical gap | LM Roman 10 at 10 pt, justified, 11.95 pt exact interval. Continuations use 9.96 pt first-line indent | Same | Near match. Word and TeX line breaking differ |
| Equation | Centered expression with number near column right. Latin Modern math faces, main glyphs around 9.96 pt | Editable Word equation, centered with right-side number, Cambria Math ≈10 pt | Same | Alignment reproduced. Math face differs |
| Captions and tables | Centered table caption above table, regular Latin Modern Roman. Horizontal rules and compact table body. No figure caption to measure | Centered 10 pt table caption, 9 pt table cells, horizontal rules. 10 pt caption below two-panel figure | Same | Table convention reproduced. Figure layout is an adaptation |
| Bibliography | Left-aligned 14.35 pt bold heading in right column on page 2 at y≈118.23. 9.96 pt regular and italic text, ≈11.95 pt leading, bracket labels with ≈15.24 pt hanging indent and visible inter-entry gaps | Right column page 2, heading top ≈360, 14.5 pt bold. 10 pt regular and italic, 11.95 pt leading, 15.24 pt hanging indent, 6 pt after entries | Same styling, heading top ≈371 | Typography and alignment reproduced. Vertical position follows this manuscript's text |
| Page numbers | Centered, LMRoman10-Regular 9.96 pt, glyph top y≈807.06 on both pages | 10 pt, centered, top ≈807.69 on both pages | Same | Near match |

## B. Bibliography verification

| Ref. | Exists? | Metadata correct? | Citation relevant? | Official verification source | Action |
|---|---|---|---|---|---|
| [1] | Yes | David G. Lowe, exact title, IJCV 60(2), 91–110, 2004 | Yes. SIFT and the nearest-neighbor ratio test | [Springer article](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94) | Standardized title and journal name, added DOI |
| [2] | Yes | Richard Hartley and Andrew Zisserman, second edition, Cambridge University Press, print 2004. Paperback ISBN 9780521540513 | Yes. Camera models, epipolar geometry and triangulation | [Cambridge book record](https://doi.org/10.1017/CBO9780511811685), [authors' edition page](https://robots.ox.ac.uk/~vgg/hzbook/) | Retained 2004 print year and added DOI. Cambridge lists a later online publication date |
| [3] | Yes | Johannes L. Schönberger and Jan-Michael Frahm, CVPR 2016, 4104–4113 | Yes. Incremental initialization, registration and triangulation | [CVF proceedings](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html), [IEEE DOI registration](https://api.crossref.org/works/10.1109/CVPR.2016.445) | Standardized title and added DOI |
| [4] | Yes | Bill Triggs, Philip F. McLauchlan, Richard I. Hartley, Andrew W. Fitzgibbon, LNCS 1883, 298–372. Springer cites the chapter as 2000 | Yes. Robust optimization, sparsity and gauge freedom in bundle adjustment | [Springer chapter and recommended citation](https://link.springer.com/chapter/10.1007/3-540-44480-7_21) | Added DOI and retained the publisher's citation year. The workshop took place in 1999 and online records list 2002 |
| [5] | Yes, as a dataset repository | Repository credits Pierre Moulon as photographer and gives a 2012 photograph copyright. Formal publication year and preferred scholarly citation were not established | Yes. Castle input sequence and photo credit | [Official openMVG repository](https://github.com/openMVG/ImageDataset_SceauxCastle) | Used the exact repository name and a clickable canonical URL in both DOCX files. Identified Moulon as photographer. Removed 2012 as a publication year and retained it in the copyright caption |
| [6] | Yes | Steven M. Seitz, Brian Curless, James Diebel, Daniel Scharstein, Richard Szeliski, CVPR 2006, vol. 1, 519–528 | Yes, for TempleRing provenance. The text cites it at the dataset identification and makes no claim of performing dense stereo evaluation | [Original benchmark paper](https://vision.middlebury.edu/mview/seitz_mview_cvpr06.pdf), [TempleRing dataset page](https://vision.middlebury.edu/mview/data/), [IEEE DOI registration](https://api.crossref.org/works/10.1109/CVPR.2006.19) | Standardized title and added DOI. Checked volume and page span in the DOI registration because the accessible IEEE Xplore view was incomplete |

The two manuscripts retain the same [1]–[6] numbering. The papers' own registration, landmark and reprojection measurements remain project results. The cited sources supply method context and dataset provenance.

## C. Final validation

- Microsoft Word repagination and independent PDF inspection each confirmed **two A4 pages per language**. The four final pages were checked visually for overflow, column order, Turkish glyphs, figure and table placement, equation alignment and bibliography indentation.
- The text uses the original Latin Modern Roman family from the [CTAN Latin Modern distribution](https://ctan.org/pkg/lm). The equation remains editable in Word and uses Cambria Math, the closest available layout within the existing editable equation. Word did not embed the Latin Modern OpenType faces in the DOCX package. Install LM Roman 9, 10 and 12 when editing on another machine to preserve pagination.
- Word's PDF/A export preserves the Latin Modern appearance in independent viewers by rasterizing affected text and supplying a searchable text layer. PDF font inspection may therefore report the hidden layer's fonts. The visible glyphs were inspected separately. The editable DOCX files retain actual text, table and equation objects.
- The main visible differences are TeX versus Word line breaking and justification, small vertical offsets in the title and author block, the later bibliography position, and the two-panel figure adaptation. No reference figure caption was present to measure.
- The only unresolved bibliographic detail is a formal publication date or preferred scholarly citation for the Castle repository. The source is cited without a publication year. The reconstruction results, images and algorithms were preserved.
