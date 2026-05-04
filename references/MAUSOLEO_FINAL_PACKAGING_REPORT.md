# Mausoleo v132 — final packaging report

**Date:** 2026-05-04 (Monday) — submission window: Wed 6 May 2026 17:00 BST
**Candidate code:** YCMG5 — **Module:** BASC0024 — **Supervisor:** Dr Yi Gong

## Counts

| Metric | Value |
|---|---|
| Word count (excl. Abstract + Preface) | **9,311** |
| Abstract words | 366 (cap 300 — slightly over; v132 ship-state, no prose edits per rules) |
| Preface words | 203 (cap 500) |
| Total dissertation pages (PDF) | **22** |
| Title page is page 1 (no footer); body starts page 2 with `YCMG5 / Page N of 22` footer. |

## Artefacts

| File | Size | Path |
|---|---|---|
| MAUSOLEO_DISSERTATION_FINAL.pdf | 528.5 KiB (541,210 B) | /tmp/file_host/ |
| MAUSOLEO_COVERSHEET_FINAL.pdf | 54.0 KiB (55,305 B) | /tmp/file_host/ |
| MAUSOLEO_FULL_DRAFT_v132.md | 60.5 KiB (61,994 B) | /tmp/file_host/ |

## OneDrive uploads (verified via `rclone lsf`)

Path: `onedrive:Schools/UCL/Y3/BASC0024 Dissertation/Submission/Final/`

```
MAUSOLEO_COVERSHEET_FINAL.pdf
MAUSOLEO_DISSERTATION_FINAL.pdf
MAUSOLEO_FULL_DRAFT_v132.md
```

## Figures (Seaborn re-render)

All four PNGs in `/tmp/mausoleo/references/figures/` were re-rendered with
`sns.set_theme(style="whitegrid", context="paper")`, fitted to ~6 in width
for the 65% pandoc width attribute on A4 main column, DPI 160.
Each PNG visually verified (read back as image) for label / legend /
annotation overlap before commit.

| Figure | Size | Notes |
|---|---|---|
| fig1_calendar_tree.png | 63.1 KB | matplotlib FancyBboxPatch on seaborn-styled axes (sns has no native tree). Title-bar left, bottom-up arrow stack right; no overlap. |
| fig2_tool_calls.png | 47.2 KB | sns.barplot, 3 cases × 2 systems with bar_label value annotations + 30-call cap line. |
| fig3_quality_rubric.png | 47.1 KB | 3-panel grouped sns.barplot, three rubric dims × Mausoleo/baseline. Panel titles split onto two lines to clear neighbours. |
| fig4_ocr_composite.png | 65.1 KB | sns.barplot dodge=False, ensemble-add (blue) vs LLM-postcorr (purple) hue, headline 0.899 reference line. |

## Pipeline

1. Re-rendered figures with seaborn (`render_appendix_figures.py` rewritten).
2. Title-page block (centred, page 1, no footer) + Pandoc HTML body, with
   `@page` CSS providing `YCMG5` bottom-left and `Page N of M` bottom-right
   on subsequent pages. Rendered via `chromium --headless --print-to-pdf`.
3. Cover sheet reconstructed as standalone HTML with verbatim declaration
   text (extracted via python-zipfile + regex over `word/document.xml`),
   archive-consent box ticked (`X`), typed signature `Elio Pascarelli`,
   date `5 May 2026`. Rendered via the same chromium pipeline.
4. `rclone copy` of all three artefacts to OneDrive Final/ subfolder;
   verified by `rclone lsf`.

## Skipped / deviations

- **Abstract over the 300-word cap** (366 words). Operating rule 4
  forbids prose edits to v132; the 6 critics ship-confirmed v132 with
  the abstract as-is. Flagged here for your awareness; if BASc enforces
  the cap strictly, we have time before Wed 17:00 BST to trim ~70 words.
- **Body keeps its top-level title heading** (`# Mausoleo: a calendar-shaped index for reading newspaper history`) which now appears both on the
  title page and at the top of page 2. Removing the body heading would
  count as a prose edit, so it was kept. The duplication is conventional
  for dissertations rendered from a single markdown source.
- **Disk** was tight (~4.4 GB free, not the expected 11 GB). Stayed under
  the budget — no extra temp PDFs retained beyond the final two.
- Figure source-of-truth lives in `/tmp/mausoleo/references/figures/`;
  build directory `/tmp/wed_packaging/build/` mirrors only what was
  needed for the dissertation HTML.

## Commits + tag

- `5016d3d` v132: re-render appendix figures with seaborn (pre-tag)
- A second commit will record the packaging artefacts (build/ inputs +
  this report) and the tag `v132-submission-final` will sit on it.
