# Mausoleo v123 — Phase 2: PDF mapping

For each distinct cited work, the PDF located in `/tmp/mausoleo/references/papers/`. Provenance gaps flagged.

| Key | PDF path | Match notes |
|---|---|---|
| bartlett-1932 | cognitive_science/bartlett-1932-remembering.pdf | Direct match |
| miller-1956 | cognitive_science/miller-1956-magical-seven.pdf | Direct match |
| cowan-2001 | cognitive_science/cowan-2001-magical-four.pdf | Direct match |
| tolman-1948 | cognitive_science/tolman-1948-cognitive-maps.pdf | Direct match |
| eichenbaum-2017 | cognitive_science/eichenbaum-2017-hippocampus-navigation.pdf | Direct match |
| whittington-2020 | cognitive_science/whittington-2020-tem.pdf | Direct match |
| sarthi-2024 | digital_humanities_ir/sarthi-2024-raptor.pdf (also technical/) | Direct match |
| edge-2024 | digital_humanities_ir/edge-2024-graphrag.pdf (also technical/) | Direct match |
| zhang-tang-2025 | technical/vectifyai-2025-pageindex.pdf | PROVENANCE FLAG: file is `vectifyai-2025-pageindex.pdf`, but reference list attributes to "Zhang, M. and Tang, J. (2025)" with arXiv:2510.13347. Need to verify arXiv:2510.13347 author list matches "Zhang and Tang". |
| ehrmann-2020 | digital_humanities_ir/ehrmann-2020-impresso-lrec.pdf | Direct match |
| during-2024 | digital_humanities_ir/during-2024-impresso-interface.pdf (or technical/ehrmann-2024-impresso-transparent.pdf, or historiography/ehrmann-2024-impresso.pdf) | POTENTIAL CO-AUTHOR CONFLATION FLAG: draft cites Düring et al. (2024) for "Transparent generosity" article, but two other "ehrmann-2024" PDFs exist that may be the actual paper. Need to verify lead author. |
| salton-1975 | digital_humanities_ir/salton-1975-vsm.pdf | Direct match |
| robertson-2009 | digital_humanities_ir/robertson-2009-bm25.pdf | Direct match |
| murugaraj-2025 | digital_humanities_ir/murugaraj-2025-topic-rag-newspapers.pdf | Direct match — but flagged for LT4HALA co-author conflation risk per memory rule |
| craik-lockhart-1972 | cognitive_science/craik-lockhart-1972-levels-of-processing.pdf | Direct match |
| behrens-2018 | cognitive_science/behrens-2018-cognitive-map.pdf | Direct match |
| schudson-1978 | historiography/schudson-1981-discovering-the-news.pdf | YEAR MISMATCH FLAG: draft cites "Schudson (1978)", reference list cites "Schudson, M. (1978)", but PDF is dated 1981. Discovering the News by Schudson published 1978 (Basic Books); reissues exist. Possibly PDF is reprint or filename error; need to verify edition pagination. |
| murialdi-1986 | historiography/murialdi-1986-stampa-regime-fascista.pdf (duplicate in philosophy_media_1943/) | NAME MISMATCH FLAG: filename `murialdi-1986-stampa-regime-fascista` suggests *La stampa del regime fascista* (1986, Laterza), but reference list cites *Storia del giornalismo italiano: Dalle gazzette a Internet* (Il Mulino). These are different Murialdi books — *Storia del giornalismo italiano* is 1996/2014 from Il Mulino, not 1986. Likely citation-list bibliographic error. |
| ica-2000 | digital_humanities_ir/ica-2000-isadg.pdf | Direct match |
| bai-2025 | technical/bai-2025-qwen25-vl.pdf | Direct match |
| doucet-2020 | technical/doucet-2020-newseye.pdf | Direct match |
| ketelaar-2001 | digital_humanities_ir/ketelaar-2001-tacit-narratives.pdf | Direct match |
| wu-2021 | technical/wu-2021-recursive-book-summarization.pdf | Direct match |
| yao-2022 | technical/yao-2022-react.pdf | Direct match |
| lewis-2020 | digital_humanities_ir/lewis-2020-rag.pdf (also technical/) | Direct match |
| pavone-1991 | historiography/pavone-1991-guerra-civile.pdf | Direct match |
| bosworth-2005 | philosophy_media_1943/bosworth-2005-mussolinis-italy.pdf | Direct match |
| deakin-1962 | philosophy_media_1943/deakin-1962-brutal-friendship.pdf | Direct match |
| schudson-1978 (in §References) | historiography/schudson-1981-discovering-the-news.pdf | See above flag |

## Provenance flags summary

1. **schudson-1978**: PDF year is 1981; the canonical first edition of *Discovering the News* is 1978 (Basic Books). Possibly later reprint shipped instead of original edition. Verify content / edition.
2. **murialdi-1986**: PDF title (*La stampa del regime fascista*) does not match the reference-list title (*Storia del giornalismo italiano: Dalle gazzette a Internet*, Il Mulino). The 1986 book is by Murialdi but is a different work. The reference-list entry is bibliographically wrong; the PDF is the correct standard source for the in-text claim about MinCulPop directives.
3. **during-2024**: Need to confirm whether the actual "Transparent generosity" article is led by Düring or Ehrmann (memory rule about Düring 2024 vs Ehrmann 2020 at Impresso conflation).
4. **zhang-tang-2025**: PDF filename is `vectifyai-2025-pageindex.pdf`; need to verify arXiv:2510.13347 = Zhang & Tang.

NO_PDF citations: 0 (all 28 distinct works have a candidate PDF located).
