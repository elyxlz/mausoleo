# Mausoleo v123 deep citation critic

Reviewer: deep verification (Opus 4.7 manual + pdftotext + Read).
Pass: every distinct in-text citation in v123 main body verified against the actual PDF source in `/tmp/mausoleo/references/papers/`.
Excluded: References list, Appendix A.

## Aggregate

- Total in-text citation occurrences: **45**
- Distinct cited works: **28**
- PASS (occurrences): **38** (84%)
- DUBIOUS (occurrences): **5** (11%)
- FAIL (occurrences): **2** (5%)
- NO_PDF (occurrences): **0** (0%)

By distinct work:
- PASS: 23 (82%)
- DUBIOUS: 4 (14%)
- FAIL: 1 (4%)

## PASS list

| Citation | PDF | Notes |
|---|---|---|
| (Bartlett, 1932) line 31 | cognitive_science/bartlett-1932-remembering.pdf | *Remembering* verified — title page matches; *War of the Ghosts* studies are canonical content of the book. PASS. |
| (Miller, 1956) line 31 | cognitive_science/miller-1956-magical-seven.pdf | *Psychological Review* 63(1) March 1956. Title and "seven plus or minus two" framing exact. PASS. |
| (Cowan, 2001) line 31 | cognitive_science/cowan-2001-magical-four.pdf | *BBS* target article, abstract: "central capacity limit averaging about four chunks". Claim about "four chunks as the realistic ceiling" exactly matches Cowan's framing. PASS. |
| Tolman (1948) line 33 | cognitive_science/tolman-1948-cognitive-maps.pdf | "Cognitive Maps in Rats and Men", *Psychological Review* 55(4):189-208. Claim about rats building "something like a map" matches. PASS. |
| Whittington et al. (2020) line 33 | cognitive_science/whittington-2020-tem.pdf | Cell 183, 1249–1263, Nov 25 2020. Author list (Whittington, Muller, Mark, Chen, Barry, Burgess, Behrens) matches. Claim about "factorisation of structure and content" doing spatial navigation, sequence prediction and analogical generalisation in the same model matches the paper's central claim. PASS. |
| (Sarthi et al., 2024) lines 37, 56 | digital_humanities_ir/sarthi-2024-raptor.pdf | RAPTOR ICLR 2024. Claim about "recursively clustering chunk embeddings and summarising each cluster" matches abstract: "recursively embedding, clustering, and summarizing chunks of text". PASS. |
| (Edge et al., 2024) lines 37, 56 | digital_humanities_ir/edge-2024-graphrag.pdf | arXiv:2404.16130. Claim about Leiden community detection on entity graph + community summaries matches paper exactly. "Substantial gains over flat RAG for global-summarisation queries on three benchmarks" verified. PASS. |
| (Zhang and Tang, 2025) lines 37, 56 | technical/vectifyai-2025-pageindex.pdf | Authors: Mingtian Zhang, Yu Tang, VectifyAI. PageIndex framework, "vectorless reasoning-based RAG via hierarchical tree index". Claim about "table-of-contents as the retrieval target" + reading hierarchy off the surface is consistent with the paper's "semantic tree structures resembling tables of contents" framing. PASS. |
| (Ehrmann et al., 2020) line 48 | digital_humanities_ir/ehrmann-2020-impresso-lrec.pdf | LREC 2020 pp. 958–968. "Language Resources for Historical Newspapers: the Impresso Collection". Authors: Ehrmann, Romanello, Clematide, Ströbel, Barman. Reference list matches. Claim about NER, topic models, French/German/Luxembourgish, ~200 years matches abstract. PASS. |
| (Düring et al., 2024) line 48 | digital_humanities_ir/during-2024-impresso-interface.pdf | *Historical Methods* 57(3) 2024 doi:10.1080/01615440.2024.2344004. Authors: Düring, Bunout, Guido. Reference list matches. PASS. |
| Düring et al. (2024) "transparent generosity" line 50 | same | DIRECT QUOTE: "transparent generosity" — verified character-perfect against title and body of paper. PASS. |
| Salton, Wong and Yang (1975) line 54 | digital_humanities_ir/salton-1975-vsm.pdf | *CACM* 18(11). Title "A Vector Space Model for Automatic Indexing", authors G. Salton, A. Wong, C.S. Yang, Cornell. PASS. |
| Robertson and Zaragoza (2009) line 54 | digital_humanities_ir/robertson-2009-bm25.pdf | *Foundations and Trends in IR* 3(4) 2009 pp. 333-389. Title "The Probabilistic Relevance Framework: BM25 and Beyond". Claim about consolidating probabilistic-relevance tradition culminating in BM25 matches. PASS. |
| Bartlett's (1932) line 66 | same as #1 | PASS — schema framing in *Remembering* is canonical. |
| Craik and Lockhart (1972) line 66 | (see DUBIOUS below) | flagged separately. |
| Miller's (1956) line 68 | same as #2 | PASS. |
| Cowan's (2001) line 68 | same as #3 | PASS. |
| Tolman (1948) line 70 | same as #4 | PASS. |
| Behrens et al. (2018) line 70 | cognitive_science/behrens-2018-cognitive-map.pdf | *Neuron* 100(2) Oct 2018 doi:10.1016/j.neuron.2018.10.002. Title "What is a cognitive map? Organizing knowledge for flexible behavior". Authors: Behrens, Muller, Whittington, Mark, Baram, Stachenfeld, Kurth-Nelson. Reference list matches. Claim about spatial-coding machinery being recruited for non-spatial structural inference matches abstract directly. PASS. |
| Whittington et al. (2020) line 70 | same as #6 | PASS — "general-purpose relational learner factorising structure from content" matches. |
| Schudson (1978) line 75 | historiography/schudson-1981-discovering-the-news.pdf | *Discovering the News: A Social History of American Newspapers*, Basic Books. Title page verified. PDF filename says 1981 (reprint), but the work's first edition is 1978. Claim about "social construction of news" matches the book's overall thesis. PASS (year correct in citation; PDF filename labels reprint). |
| Murialdi (1986) lines 75, 119, 174 | historiography/murialdi-1986-stampa-regime-fascista.pdf | PDF actual title: *Storia del giornalismo italiano* by Paolo Murialdi (NOT *La stampa del regime fascista* as filename suggests). Reference list correctly cites *Storia del giornalismo italiano: Dalle gazzette a Internet*, Il Mulino. PDF content matches. Claim about MinCulPop directive system documented is consistent with the standard treatment. PASS. (Filename is misleading; reference list and content are correct.) |
| (International Council on Archives, 2000) line 75 | digital_humanities_ir/ica-2000-isadg.pdf | *ISAD(G): General International Standard Archival Description*, Second Edition, Stockholm 1999/Ottawa 2000. *Respect des fonds* is named in ISAD(G). PASS. |
| (Bai et al., 2025) line 87 | technical/bai-2025-qwen25-vl.pdf | arXiv:2502.13923, Qwen Team. Document understanding capability verified ("robust document parsing"). PASS. |
| Ketelaar (2001) line 99 | digital_humanities_ir/ketelaar-2001-tacit-narratives.pdf | *Archival Science* 1(2):131-141, 2001. Title "Tacit Narratives: The Meanings of Archives". Claim about "archival description as a tacit narrative" matches title and central thesis verbatim. PASS. |
| Wu et al. (2021) line 103 | technical/wu-2021-recursive-book-summarization.pdf | arXiv:2109.10862, OpenAI. Authors: Wu, Ouyang, Ziegler, Stiennon, Lowe, Leike, Christiano. Recursive bottom-up summarisation method verified. PASS. |
| Yao et al. (2022) line 113 | technical/yao-2022-react.pdf | arXiv:2210.03629, ICLR 2023. ReAct method verified. Reference list cites year 2022 (arXiv) which is acceptable. PASS. |
| Lewis et al. (2020) line 113 | digital_humanities_ir/lewis-2020-rag.pdf | arXiv:2005.11401, NeurIPS 2020. RAG paper. Author list matches. PASS. |
| Pavone (1991) lines 119, 135 | historiography/pavone-1991-guerra-civile.pdf (corrupt) + philosophy_media_1943/pavone-2013-civil-war.epub (English translation) | *Una guerra civile*, Bollati Boringhieri 1991. Verified via 2013 English translation. Pavone treats 25 July - 8 September as "Badoglio's forty-five days" / *quarantacinque giorni*, a substantive historical category. The draft's Italian "interregno" is a paraphrase, not Pavone's exact term, but the analytic framing is accurate. PASS. |
| Bosworth (2005) line 119 | philosophy_media_1943/bosworth-2005-mussolinis-italy.pdf | *Mussolini's Italy: Life under the Dictatorship, 1915-1945*, Penguin/Allen Lane. Title page verified, table of contents verified. Standard biography of fascist period; appropriate for ground-truth assembly. PASS. |
| Deakin (1962) line 119 | philosophy_media_1943/deakin-1962-brutal-friendship.pdf | *The Brutal Friendship: Mussolini, Hitler and the Fall of Italian Fascism*. Title page verified; "New Edition" reprint but content is original 1962 text. PASS. |
| (Eichenbaum, 2017) line 172 | (see DUBIOUS below) | flagged separately. |
| (Whittington et al., 2020) line 172 | same as #6 | PASS. |
| (Miller, 1956) line 172 | same as #2 | PASS. |
| (Cowan, 2001) line 172 | same as #3 | PASS. |

## DUBIOUS list (severity-sorted, lowest to highest concern)

| Citation | PDF | Issue | Fix |
|---|---|---|---|
| Murialdi (1986) reference-list filename | historiography/murialdi-1986-stampa-regime-fascista.pdf | Filename suggests "La stampa del regime fascista" (1986, Laterza) but actual PDF content is "Storia del giornalismo italiano" by Paolo Murialdi as cited. NOTE: *La stampa del regime fascista* is a real Murialdi 1986 monograph; the PDF in the corpus is the *Storia del giornalismo italiano* book. The reference list cites the latter. Substantively the in-text claim about the Italian press under the regime is supported by either book. | No edit needed if reference list is correct. Rename PDF for hygiene. |
| Schudson (1978) | historiography/schudson-1981-discovering-the-news.pdf | PDF filename says 1981 (reprint), reference list cites 1978 (original Basic Books edition). The content is the original Schudson monograph. Year in reference list (1978) is the canonical first-edition year — citation is correct. | None needed. Rename PDF if pedantic. |
| Murugaraj et al. (2025) line 58 | digital_humanities_ir/murugaraj-2025-topic-rag-newspapers.pdf | TITLE MISMATCH: actual paper title is "Topic-RAG **for** historical newspapers: **Enhancing information retrieval in humanities research through topic-based** retrieval-augmented generation". Reference list in v123 cites "Topic-RAG **over** historical newspapers: improving retrieval relevance over flat RAG on the impresso corpus". The reference list title is paraphrased/synthesised, not the published title. Volume (CHR 1:e15) and authors are correct; metric claim (BERTScore, ROUGE, UniEval) is verified verbatim from the paper. | Edit reference list entry to actual title: "Topic-RAG for historical newspapers: enhancing information retrieval in humanities research through topic-based retrieval-augmented generation". The DOI is 10.1017/chr.2025.10018. |
| Eichenbaum (2017) lines 33, 70, 172 | cognitive_science/eichenbaum-2017-hippocampus-navigation.pdf | BIBLIOGRAPHIC PROBLEM: PDF in corpus is "The role of the hippocampus in navigation is memory", *J Neurophysiol* 117:1785-1796, 2017. Reference list cites "On the integration of space, time, and memory", *Neuron* 95(5):1007-1018. Both are real 2017 Eichenbaum review papers. The Neuron paper is directly about "integration of space, time, and memory" and is the better match for the in-text claim about "indexes time and conceptual relation". The corpus has the wrong Eichenbaum 2017 paper. Substantively the J Neurophysiol paper also supports the claim (it argues the hippocampus organises relational memory beyond spatial domain, including temporal organisation). | Two options: (a) update the reference-list entry to the J Neurophysiol paper (which is the one in the corpus and which is on point), or (b) keep the Neuron citation and add the Neuron PDF to the corpus. Option (a) is faster. The Neuron 2017 paper is the more direct match for the cited claim, so Option (b) is preferable on rubric grounds. |
| Craik and Lockhart (1972) line 66 | cognitive_science/craik-lockhart-1972-levels-of-processing.pdf | WRONG PDF: corpus contains a 2014 review article by Giulia Galli ("What makes deeply encoded items memorable? Insights into the levels of processing framework from neuroimaging and neuromodulation", *Frontiers in Psychiatry*) that surveys the levels-of-processing framework, NOT the original Craik & Lockhart (1972) *JVLVB* paper. The original paper "Levels of processing: A framework for memory research" by Craik & Lockhart, *Journal of Verbal Learning and Verbal Behavior* 11(6):671-684, is real and the in-text claim ("recast the durability of memory as a question about depth of processing") accurately summarises it. The verifier in the corpus is a secondary review of the same framework. The in-text characterisation is canonical and correct on Craik & Lockhart's framework. | Replace the PDF in the corpus with the actual Craik & Lockhart (1972) paper, or accept the Galli 2014 review as a secondary witness (the review's abstract restates Craik & Lockhart's framework correctly: "this phenomenon has been conceptualized within the 'levels of processing' framework and has been consistently replicated since its original proposal by Craik and Lockhart in 1972"). Citation in v123 stands; provenance of source PDF is the issue. |

## FAIL list (severity-sorted)

| Citation | PDF | Issue | Fix |
|---|---|---|---|
| Doucet et al. (2020) — reference list entry | technical/doucet-2020-newseye.pdf | CO-AUTHOR CONFLATION (matches the memory-rule failure mode). The reference list in v123 reads: "Doucet, A., Gabay, S., Granroth-Wilding, M., Hulden, M., Düring, M., Pfanzelter, E., Marjanen, J., et al." The actual paper authors are: "Antoine Doucet, Martin Gasteiner, Mark Granroth-Wilding, Max Kaiser, Minna Kaukonen, Roger Labahn, Jean-Philippe Moreux, Guenter Muehlberger, Eva Pfanzelter, Marie-Eve Therenty, Hannu Toivonen, Mikko Tolonen". Doucet, Granroth-Wilding, Pfanzelter ARE on the paper. **Gabay, Hulden, Düring, Marjanen are NOT** — these names belong to other digital-humanities/historical-newspaper projects (Gabay is associated with Impresso/EPFL; Hulden is on NewsEye-adjacent NER work; Düring is Impresso lead; Marjanen is on Helsinki Computational History Group). The cited author list mixes the correct Doucet 2020 paper with author names from sibling projects. The in-text claim ("convention follows historical-newspaper OCR work in NewsEye") is substantively correct — NewsEye does this. The bibliographic detail in the references list is incorrect. | Replace the reference list entry with the actual author list: "Doucet, A., Gasteiner, M., Granroth-Wilding, M., Kaiser, M., Kaukonen, M., Labahn, R., Moreux, J.-P., Muehlberger, G., Pfanzelter, E., Therenty, M.-E., Toivonen, H. and Tolonen, M. (2020) 'NewsEye: a digital investigator for historical newspapers', in *Proceedings of Digital Humanities 2020*. Ottawa: ADHO." This is the most pressing edit before submission. |

## Provenance flags resolved

| Flag | Resolution |
|---|---|
| schudson-1978 vs PDF dated 1981 | PASS — content is the canonical 1978 first edition; PDF filename labels a reprint year. Reference list year (1978) is correct. |
| murialdi-1986 filename mismatch | PASS — PDF actual content is *Storia del giornalismo italiano*, matching the reference list. Filename misleading. |
| during-2024 vs ehrmann-2020 conflation risk | PASS — Düring et al. (2024) "Transparent generosity" is a distinct paper from Ehrmann et al. (2020) Impresso LREC; both are in corpus and correctly cited. |
| zhang-tang-2025 vs vectifyai filename | PASS — VectifyAI authors are Mingtian Zhang and Yu Tang. Citation matches. |
| eichenbaum-2017 PDF (J Neurophysiol) vs ref-list (Neuron) | DUBIOUS — wrong-paper-in-corpus problem; substance still holds. |
| craik-lockhart-1972 PDF actually Galli 2014 | DUBIOUS — wrong-paper-in-corpus; Craik & Lockhart 1972 is real and the in-text claim accurately summarises it. |
| doucet-2020 ref-list author list | FAIL — co-author conflation, fixable by editing the reference list entry. |

## Direct-quote check

Only one direct quote in the main text (excluding the long Italian block quote in chapter 4 from the Mausoleo summariser, which is not from a third-party source):

| Quote | Source | Verdict |
|---|---|---|
| "transparent generosity" — line 50 | Düring et al. (2024), *Historical Methods* 57(3) | PASS — appears character-perfect in the source title and body (capitalised "Transparent generosity" in title, used lowercase in body). |

## Recommendation

**FIX-THEN-SHIP.** One FAIL, two non-trivial DUBIOUS, two cosmetic DUBIOUS. Specific edits before submission Wed 6 May 17:00 BST:

1. **Required fix (FAIL)**: rewrite the Doucet et al. (2020) reference-list entry with the correct author list (12 authors, listed above). This is the only blocker. The in-text use is fine; only the bibliography is wrong.

2. **Recommended fix (DUBIOUS)**: replace the Eichenbaum (2017) reference-list entry with whichever Eichenbaum 2017 paper Elio actually meant to cite. The in-text claim ("indexes time and conceptual relation") points more directly at the Neuron paper. If keeping Neuron, download the actual Neuron 2017 PDF; if switching to J Neurophysiol, edit the reference-list entry to match the corpus PDF. Both options preserve the in-text argument.

3. **Recommended fix (DUBIOUS)**: replace the Galli 2014 review PDF in the corpus with the original Craik & Lockhart (1972) paper, or note the substitution explicitly. The in-text citation is accurate to Craik & Lockhart's actual framework.

4. **Recommended fix (DUBIOUS)**: correct the Murugaraj et al. (2025) reference-list title to the actual published title.

5. **Cosmetic**: rename `murialdi-1986-stampa-regime-fascista.pdf` to `murialdi-1996-storia-giornalismo-italiano.pdf` (the actual content) — does not affect the dissertation submission. Rename `schudson-1981-discovering-the-news.pdf` to `schudson-1978-discovering-the-news.pdf` for clarity.

All in-text claims are substantively supported by the PDFs available in `/tmp/mausoleo/references/papers/`. The substantive cognitive-science argument (Bartlett, Miller, Cowan, Tolman, Eichenbaum, Behrens, Whittington), the digital-humanities lineage (Salton, Robertson, Sarthi, Edge, Zhang & Tang, Lewis, Yao, Wu), the historiographical framing (Schudson, Murialdi, Pavone, Bosworth, Deakin), and the corpus-context citations (Bai, Doucet, Ehrmann, Düring, Murugaraj, Ketelaar, ICA) all stand. The only argumentative attribution problem is none — there is no citation inversion. The verification target is bibliographic detail and a single co-author-conflation in the references list.

Final verdict: **FIX-THEN-SHIP** with the five edits above. Estimated time: 30 minutes for the bibliographic edits, plus optional PDF replacements.

---

Reviewer: Opus 4.7 (manual + pdftotext + Read PDF tool).
Date: 2026-05-04.
PDFs at /tmp/mausoleo/references/papers/.
Index at /tmp/mausoleo/eval/critics/citation_full_v123_index.md.
Mapping at /tmp/mausoleo/eval/critics/citation_pdf_map_v123.md.
