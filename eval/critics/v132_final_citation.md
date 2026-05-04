# v132 final citation verdict

Model: claude-haiku-4-5
Elapsed: 31.4s
Draft: /tmp/mausoleo/references/MAUSOLEO_FULL_DRAFT_v132.md

---

# Citation Verification Report — v132 Final Check

## Bibliography Fix Re-verification

**1. Doucet et al. (2020) NewsEye**

In-text: "This convention follows historical-newspaper OCR work in NewsEye (Doucet et al., 2020)."

Reference listed: "Doucet, A., Gasteiner, M., Granroth-Wilding, M., Kaiser, M., Kaukonen, M., Labahn, R., Moreux, J.-P., Muehlberger, G., Pfanzelter, E., Therenty, M.-E., Toivonen, H. and Tolonen, M. (2020) 'NewsEye: a digital investigator for historical newspapers', in *Proceedings of Digital Humanities 2020*. Ottawa: ADHO."

**Verdict: HOLDS** ✓
The author list is correct; no conflation with sibling projects (Gabay, Hulden, Düring, Marjanen absent). Correct venue, year, and title format.

---

**2. Eichenbaum (2017)**

In-text: "Eichenbaum (2017), reviewing decades of single-cell hippocampal recording, argued that the same circuit also indexes time and conceptual relation."

Reference listed: "Eichenbaum, H. (2017) 'The role of the hippocampus in navigation is memory', *Journal of Neurophysiology*, 117(4), pp. 1785–1796. doi:10.1152/jn.00005.2017."

**Verdict: HOLDS** ✓
Corrected from erroneous Neuron 95(5):1007-1018. Title, journal, volume, issue, pages, and DOI all match the required fix. The subject matter (hippocampal encoding of temporal and relational structure) aligns with in-text claim.

---

**3. Murugaraj et al. (2025)**

In-text: "Murugaraj et al. (2025) is the closest prior work on retrieval-augmented generation over historical newspapers; they apply a topic-restricted retrieval pipeline to the Impresso Swiss corpus and report improved retrieval relevance over flat retrieval-augmented generation, measured by BERTScore, ROUGE and UniEval."

Reference listed: "Murugaraj, M., Lamsiyah, S., Düring, M. and Theobald, M. (2025) 'Topic-RAG for historical newspapers: enhancing information retrieval in humanities research through topic-based retrieval-augmented generation', *Computational Humanities Research*, 1(e15), pp. 1–24. doi:10.1017/chr.2025.10018."

**Verdict: HOLDS** ✓
Title, journal, volume, article ID, page range, and DOI all match the specified fix. In-text paraphrase (topic-restricted retrieval, BERTScore/ROUGE/UniEval metrics) is consistent with the full title's promise of "topic-based retrieval-augmented generation." Earlier version with paraphrased title successfully corrected.

---

## Spot-Check of 5 Additional Citations

I will check 5 citations from beyond the deep-verified list (Bartlett-1932, Miller-1956, Cowan-2001, Tolman-1948, Whittington-2020, Sarthi-2024, Edge-2024, Zhang-Tang-2025, Ehrmann-2020, Düring-2024, Salton-1975, Robertson-2009, Behrens-2018, Schudson-1978, Murialdi-1986, ICA-2000, Bai-2025, Ketelaar-2001, Wu-2021, Yao-2022, Lewis-2020, Pavone-1991, Bosworth-2005, Deakin-1962, Craik-Lockhart-1972).

---

### Spot-Check 1: Craik & Lockhart (1972)

**In-text claim + citation:**
"Craik and Lockhart (1972) recast the durability of memory as a question about depth of processing, where deeper semantic encoding leaves a more durable trace."

**Expected source content:**
*Levels of processing: a framework for memory research* is a foundational psychology paper proposing that memory durability correlates with depth of processing (shallow → deep), with semantic (deep) encoding producing longer-lasting retention than phonemic or structural (shallow) processing.

**Reference as listed:**
"Craik, F.I.M. and Lockhart, R.S. (1972) 'Levels of processing: a framework for memory research', *Journal of Verbal Learning and Verbal Behavior*, 11(6), pp. 671–684."

**Verdict: VERIFIED** ✓
Title, authors, journal, year, volume, issue, and page range are all standard and correct. The in-text paraphrase ("depth of processing" → durability) is the core claim of the paper.

---

### Spot-Check 2: Ketelaar (2001)

**In-text claim + citation:**
"I follow Ketelaar (2001) on archival description as a tacit narrative: summaries are treated as derived and authority rests at the leaf level paragraphs."

**Expected source content:**
*Tacit narratives: the meanings of archives* discusses how archival description itself is a form of narrative construction, and that archives carry meanings beyond explicit cataloguing—this fits the dissertation's use of the principle that summaries are secondary and raw text (leaves) hold primary authority.

**Reference as listed:**
"Ketelaar, E. (2001) 'Tacit narratives: the meanings of archives', *Archival Science*, 1(2), pp. 131–141."

**Verdict: VERIFIED** ✓
Title, author, year, journal, volume, issue, and page range standard. The cited principle aligns with archival theory on respecting source material as primary and derived descriptions as secondary.

---

### Spot-Check 3: Bai et al. (2025)

**In-text claim + citation:**
"Trained for dense document understanding (Bai et al., 2025), the Qwen2.5-VL backbone functions as a black-box OCR engine prompted to emit structured article JSON."

**Expected source content:**
A technical report on the Qwen2.5-VL vision-language model, documenting its architecture and performance on document understanding / dense visual tasks. The model is described as suitable for OCR and structured extraction from images.

**Reference as listed:**
"Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., et al. (2025) 'Qwen2.5-VL Technical Report', *arXiv preprint* arXiv:2502.13923."

**Verdict: VERIFIED** ✓
Author list (et al. form is acceptable for preprints with many authors), year, title, and arXiv ID are consistent with a real technical report on that model. The in-text claim (document understanding, suitable for OCR) matches Qwen2.5-VL's advertised capabilities.

---

### Spot-Check 4: Wu et al. (2021)

**In-text claim + citation:**
"The procedure follows Wu et al. (2021), whose recursive book summarisation showed that fixed-branch bottom-up summarisation produces usable abstractions without ever blowing the context window."

**Expected source content:**
"Recursively summarizing books with human feedback" — a paper on scaling summarisation to long documents through hierarchical, bottom-up recursive summarisation. The key finding is that fixed-branching recursive summarisation avoids exceeding LLM context limits while maintaining quality.

**Reference as listed:**
"Wu, J., Ouyang, L., Ziegler, D.M., Stiennon, N., Lowe, R., Leike, J. and Christiano, P. (2021) 'Recursively summarizing books with human feedback', *arXiv preprint* arXiv:2109.10862."

**Verdict: VERIFIED** ✓
Authors, year, title, and arXiv ID standard. The in-text paraphrase ("fixed-branch bottom-up summarisation," avoiding context-window overflow) correctly captures the paper's core contribution.

---

### Spot-Check 5: Yao et al. (2022)

**In-text claim + citation:**
"The reading loop is a ReAct loop in the sense of Yao et al. (2022), not single-shot retrieval-augmented generation in the sense of Lewis et al. (2020)."

**Expected source content:**
"ReAct: Synergizing Reasoning and Acting in Language Models" — a landmark paper introducing the ReAct (Reasoning + Acting) framework for LLM agents that iteratively reason about observations and choose actions, contrasted with single-pass question-answering or retrieval-augmented generation that does not loop.

**Reference as listed:**
"Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. and Cao, Y. (2022) 'ReAct: Synergizing reasoning and acting in language models', *arXiv preprint* arXiv:2210.03629."

**Verdict: VERIFIED** ✓
Authors, year, title, and arXiv ID all standard. The in-text distinction (ReAct loop vs. single-shot RAG) directly maps to the paper's core methodological contrast.

---

## Summary Table

| Citation | In-text Claim | Expected Source | Verdict | Note |
|---|---|---|---|---|
| Doucet et al. 2020 | NewsEye OCR history | Digital newspaper project with full author list | VERIFIED ✓ | Fix holds; correct 12-author list |
| Eichenbaum 2017 | Hippocampal temporal/relational encoding | J Neurophysiol article on hippocampus & navigation | VERIFIED ✓ | Fix holds; correct venue/DOI |
| Murugaraj et al. 2025 | Topic-RAG on historical newspapers | CHR paper on topic-based retrieval-augmented generation | VERIFIED ✓ | Fix holds; correct journal/DOI |
| Craik–Lockhart 1972 | Depth-of-processing framework | Classic memory psychology paper | VERIFIED ✓ | Standard foundational reference |
| Ketelaar 2001 | Archives as tacit narrative | Archival science on narrative meaning-making | VERIFIED ✓ | Aligns with dissertation's archival principle |
| Bai et al. 2025 | Qwen2.5-VL for document understanding | Vision-language model technical report | VERIFIED ✓ | arXiv ID correct; model documented |
| Wu et al. 2021 | Recursive book summarisation, fixed-branch | Scaling summarisation via hierarchy | VERIFIED ✓ | arXiv ID correct; methodology matches |
| Yao et al. 2022 | ReAct reasoning-and-acting loops | LLM agentic framework paper | VERIFIED ✓ | arXiv ID correct; contrasts with Lewis 2020 RAG |

---

## Final Verdict

**Bibliography fix re-verification:**
- Doucet 2020: **HOLDS** ✓
- Eichenbaum 2017: **HOLDS** ✓
- Murugaraj 2025: **HOLDS** ✓

**Spot-check summary:** 
- **5 verified** / 0 unverifiable / 0 mismatch / 0 hallucinated

**Citation verdict: PASS** ✓

**Comparison to v123 deep critic baseline (38/45 PASS, fixes shipped in v131):**
- The three deep-verified fixes (Doucet, Eichenbaum, Murugaraj) all hold correctly in v132.
- Spot-checks on 5 additional citations (Craik–Lockhart, Ketelaar, Bai, Wu, Yao) are all verified.
- No new errors or citation inversions detected.
- The v131 repairs were successful and stable through v132.

**CONFIRM** — v132 maintains v131's citation corrections and shows no degradation. Ready for submission.
