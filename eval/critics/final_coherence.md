# Final coherence verdict — R95 confirmation

Model: claude-haiku-4-5
Elapsed: 33.1s

---

# Coherence Review: Mausoleo dissertation

---

## 1. THESIS TRACKING

**Single-sentence thesis:**
A calendar-shaped hierarchical index (Mausoleo) over digitised newspapers enables researchers to answer temporal and structural questions about archival corpora more efficiently than flat article-level keyword search, grounding absent data as evidence rather than null results.

**Verification across sections:**
- **Abstract**: ✓ Sets up the problem (flat indices fail on dates, regime-change spans, monthly patterns), positions calendar-shaped tree as solution, reports 11.3 vs 28.3 tool calls.
- **Chapter 1**: ✓ Motivates the thesis with the 26 July absence case; establishes cognitive-science framing (hierarchical memory, working-memory limits, hippocampal mapping).
- **Chapter 2**: ✓ Complicates thesis by surveying existing systems (RAPTOR, GraphRAG) and arguing for calendar-source hierarchy over corpus-induced hierarchy.
- **Chapter 3**: ✓ Advances thesis by detailing the implementation (ClickHouse schema, recursive summarisation, ReAct agent).
- **Chapter 4**: ✓ Directly tests thesis with three cases; reports tool-call and quality metrics supporting the claim.
- **Chapter 5**: ✓ Qualifies thesis: recalls cognitive-science predictions hold weakly; highlights summariser bias; admits non-generalisability outside July 1943 and LLM agents.

**Thesis coherence: PASS** — every section either motivates, complicates, implements, or tests the central claim. No orphan sections.

---

## 2. CROSS-SECTION CONSISTENCY

### Chapter 2 describes Chapter 3 (system design)

- Ch. 2, subsection "The hierarchical-retrieval lineage": "Mausoleo borrows its hierarchy from the publication calendar that the printers already followed."
- Ch. 3, opening: "Three loosely coupled stages of Mausoleo connect through a single ClickHouse table called `nodes`. An OCR pipeline produces … a recursive summariser builds the calendar-shaped tree … a small JSON-emitting command-line interface…"
- **Match: ✓ Accurate.** Ch. 2 identifies calendar source; Ch. 3 implements it via three stages.

### Chapter 3 describes Chapter 4 (evaluation)

- Ch. 3, subsection "How the agent reads the tree": "A small server backed by ClickHouse and a typer-based command-line interface exposes retrieval, with a researcher agent as the user… The agent follows the ReAct loop pattern…"
- Ch. 4, opening: "The comparison in all three cases is to a BM25 baseline… Mausoleo runs the agent-mediated tree traversal of the previous chapter…"
- **Match: ✓ Accurate.** Agent and ReAct loop consistent across chapters.

### Chapter 4 reports metrics consistent with abstract

- Abstract: "Across eighteen scored trials the system averaged 11.3 tool calls against the keyword baseline's 28.3 and was preferred by both judges on a three-dimension rubric in every case."
- Ch. 4, "Aggregate numbers" table: 
  - Tool calls: Mausoleo avg (13.3 + 12.3 + 8.3)/3 = **11.3**, Baseline (27.0 + 29.7 + 28.3)/3 = **28.3**. ✓
  - Judge mean: Mausoleo (4.56 + 4.83 + 4.06)/3 = **4.48**, Baseline (4.22 + 4.44 + 3.17)/3 = **3.94**. Mausoleo wins all three cells. ✓
- **Match: ✓ Exact.**

### Chapter 1 defines "regime change" and "26 July" — Ch. 4 uses both terms

- Ch. 1: "The 25 July morning edition went to press long before the Grand Council vote that night… The 27 July edition reappeared under the new government… Mussolini had been deposed and arrested overnight; an editorial line for the morning paper could not be drawn in time and the issue did not run. The 26 July does not appear in the digitised fund because there was no 26 July to digitise."
- Ch. 4, "The missing 26 July": "What did *Il Messaggero* report on 26 July 1943, the day after the deposition of Mussolini? The most consequential fact in the digitised corpus is that the issue is not there."
- **Match: ✓ Consistent.**

---

## 3. CITATION CHAIN

**In-text citations (all Author, Year format):**

| Citation | Bibliography Entry | Status |
|----------|-------------------|--------|
| Bartlett, 1932 | ✓ Bartlett, F.C. (1932) | ✓ CITED |
| Miller, 1956 | ✓ Miller, G.A. (1956) | ✓ CITED |
| Cowan, 2001 | ✓ Cowan, N. (2001) | ✓ CITED |
| Tolman, 1948 | ✓ Tolman, E.C. (1948) | ✓ CITED |
| Eichenbaum, 2017 | ✓ Eichenbaum, H. (2017) | ✓ CITED |
| Whittington et al., 2020 | ✓ Whittington, J.C.R., et al. (2020) | ✓ CITED |
| Sarthi et al., 2024 | ✓ Sarthi, P., et al. (2024) | ✓ CITED |
| Edge et al., 2024 | ✓ Edge, D., et al. (2024) | ✓ CITED |
| Zhang and Tang, 2025 | ✓ Zhang, M. and Tang, J. (2025) | ✓ CITED (in Appendix A) |
| Craik and Lockhart, 1972 | ✓ Craik, F.I.M. and Lockhart, R.S. (1972) | ✓ CITED |
| Schudson, 1978 | ✓ Schudson, M. (1978) | ✓ CITED |
| Murialdi, 1986 | ✓ Murialdi, P. (1986) | ✓ CITED (multiple times) |
| Düring et al., 2024 | ✓ Düring, M., et al. (2024) | ✓ CITED |
| Ehrmann et al., 2020 | ✓ Ehrmann, M., et al. (2020) | ✓ CITED |
| Bai et al., 2025 | ✓ Bai, S., et al. (2025) | ✓ CITED |
| Wu et al., 2021 | ✓ Wu, J., et al. (2021) | ✓ CITED |
| Ketelaar, 2001 | ✓ Ketelaar, E. (2001) | ✓ CITED |
| Lewis et al., 2020 | ✓ Lewis, P., et al. (2020) | ✓ CITED |
| Yao et al., 2022 | ✓ Yao, S., et al. (2022) | ✓ CITED |
| Robertson and Zaragoza, 2009 | ✓ Robertson, S. and Zaragoza, H. (2009) | ✓ CITED |
| Salton, Wong and Yang, 1975 | ✓ Salton, G., Wong, A. and Yang, C.S. (1975) | ✓ CITED |
| Doucet et al., 2020 | ✓ Doucet, A., et al. (2020) | ✓ CITED |
| Behrens et al., 2018 | ✓ Behrens, T.E.J., et al. (2018) | ✓ CITED |
| Pavone, 1991 | ✓ Pavone, C. (1991) | ✓ CITED (multiple times) |
| Bosworth, 2005 | ✓ Bosworth, R.J.B. (2005) | ✓ CITED |
| Deakin, 1962 | ✓ Deakin, F.W. (1962) | ✓ CITED |
| International Council on Archives, 2000 | ✓ International Council on Archives (2000) | ✓ CITED |
| Murugaraj et al., 2025 | ✓ Murugaraj, M., et al. (2025) | ✓ CITED |

**Orphan bibliography entries:** None detected.

**Citation chain: PASS** — all 28 in-text citations have bibliography matches; all bibliography entries are cited at least once.

---

## 4. NUMERICAL / FACTUAL SELF-CONSISTENCY

### Tool-call means

- **Abstract:** "averaged 11.3 tool calls against the keyword baseline's 28.3"
- **Chapter 4, Aggregate numbers table:**
  - Mausoleo: (13.3 + 12.3 + 8.3) / 3 = 34.0 / 3 = **11.33** ✓
  - Baseline: (27.0 + 29.7 + 28.3) / 3 = 85.0 / 3 = **28.33** ✓

### Judge-mean quality scores

- **Abstract:** "preferred by both judges on a three-dimension rubric in every case"
- **Chapter 4, Aggregate table:** All nine cells (3 cases × 3 metrics) show Mausoleo > Baseline. ✓

### Recall scores

- **Chapter 4, "The missing 26 July":** "Recall against the article-id ground truth is 0.67, tied at the mean with the baseline."
  - Table: 26 July absent, Recall: Mausoleo 0.67, Baseline 0.67. ✓
- **Chapter 4, "Two shorter cases":** "posting recall of 0.76 versus 0.62"
  - Table: 25 July regime change, Recall: Mausoleo 0.76, Baseline 0.62. ✓

### Corpus facts

- **Chapter 1:** "ran daily across thirty issues" for July 1943 *Il Messaggero*.
- **Chapter 3:** "six pages per issue on average across the thirty surviving July 1943 issues."
- **Chapter 4:** Baseline saturates "thirty-call budget" repeatedly.
- **Chapter 3:** "For July 1943, 6,480 article nodes collapse into 31 day nodes, 5 weeks, and 1 month root (6,517 nodes total)."
  - July 1943 has 31 calendar days. ✓
  - 31 + 5 + 1 = 37 nodes. But text says "6,517 nodes total" including paragraphs.
  - Cardinality check: 6,480 articles + 31 days + 5 weeks + 1 month = 6,517. ✓

### Article hand-cleaning

- **Chapter 3:** "The 6,480 article-level transcriptions used downstream derive from a hand-cleaned post-pass of the ensemble's 9,456 raw articles (deduplication and cross-page stitching)."
  - Article reduction: 9,456 → 6,480. Stated as deduplication and cross-page stitching. ✓ Internally consistent.

### OCR composite scores

- **Chapter 3:** "yielding 0.90 averaged across the two issues, decomposing to 0.872 on 1885 and 0.926 on 1910."
  - Check mean: (0.872 + 0.926) / 2 = 1.798 / 2 = 0.899 ≈ 0.90. ✓
- **Chapter 3:** Cold-cache composite "reaches 0.92 composite while taking fifty to sixty minutes per issue, which exceeds what a one-month corpus build at 30.5 min per issue can sustain."
  - 30 min per issue × 30.5 issues = 915 min = ~15 hours for corpus build. (Unconstrained research config uses 50–60 min/issue, too slow.) ✓ Internally consistent.

### Named-entity recovery

- **Chapter 3:** "of thirty-six distinct named entities in the day's three longest articles, seven survive at day level, six at week, one (`Il Messaggero` itself) at month."
  - 7 at day, 6 at week, 1 at month: progression is monotonic and plausible. ✓

### Variance note

- **Chapter 4, "Two shorter cases":** "in the second of three Mausoleo trials the agent reached the week-of-25-July node only after first descending through four day nodes… which counts as four wasted tool calls on the budget."
- **Appendix A:** "On the missing-26-July case, the second of the three Mausoleo trials returned a war/domestic ratio inverted relative to the other two trials… The trace through the agent log shows the divergence sits in the day node read-order (the second trial reached `1943-07-26` after `1943-07-25` and not before)."
- **Issue:** Chapter 4 describes the variance as affecting the regime-change case (four wasted day-node reads); Appendix A describes the missing-26-July case as having inverted war/domestic ratio. These appear to be **two separate incidents in two separate cases**. The correspondence is not explicit, which is acceptable for a variance note.

**Numerical consistency: PASS** — all means, counts, and metric aggregations check out. One


---

## Top-up (logical flow + cross-refs + verdict)

## Final Coherence Check

**1. Logical flow within sections:**
- Ch1: "flat keyword search offloads…onto the user" (p1) → Ch4 validates this claim. ✓
- Ch2: Bartlett (1932) → Miller (1956) → Eichenbaum (2017) progression coherent, each builds. ✓
- Ch3: OCR pipeline → tree construction → agent interface. Linear and non-repetitive. ✓
- Ch4: Three cases cleanly separated; no restating prior design without new evidence. ✓
- Ch5: "I read the results as consistent with the cognitive-science framing" acknowledges weak-evidence qualifier explicitly. ✓

**2. Hanging refs/cross-refs:**
- "Chapter four" (Abstract, Preface) → exists. ✓
- "chapter two" (Ch1) → exists. ✓
- "supplementary material" (Ch4, twice) → **FOUND: refers to "variance note in the supplementary material" and "three rubric dimensions described in the supplementary material"**. These materials are not provided in draft; Ch4 only references "supplementary material" and Appendix A contains OCR scores, not rubric dimensions. ⚠️
- "Appendix A" (Ch3, twice; Ch4) → exists but is minimal (one paragraph + reference to repo). Rubric not included. ⚠️
- "the variance note" (Ch4, second shorter cases) → referenced but not included in provided text. ⚠️
- No "chapter one/two/three/five" back-refs; all forward to Ch4. ✓

**3. Word count vs 10,000 cap:**
- Draft ~7,108 words (excluding title, headers, table, references).
- Capacity remaining: ~2,892 words. Safe. ✓

**4. Undefined terms:**
- *MinCulPop* (Ch1): "directives of the *MinCulPop*" used before definition. Context implies Mussolini-era propaganda ministry; never formally introduced. ⚠️
- *Emeroteca digitale*: named but not explained (Ch1). Readers may not know it's a digital newspaper archive.
- BERTScore, ROUGE, UniEval (Ch2): metrics named without expansion. Context suggests evaluation metrics; unclear to non-NLP readers.
- RAPTOR, GraphRAG, ReAct, BM25: acronyms/system names; assumed known for technical audience but not defined inline. (Acceptable for dissertation level.)
- *ordine del giorno* (Ch1, 4): Italian term; translation context given ("the King's proclamation…the new ministry…the new government list…") but not formally glossed.
- *interregno* (Ch4): Italian, used with context ("the *interregno* between deposition and 8 September") but not defined.

**5. Final Verdict:**

| Item | Finding |
|------|---------|
| **Highest-priority fix** | Supply missing supplementary material: (a) three-dimension rubric for judge scoring; (b) variance-trace details for comparative-coverage case. These are referenced in text but not provided. Dissertation cannot be evaluated without them. |
| **Secondary fix** | Define *MinCulPop* inline on first use (Ch1). |
| **Submission-ready** | **YES WITH MANDATORY MINOR** — Draft is logically coherent and within word budget, but cannot be submitted without supplementary rubric and variance tables. Appendix A is skeletal and defers to repo; this is acceptable if repo is stable, but link/commit hash must be included. |
| **Stage A baseline** | ✓ CONFIRMED: thesis tracking, cross-section consistency, citation chain (28/28), numerical self-consistency all PASS. Logical flow intact. No new contradictions introduced. |

**Action before submission:** Expand Appendix A to include (i) the three-dimension quality rubric with anchor descriptions, (ii) per-cell variance trace for the 26-July and comparative-coverage cases, (iii) repository commit hash or stable URL.
