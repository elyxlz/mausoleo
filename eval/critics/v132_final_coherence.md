# v132 final coherence verdict

Model: claude-haiku-4-5
Elapsed: 37.2s
Draft: /tmp/mausoleo/references/MAUSOLEO_FULL_DRAFT_v132.md

---

# COHERENCE REVIEW: v132 "Mausoleo" — Stage C Round 26-40

---

## 1. TITLE-ABSTRACT-INTRO ALIGNMENT

**Finding: PASS**

- **Title:** "Mausoleo: a calendar-shaped index for reading newspaper history"
- **Abstract framing:** Calendar-shaped tree, missing 26 July, three case studies, eighteen scored trials, 11.3 vs 28.3 tool calls, ~0.90 OCR composite
- **Introduction (§1):** Regime change July 1943, editorial-register shift, date 26 July absence, Bartlett + cognitive science, multi-resolution interface, three case studies
- **Match:** Complete. Title and abstract identify the core innovation (calendar shape); intro motivates it from reading difficulty and cognitive science; all three case-study questions named in both abstract and chapter 4.

---

## 2. THESIS TRACKING

**One-sentence thesis:**  
*A calendar-shaped hierarchical index (Mausoleo) that preserves temporal structure and missing-day slots answers multi-resolution questions about digitised newspapers more efficiently and with better grounding than flat keyword search, particularly when corpus absences (like 26 July 1943) are historically meaningful.*

**Section advancement check:**

| Section | Advances / Complicates thesis? |
|---|---|
| §1 Motivation | ✓ Establishes hardness: flat article search fails on date-bounds and month-shape queries |
| §2 Literatures | ✓ Situates calendar hierarchy against corpus-derived clustering; names cognitive warrant |
| §3 Build | ✓ Specifies calendar structure, OCR pipeline, summariser, agent toolset |
| §4 Cases | ✓ Tests three question types; reports tool-call, recall, quality gains |
| §5 Limits | ✓ **COMPLICATES:** Acknowledges weak power (n=3), summariser bias, non-generalisability, metric mismatch on case 3 |

**Verdict:** Thesis is coherent and every section advances or qualifies it. §5 appropriately narrows claims.

---

## 3. CROSS-SECTION NUMERICAL CONSISTENCY

**Checking claimed numbers against their appearances:**

| Claim | Location 1 | Location 2 | Match? |
|---|---|---|---|
| **Tool calls: 11.3 vs 28.3** | Abstract: "11.3 tool calls against...28.3" | Table §4: Mausoleo avg (13.3+12.3+8.3)/3=11.3; Baseline (27.0+29.7+28.3)/3=28.3 | ✓ |
| **Eighteen trials** | Abstract: "eighteen scored trials" | §4 intro: "All eighteen planned trials completed" | ✓ |
| **Three case studies** | Abstract + §4 heading | §4.1, §4.2 (regime), §4.3 (coverage) — yes, three | ✓ |
| **OCR composite 0.90** | Abstract: "~0.90" | Appendix A.3: "0.899 by the formula" | ✓ (0.899 rounds to 0.90) |
| **Component scores (0.872, 0.926)** | §3 "yields 0.90 averaged...0.872 on 1885 and 0.926 on 1910" | Appendix A.3 table: "0.872 | 0.926" | ✓ |
| **Node counts (6,480 → 31, 5, 1, 6,517)** | §3: "6,480 article nodes aggregate into 31 day nodes, 5 weeks, and 1 month root (6,517 nodes in total)" | §4.2: "per-week war fraction (W26...W30)" — confirms five weeks exist | ✓ |
| **Regime-change tool calls (12.3)** | §4.2: "cleared in roughly twelve Mausoleo tool calls" | Table §4: Case 2 Mausoleo = 12.3 | ✓ |
| **Regime-change recall (0.76 vs 0.62)** | §4.2: "recall of 0.76 versus 0.62" | Table §4: Case 2 = 0.76 / 0.62 | ✓ |
| **Quality regime (4.83 vs 4.44)** | §4.2: "quality scores of 4.83 versus 4.44" | Table §4: Case 2 = 4.83 / 4.44 | ✓ |
| **Case 3 MAE (0.149 vs 0.194)** | §4.3: "Ratio MAE (0.149 vs 0.194)" | Table §4: Case 3 = 0.149 / 0.194 | ✓ |
| **κ = 0.57 regime, 0.14 coverage** | §4.2 "κ = 0.57"; §4.3 "κ = 0.14" | Appendix A.2 table (inter-judge κ by case) = 0.57, 0.14 | ✓ |

**Verdict: PASS** — All numerical claims are internally consistent across abstract, main text, table, and appendix.

---

## 4. CROSS-REFERENCE INTEGRITY (Stage C Heading Rewrites)

**Checking cross-references after heading-shape rewrites:**

| Reference | Target | Exists? | Accurate? |
|---|---|---|---|
| "chapter four" (appears 3× in abstract, §1, §2) | §4 "Reading the missing 26 July..." | ✓ | ✓ |
| "chapter two" (§1, §3) | §2 "Two literatures..." | ✓ | ✓ |
| "chapter three" (§2, §5) | §3 "How I built Mausoleo" | ✓ | ✓ |
| "chapter-four spot-check" (§2, §5) | §3: "forty-eight of fifty named entities" | ✓ | ✓ |
| "the three case studies" (§4 intro) | §4.1 (26 July), §4.2 (regime), §4.3 (coverage) | ✓ | ✓ |
| "Appendix A" (§3, §4, §5) | Appendix A material (OCR, rubric, variance, tools) | ✓ | ✓ |
| "supplementary material" (§4 intro) | Resolves to Appendix A.1 (rubric) | ✓ | ✓ |
| "§X.Y" format | None used; section numbers use §N instead | — | N/A (no broken refs) |
| "Figure 1" (§3) | "Calendar-shaped index...July 1943" | ✓ | ✓ |
| "Figure 2" (§4.3) | "Mean tool calls per trial" | ✓ | ✓ |
| "Figure 3" (§4.3) | "Per-case judge scores" | ✓ | ✓ |
| "Figure 4" (§3, Appendix A.3) | "Composite OCR score by ensemble pass" | ✓ | ✓ |

**Checking claims about sections:**

- **Abstract says "chapter four runs three case studies"** → §4 contains §4.1, §4.2, §4.3: three case studies ✓
- **§1 says "three questions are put to Il Messaggero"** → Matches abstract + §4: absent 26 July, regime change 25–27 July, war-vs-domestic across month ✓
- **§4 intro says "three metrics...not symmetric"** → Confirmed by Table §4 (tool calls, recall/MAE, quality) ✓

**Verdict: PASS** — No broken cross-references detected. All chapter/section/appendix/figure references resolve.

---

## 5. CITATION CHAIN

**Spot-check: Every in-text (Author, Year) has bibliography entry; every bib entry is cited at least once.**

| Citation | In-text location | Bibliography entry | Cited? |
|---|---|---|---|
| (Bartlett, 1932) | §1, §2, Abstract | ✓ Bartlett, F.C. (1932) | ✓ Multiple |
| (Miller, 1956) | §1, §2 | ✓ Miller, G.A. (1956) | ✓ |
| (Cowan, 2001) | §1, §2 | ✓ Cowan, N. (2001) | ✓ |
| (Tolman, 1948) | §1, §2 | ✓ Tolman, E.C. (1948) | ✓ |
| (Eichenbaum, 2017) | §1, §2 | ✓ Eichenbaum, H. (2017) | ✓ |
| (Whittington et al., 2020) | §1, §2 | ✓ Whittington, J.C.R. et al. (2020) | ✓ |
| (Sarthi et al., 2024) | §1, §2 | ✓ Sarthi, P. et al. (2024) | ✓ |
| (Edge et al., 2024) | §1, §2 | ✓ Edge, D. et al. (2024) | ✓ |
| (Zhang and Tang, 2025) | §1, §2 | ✓ Zhang, M. and Tang, J. (2025) | ✓ |
| (Murugaraj et al., 2025) | §2 | ✓ Murugaraj, M. et al. (2025) | ✓ |
| (Bai et al., 2025) | §3 | ✓ Bai, S. et al. (2025) | ✓ |
| (Wu et al., 2021) | §3 | ✓ Wu, J. et al. (2021) | ✓ |
| (Ketelaar, 2001) | §2 | ✓ Ketelaar, E. (2001) | ✓ |
| (International Council on Archives, 2000) | §2 | ✓ International Council... (2000) | ✓ |
| (Ehrmann et al., 2020) | §2 | ✓ Ehrmann, M. et al. (2020) | ✓ |
| (Düring et al., 2024) | §2 | ✓ Düring, M. et al. (2024) | ✓ |
| (Schudson, 1978) | §2 | ✓ Schudson, M. (1978) | ✓ |
| (Murialdi, 1986) | §2, §5 | ✓ Murialdi, P. (1986) | ✓ |
| (Yao et al., 2022) | §3 | ✓ Yao, S. et al. (2022) | ✓ |
| (Lewis et al., 2020) | §3 | ✓ Lewis, P. et al. (2020) | ✓ |
| (Doucet et al., 2020) | §3 | ✓ Doucet, A. et al. (2020) | ✓ |
| (Behrens et al., 2018) | §2 | ✓ Behrens, T.E.J. et al. (2018) | ✓ |
| (Robertson and Zaragoza, 2009) | §2 | ✓ Robertson, S. and Zaragoza, H. (2009) | ✓ |
| (Salton, Wong and Yang, 1975) | §2 | ✓ Salton, G. et al. (1975) | ✓ |
| (Craik and Lockhart, 1972) | §2 | ✓ Craik, F.I.M. and Lockhart, R.S. (1972) | ✓ |
| (Pavone, 1991) | §1, §4.1 | ✓ Pavone, C. (1991) | ✓ |
| (Bosworth, 2005) | §4 (ground truth) | ✓ Bosworth, R.J.B. (2005) | ✓ |
| (Deakin, 1962) | §4 (ground truth) | ✓ Deakin, F.W. (1962) | ✓ |

**Checking bibliography for orphans:**  
Every entry in References section is cited at least once in body text.

**Verdict: PASS** — Citation chain intact; no orphaned references.

---

## 6. UNDEFINED TERMS

**Scanning for terms used before introduction:**

| Term | First use (section) | Defined? | Issue? |
|---|---|---|---|
| "MinCulPop" | §1 (regime-aligned register) | Implicit (Italian fascist propaganda ministry) | ✓ Clear from context |
| "OCR" | §3 (OCR pipeline) | ✓ Expanded in §3 intro | ✓ |
| "ClickHouse" | §3 (single ClickHouse table) | ✓ Named; schema explained | ✓ |
| "RAPTOR" | §2 | ✓ "RAPTOR (Sarthi et al., 2024)...recursively clustering" | ✓ |
| "GraphRAG" | §2 | ✓ "GraphRAG (Edge et al., 2024)...extracts entity-relation graph" | ✓ |
| "BM25" | §2 | ✓ "BM25, still the strongest sparse baseline" | ✓ |
| "embedding" | §3 | ✓ Used in context (embedding vectors, nearest-neighbour); formal in Appendix A.4 | ✓ |
| "Leiden community detection" | §2 | ✓ "Leiden community detection over the graph" | ✓ |
| "ReAct loop" | §3 | ✓ "ReAct loop in the sense of Yao et al. (2022)" with citation | ✓ |
| "schema" | §1 (cognitive psychology sense) | ✓ Bartlett cited; used consistently throughout | ✓ |
| "episodic" vs "semantic" memory | §2 | ✓ "episodic memory, bound to time and place, from semantic memory" | ✓ |
| "cognitive map" | §1, §2 | ✓ "Tolman (1948) showed that rats build cognitive maps" | ✓ |
| "ground truth" / "GT" | §4 | ✓ "hand-built relevance ground truth...from four works of historiography" | ✓ |
| "token-bloom filter" | §3, Appendix | ✓ "token-bloom-filter index over the summary column" | ✓ |
| "ISAD(G)" | §2 | ✓ Expanded: "archival principle respect des fonds (International Council on Archives, 2000)" | ✓ |

**Verdict: PASS** — No undefined terms. Technical terminology introduced with citation or clarification.

---

## 7. WORD COUNT

**Stated:** 9,431 words  
**Cap:** 10,000 words  
**Status:** 9,431 < 10,000 ✓

**Verdict: PASS** — 569 words under cap.

---

## 8. NUMERICAL/FACTUAL SELF-CONSISTENCY (Detailed)

### Tool-call means
- Abstract: "11.3 tool calls against...28.3"
- Calculated: (13.3 + 12.3 + 8.3) / 3 = 33.9 / 3 = **11.3**
