# Final plagiarism verdict — R95 confirmation

Model: claude-haiku-4-5
Elapsed: 29.4s

---

# PLAGIARISM REVIEW: Mausoleo dissertation

## STATEMENT OF LIMITATIONS

I cannot access `/tmp/mausoleo/references/papers/` or verify against the actual 149-PDF corpus. I will flag only passages where I can detect issues **from the draft text itself** — internal inconsistencies, citation format problems, or claims that invite verification against standard sources. I will **not fabricate matches** to papers I cannot read.

---

## PARAGRAPH-LEVEL SCAN

### **1. Chapter 1, Cognitive science section (Bartlett onwards)**

**Text:**
> "Bartlett's *Remembering* (Bartlett, 1932) opens the cognitive-science line that bears on this reading. Bartlett's *War of the Ghosts* studies described how recall reconstructs a generic schema in place of the verbatim event."

**Status:** ✓ PASS — standard reference, correctly cited. The *War of the Ghosts* framing is foundational cognitive-science material.

---

### **2. Chapter 1, Miller / Cowan passage**

**Text:**
> "Subsequent working-memory research established a small functional capacity for the active workspace: roughly seven plus or minus two items in Miller's original framing (Miller, 1956), revised downwards to about four chunks in Cowan (2001)."

**Status:** ✓ PASS — direct attribution to Miller & Cowan. Accurate capsule of canonical results.

---

### **3. Chapter 1, Tolman / Eichenbaum / Whittington section**

**Text:**
> "A converging line of research, from Tolman's (1948) spatial cognitive-map experiments to Eichenbaum (2017) on the hippocampal integration of space, time and conceptual relation, suggests that the same neural machinery handles hierarchical structure across these domains. Whittington et al. (2020) modelled the circuit as a general-purpose relational learner."

**Status:** ✓ PASS — all three cited. Claims are framed as paraphrases of cited work, not as original synthesis.

---

### **4. Chapter 2, Schudson / Murialdi passage**

**Text:**
> "Schudson (1978) treats the social construction of news as itself a historical process; Murialdi (1986) remains the standard treatment of the Italian press across the regime period, documenting the directive system that shaped what could and could not appear in print."

**Status:** ✓ PASS — attributed. Characterisation is standard in media-history discourse.

---

### **5. Chapter 2, RAPTOR / GraphRAG / Zhang summaries**

**Text:**
> "RAPTOR (Sarthi et al., 2024) builds a retrieval tree by recursively clustering chunk embeddings and summarising each cluster, allowing the same query to hit a leaf passage or a higher-level summary depending on its scope. GraphRAG (Edge et al., 2024) takes a different route: it extracts an entity-relation graph from the corpus, runs Leiden community detection over the graph, and writes summaries at each community level."

**Status:** ✓ PASS — paraphrased system descriptions, cited. Not verbatim.

---

### **6. Chapter 2, Murugaraj et al. (2025) critique**

**Text:**
> "Murugaraj et al. (2025) is the closest prior work on retrieval-augmented generation over historical newspapers; they apply a topic-restricted retrieval pipeline to the Impresso Swiss corpus and report improved retrieval relevance over flat retrieval-augmented generation, measured by BERTScore, ROUGE and UniEval. Their pipeline restricts retrieval to documents matching the query's inferred topic, on the assumption that topical coherence between query and retrieved articles is the dominant signal of relevance."

**Status:** ✓ PASS — attributed. The critique ("topically most-relevant articles cluster around…") reads as interpretive, not as plagiarised framing.

---

### **7. Chapter 3, Wu et al. (2021) lineage**

**Text:**
> "This lineage derives from the recursive book summarisation of Wu et al. (2021), which shows that bottom-up summarisation over a fixed branching factor produces coherent abstractions without exceeding any context window."

**Status:** ✓ PASS — cited. Claim is appropriately attributed.

---

### **8. Chapter 3, Ketelaar (2001) archival principle**

**Text:**
> "ISAD(G) names this archival principle *respect des fonds* (International Council on Archives, 2000)."

**Status:** ⚠ **MINOR FLAG: Ketelaar cited earlier but not here.** 

The passage earlier states: "I follow Ketelaar's (2001) treatment of archival description as a tacit narrative, an activation of the source and not a replacement for it." The ISAD(G) citation immediately after should clarify whether the principle comes from Ketelaar, ISAD(G), or both. As written, it's **not incorrect** but leaves attribution ambiguous. No fix needed if intentional; if the *respect des fonds* framing derives from Ketelaar, add: "(Ketelaar, 2001; International Council on Archives, 2000)."

---

### **9. Chapter 5, Pavone (1991) characterisation**

**Text:**
> "Romans reading the paper that morning learned of the deposition before the morning paper would normally have arrived, and registered that no paper had arrived; that registration was part of the experience of the day, in the way Pavone (1991) treats the *interregno* between deposition and 8 September as a historical category in its own right and not a mere gap in the record."

**Status:** ⚠ **MEDIUM FLAG: Paraphrase proximity to Pavone.**

**Issue:** This is a sophisticated historiographic claim attributed to Pavone. Without access to Pavone (1991), I **cannot verify** whether the framing of the *interregno* as "a historical category in its own right and not a mere gap in the record" is Pavone's own language or the author's interpretation of Pavone's argument. 

**Verdict:** If the specific **phrasing** — "not a mere gap in the record" — appears verbatim or near-verbatim in Pavone, this is an **unattributed paraphrase (E-class)** and should be reformulated or block-quoted. 

**Recommended fix:**
- Verify against Pavone's actual text in `/tmp/mausoleo/references/papers/`.
- If close paraphrase: block-quote the passage + page number.
- If loose interpretation: retain but add interpolation signal: "Pavone treats the *interregno* as [my gloss: not a gap in the record but] a historical category…"

---

## BLOCK-QUOTE AUDIT

**No block-quoted passages found** except:

1. **Chapter 4, the 26 July summary node** — flagged as a *quoted span* from generated index content (not a source):
   > [edizione assente: il fondo archivistico…]

   This is correctly presented as **output from the system**, not as a cited source. ✓ PASS.

2. **Chapter 2, Düring et al. quote** — **MISSING ATTRIBUTION**:
   > "transparent generosity" about exposing intermediate structure

   **Status:** ✗ **FAIL — Block-quote rule violated.**
   
   **Issue:** The phrase `"transparent generosity"` appears to be a direct quotation from Düring et al. (2024) but **lacks quotation marks and page number**. The passage reads:
   > "Düring et al. (2024) describe the Impresso interface as one of 'transparent generosity' about exposing intermediate structure"
   
   **Fix required:** Either:
   - Add page number: `Düring et al. (2024: [p. X]) describe the Impresso interface as "transparent generosity"…`
   - OR reformulate as paraphrase: "Düring et al. (2024) emphasise the interface's candour in exposing intermediate structure…"

---

## CITATION INVERSION CHECK

**Scanned for reversed arguments.** None detected. All cited sources are characterized in alignment with standard scholarship (Bartlett on schema, Tolman on cognitive maps, etc.).

---

## COMMON-KNOWLEDGE PHRASINGS CHECK

Phrases like:
- "digitised newspaper corpus"
- "working memory"
- "regime-aligned daily"
- "OCR pipeline"

are standard jargon in digital humanities and cognitive science, properly contextualized, not misrepresented as original. ✓ PASS.

---

## STAGE A BASELINE CONFIRMATION

The preface states: **"Stage A baseline (May 3): PASS"** — awaiting confirmation or dispute.

**From my scan: No fabricated findings, verified against text only.**

---

## VERDICT

**CONDITIONAL PASS with two items for author revision:**

| Item | Class | Severity | Fix |
|------|-------|----------|-----|
| Düring et al. "transparent generosity" phrase | D (unattributed quasi-quote) | **Medium** | Add quotation marks + page number, or paraphrase |
| Pavone (1991) *interregno* framing | E (paraphrase proximity uncertain) | **Medium** | Verify phrasing against source; block-quote if verbatim, signal if interpretive |
| Ketelaar/ISAD(G) dual attribution | Ambiguous | **Low** | Clarify which source introduces *respect des fonds* |

**Summary:** The dissertation demonstrates strong citation discipline overall. The two flagged issues are **not egregious plagiarism** but rather **citation-hygiene gaps** common in final drafts — missing quotation marks and attributional fuzziness on interpretive claims. Both are **readily fixable** before submission.

**Recommendation:** Verify the three flagged passages against `/tmp/mausoleo/references/papers/` and revise citations accordingly. No deeper structural issues detected.
