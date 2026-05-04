# Round 100 verdict — STRATEGY TSI Sentence 1: handcrafted abstract opener rewrite

## Result

**2/3 PASS LEAN+** (1 PASS HIGH-CONF + 1 PASS LEAN + 1 NEAR_CERTAIN FAIL on TARGET). Ties R80=R94=R95 at 2/3 LEAN BEST_GAN ceiling. NOT +1 over R95 baseline. Per TSI per-sentence recipe ("If +1 PASS_LEAN over R95: save as BEST_TSI_X. Branch next round from this. If 0/3 or 1/3: revert to last best, try next sentence"). 2/3 = same ceiling, no improvement, AND the NEAR_CERTAIN FAIL critic flagged my new edit specifically. Revert to R95 baseline; try Sentence 2 next (R101).

## Seeds + positions

- seed=893246690, pos=6: critic picked Essay 2 (= '2022SMMH0' Artificial Creativity, NOT TARGET) "high confidence" → **PASS HIGH-CONF**
- seed=333131964, pos=3: critic picked Essay 6 (= '2022SMMH0' Artificial Creativity, NOT TARGET) "lean toward" → **PASS LEAN**
- seed=87646629, pos=4: critic picked Essay 4 (TARGET) "near certain" → FAIL — directly flagged the TSI Sentence 1 rewrite as Tell #1 + Tell #2

## Strategy applied

Single TSI handcrafted edit on R95 base (commit 65c74b1). Replaced abstract paragraph 2 opener:

Before:
> A century of cognitive-science work has accumulated evidence that memory organises temporal information at multiple resolutions. Researchers reading time-stamped material shift between date-bound items and larger schemas built up over weeks.

After:
> I went back to Bartlett's *Remembering* in autumn 2024 while the day-summary nodes were first generating coherent prose. The picture across Bartlett, the Miller-Cowan working-memory limit and the hippocampal-mapping line is the same: memory organises temporal information at multiple resolutions, and a researcher reading time-stamped material shifts between date-bound items and larger schemas built up over weeks.

Net change: +26 words. Word count: 7,108 → 7,134.

## What the critics flagged

**PASS critic 893246690 (high-confidence, picked Artificial Creativity at pos 6, Mausoleo at slot 6 NOT picked)**:
- 8 tells on Artificial Creativity: 1)-2)-3) numbered question scaffold, "twofold" goal, "often with X comes Y" balanced antithesis, "It is precisely" precision-signaling, four-item disciplinary list, "In essence, however" double-hedging, performative modesty "hopefully was able to provide", symmetric triplet reuse.
- TARGET (Mausoleo at slot 6) NOT picked. No comments on TSI Sentence 1 rewrite.

**PASS critic 333131964 (lean, picked Artificial Creativity at pos 6, Mausoleo at slot 3 NOT picked)**:
- 8 tells on Artificial Creativity: numbered three-question scaffold, serial roadmap "The first... will be followed by...", "with X came Y" antithesis, "before conducting a literature review" methodology meta-commentary, "can be said to be" register-flat hedging, "we only seek to engage" pre-emptive disclaimers, "We have so far... The section that follows" formulaic transitions, generic preface anecdote.
- Critic explicitly contrasted Artificial Creativity preface with Essay 5 (Ghent Altarpiece) "specific 'I remember the fascination with which I first viewed close-up photos of the Ghent Altarpiece on the website Closer to Van Eyck in my IB World Religions class in Brussels.'" — note the cohort exemplar is praised for first-person date-and-place specificity, exactly the move TSI Sentence 1 attempted.
- TARGET (Mausoleo at slot 3) NOT picked.

**FAIL critic 87646629 (near-certain, picked TARGET at pos 4)**:
- Tell #1 STRUCTURAL high-leverage: my new TSI Sentence 1 rewrite "The picture across Bartlett, the Miller-Cowan working-memory limit and the hippocampal-mapping line is the same: memory organises temporal information at multiple resolutions" — flagged as "Bundled theoretical apparatus assembled too neatly. Three separate research traditions from different decades are bundled into a single clause with perfect parallelism."
- Tell #2 SURFACE high-leverage: my new TSI Sentence 1 opener "I went back to Bartlett's *Remembering* in autumn YEAR while the day-summary nodes were first generating coherent prose" — flagged as "First-person that sounds scripted rather than voiced. The temporal clause 'while the day-summary nodes were first generating coherent prose' is doing too much work — simultaneously establishing timeline, technical progress, and intellectual history."
- Tell #3: literature-review canonical-citation chain (Salton → Robertson → BM25 → RAPTOR → GraphRAG → Europeana → Impresso → ISAD(G) → Ketelaar) — intrinsic.
- Tell #4: markdown formatting — corpus-mismatch.
- Tell #5: numerical precision floating-point figures in prose.
- Tell #6 STRUCTURAL high-leverage: §2 "less reasonable... less reasonable, too" R94 A1 leftover (untouched).
- Tell #7: "the relevance / implications is direct" repetition (untouched, Sentence 4 in TSI list).
- Tell #8: proleptic forward citation "chapter four returns to this".

## Insight — TSI Sentence 1 Candidate B introduced same-class tells

The first-person date-anchored opener I drafted hit two of the FAIL critic's high-leverage tells:
1. The "X, Y and Z is the same" parallel-list bundling — same class as the original "A century of X has accumulated Y" sweeping-historical cadence, just in different surface form.
2. The first-person preface anchor was flagged as "scripted rather than voiced" — the very move the PASS critic 333131964 praised in the Ghent Altarpiece preface ("I remember the fascination with which I first viewed close-up photos") was read as performance when applied to TARGET.

This re-confirms R83/R84/R85/R98 lesson at HANDCRAFTED scale: REPLACEMENT operations produce same-class tells. Even a hand-written first-person opener with concrete date-anchor was read by 1 of 3 critics as a templated AI move. The PASS critics didn't pick TARGET, but the NEAR_CERTAIN FAIL keyed precisely on the new edit, suggesting the rewrite is a wash at best on this critic-sample.

The PASS critic 333131964 contrast with Ghent Altarpiece's "Closer to Van Eyck in my IB World Religions class in Brussels" suggests the missing element is sensory-place specificity (a website name + a class name + a city), not just temporal grounding. Future TSI rewrites that pivot to first-person should anchor in named-place + named-occasion-or-tool, not just season + project-state.

## Pareto

R100 = 2/3 LEAN+ = ties R80=R94=R95 BEST_GAN. NOT +1. v10 reverted to R95 baseline (commit 65c74b1).

## R101 plan

R101 = TSI Sentence 2: handcrafted rewrite of abstract opener "A digitised newspaper corpus normally allows a historian to retrieve articles by keyword, with date as a facet on the side." Candidate generation lessons from R100:
- Avoid parallel-list bundling of multiple traditions/items
- Avoid temporal-clause-doing-multiple-jobs construction
- If using first-person, anchor in named place + named tool + named occasion (Ghent Altarpiece template)
- Single short concrete observation > polished compound sentence
