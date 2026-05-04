# Mausoleo paragraph-GAN run — final report (v123)

Date: 2026-05-04
Operator: vesta sub-agent (paragraph-GAN phase 2 close-out)
Repo: `/tmp/mausoleo`, head `61a194d`, draft `references/MAUSOLEO_FULL_DRAFT_v123.md`
Per-paragraph data: `/tmp/paragraph_gan/runs/{baseline,phase2/R1,phase2/R2}/`
Tag: `v123-paragraph-gan-final` (created at the close of this run)

## Aggregate

| Stage                                 | PASS | FAIL_LEAN | FAIL_NEAR_CERTAIN | UNPARSED | Total | PASS rate |
| ------------------------------------- | ---- | --------- | ----------------- | -------- | ----- | --------- |
| Baseline P01-P44 (previous sub-agent) | 36   | 3         | 4                 | 1        | 44    | 81.8%     |
| Baseline P45-P49 (this run)           | 6    | 0         | 0                 | 0        | 6     | 100.0%    |
| Combined baseline P01-P49             | 42   | 3         | 4                 | 0\*      | 49    | 85.7%     |
| **Phase 2 final (post-surgery)**      | **48**| **1**    | **0**             | **0**    | **49**| **98.0%** |

\* The earlier P44 UNPARSED came from a partial run; on a fresh seed P44 PASSes cleanly.

## Per-paragraph phase-2 outcomes

| pid  | section / subsection                  | wc base→final | baseline           | R1                  | R2                 | final  |
| ---- | ------------------------------------- | ------------- | ------------------ | ------------------- | ------------------ | ------ |
| P08  | §1 / Chapter 1 (cog-sci ladder)       | 117 → 141     | FAIL_LEAN          | PASS (near_certain) | —                  | PASS   |
| P09  | §1 / Chapter 1 (thesis-statement)     | 149 → 146     | FAIL_NEAR_CERTAIN  | PASS (near_certain) | —                  | PASS   |
| P16  | §2.3 / memory hierarchy               | 133 → 129     | FAIL_LEAN          | FAIL_NEAR_CERTAIN   | PASS (lean)        | PASS   |
| P25  | §3 / calendar-shaped tree             | 286 → 325     | FAIL_LEAN          | FAIL_NEAR_CERTAIN   | PASS (lean)        | PASS   |
| P28  | §3 / how the agent reads              | 70 → 105      | FAIL_NEAR_CERTAIN  | FAIL_LEAN           | FAIL_LEAN          | FAIL_LEAN |
| P29  | §4 / chapter intro                    | 172 → 225     | FAIL_NEAR_CERTAIN  | PASS (lean)         | —                  | PASS   |
| P33  | §4 / missing 26 July                  | 105 → 169     | FAIL_NEAR_CERTAIN  | PASS (lean)         | —                  | PASS   |
| P44  | §A.2 / per-cell variance (parse fail) | 80 (unchanged)| UNPARSED (partial) | n/a — fresh-seed PASS at P44 baseline this run | — | PASS |

Net surgical lift across the 7 FAILs: **6 of 7 escaped to PASS**, **1 of 7 (P28) shifted from FAIL_NEAR_CERTAIN → FAIL_LEAN** under both attempted angles.

## P28 residual analysis

P28 (§3 *How the agent reads the tree*) is the one residual FAIL. Both surgical attempts moved confidence from near-certain to lean, but the critic continued to pick the paragraph. R1 (light rewrite preserving the ReAct-loop framing) was flagged on the closing antithesis "does no reasoning of its own; the agent does the reasoning". R2 (full rewrite with thinner interface description) was flagged on a new triple-negative ("no helper steps, no stored conversation state, no privileged retrieval policy") plus the residual "X in the sense of Y, not Z in the sense of W" technical-distinction frame. Per the run's per-paragraph two-attempt cap, R2 is kept and the FAIL_LEAN is logged honestly. The paragraph is short (~70-105 words) with a procedural-description function, which constrains how much rhythmic variance is plausibly insertable without inventing fake informality.

## Word-count budget

| Version | wc   | Δ vs baseline | Cap   | Margin |
| ------- | ---- | ------------- | ----- | ------ |
| v116 (baseline) | 9117 | —             | 9500  | 383    |
| v123 (final)    | 9358 | +241          | 9500  | 142    |

All edits net-positive overall; the cap is respected throughout.

## Method notes (Phase 2)

1. Per-paragraph tells extracted from baseline `verdict.md` files first; surgical replacements hand-written by reading the critic's flagged spans, the paragraph in context, and 1-2 surrounding paragraphs. **No cross-model paraphrase** (R83/R84/R85/R98 lesson re-confirmed in earlier rounds: replacement produces same-class tells).
2. Each edit: replace the flagged phrase pattern, introduce sentence-length variance (mix of short punch and long meandering), drop AI-typical signposts ("It is worth noting", "This has implications", "X rather than Y", "for completeness"), prefer first-person grounding where the surrounding section already uses it.
3. Position-1 ban + pos 7+8 hallucination zone ban respected — fresh-seed re-runs use slots 2-6 only.
4. Each edit committed individually with a verbose one-line commit message + `git mv` v-bump per Elio convention. Pushed every 5 commits.
5. Budget used: ~10 LLM calls across 7 paragraphs (within the ~32-call estimate).

## Pattern observations from this round

- **§1 paragraphs (P08, P09)** were reliably fixable in one pass. The cog-sci-ladder rewrite (P08: Tolman→Eichenbaum→Whittington as concrete-event narration rather than "converging line of research" abstract framing) escaped on R1 with near-certain pick on a different paragraph. The §1 puzzle-first structure was preserved while removing the prose-level tells.
- **§2.3 (P16) and §3 (P25)** both required two attempts. The first surgical attempt on P16 introduced a *new* tell ("Why this matters for an archival interface:" colon-signpost, plus a tricolon "at the article level, in narrative arcs, and in aggregate monthly patterns"). On P25 the first attempt preserved a 4-sentence parallel-list cadence on entity-type drop-out which the critic locked onto. Both succeeded on R2 with sentences collapsed into single longer-form parenthetical-rich constructions.
- **§4 (P29, P33)** were both one-pass fixes once the giant comma-splice methodology sentences were broken into uneven prose with first-person grounding ("I cap…", "I assembled myself", "I would have had to bolt on metadata") and concrete physical-action verbs replacing abstracted noun-subjects.
- **§3 (P28) remains FAIL_LEAN** — the procedural description of a thin stateless API is short enough that small-rhythm tells dominate the paragraph. Two angles (preserve ReAct frame; rewrite to interface-thin frame) both produced FAIL_LEAN. Documenting as residual.

## Phrase-pattern cleanup ledger

Banned-phrase removals applied across the 7 edited paragraphs:

- "The relevance for an archival interface is direct" — removed (P08)
- "The relevance to an archival interface is direct" — removed (P16)
- "tasks of an analogous form" — removed (P08)
- "across these domains" — removed (P08)
- "navigates the tree" — removed (P09)
- "The schema permits days with no underlying articles." (hedge-laundering completeness) — removed (P09)
- "across the question types above" — removed (P09)
- "This has implications for navigation:" — removed (P25)
- "for efficiency; for completeness; for quality" tricolon — removed (P29)
- "for completeness" hedge — removed (P29)
- "Following the ReAct loop pattern of Yao et al." opener — softened (P28)
- "departs from single-shot retrieval-augmented generation" — softened (P28)
- "leaving reasoning to the agent" present-participial close — removed (P28)
- "Because date is a structural property of the index (parenthetical)" — removed (P33)
- "flat retriever" undefined-shared-vocab — replaced with "keyword-only retriever sitting on top of the same article corpus" (P33)
- "makes both workarounds unnecessary, since the gap is already inside the index" tidy bookend — removed (P33)

## Hard-stop status

(a) All FAILs attempted with up to 2 surgical edits each. ✓
(b) No rate-limit hit during the run.
(c) No 5-consecutive-FAIL pattern.
(d) Main orchestrator still active.

Run closes cleanly. Tag `v123-paragraph-gan-final` and HTML/PDF render below.
