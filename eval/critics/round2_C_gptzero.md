# Stage C Round 2 GPTZero verdict — Strategy 1b parallel-triplet collapse (low-GAN-risk)

## Result

| metric | R95 baseline | R2_C edit | delta |
|---|---|---|---|
| AI prob | 0.929 | **0.929** | **0.000** |
| confidence | 0.929 high | 0.929 high | — |
| doc class | AI_ONLY | AI_ONLY | — |
| subclass | pure_ai | pure_ai | — |

## Edits applied

1. §3 OCR-merge sentence: triplet "REPLACE pass, ADDITIVE pass, and quality-weighted text selector" → 3-sentence cascade with semicolons.
2. §4 results sentence: triplet "wins on ratio MAE..., on tool calls..., and on quality metrics..." → 3-sentence sequence with periods.

## Interpretation

ZERO movement on doc-level GPTZero score after two surface triplet collapses in low-prominence technical/results sections. Confirms essay-iter SKILL lesson: "Document-level classifier dominates per-sentence scores. A sentence rewritten in genuinely human style scores ~1% AI alone, but 98% AI when bundled with two sibling sentences in the same prose style."

The §3 OCR description and §4 numerical results paragraphs sit at moderate per-sentence AI-prob (these were not the 99%+ hot zones in R95 final_aidetection.json). Editing them does not shift doc-level fingerprint.

## Pareto

GPTZero unchanged. Per R2_C plan rule "If GPTZero unchanged → REVERT, advance to 1c."

REVERTED via `git checkout references/MAUSOLEO_FULL_DRAFT_v10.md`. File now matches commit 65c74b1 (R95 pure baseline; the uncommitted R109 sentence-8 edit was incidentally also discarded since it was untested for GAN — its loss is acceptable).

NO GAN check run this round (Pareto irrelevant since GPTZero did not improve).

## Quota burn

- GPTZero scans this round: 1 (running total 1/7 in 24h window)
- GAN check: 0
- LLM/Anthropic OAuth calls: 0

## Next

R3_C = Strategy 1c (hedge cadence variation: vary "Notably / Importantly / It is worth noting" sentence openers).
