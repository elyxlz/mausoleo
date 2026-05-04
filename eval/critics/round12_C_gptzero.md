# R12_C GPTZero — REVERTED

## Setup
- Surgical single-sentence anti-template (line 72 bridge sentence)
- Replaced "The case studies in chapter four ask whether the prediction implied by these three strands of cognitive science shows up in the metrics." (28w)
- With "Chapter four tests the framing on three corpus questions and reports the tool-call counts and rubric scores." (18w)

## GPTZero
- Pre: 0.7600 / 0.7603 (R10_C / R0_C resume)
- Post: 0.7611 (identical 12-dec to R11_C post-edit, scan-cache lock at this regime)
- Δ = +0.0008 noise

## GAN Pareto (round 111, master_seed=111700, positions 3/4/5)
- seed=598332488 pos=4 → critic picked Essay 1 (NOT TARGET) → **PASS**
- seed=633365989 pos=3 → critic picked Essay 3 (TARGET) → **FAIL**
- seed=970381379 pos=5 → critic picked Essay 5 (TARGET) → **FAIL**
- Result: **1/3 PASS** (regression from R10_C 3/3 PASS, but +1 vs R11_C)

## Critical finding
pos5 critic explicitly flagged the R12_C edit (line 72 replacement) as Tell #7 [STRUCTURAL]:
> "Chapter four tests the framing on three corpus questions and reports the tool-call counts and rubric scores." — "Human dissertations rarely announce coming content with such mechanical precision; the proleptic summary structure is an AI hallmark."

The replacement was MORE AI-typical than the original. The "tests / reports" verb pair + concrete-noun stack reads as algorithmically-generated specificity.

## Verdict — REVERT

R10_C zero-movement-KEEP precedent does NOT apply here because R10_C's edit removed a 32w R103-flagged Tell #2 ("There is an episodic-memory analogue: ...") and replaced it with a less-templated alternative ("Episodic memory works the same way at the cognitive level: events whose temporal slot..."). R12_C edit replaced one templated bridge with a different templated bridge.

## Lessons (cumulative R11_C + R12_C)
1. The R8_C 3/3 PASS state is NOT robust. Two consecutive surgical edits both regressed GAN by 1-3 PASS.
2. GPTZero is **scan-locked at 0.7611±0.001** for surface edits in this prose-corpus class.
3. The locked GPTZero floor implies surface-only Strategy 3b is exhausted at this layer.
4. Structural rewrites are the only remaining path to push GPTZero below 0.74. But structural rewrites of comparable scope (R6_C +189w preface) historically RAISE the score.
5. Strategy 4 (AuthorMist Qwen2.5-3B) remains untested in Stage C and is the documented "strongest known move" per skill memo, with cache deletion meaning ~10G re-download.

## Files
- /tmp/mausoleo/eval/critics/round12_C_plan.md
- /tmp/gan_round111/seed_598332488_pos4/verdict.md (PASS)
- /tmp/gan_round111/seed_633365989_pos3/verdict.md (FAIL)
- /tmp/gan_round111/seed_970381379_pos5/verdict.md (FAIL — flagged the edit)
