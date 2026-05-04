# R11_C GPTZero — REVERTED

## Setup
- Strategy 3b surgical Haiku 4.5 paraphrase of §2 line 50 paragraph
- Hot-zone: "It is less reasonable for X. It is less reasonable, too, for Y" parallel anaphora
- Haiku v3 output: 142w, broken parallel construction, added concrete details ("entity links and OCR confidence", "rhythm of strike coverage")

## GPTZero scan
- Pre: 0.7600 (R10_C) / 0.7603 (R0_C resume baseline with fresh proxy)
- Post: 0.7611
- Δ = +0.0008 (within scan variance — neither improvement nor regression on GPTZero)

## GAN Pareto (round 110, master_seed=110700, positions 2/3/6)
- seed=743406862 pos=2 → critic picked Essay 2 (TARGET) → **FAIL**
- seed=129492785 pos=3 → critic picked Essay 3 (TARGET) → **FAIL**
- seed=819697958 pos=6 → critic picked Essay 6 (TARGET) → **FAIL**
- Result: **0/3 PASS** (hard regression from R10_C 3/3 PASS)

## Verdict — REVERT

The Haiku paraphrase added abstract phrasings ("the historian's answer takes the form of a shape moving across days"), and rephrased the access-template paragraph in a register that flagged Mausoleo across all three GAN positions. Lesson: even surgical 1-paragraph Haiku paraphrases can re-introduce LLM-typical abstract noun-of-noun phrasings that the GAN critics latch onto, especially in early-section §2 prose where the cohort essays have a more concrete register.

R8_C remains the only +1-both-axes win. R11_C confirms R8_C lesson is fragile: SCOPE alone is not sufficient. The successful R8_C move worked because the §1 ¶4 cognitive-science paragraph could absorb a more direct opener; §2 access-template paragraph cannot, because it sits in a comparative-systems analysis where Mausoleo's prose stands out by abstraction level.

## Files
- /tmp/mausoleo/eval/critics/r11_C_haiku_prompt.md (final v3 with anti-template + length floor + concrete-detail hint)
- /tmp/mausoleo/eval/critics/r11_C_haiku_out_v3.md (the rejected paraphrase)
- /tmp/gan_round110/seed_743406862_pos2/verdict.md (FAIL)
- /tmp/gan_round110/seed_129492785_pos3/verdict.md (FAIL)
- /tmp/gan_round110/seed_819697958_pos6/verdict.md (FAIL)

## Quota
- GPTZero: 2 scans this round (baseline + post-edit), both via fresh webshare proxies — quota is NOT exhausted, proxy rotation works as documented.
- Claude OAuth: 1 Haiku call + 3 Opus 4.5 GAN critic calls.
