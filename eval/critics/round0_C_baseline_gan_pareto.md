# Stage C Round 0 baseline — GAN Pareto check on R95+ (commit 36b2df5)

## Setup

master_seed=504001, target_words=6454, positions constrained to 2-6 (R96-R97 lesson).

## Verdicts

| seed | pos | TARGET slot | critic picked | result |
|---|---|---|---|---|
| 124273414 | 2 | Essay 2 | Essay 8 (Artificial Creativity, NEAR_CERTAIN) | **PASS** |
| 306170356 | 5 | Essay 5 | Essay 5 TARGET (NEAR_CERTAIN) | **FAIL** |
| 414656904 | 3 | Essay 3 | Essay 8 (Spanish Transition Films, LEAN) | **PASS** |

## Result

**2/3 PASS_LEAN+** — ties R95=R94=R80 BEST_GAN ceiling.

Note: pos2 critic and pos3 critic each picked Essay 8 (different essays in their cohorts: Artificial Creativity for pos2, Spanish Transition Films for pos3). TARGET passed both.

pos5 FAIL critic flagged STRUCTURAL tells:
- Markdown formatting (## Chapter 1, ### Existing, inline backticks for API endpoints)
- Preface attribution-light (single supervisor thank vs Essay 1's named-person list)
- "Calendar-tree" central conceit
- Three OCR sub-pipelines pattern

These are MOSTLY R95-baseline floor tells already known. The appendix expansion + figures additions did NOT introduce new same-class tells (no Tell #1 keyed on appendix or figures).

## Pareto status

GAN INTACT at R95 ceiling. Stage C edits in subsequent rounds MUST NOT regress below 2/3.

## Files

- /tmp/gan_round0_C/seed_124273414_pos2/verdict.md (PASS)
- /tmp/gan_round0_C/seed_306170356_pos5/verdict.md (FAIL)
- /tmp/gan_round0_C/seed_414656904_pos3/verdict.md (PASS)
