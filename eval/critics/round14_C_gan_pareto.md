# R14_C GAN Pareto — 0/3 PASS (REVERT trigger)

## Setup
master_seed=1777895593 (epoch), positions 5/6/4 (all in safe 2-6 zone, BANNED 1).

## Verdicts

| seed | pos | TARGET slot | critic picked | confidence | result |
|---|---|---|---|---|---|
| 641402409 | 5 | Essay 5 | Essay 5 | NEAR_CERTAIN | **FAIL** |
| 333046314 | 6 | Essay 6 | Essay 6 | NEAR_CERTAIN | **FAIL** |
| 856460033 | 4 | Essay 4 | Essay 4 | LEAN | **FAIL** |

## Result

**0/3 PASS** — hard regression below R8_C 3/3 baseline.

## Tells flagged

CRITICAL FINDING: across all 3 critic verdicts, the flagged spans cite:
- §1 ¶4 cog-sci paragraph (Bartlett, Miller, Cowan, Tolman, Eichenbaum tells) — UNTOUCHED in R14_C
- §5 case-study disclaimer ("What I have not shown:") — UNTOUCHED in R14_C
- Italian block quote (`[edizione assente]`) — UNTOUCHED in R14_C
- Preface ("Two weeks went into a post-correction pass") — UNTOUCHED in R14_C
- Em-dash clause insertion in cog-sci section — UNTOUCHED in R14_C

NONE of the flagged spans cite the §3 line 99 ClickHouse paragraph that R14_C edited. The R14_C paraphrase is invisible to the critic verdicts. The 0/3 hard regression suggests **R8_C 3/3 PASS was seed-fragile**, NOT robust under any perturbation including this AuthorMist surgical edit on a non-cohort-fit zone.

## Decision: REVERT

Per protocol "If GAN drops below 2/3, REVERT". R14_C edit reverted; v112 file restored to R8_C state. Word count back to 9092w.

## Diagnostic next step

Re-run GAN Pareto on REVERTED v112 with fresh seeds (baseline_v112_recheck) to determine:
- If baseline ALSO = 0/3 → R8_C 3/3 was seed-fragile fluke, GAN ceiling is actually ~0/3 at v112 prose, and R14_C-revert decision was driven by a critic that flags this draft regardless of edit.
- If baseline = 2/3 or 3/3 → R14_C edit specifically broke GAN despite touching no cohort-fit zones, which would be an unexpected finding.

## Files
- /tmp/gan_round14_C/seed_641402409_pos5/verdict.md (FAIL Essay 5)
- /tmp/gan_round14_C/seed_333046314_pos6/verdict.md (FAIL Essay 6)
- /tmp/gan_round14_C/seed_856460033_pos4/verdict.md (FAIL Essay 4)
- /tmp/gan_round14_C/PARETO_SUMMARY.md
