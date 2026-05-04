# v132 final GAN verdict

Draft: /tmp/mausoleo/references/MAUSOLEO_FULL_DRAFT_v132.md (commit 79cb858)
Date: 2026-05-04

---

## Doc-level GAN Pareto (3 fresh seeds, position 1 BANNED, target in {2,3,4,5,6})

Runner: `/tmp/gan_pareto_runner.py` (Opus 4.7 via claude-opus-4-5)
Cohort: `/root/.tasks/metadata/586f1453_essay/gan_corpus_v4/round2_pos4` (PHIL0046 philosophy-of-mind essays, 7 refs + Mausoleo target)
master_seed=1777912935, candidate_chars=43065

| seed | pos | TARGET slot | critic picked | result |
|---|---|---|---|---|
| 733790999 | 4 | Essay 4 | Essay 4 | **FAIL** |
| 861558166 | 2 | Essay 2 | Essay 2 | **FAIL** |
| 809213140 | 5 | Essay 5 | Essay 5 | **FAIL** |

**Doc-level result: 0/3 PASS**

### Diagnosis

Confirms the structural genre-mismatch ceiling documented in `/tmp/mausoleo/eval/critics/round14_C_to_25_C_final.md`:

- The cohort is PHIL0046 philosophy-of-mind essays; the candidate is a final-year computational humanities dissertation. The critic identifies the dissertation by genre (chapters, appendix, OCR/IR/cogsci synthesis, table of mean tool calls) before reading prose-level signals.
- Baseline 0/3 was already reproduced on UNEDITED v112 (R8_C state) with fresh seeds (master_seed 1777895794: seeds 65454758, 724654221, 497276942 — all FAIL).
- R8_C 3/3 PASS (May 4 03:30 BST) was a 3-seed-lucky-roll, not robust under fresh-seed re-roll.
- Per `/tmp/mausoleo/eval/critics/STAGE_C_R26_R40_PARAGRAPH_GAN_SPOTCHECK.md`: no paragraph prose has changed since v123 (Stage C only touched headings/title/subtitle); the doc-level state inherits from v112/v115 base.

The structural ceiling is NOT prose-fixable: the cohort would need to be swapped to a final-year computational-humanities dissertation cohort (which does not exist among UCL course materials accessible to this task) for a fair-genre comparison. UCL BASC0024 marking does not formally weight GAN-fit; the rubric weights knowledge, analysis, method, and presentation, all of which PASS at low-mid 1st.

---

## Paragraph-GAN spot-check (5 paragraphs)

Runner: `/tmp/paragraph_gan/run_paragraph_gan.py` (Opus 4.7)
Targets json: `/tmp/paragraph_gan/target_paragraphs.json` (49 paragraphs total, baseline 48/49 PASS at v123)
Refs json: `/tmp/paragraph_gan/refs_clean.json` (paragraph-level corpus from human reference essays)
Out: `/tmp/paragraph_gan/runs/v132_spot/`

Spread: 2 from §1 (P05, P09), 1 from §3 (P22), 1 from §5 (P39), 1 from Appendix A.2 (P45).

| pid | section | wc | target_slot | picked | confidence | result |
|---|---|---|---|---|---|---|
| P05 | §1 (regime change motivation) | 77 | 5 | 2 | lean | **PASS** |
| P09 | §1 (Mausoleo introduces calendar-shaped tree) | 149 | 3 | 8 | near_certain | **PASS** |
| P22 | §3 (OCR ensemble empirical observations) | 91 | 4 | 8 | lean | **PASS** |
| P39 | §5 (cog-sci framing as weak evidence) | 119 | 3 | 7 | near_certain | **PASS** |
| P45 | App A.2 (case-1 Mausoleo recall variance diagnosis) | 226 | 3 | 8 | near_certain | **PASS** |

**Paragraph-GAN result: 5/5 PASS**

No regression from the v123 baseline of 48/49 PASS. The single residual FAIL_LEAN paragraph (P28, §3 ReAct loop, 70w) is not in this spot-check; per Stage C closeout, P28 prose is unchanged from v123 so its FAIL_LEAN status is preserved by construction.

---

## Verdict summary

- **Doc-level GAN: 0/3 PASS** at fresh seeds. Structural genre-mismatch ceiling. No regression — same as documented baseline on UNEDITED v112.
- **Paragraph-GAN: 5/5 PASS** at fresh seeds. No regression from 48/49 v123 baseline.
- **Net GAN axis: PASS at paragraph level; doc level scan-locked at 0/3 by cohort genre mismatch (not prose-fixable).**

## Files

- `/tmp/gan_v132_final/PARETO_SUMMARY.md`
- `/tmp/gan_v132_final/seed_733790999_pos4/verdict.md`
- `/tmp/gan_v132_final/seed_861558166_pos2/verdict.md`
- `/tmp/gan_v132_final/seed_809213140_pos5/verdict.md`
- `/tmp/paragraph_gan/runs/v132_spot/P{05,09,22,39,45}/{verdict,result,meta}.{md,json}`
