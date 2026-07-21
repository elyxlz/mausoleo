# Eval Harness Audit #2 — 2026-07-21 (Fable agent)

Audit of composite_v2 against the new 6-issue all-article GT. Motivated composite_v3 (clean-slate restart, per Elio — no dual-metric period). Full findings below; the parent implemented the DO-NOW items as composite_v3.

## Headline findings
- **P0 null-headline crash** (evaluate.py:150) on all-article GT with headline-less units — FIXED (parent, 2026-07-21).
- **Garbage-floor exploit still OPEN in v2**: scrambling word order inside every article (CER→1, word-set Jaccard unchanged → matching identical) only drops composite 1885 0.585→0.478, 1910 0.651→0.517. Half the composite (F1 0.25 + ordering 0.15 + page 0.10) has no text-quality gate. → composite_v3 gates structure credit by match quality max(0,1-cer).
- **Board mixed-denominator ranking** (research.py cmd_board): averages over whichever dates a config has, so exp_164 (2 dates, 0.698) outranks prune5 (6 dates, 0.692) on the live board. → rank only full-coverage configs.
- **Greedy-in-GT-order matcher** steals ~6pp recall (an early GT takes a pred a later GT overlaps more). → best-first global greedy.
- **Ordering is a saturated constant** (min 0.986/median 0.999 across configs) — 0.15 of dead weight. → demote to 0.05, reallocate to F1.

## composite_v3 (implemented)
`0.40·(1−wCER_all) + 0.35·gated_F1 + 0.05·ordering + 0.10·(1−hCER) + 0.10·gated_page`
- match quality q = max(0, 1−cer) per matched article; gated recall/precision/F1 weight each match by q; gated page = Σ(q·page_correct)/N_gt; ordering computed only over good matches (cer≤0.75) so scrambled text yields <2 and ordering→0; hCER charges unmatched-with-headline as 1.0.
- best-first global-greedy matching (sort all (gt,pred,overlap) pairs desc).
- Adversarial invariants (pinned in scripts/eval_probes.py): word-scramble ≲0.10, one-blob-per-page ≲0.25, full-issue blob = 0.

## Protocol (folded into program.md)
- 6-date objective; whole-issue era-stratified holdout as primary anti-overfit tool (even/odd demoted to secondary).
- Effect-size floors kept (0.002/0.005); sign rule restated: 6-date avg ≥ floor, no single date regresses >0.01, no era bucket regresses across its issues.
- Probes cut to one (1943-07-15) — the only unseen-data check now that all 6 GT issues are in the objective.
- `evaluate_all`/`print_results` (dead, hardcoded old dates) removed; audit line adds matched-only mean_cer.

## CONSIDER (not yet done)
- surface full_text_cer in audit (text-present-but-misgrouped vs absent); per-length-bucket recall; page-set Jaccard when cross-page stitching becomes active.
