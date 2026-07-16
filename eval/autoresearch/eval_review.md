# Eval Reward-Hacking Review — 2026-07-16

Adversarial audit of `src/mausoleo/eval/evaluate.py` (v1 composite). All exploits demonstrated numerically on 1910-06-15 GT. Result: three confirmed exploit classes → **composite_v2 implemented** (same weights, three targeted changes). v1 remains available as `composite_v1_score` for comparing against historical log entries.

## Confirmed exploits (v1)

1. **CRITICAL — cherry-picking.** wCER/mean_cer averaged over MATCHED articles only: dropping your own worst 10% of articles (by the repo's own GT-free `quality_score`) gained +0.064 composite (30x the acceptance floor) while producing strictly less OCR. Degenerate limit: one clean article scored 0.6511.
2. **CRITICAL — spam is free.** No precision term: +2,000 fabricated articles changed composite by 0.0000; noisy 3x duplication RAISED recall 0.979→0.990. Production ensemble's 837 preds vs 193 GT (precision 0.226) is this exploit institutionalized.
3. **HIGH — garbage floor ~0.60.** Aggregate min(wCER,1) cap + Jaccard matching: repetition-padded matches (per-article CER 10) still earn full structure credit; a GT-word-skeleton graft gained +0.11 while wCER worsened.
4. **MEDIUM — degenerate edges.** ordering=1.0 for <2 matches (empty prediction scored 0.15); config averages computed only over dates with predictions (failing your bad date inflates the average).
5. **Verified clean:** no GT leakage — no pipeline code reads eval/ground_truth; rsync excludes it from ripperred.

## composite_v2 (implemented 2026-07-16)

- wCER over ALL GT articles, per-article cap min(cer,1.0) (unmatched = 1.0) — kills cherry-picking, removes the aggregate-cap cliff
- F1 replaces recall — kills spam
- ordering = 0.0 when <2 matches — kills empty/single degenerates
- hCER per-article capped at 1.0

Verified: spam now −0.061; drop-worst-10% now −0.046; empty → 0.000. Rank order of existing configs unchanged; gaps compress (ensemble_30min 1910: 0.9454 v1 → 0.7892 v2, its lead over exp_045 shrinks +0.27 → +0.07 as spam gets charged).

## Re-based reference scores (v2)

| config | dates | v1 | v2 |
|---|---|---|---|
| ensemble_30min | 1910 only | 0.9454 | 0.7892 |
| exp_045_qwen3vl_vllm | both | 0.5372 | 0.6305 |
| exp_148_paddleocr_yolo | 1885 only | 0.3020 | 0.3987 |
| exp_149_paddleocr_yolo_headline | both | 0.2878 | 0.3973 |

All pre-2026-07-16 numbers in log.jsonl are v1. From now on log composite (v2) and note v1 only when comparing to history.

## Protocol additions (also in program.md)

1. Report precision/F1 alongside composite; accepted changes that drop precision >5pts need explicit justification.
2. Holdout rule extends to structural changes that filter/drop articles, not just hyperparameters.
3. Pipeline code must never read `eval/ground_truth/*/ground_truth.json` at inference, nor re-emit another config's `eval/predictions` file as its own output.
