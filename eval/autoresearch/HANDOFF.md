# Session Handoff — 2026-07-16

Continuation point for the autoresearch loop running on ripperred (tmux). Read `program.md` (objective, budget, protocol), `registry.md` (approach families), and the last entries of `log.jsonl` first. Work the loop per `.claude/skills/ocr-autoresearch/SKILL.md`; self-pace with ScheduleWakeup.

## Immediate verification queue (do before new experiments)
1. **Ensemble rewrite equivalence**: `ParallelEnsembleOcr` was rewritten (inline sub_configs + dynamic GPU queue + subprocess isolation via `scripts/run_sub_pipeline.py`); `configs/ocr/ensemble_30min.py` is now self-contained. Merge-only rerun on cached predictions completed for both dates, but byte-equivalence was only backed up for 1910 (`/tmp/ens_1910_before.json` on ripperred). Compare it against `eval/predictions/ensemble_30min_1910-06-15.json` — articles must be identical. Also verify a fresh sub-pipeline run works end-to-end (delete one cached sub prediction, rerun, watch the queue schedule it).
2. **1 failing test**: `pytest tests/test_ocr_pipeline.py tests/test_all_pipelines.py` on ripperred showed "1 failed, 15 passed" (before dead-operator removal). Identify and fix; also rerun after the operator deletions (ensemble_ocr, sub_pipeline, merge_ensemble, page_pairs_vlm, chandra_layout, llm_post_correct were deleted — archive configs referencing them will no longer import, which is accepted).
3. **Eval reward-hack review**: an adversarial review of `src/mausoleo/eval/evaluate.py` was in flight at handoff. Key suspicions to verify and fix: composite has NO precision term (article spam is free — 837 preds vs 193 GT on 1910), `min(wCER,1)` cap makes CER-20 blob matches cost the same as CER-1, greedy Jaccard matching rewards giant articles. If confirmed: define `composite_v2` (e.g. F1 replacing recall; document rationale), recompute the leaderboard (`scripts/research.py board`), record both scores during transition. Never silently change historical numbers.

## Experiment queue (registry priorities)
- **exp_045 steady-state throughput**: benchmark Qwen3-VL-8B col3 vllm GPU-s/page (2 issues back-to-back, exclude load) — the production reference vs the 6.9–13.9 budget.
- **olmOCR-2-7B native prompt + MergeMarkdownPages** (cached on ripperred, F1 family, untested with correct usage).
- **Unlimited-OCR column-crop sequence** (F2 unblock condition: legible crops as page sequence via `scripts/run_unlimited_standalone.py`, `~/unlimited_env`).
- **PaddleOCR-VL segmentation**: recall/headlines are the bottleneck (registry F1). Try YOLO title-class regions for headlines instead of first-line heuristic.

## Constraints (already in CLAUDE.md/program.md — enforced)
- Budget 6.9–13.9 GPU-s/page; record timing every run (`scripts/research.py run <config>` does run→fetch→eval→audit→runs.jsonl).
- One config → one run → one result. All compute on ripperred. One variable per experiment. Log everything with mechanism lines; update registry.md every iteration; never stop the loop on failed waves.
