# Loop state — live queue for the autoresearch loop

Rewritten by the loop every iteration (see program.md §Running the Loop). Rules live in `program.md` (+ `registry.md`); this file is only the CURRENT state and queue.

## Standing context
- Run everything locally on ripperred (`.venv/bin/python`); GPU1 preferred (`CUDA_VISIBLE_DEVICES=1`, eve holds ~350MB on GPU0). paddle_env at `~/paddle_env` (PP-DocLayoutV3).
- Fable spend limit may still bite → spawn subagents with `model: "opus"` if a fable agent errors on limit.
- **Eval is now 6 human-verified issues** (1885, 1895, 1910, 1925, 1935, 1952-06-15), all units `unit_type=article`. research.py EVAL_DATES updated. Tentative GTs promoted; `eval/tentative_gt/` removed.
- Review server (scripts/review_server.py) serves the real GTs on :8077 via the trycloudflare tunnel.
- Commit and push as you go; log to `log.jsonl` with mechanism lines; update `registry.md` every iteration.

## State (2026-07-21)
- GT set expanded 2→6. Predictions on the 4 new issues already exist for: exp_157, exp_045, ensemble_30min, ensemble_prune5. Running now: **exp_160 on the 4 new issues** (task blhk7cn7n) to complete the production candidate's 6-issue coverage.
- **Fable eval-review agent running** (a9c3a7873bcee2b4c) — auditing evaluate.py/composite_v2 and proposing improvements; on completion save its report to `eval/autoresearch/eval_review_2.md`, commit, and fold any accepted changes into program.md (metric changes only via documented audit; keep composite_v2 comparable, add v3 alongside if changing).
- Prior production candidate exp_160 was 0.6180 on the OLD 2-issue eval; must be re-measured on 6.

## State (2026-07-21 11:35) — re-baselined
- 6-issue composite_v2: ensemble_30min 0.7091 (recall oracle), ensemble_prune5 0.6917, exp_045 0.6087, **production exp_160 0.5756** (was 0.6180 on 2). Bottleneck = segmentation; exp_160 recall 0.37-0.66, worst on dense classifieds (1952 comp 0.414).
- Cheap grouper approaches are plateaued: geometric 0/3, local-LLM grouper 0/2 (all on 2 issues). Opus-grouper ceiling 0.698 (2 issues). Elio authorized training.
- Region dumps (regions_<date>.json = PP-DocLayout regions + PaddleOCR-VL text) running for the 4 new issues (task bb3af4bc8); 1885/1910 already dumped.
- Fable eval-review agent (a9c3a7873bcee2b4c) still auditing.

## Queue (in order)
1. **Trained boundary grouper (F3 escalation, top priority)** — the recall bottleneck, budget-friendly design: frame grouping as per-region sequence labeling "does region i START a new article?" Steps:
   (a) Align GT articles -> region indices for all 6 issues by text overlap (GT article text vs concatenated region texts) to label each region's start/continue and whether it's a headline region. Build a features+labels table: region class (title/text), bbox geometry (column index, normalized y-gap to prev region, x-overlap with prev, page position), short text signals (length, starts-uppercase, has-dateline), + optional text embedding.
   (b) Train a small classifier (gradient-boosted trees or a tiny MLP — NOT an 8B LLM; inference cost ~0 so OCR dominates the budget). Cross-validate by-issue (train on 5, test on held-out issue) to avoid overfitting the 6.
   (c) Wire as experiments/exp_167: exp_160 pipeline but grouping = classifier boundaries + head-block merge. Evaluate all 6 issues + probe; compare to exp_160 0.5756 and the exp_164 ceiling.
2. **Complete the ceiling/rejected-approach re-measurement on 6 issues** (parallel, cheap): run exp_159 on 4 new issues; re-run exp_164 grouping only if we want the true 6-issue ceiling (Opus, expensive — optional). Confirm exp_161/162/165/166 verdicts hold on 6 using the new region dumps.
3. Fold eval-review recommendations (eval_review_2.md) into the metric/protocol before heavy climbing.
4. Ship-bar re-check on 6 issues.
