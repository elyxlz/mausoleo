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

## Queue (in order)
1. **Re-baseline on 6 issues**: when exp_160 new-date run finishes, `scripts/research.py board` over all 6. Also run exp_159 on the 4 new issues (same pipeline, cheaper grouping) for comparison. Re-base program.md "Current Baselines" table + registry with 6-issue composites. Log the re-baseline with a mechanism line.
2. **Retest previous approaches on the new eval**: the geometric/grouping family (exp_161/162/163) and the local-grouper family (exp_165/166) were rejected on 2 issues — re-confirm their verdicts hold on 6 (they reuse exp_164 region dumps; dump the 4 new issues first via `experiments/exp_164_ppdoclayout_semgroup.py dump <date>`). Ensemble F4 prune re-verify on 6.
3. **Resume hillclimbing** the recall/segmentation bottleneck (registry F3). The Opus-grouper CEILING was 0.6980 (exp_164) — the open prize. Local-grouper route rejected 0/2 (exp_165/166). **Escalation allowed (per Elio 2026-07-21): if rule/prompt grouper approaches plateau, TRAIN a small grouper** — fine-tune a compact model on (region-listing → article-grouping) pairs derived from the 6 promoted GTs (align GT articles to PP-DocLayout regions by text overlap to build labels; ~900 units of supervision). Keep OCR (PaddleOCR-VL) fixed; train grouping only. Measure the trained grouper's GPU-s/page vs the 6.9-13.9 budget.
4. Incorporate the eval-review recommendations once saved and sanity-checked (may reshape the loop's target — do this before heavy hillclimbing so we climb the right hill).
5. Ship-bar re-check on 6 issues (plan/01) after re-baseline.
