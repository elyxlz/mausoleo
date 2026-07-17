# Loop state — live queue for the autoresearch loop

Rewritten by the loop every iteration (see program.md §Running the Loop). Rules live in `program.md` (+ `registry.md`); this file is only the CURRENT state and queue.

## Standing context
- Run everything locally on ripperred (`.venv/bin/python`); GPU1 preferred (`CUDA_VISIBLE_DEVICES=1`, eve holds ~350MB on GPU0).
- Fable spend limit hit → spawn subagents with `model: "opus"`.
- Elio's decisions (final): eval GT set = 6 issues; NO issue-level holdout; 1925 Il Meridiano accepted.
- Tentative GTs (1895/129u, 1925/108u, 1935/256u, 1952/197u) in `eval/tentative_gt/` — awaiting Elio's review; do NOT promote.
- plan/01 ship bar + plan/02 corpus-v0-early await Elio's sign-off — no corpus run without it.
- Commit and push as you go; log to `log.jsonl` with mechanism lines; update `registry.md` every iteration.

## State (2026-07-17 14:35)
- **Production candidate: exp_160** (experiments/exp_160_ppdoclayout_headblocks.py) = 0.6180 avg, F1 0.75/0.68, precision 0.82/0.77, hCER 0.30/0.33, steady-state 6.22 GPU-s/page = 6.3-day corpus (under 1-week goal). Chain: exp_158 (PP-DocLayoutV3 regions, recall signal) → exp_159 (title-boundary grouping, 0.6040) → exp_160 (head-block merge).
- Ship-bar gap (plan/01, proposed): composite ≥0.60 ✓ (on the 2 verified issues), recall ≥0.70 ✗ (0.683/0.611) — recall is the open front.
- GPUs free.
- **Zoom-refinement pass DONE (2026-07-17 16:00)**: all four tentative GTs re-verified flag-by-flag at 2-14x zoom — 1895 (5 fixes), 1925 (5), 1935 (14), 1952 (7 + 1 missing ad added). Each REVIEW_NOTES has a 'Resolved by zoom pass' audit; remaining uncertain items are marked. Drafts are as clean as machine passes get — ready for Elio.

## Queue (in order)
1. **F4 (oracle-only)**: add exp_160 as diversity source to ensemble_prune5 — offline merge sweep over cached predictions (pattern: the 2026-07-16 prune search scripts in scratchpad, or rewrite small): try exp_160 in the replacement chain at several (overlap, ratio) points + as quality-select source; accept per ensemble floor (≥0.002 both dates + holdout). Then prune5 precision filtering.
2. **Recall via semantic grouping (F3 unblock candidate)**: LLM/subagent grouping pass over PP-DocLayoutV3 region texts+geometry (group region indices into articles). Only prototype if F4 stalls — model-based, era-independent.
3. **Over-split lever**: title score threshold sweep (only if evidence appears).
4. On Elio's GT promotion: board over 6 issues; re-base baselines; check ship bar (composite ✓ 0.618, recall 0.68/0.61 vs proposed 0.70).
