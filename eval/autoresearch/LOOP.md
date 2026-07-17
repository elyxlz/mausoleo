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

## Queue (in order)
1. **Recall inspection (no GPU)**: diff exp_160's 1910 predictions vs GT — list the ~75 unmatched GT units by unit_type/page. Hypotheses: ads/classifieds regions dropped by TEXT_LABELS filter (check what labels PP-DocLayoutV3 gives ads — maybe "image"/"figure" or excluded classes), tiny notices under min-area 1500, or grouping absorbing them into neighbors. Write findings to log.jsonl; pick the next one-variable lever from evidence (candidate exp_161: add missing label classes or lower min-area).
2. **Over-split lever**: paragraph_title false positives splitting single articles — title score threshold sweep as its own experiment (only if evidence from item 1 points here).
3. **F4 (oracle-only)**: add exp_160 as diversity source to ensemble_prune5; prune5 precision filtering.
4. On Elio's GT promotion: board over 6 issues; re-base baselines; check ship bar on full set.
