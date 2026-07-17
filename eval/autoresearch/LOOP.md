# Loop state — live queue for the autoresearch loop

Rewritten by the loop every iteration (see program.md §Running the Loop). Rules live in `program.md` (+ `registry.md`); this file is only the CURRENT state and queue.

## Standing context
- Run everything locally on ripperred (`.venv/bin/python`); GPU1 preferred (`CUDA_VISIBLE_DEVICES=1`, eve holds ~350MB on GPU0).
- Fable spend limit hit → spawn subagents with `model: "opus"`.
- Elio's decisions (final): eval GT set = 6 issues; NO issue-level holdout; 1925 Il Meridiano accepted.
- Tentative GTs (1895/129u, 1925/108u, 1935/256u, 1952/197u) delivered in `eval/tentative_gt/` — awaiting Elio's human review; do NOT promote to `eval/ground_truth/`.
- plan/01 ship bar + plan/02 corpus-v0-early stance await Elio's sign-off — no corpus run without it.
- Commit and push as you go; log every result to `log.jsonl` with a mechanism line; update `registry.md` every iteration.

## In flight
- **exp_158 running** (task bgomm858c, log scratchpad/exp158.log): both GT dates on GPU1. Paddle env verified (2 GPUs); PP-DocLayoutV3 smoke-tested on 1885 p1: 106 boxes (99 text, 6 paragraph_title, 1 header excluded) — labels match the script's sets. Note the granularity: ~100 regions/page (vs YOLO's ~11 merged columns) — expect many more, smaller OCR calls.

## Queue
1. Evaluate: `scripts/research.py eval exp_158_ppdoclayout_paddle` vs exp_157 (0.4284 avg; recall 0.488/0.358). Inspect concrete predictions vs GT; audit giant blobs; holdout; probe on 1943-07-15 (`research.py probe exp_158_ppdoclayout_paddle_1943-07-15` vs exp_157: lexicon 0.6757, high-rep 0.085). Accept per program.md floors (single-source ≥0.005 both dates). Log + registry F3 either way; commit+push.
4. If paddle is a dependency wall: log exact failure + unblock condition in registry F3; move to F1 abandon-class filtering (`experiments/exp_159`: exp_157 with "abandon" removed from YoloCrop text_classes — one variable).
5. Backlog after that: horizontal_overlap lever (F1); prune5 precision filtering (F4, oracle-only); Paddle as prune5 diversity source (F4).
