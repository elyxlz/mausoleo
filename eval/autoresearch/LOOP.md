# Loop state — live queue for the autoresearch loop

Rewritten by the loop every iteration (see program.md §Running the Loop). Rules live in `program.md` (+ `registry.md`); this file is only the CURRENT state and queue.

## Standing context
- Run everything locally on ripperred (`.venv/bin/python`); GPU1 preferred (`CUDA_VISIBLE_DEVICES=1`, eve holds ~350MB on GPU0).
- Fable spend limit hit → spawn subagents with `model: "opus"`.
- Elio's decisions (final): eval GT set = 6 issues; NO issue-level holdout; 1925 Il Meridiano accepted.
- Tentative GTs (1895/129u, 1925/108u, 1935/256u, 1952/197u) in `eval/tentative_gt/` — awaiting Elio's review; do NOT promote.
- plan/01 ship bar + plan/02 corpus-v0-early await Elio's sign-off — no corpus run without it.
- Commit and push as you go; log to `log.jsonl` with mechanism lines; update `registry.md` every iteration.

## State (2026-07-17 19:05)
- **Production candidate: exp_160** = 0.6180 avg, F1 0.75/0.68, 6.22 GPU-s/page = 6.3-day corpus. Oracle: ensemble_prune5 0.7776. Semantic-grouping ceiling documented at 0.6980 (exp_164 Opus probe); local replication BLOCKED 0/2 (see registry F3).
- Today's arc complete: geometric grouping blocked 0/3, F4 diversity negative, local grouper blocked 0/2 — all logged with unblock conditions. Zoom-refinement of all 4 tentative GTs done and pushed.
- **Main gate is now Elio**: review/promote eval/tentative_gt/{1895,1925,1935,1952}-06-15 -> eval/ground_truth; sign off plan/01 ship bar + plan/02 corpus-v0-early.

## Queue (in order)
1. On Elio's GT promotion: run board over 6 issues; re-base program.md baselines; re-check ship bar (composite 0.618 ✓ 0.60; recall 0.68/0.61 vs proposed 0.70) and decide corpus v0 with Elio.
2. Idle-time levers (only with fresh evidence, avoid noise-fitting): title-score threshold sweep; distill-grouper prototype using promoted GT groupings as training data (the F3 unblock).
3. Keep loop heartbeat long (~30 min) while gated on Elio.
