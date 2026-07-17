# Loop state — live queue for the autoresearch loop

Rewritten by the loop every iteration (see program.md §Running the Loop). Rules live in `program.md` (+ `registry.md`); this file is only the CURRENT state and queue.

## Standing context
- Run everything locally on ripperred (`.venv/bin/python`); GPU1 preferred (`CUDA_VISIBLE_DEVICES=1`, eve holds ~350MB on GPU0).
- Fable spend limit hit → spawn subagents with `model: "opus"`.
- Elio's decisions (final): eval GT set = 6 issues; NO issue-level holdout; 1925 Il Meridiano accepted.
- Tentative GTs (1895/129u, 1925/108u, 1935/256u, 1952/197u) in `eval/tentative_gt/` — awaiting Elio's review; do NOT promote.
- plan/01 ship bar + plan/02 corpus-v0-early await Elio's sign-off — no corpus run without it.
- Commit and push as you go; log to `log.jsonl` with mechanism lines; update `registry.md` every iteration.

## State (2026-07-17 14:15)
- **exp_159 accepted as production candidate**: 0.6040 avg (0.5663/0.6416), F1 0.71/0.66, ~5.1 GPU-s/page warm. PP-DocLayoutV3 regions (paddle_env subprocess) + PaddleOCR-VL vllm OCR + title-boundary grouping. Baselines table in program.md updated.
- GPUs free. paddle_env verified; PP-DocLayoutV3 cached in ~/.paddlex.

## Queue (in order)
1. **Formal steady-state bench for exp_159**: adapt scripts/bench_throughput.py (it drives config-registry operators; exp_159 is an experiments/ script — simplest: add a bench mode/flag to exp_159 that loops the 5-issue era-diverse set [1885-06-15, 1910-06-15, 1943-07-03, 1943-07-15, 1943-07-25] in one process, reporting warm GPU-s/page excluding first issue). Record corpus extrapolation in log.jsonl.
2. **share_tiny lever (exp_160)**: 1943 probe shows 14% tiny (<50 char) units — mostly title-only fragments. One variable: merge a title-article with empty/near-empty body into the NEXT article as its headline when that next article is headline-less (or drop <20-char orphans). Evaluate both dates + probe.
3. **Over-split lever**: paragraph_title false positives may split single articles (check 1910 GT misses); consider min title score threshold sweep as its own experiment.
4. **F4 (oracle-only)**: add exp_159 as a diversity source to ensemble_prune5; prune5 precision filtering.
5. On Elio's GT promotion: re-run board over 6 issues; re-base baselines.
