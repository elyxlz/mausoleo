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

## State addendum (18:00)
- **exp_164 ceiling measured: 0.6980 avg, recall 0.90/0.79** (Opus grouper probe over exp_160's layout+OCR; research-only). Semantic grouping is the confirmed route for the recall bar.

## Queue (in order)
1. **exp_165 local grouper**: replicate exp_164 with a LOCAL model as grouper — Qwen3-VL-8B (cached) text-only via vllm: per-page prompt = region listing (idx/class/bbox/text first ~100 chars) -> JSON groups. Reuse exp_164's dump/assemble; new `group` stage calling vllm. Port exp_160's head-block merge into the assembler first (fixes hCER 0.48). Measure GPU-s/page of the grouping pass (region listings ~15K tok/page input, small output; prefill-dominated so likely cheap). Evaluate both dates + 1943 probe; compare to the 0.6980 ceiling and exp_160's 0.6180.
2. **Over-split lever**: title score threshold sweep (only if evidence appears).
3. On Elio's GT promotion: board over 6 issues; re-base baselines; ship-bar check.
(F4 exp_160-diversity closed NEGATIVE; geometric grouping BLOCKED 0/3 — see log.)
