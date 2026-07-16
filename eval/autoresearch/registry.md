# Approach Registry

Families grouped by underlying mechanism. Statuses: ACTIVE (worth iterating), BLOCKED (hard failure — reopen only with the stated unblock condition), SATURATED (works, no headroom without new inputs). Update every iteration.

## F1 — Fast specialized OCR models as sources (0.9–3B, markdown-native)
**Status: ACTIVE — top priority under the corpus-scale budget (small = fast).**
- PaddleOCR-VL-1.6 (0.9B, vllm-supported): col3 misuse FAILS (exp_147, 0.23); YOLO-region usage matches design (exp_148). **exp_152 WIN (2026-07-16): YOLO title-class regions as headlines → 0.4305 v2 avg (+0.025 over exp_149). exp_155 = exp_152 + CUDA graphs: 0.4285 at 5.13 GPU-s/page = 5.2-day corpus. exp_157 = exp_155 + char-run squeeze (F6 guard): eval-neutral, probe junk capped — CURRENT PRODUCTION CANDIDATE.** Dead levers: conf 0.15 (worse, noise boxes glue columns, exp_153); merge_vertical_gap 30/40 (tune-half-only gains, holdout-rejected, exp_154/156). Remaining bottleneck: recall 0.36–0.49 and wCER 0.74–0.77. Next levers: PP-DocLayoutV3 regions (F3); abandon-class filtering; horizontal_overlap.
- HunyuanOCR (1B, cached): **BLOCKED** — native markdown prompt on col3 crops yields hallucinated gibberish (exp_150, composite 0.15) and transformers-eager runs 35min/issue (~50x over budget). Unblock: vllm HunYuanVL support AND evidence it reads broadsheet type.
- olmOCR-2-7B (cached): **BLOCKED** — native front-matter prompt on col3 crops emits zero markdown headings (exp_151, 0.2761 v2 avg, hCER 1.0, one blob per crop); ~65 GPU-s/page, over budget even as a source. The exp_047 "misuse" hypothesis is settled — the model linearizes, it does not segment. Unblock: only as YOLO-region-level oracle diversity source if F4 pruning ever needs a non-Qwen text opinion.
- GLM-OCR (0.9B, downloaded): **BLOCKED** — degenerate repetition loops on broadsheet column crops (exp_145, composite 0.15). Unblock: vllm no-repeat logit processor from the GLM-OCR recipe, or its MTP path, or evidence it works on sub-page crops of modern-density text.

## F2 — Long-horizon multi-page parsing (cross-page continuity)
**Status: BLOCKED (2026-07-16) — both Unlimited-OCR routes dead.**
- Multi-page mode at 1024px/page: BLOCKED — broadsheet text illegible at 1024px (exp_146, hallucinated numerics).
- Legible-tile sequence (the former unblock condition): **tested and FAILED** — square 730px tiles are legible to the model (real headlines read) but output garbles far below Paddle quality and degenerates into repetition loops at any sequence length (12 or 4 tiles) and anti-repeat setting. `infer_multi` letterboxes every image to a square canvas; tiles were the correct input shape, failure is distributional.
- Unblock: a NEW long-horizon model release, not parameter/prompt nudges. Cross-page continuity must meanwhile come from layout/reading-order (F3) or merge-level stitching (F6).

## F3 — Layout detection & reading order
**Status: ACTIVE, underexplored.**
- PP-DocLayoutV3 (31M, newspaper class, predicts reading order): untested; needs paddle runtime in own env.
- DocLayout-YOLO param tuning: SATURATED (defaults near-optimal, two sweeps failed).
- Chandra layout operator exists (chandra_layout.py), integration untested.

## F4 — Ensemble merge/quality-select tuning
**Status: PRUNING DONE (2026-07-16) — `ensemble_prune5` is the v2 leader at 0.7776 (+0.026 over 8-source).** Greedy v2 backward-elimination removed exp_055 (additive ratio-100 spam, LOO +0.0148), exp_142, exp_140; selection split-stable, holdout flat. `ensemble_30min` retained as recall-oracle (recall 1.0/0.98) for GT building. Remaining F4 headroom: precision-filtering the survivors (dedup/confidence gating on the 88/301 remaining preds), or swapping a Qwen source for exp_155-Paddle as a cheap diversity source. Still oracle-tier cost — not production.

## F5 — Prompt engineering on the structured-JSON VLM path
**Status: SATURATED/BLOCKED — V2 is the optimum; complex prompts hallucinate (V3), /no_think degrades. Unblock: a new model family with different prompt affordances.**

## F6 — Post-processing (trim, repair, stitching)
**Status: ACTIVE minor.** trim_predictions is the biggest historical win (+0.0165). squeeze_char_runs (2026-07-16) guards model degeneration on degraded scans (exp_157). Heuristic cross-page stitching BLOCKED (VLM naturalizes text; zero stitches).

## F7 — Segmentation adapters (model output → article JSON)
**Status: ACTIVE.** MergeMarkdownPages written (2026-07-16), validated on synthetic input; unlocks F1. Embedding-similarity article grouping (STRAS-style) unexplored.

## Cross-cutting constraint (2026-07-16)
Corpus-scale production budget (program.md): the score-maximization target is now composite-per-GPU-second, not composite alone. Every family's value is re-weighted by throughput.

**Throughput ground truth (measured 2026-07-16, exp_045 benchmark):** Qwen3-VL-8B col3 single pass = ~136 GPU-s/page avg (74 on 1885, 178 on dense 1910) — 5–26× over budget. Every ≥7B full-coverage pass is production-infeasible on 2×3090; 7–8B models remain useful only as oracle/GT/ensemble-reference sources. Production quality must come from F1 sub-1B models (Paddle ~10 GPU-s/page) + cheap layout (F3). Open lever: enforce_eager=False (CUDA graphs) could recover 2–3× on decode, still leaves 8B marginal at best.
