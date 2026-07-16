# Approach Registry

Families grouped by underlying mechanism. Statuses: ACTIVE (worth iterating), BLOCKED (hard failure — reopen only with the stated unblock condition), SATURATED (works, no headroom without new inputs). Update every iteration.

## F1 — Fast specialized OCR models as sources (0.9–3B, markdown-native)
**Status: ACTIVE — top priority under the corpus-scale budget (small = fast).**
- PaddleOCR-VL-1.6 (0.9B, vllm-supported, downloaded): UNTESTED → exp_147 next.
- HunyuanOCR (1B, cached): retry with NATIVE markdown prompt + MergeMarkdownPages (previous attempts misused the V2 JSON prompt). Needs transformers ≥5.13 (main venv has 5.5.4) or own env. UNTESTED with correct usage.
- olmOCR-2-7B (cached): retry with native prompt + adapter. UNTESTED with correct usage.
- GLM-OCR (0.9B, downloaded): **BLOCKED** — degenerate repetition loops on broadsheet column crops (exp_145, composite 0.15). Unblock: vllm no-repeat logit processor from the GLM-OCR recipe, or its MTP path, or evidence it works on sub-page crops of modern-density text.

## F2 — Long-horizon multi-page parsing (cross-page continuity)
**Status: ACTIVE, one route BLOCKED.**
- Unlimited-OCR multi-page mode at 1024px/page: **BLOCKED** — broadsheet text illegible at 1024px; model hallucinates numeric sequences (exp_146 smoke, 45KB of "A. NNNN"). Unblock: feed legible sub-page crops (column/half-column) as the page sequence so R-SWA context spans crops — materially different input distribution, untested.
- Caveat: 32K context bounds whole-issue output (1910 issue ≈ 185K chars ≈ 60K+ tokens) → windowed passes required regardless.

## F3 — Layout detection & reading order
**Status: ACTIVE, underexplored.**
- PP-DocLayoutV3 (31M, newspaper class, predicts reading order): untested; needs paddle runtime in own env.
- DocLayout-YOLO param tuning: SATURATED (defaults near-optimal, two sweeps failed).
- Chandra layout operator exists (chandra_layout.py), integration untested.

## F4 — Ensemble merge/quality-select tuning
**Status: SATURATED at 0.89878 (50+ evals tied within ±0.0001). Do not micro-tune without new diverse sources. NOTE: the 8-source design violates the new corpus-scale budget — production architecture must shrink drastically (see program.md Resource Budget).**

## F5 — Prompt engineering on the structured-JSON VLM path
**Status: SATURATED/BLOCKED — V2 is the optimum; complex prompts hallucinate (V3), /no_think degrades. Unblock: a new model family with different prompt affordances.**

## F6 — Post-processing (trim, repair, stitching)
**Status: ACTIVE minor.** trim_predictions is the biggest historical win (+0.0165). Heuristic cross-page stitching BLOCKED (VLM naturalizes text; zero stitches).

## F7 — Segmentation adapters (model output → article JSON)
**Status: ACTIVE.** MergeMarkdownPages written (2026-07-16), validated on synthetic input; unlocks F1. Embedding-similarity article grouping (STRAS-style) unexplored.

## Cross-cutting constraint (2026-07-16)
Corpus-scale production budget (program.md): the score-maximization target is now composite-per-GPU-second, not composite alone. Every family's value is re-weighted by throughput.
