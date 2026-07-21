# Approach Registry — MausoleoBench fresh slate (2026-07-21)

Scores are **MausoleoBench** (6-issue). composite_v2-era registry archived at `archive/registry_composite_v2.md`; old experiment scripts at `experiments/archive/`. Statuses: ACTIVE (worth iterating), BLOCKED (hard failure — reopen only with the stated unblock), SATURATED (works, no headroom without new inputs). Update every iteration.

Board (fresh): oracle ceiling `ensemble_30min` 0.5941 · **best real `exp_045_qwen3vl_vllm` 0.4615** · exp_140 0.4170 · exp_160 0.3576. Live graph: `scripts/progress_server.py` (:8078 + cloudflare).

## F1 — Fast specialized OCR models as sources (0.9–3B)
**Status: ACTIVE — cheapest route under corpus budget.** PaddleOCR-VL-1.6 (0.9B, vllm) works as a source; YOLO-title-class regions as headlines is the right usage. Budget-independent BLOCKS (carried forward, physical facts): HunyuanOCR — transformers-eager 35min/issue + col-crop gibberish; olmOCR-2-7B — linearizes, won't segment, ~65 GPU-s/page; GLM-OCR — degenerate repetition loops on column crops. Unblock any: vllm support + evidence it reads broadsheet type.

## F2 — Long-horizon multi-page parsing
**Status: BLOCKED (physical).** Broadsheet text illegible at 1024px/page; legible square tiles garble + repetition-loop. Unblock: a NEW long-horizon model release, not parameter nudges. Cross-page continuity must come from F3/F6 instead.

## F3 — Layout detection & reading order
**Status: ACTIVE — grouping is THE bottleneck.** PP-DocLayoutV3 (31M, `~/paddle_env`) gives strong detection recall (0.83/0.78) at 4–6s/issue, but ~100 paragraph regions/page → precision collapse unless grouped. Head-block grouping (ex-exp_160) reached composite_v2 0.5756 but only **0.3576 on MausoleoBench** — its matched text is high-CER, so the quality gate discounts it. Geometric region-grouping BLOCKED (intra-article vs inter-ad gaps overlap at every threshold). Local-LLM grouper BLOCKED (over-splits + 11–47 s/page busts budget). **Top unblock: train a small boundary grouper** (per-region "starts new article?" classifier, features in `experiments/grouper_features.py`, labels by aligning the 6 GTs to region dumps `semgroup/regions_<date>.json`). Region dumps ready for all 6 issues.

## F4 — Ensemble merge/quality-select
**Status: reference only.** ensemble_30min/prune5 are oracle ceilings (recall ~1.0) used for GT-building and as the score ceiling; ~600 GPU-s/page = not production.

## F5 — Structured-JSON VLM path (Qwen3-VL)
**Status: ACTIVE — now the leading real pipeline on MausoleoBench (exp_045 0.4615).** The quality gate favors this route's clean per-article text. Column-split matters: col4/col6 variants (exp_055/097) score below the default; full-page (exp_102/107) collapses on dense pages. Cost ~136 GPU-s/page is over the 6.9–13.9 budget — a production win needs the same text quality at a fraction of the cost (smaller model, fewer crops, or PP-DocLayout regions feeding the VLM). Levers: better column/region cropping, ads-aware prompt that doesn't hurt precision, cheaper backbone.

## F6 — Post-processing (trim, repair, stitching)
**Status: ACTIVE, low-cost.** Char-run squeeze is a cheap CER guard. Cross-page stitching at merge level is the F2 substitute. Open: dedup/confidence gating on over-generated predictions to lift gated-precision.

## Cross-cutting insight (2026-07-21)
MausoleoBench scores text-quality × correct-segmentation jointly. The two strong routes are complementary: **F3 (PP-DocLayout) has recall, F5 (Qwen3-VL) has text quality.** The highest-EV unexplored direction is grafting them — PP-DocLayout regions/reading-order as the crop plan feeding Qwen3-VL OCR, or a trained grouper over Qwen3-VL region text — pursued under the corpus budget.
