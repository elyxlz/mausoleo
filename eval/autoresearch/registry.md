# Approach Registry

Approach families with current status. Statuses: **ACTIVE** (worth iterating), **BLOCKED** (hard failure — reopen only with the stated unblock condition), **SATURATED** (works, no headroom without new inputs). Update every iteration. Budget-compliant record: **exp_167 0.3815**.

## F1 — Fast specialized OCR models as sources (≤1B)
**ACTIVE — cheapest route within budget.** PaddleOCR-VL-1.6 (0.9B, vllm) is the working budget-fit OCR source; its per-region text is the current record's input. Blocked alternatives (physical, not budget): HunyuanOCR (transformers-eager too slow + column-crop gibberish); olmOCR-2-7B (linearizes, won't segment); GLM-OCR (repetition loops on column crops). Unblock any: vllm support + evidence it reads broadsheet type.

## F2 — Long-horizon multi-page parsing
**BLOCKED (physical).** Broadsheet text illegible at low resolution; legible tiles garble + repetition-loop. Unblock: a new long-horizon model. Cross-page continuity comes from F3/F6 instead.

## F3 — Layout detection & reading order
**ACTIVE.** PP-DocLayoutV3 (`~/paddle_env`) gives strong detection recall at low cost, but paragraph-level regions must be grouped into articles. **Trained boundary grouper is the solution**: a per-region "does region i start a new article?" classifier (features in `experiments/grouper_features.py`, labels by aligning the 6 GTs to region decompositions, by-issue cross-validation). It solves segmentation cheaply — recall 0.57–0.89 at ~0 inference cost. Geometric grouping is BLOCKED (intra-article vs inter-ad gaps overlap at every threshold).

## F4 — Ensemble merge/quality-select
**Reference only.** Multi-source oracle ensembles (`ensemble_30min`, `ensemble_prune5`) are recall ceilings used for GT-building; cost is far over budget — not production.

## F5 — VLM OCR text quality
**ACTIVE — the open bottleneck.** Per-region text quality scales with model size (2B < 4B < 8B), and the specialized 0.9B PaddleOCR-VL sits between 4B and 8B general VLMs. The goal is the best text quality that fits ≤50 sec/page caller-measured. Levers: better budget-fit models, region-crop resolution/prompt, and merging complementary budget-fit sources.

## F6 — Post-processing (trim, repair, stitching)
**ACTIVE, low-cost.** Char-run squeeze / trailing-garbage trim are cheap CER guards (no-ops on already-clean PaddleOCR text). Cross-page stitching at merge level substitutes for F2. Open: dedup/confidence gating to lift gated-precision.

## Cross-cutting
MausoleoBench scores text-quality × correct-segmentation jointly. Segmentation is solved cheaply (F3 trained grouper); the climb is now **F5 budget-fit text quality**, pursued under the 50 sec/page cap.
