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
**ACTIVE — the open bottleneck.** Two evidence-backed hypotheses (Fable research, 2026-07-21):
- **H1 context-per-crop is the dominant lever**: region OCR caps at 0.32–0.38 regardless of model size (Qwen3-VL 2B/4B/8B all ≈0.33); the archived column-structured route (exp_045) hit **0.46**. Region crops sever the decoder's language context. The quality is in the scans; the problem is delivering it ≤50 sec/page.
- **H2 historical-print domain mismatch**: modern OCR models are trained on modern PDFs (CER multiplies 3–4× on historical print). CHURRO-3B (Stanford, Qwen2.5-VL-3B fine-tuned on 100K historical pages) and LightOnOCR-2-1B (distillable) target this directly.

Prioritized queue (details in LOOP.md): **E1** cheapen the 0.46 column route (batched + plain-text + AWQ Qwen3-VL-8B) → within budget; **E2** CHURRO-3B on columns; **E3** distill the column teacher into LightOnOCR-2-1B (offline, highest ceiling); **E4** DeepSeek-OCR Gundam; **E5** cheap scan preprocessing (CLAHE/upscale, no binarization) for 1952; **E6** PaddleOCR-VL on column crops. Dead ends: scaling general VLMs on region crops; Nanonets-OCR2 (vllm lm_head bug); whole-page long-horizon.

## F6 — Post-processing (trim, repair, stitching)
**ACTIVE, low-cost.** Char-run squeeze / trailing-garbage trim are cheap CER guards (no-ops on already-clean PaddleOCR text). Cross-page stitching at merge level substitutes for F2. Open: dedup/confidence gating to lift gated-precision.

## Cross-cutting
MausoleoBench scores text-quality × correct-segmentation jointly. Segmentation is solved cheaply (F3 trained grouper); the climb is now **F5 budget-fit text quality**, pursued under the 50 sec/page cap.
