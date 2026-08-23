# Approach Registry

Approach families with current status. Statuses: **ACTIVE** (worth iterating), **BLOCKED** (hard failure — reopen only with the stated unblock condition), **SATURATED** (works, no headroom without new inputs). Update every iteration. Budget-compliant record: **exp_009 0.4071** (article-level OCR + PaddleOCR-VL + fill-ratio guard).

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

## F7 — Domain adaptation (fine-tune / distill) — the ceiling lever
**FINE-TUNING FRAGILE / near-exhausted.** Naive LoRA on real GT FAILS (exp_012, 6-fold LOIO = 0.4041, net -0.003 vs base 0.4071): 261 pairs/fold overfits the training issues' typography (+0.013 on 1895 but -0.016 on 1885). Adaptation is real but data-starved. Synthetic period-Italian augmentation FAILS WORSE (exp_013: -0.130 on held-out 1935, meanCER 1.663 = HALLUCINATION — literary-prose synth pushes the OCR model to generate, not transcribe; PaddleOCR-VL RL-post-trained transcription breaks under SFT). Only untried lever: **oracle-ensemble distillation** on non-eval corpus pages — REAL newspaper transcription labels (right domain; naive real-GT SFT did NOT hallucinate, only literary synth did), ~1 day teacher labeling, uncertain. 0.4071 was the practical ceiling under the old 50 sec/page budget; since the 5x raise (cap 250, 2026-07-22) the record is exp_017 = 0.4275 @ 181.98 sec/page via the 8B column route. Strict LOIO correctly caught the overfit (non-LOIO would show a fake memorization gain). Incremental PaddleOCR-VL tweaks saturate ~0.41; the gap to the oracle ceiling (0.594) needs adapting the model to historical Italian broadsheet type (Fable research 2026-07-21). **PaddleOCR-VL-1.6 IS LoRA-fine-tunable** — ms-swift official support (`--train_type lora`) + HF transformers native (`AutoModelForImageTextToText`) + a working precedent ([jzhang533/paddleocr-vl-sft], manga region-crop SFT 27%→70%). A merged LoRA serves in the existing vllm path unchanged → **zero inference-cost delta, automatically budget-fit.** Ranked plan:
1. **CHURRO-3B zero-shot on ARTICLE crops** (no training, cheapest go/no-go): page-trained on 97K historical pages incl. newspapers/Italian, zero-hallucination on in-distribution input; article crops ≈ its distribution (unlike the region crops that broke it). Cap max_pixels ~2–2.5M. Needs PaddleOCR region OCR (grouper features) + CHURRO article OCR (two models, sequential load).
2. **LoRA PaddleOCR-VL-1.6** on GT article-crops→GT-text, strict by-issue LOIO (6 folds), vision encoder frozen, low LR, + synthetic period-Italian degraded-text augmentation (Wikisource/Liber Liber, never eval text). Single-fold (test on 1935) first as go/no-go. ~1–2 h/fold train, same 8.66 sec/page inference.
3. **Distill oracle (0.594) → PaddleOCR-VL** on non-eval corpus pages (teacher labels; ~1.7 days for 500 pages). Cleanest integrity (eval never touched). Student backup: lightonai/LightOnOCR-2-1B.
4. **ByT5 post-OCR corrector** (stacks, +1–2 sec/page; watch fluent-hallucination via edit-distance caps).

**Anti-overfit protocol (mandatory for every fine-tune)**: strict LOIO (score for issue X only from the fold excluding X); cross-decade split (train 1885/1895/1910 → test 1935/1952 & reverse); GT-free probes on the **5 image-only 1943-07 issues** in eval/ground_truth/ (01, 08, 15, 22, 31) (lexicon hit, LM perplexity, blob/repetition rate) — never sample 1943-07 for teacher labels; standard matcher-gaming audit.

## Cross-cutting
MausoleoBench scores text-quality × correct-segmentation jointly. Segmentation is solved cheaply (F3 trained grouper); text-quality within budget saturated at ~0.41 via specialized PaddleOCR-VL + article context (F5). The remaining headroom is **F7 domain adaptation** (fine-tune/distill PaddleOCR-VL on historical Italian) — the path toward the 0.594 oracle.
