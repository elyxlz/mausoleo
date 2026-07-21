# Loop state — live queue for the autoresearch loop

Rewritten every iteration. Rules in `program.md` (+ `registry.md`); this file is only the CURRENT state + queue.

## Standing context
- Metric = **MausoleoBench** (`src/mausoleo/eval/evaluate.py`). Budget = **50.0 sec/page** caller-measured (`scripts/time_experiment.sh`). Full rules in `program.md`.
- Run on ripperred (`.venv/bin/python`), GPU1 (`CUDA_VISIBLE_DEVICES=1`). Experiments are self-contained, never import the eval, measured cold by the caller, and get an adversarial review.
- Experiments numbered from **exp_001**. Log each full-6-issue run to `mausoleobench_log.jsonl` (`{n, config, exp, score, description, gpu_s_per_page, budget_ok}`, n = last+1) → live graph `scripts/progress_server.py` (:8078 + cloudflare, per-experiment prediction viewer). Commit + push; update `registry.md`.

## Board
- **RECORD: exp_002 = 0.3826** (n=2) — grouper + PaddleOCR-VL with hi-res crops, 5.91 sec/page. exp_001 baseline 0.3802 (n=1).
- Oracle references (not production): `ensemble_30min` 0.5941, `ensemble_prune5` 0.5622.
- Segmentation is solved cheaply by the trained grouper. Open bottleneck: **budget-fit OCR text quality** (`registry.md` §F5).
- Note: **any gain counts** (no effect-size floor). 1952 dense-classifieds still the weak point (~0.27).

## Done this slate
- exp_001 0.3802, **exp_002 0.3826 (record)** — grouper + PaddleOCR-VL per-region.
- **exp_003 FAILED** (n=3): Qwen3-VL-8B column-structured JSON = 0.1934 @ 65.65 sec/page (DQ). Structured JSON merges columns into giant blobs (recall 0.16–0.68); did NOT reproduce archived exp_045 0.46. JSON is the wrong delivery for context.

## Current
- **exp_004 RUNNING** (waiter bxqcxwk2x): **CHURRO-3B** (`stanford-oval/churro-3B`, Qwen2.5-VL-3B fine-tuned on 100K historical pages) per-region + trained grouper — drop-in swap for PaddleOCR-VL in the working pipeline (keeps grouper segmentation), caller-measured. Tests H2 (historical domain fine-tuning) directly vs the 0.3826 baseline. CHURRO is page-trained so region crops are mildly OOD — informative either way.

## Queue (Fable research plan, ranked)
1. **exp_004 verdict** — if CHURRO-3B per-region beats 0.3826 within budget → record + H2 confirmed. If region crops too OOD, try CHURRO on column crops (needs column-text→region alignment for the grouper).
2. **PLAIN-TEXT columns + grouper** (the report's real E1, not JSON): OCR column crops as plain text with a strong model, map column text back to regions via bbox y-alignment, feed the grouper. Gets context (H1) without JSON's blob-merging; keeps grouper segmentation.
3. **E3 distill** a strong teacher → `lightonai/LightOnOCR-2-1B` (offline QLoRA on teacher-labeled non-eval corpus pages, decade-stratified; ~5–10 sec/page). Highest ceiling.
4. **E5 preprocessing** (CLAHE/upscale small-text regions, NO binarization) for 1952 — cheap CPU; ship only if ≥4/6 issues improve.
- Anti-overfit tell: a real H1/H2 gain lifts all 6 issues roughly uniformly.
