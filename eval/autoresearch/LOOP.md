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

- **exp_003 FAILED** (n=3): column-structured JSON 8B = 0.1934 @ 65.65 (DQ) — JSON blobs.
- **exp_004 FAILED** (n=4): CHURRO-3B per-region = ~0.15/cer 6.0 (DQ, hallucinates on crops + slow). Page-trained → region crops OOD.

## Current
- **exp_005 RUNNING** (waiter bnn3d755v): **article-level OCR** — the clean way to get context WITH grouper segmentation and NO fuzzy alignment. Grouper decides article boundaries from cheap per-region text, then OCR each article's union-bbox crop (context-rich) with PaddleOCR-VL for the output text. Budget-fit + robust (no OOM/blob/hallucination risk). Tests whether article-level context lifts text quality over per-region 0.3826.

## Queue (ranked)
1. **exp_005 verdict** — if article-crop context beats 0.3826 within budget → record + H1 confirmed cheaply. If PaddleOCR-VL gains from context, swap a stronger model for the article-crop OCR (fewer/bigger crops than region-level, so an 8B article pass is ~budget-fit).
2. **Stronger model on article crops** (Qwen3-VL-8B PLAIN text, not JSON) — article crops are fewer than columns/regions, so 8B article OCR may fit ≤50; gets 8B text quality at article granularity.
3. **E3 distill** a teacher → `lightonai/LightOnOCR-2-1B` (offline, highest ceiling).
4. **E5 preprocessing** (CLAHE/upscale small-text regions, NO binarization) for 1952 — cheap; ship only if ≥4/6 issues improve.
- Anti-overfit tell: a real gain lifts all 6 issues roughly uniformly.
- Infra note: kill orphaned vllm EngineCore procs (nvidia-smi --query-compute-apps) after killing a run — the parent pkill leaves them holding GPU mem.
