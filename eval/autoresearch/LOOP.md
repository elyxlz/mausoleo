# Loop state — live queue for the autoresearch loop

Rewritten every iteration. Rules in `program.md` (+ `registry.md`); this file is only the CURRENT state + queue.

## Standing context
- Metric = **MausoleoBench** (`src/mausoleo/eval/evaluate.py`). Budget = **50.0 sec/page** caller-measured (`scripts/time_experiment.sh`). Full rules in `program.md`.
- Run on ripperred (`.venv/bin/python`), GPU1 (`CUDA_VISIBLE_DEVICES=1`). Experiments are self-contained, never import the eval, measured cold by the caller, and get an adversarial review.
- Experiments numbered from **exp_001**. Log each full-6-issue run to `mausoleobench_log.jsonl` (`{n, config, exp, score, description, gpu_s_per_page, budget_ok}`, n = last+1) → live graph `scripts/progress_server.py` (:8078 + cloudflare, per-experiment prediction viewer). Commit + push; update `registry.md`.

## Board
- **RECORD: exp_001 = 0.3802** (n=1) — self-contained trained grouper + PaddleOCR-VL, 5.95 sec/page. Independent + caller-measured; reproduces the prior baseline honestly.
- Oracle references (not production): `ensemble_30min` 0.5941, `ensemble_prune5` 0.5622.
- Segmentation is solved cheaply by the trained grouper. Open bottleneck: **budget-fit OCR text quality** (`registry.md` §F5).

## Queue
1. **Budget-fit text quality (exp_002)**: screen the best OCR model that fits ≤50 sec/page caller-measured, fed through the trained grouper. Independent general VLMs measured so far (archived): Qwen3-VL 2B 0.32/6.5, 4B 0.34/10.6, 8B 0.33/20.3 — all below PaddleOCR-VL 0.38. Next angles: OCR-specialized budget models (GOT-OCR-2.0, InternVL3-2B), region-crop resolution/prompt on PaddleOCR-VL, or merging two budget-fit sources.
2. **1952 dense-classifieds**: the common weak point across every route.
