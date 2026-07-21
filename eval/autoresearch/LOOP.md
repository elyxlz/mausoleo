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

## Queue
1. **exp_003 — await Fable strategy report** (agent a64a08fb, running): ranked budget-fit text-quality plan (context strategy, quantizing a strong model into budget, current specialized OCR models, distillation of the 0.46 column-structured teacher, scan preprocessing). Design exp_003 from its top recommendation. The evidence: per-region OCR caps at 0.32–0.38 regardless of model size; **context per crop is the big lever** (whole-column structured route hit 0.46 but cost 136 sec/page) — get that quality ≤50 sec/page. Budget is generous (50) so speed-engineering / quantization / distillation are all in scope (program.md §Research Toolbox).
2. **1952 dense-classifieds**: the common weak point across every route (~0.27).
