# Loop state — live queue for the autoresearch loop

Rewritten every iteration. Rules in `program.md` (+ `registry.md`); this file is only the CURRENT state + queue.

## Standing context
- Metric = **MausoleoBench** (`src/mausoleo/eval/evaluate.py`). Budget = **50.0 sec/page** caller-measured (`scripts/time_experiment.sh`). Full rules in `program.md`.
- Run on ripperred (`.venv/bin/python`), GPU1 (`CUDA_VISIBLE_DEVICES=1`). Experiments are self-contained, never import the eval, measured cold by the caller, and get an adversarial review.
- Experiments numbered from **exp_001**. Log each full-6-issue run to `mausoleobench_log.jsonl` (`{n, config, exp, score, description, gpu_s_per_page, budget_ok}`, n = last+1) → live graph `scripts/progress_server.py` (:8078 + cloudflare, per-experiment prediction viewer). Commit + push; update `registry.md`.

## Board
- Fresh slate — attempts restart at exp_001. Baseline to re-establish: the trained per-region boundary grouper over PP-DocLayout regions + PaddleOCR-VL text (previously scored 0.3815).
- Oracle references (not production): `ensemble_30min` 0.5941, `ensemble_prune5` 0.5622.
- Segmentation is solved cheaply by the trained grouper. Open bottleneck: **budget-fit OCR text quality** (`registry.md` §F5).

## Queue
1. **exp_001 — baseline**: self-contained grouper + PaddleOCR-VL pipeline (fresh PP-DocLayout + fresh region OCR + trained boundary grouper), caller-measured. Establishes the fresh-slate baseline.
2. **Budget-fit text quality**: screen the best OCR model that fits ≤50 sec/page (measured cold via the caller) fed through the trained grouper; the open lever.
3. **1952 dense-classifieds**: the common weak point across every route.
