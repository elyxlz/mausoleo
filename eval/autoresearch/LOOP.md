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

## Current
- **exp_003 RUNNING** (waiter buuz4jyza): reproduce the 0.46 column-structured route self-contained — Qwen3-VL-8B on 3-column crops + structured-JSON article prompt (VLM_OCR_STRUCTURED_V2), batched (max_num_seqs=16), caller-measured. Tests whether the archived exp_045 quality (0.4615) fits ≤50 sec/page (per-region 8B was only 20 sec/page, so the old "136" was a stale estimate). If yes → big record jump.

## Queue (Fable research plan, ranked)
1. **exp_003 verdict** — if column-structured 8B is ≤50 sec/page AND ~0.46, huge record. If over budget, cut cost: plain-text prompt (JSON inflates decode), AWQ (`cpatonn/Qwen3-VL-8B-Instruct-AWQ-4bit`), higher max_num_seqs.
2. **E2 CHURRO-3B on columns** — `stanford-oval/churro-3B` (Qwen2.5-VL-3B fine-tuned on 100K historical pages; loads in vllm). Purpose-built for H2; ~12–20 sec/page. Download needed.
3. **E3 distill the column teacher → `lightonai/LightOnOCR-2-1B`** (offline QLoRA on teacher-labeled non-eval corpus pages, decade-stratified; ~5–10 sec/page inference). Highest ceiling; start teacher-labeling in background once E1/E2 picks the teacher.
4. **E5 preprocessing** (CLAHE/upscale small-text regions, NO binarization) for the 1952 weak point — cheap CPU; ship only if ≥4/6 issues improve (anti-overfit).
5. **E4 DeepSeek-OCR Gundam**, **E6 PaddleOCR-VL on columns** — opportunistic.
- Anti-overfit tell: a real H1/H2 gain lifts all 6 issues roughly uniformly; single-issue gains = fitting.
