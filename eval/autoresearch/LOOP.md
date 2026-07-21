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

- **exp_005 = 0.3946 RECORD** (n=5): article-level OCR (grouper boundaries + article-union-bbox crop OCR, PaddleOCR-VL) @ 8.74 sec/page. Context helps: 1885 +0.025, 1895 +0.083, 1910 +0.017, 1925 +0.027; but 1935 -0.063, 1952 -0.017 (dense pages: mis-grouped regions → huge multi-column crops → blobs).

- **exp_006 = 0.3931** (n=6): length-ratio blob guard, NOT a record — too blunt (article text longer than region text is usually the GAIN, so it reverts wins). exp_005 stays record.

- **exp_007 FAILED** (n=7): full-precision 8B on article crops OOM'd (big multi-region crops blow past 24GB even at max_num_seqs=32).

## Current
- **exp_008 RUNNING** (chained task bai31mamu, after AWQ download): **AWQ-4bit Qwen3-VL-8B on article crops** (`cpatonn/Qwen3-VL-8B-Instruct-AWQ-4bit`, ~5GB weights → room + speed, awq_marlin on Ampere, max_pixels cap). 8B-quality text at article granularity within memory+budget. The task waits for the download, then runs caller-timed + evals. Key: 8B is only better WITH context (per-region 8B 0.334 < PaddleOCR 0.38; article/column context is where 8B wins toward the 0.46 seen with columns).

## Queue (ranked)
1. **exp_008 verdict** — does AWQ-8B article text beat 0.3946 within ≤50 sec/page? Watch for OOM (lower max_num_seqs / max_pixels if so) and blobs on dense pages.
2. **E3 distill** a strong teacher → `lightonai/LightOnOCR-2-1B` (offline, highest ceiling).
3. **E5 preprocessing** (CLAHE/upscale small-text regions, NO binarization) for 1952.
4. **Geometric width-guard** for exp_005's dense regressions (skip article-crop when union bbox spans >~1.3 columns).
- Infra: to stop a run, kill the PARENT python proc (pgrep -f exp_NNN), NOT the vllm EngineCore — killing the engine orphans+hangs the parent (looks 'done' but holds host RAM). Verify GPU freed (nvidia-smi --query-compute-apps). Pre-download models so download time doesn't pollute the caller budget.
