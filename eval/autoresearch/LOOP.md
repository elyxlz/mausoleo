# Loop state — live queue for the autoresearch loop

Rewritten every iteration. Rules in `program.md` (+ `registry.md`); this is only CURRENT state + queue.

## Standing context
- Metric = **MausoleoBench** (`src/mausoleo/eval/evaluate.py`); quality-gated F1. Adversarial invariants in `scripts/eval_probes.py` — run after ANY metric change. Never modify eval/GT to move a score.
- **Contract (program.md)**: experiments NEVER import/run the eval (fixed a-priori params; score ONLY via `scripts/research.py eval`). **Independence + cold**: self-contained, reproducible from scratch, not sped up by other experiments' caches; caches cold at start. **Adversarial review** every experiment (overgeneration / giant blobs / holdout / GT-free probe) → real-vs-gamed verdict + caveats in the log.
- **Budget = 3× (2026-07-21): target 20.7, hard cap 41.7 GPU-s/page steady-state.** `BUDGET_CAP` (progress_server.py + seed_progress.py) is the single source of truth; dashboard recomputes `budget_ok` from `gpu_s_per_page`. Corpus = 172,600 pages (1880–1959; 498,252 exist 1880–2000). Measure steady-state (model load amortized ~0/page; small runs are upper bounds).
- Run on ripperred (`.venv/bin/python`), GPU1 (`CUDA_VISIBLE_DEVICES=1`). Log each full-6-issue run to `mausoleobench_log.jsonl` (`{n,config,exp,score,description,gpu_s_per_page,budget_ok}`, n=last+1) → live graph `scripts/progress_server.py` (:8078 + cloudflare, with per-experiment prediction viewer). Commit+push; update registry.

## Board (budget-compliant only counts; ≤41.7 GPU-s/page)
- **RECORD: exp_167 0.3815** (trained grouper + PaddleOCR-VL region text, 6.2). exp_160 0.3576 (6.2). exp_157 0.1718 (5.1). exp_171 0.3815 (cleanup no-op). exp_172 0.3185 (Qwen3-VL-2B, 6.5 — general 2B < specialized 0.9B).
- **Disqualified (over 41.7)**: all Qwen-8B routes (74–286) — exp_045 0.4615, exp_168 0.4342, exp_169 0.5266 (merge, ceiling diagnostic), exp_170, etc. The 3× raise unlocked none of them (cheapest is 74, still 2× over).
- Segmentation solved cheaply (grouper). **Bottleneck = budget-fit OCR text quality.** exp_172 says a general 2B is too small; the 8B (DQ) has the best text → test the middle within the new budget.

## Current
- **exp_173 RUNNING cold on GPU1** (waiter b89ya5s4x): fresh PP-DocLayout + fresh **Qwen3-VL-4B** region OCR (prefix cache off) + trained grouper. Tests whether a 4B (between the too-small 2B and the DQ-but-best 8B) gives better text AND fits ≤41.7 with the 3× budget. Nanonets-OCR2-3B was tried first but won't load in vllm (lm_head weight-tie error) — skipped.

## Next (in order) — budget-compliant only
1. **exp_173 verdict** — if Qwen3-VL-4B ≤41.7 GPU-s/page AND beats exp_167 0.3815 → new record. Log + adversarial-review.
2. **More budget-fit text**: if 4B helps, is 8B now within 41.7 too? (measure cold). Else GOT-OCR-2.0 / InternVL3-2B, each self-contained like exp_172.
3. **1952 dense-classifieds** — common weak point across every route (0.19–0.45).
