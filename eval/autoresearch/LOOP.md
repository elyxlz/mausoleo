# Loop state — live queue for the autoresearch loop

Rewritten every iteration. Rules in `program.md` (+ `registry.md`); this file is ONLY the current state + queue. Do not accumulate history here — results go to `mausoleobench_log.jsonl`, mechanisms to `registry.md`.

## Standing context
- Metric = **MausoleoBench** (`src/mausoleo/eval/evaluate.py`). Budget cap = **200.0 sec/page**, caller-measured (`scripts/time_experiment.sh`); `BUDGET_CAP` in `scripts/progress_server.py` is the single source of truth.
- Run on ripperred (`.venv/bin/python`), GPU1 (`CUDA_VISIBLE_DEVICES=1`). Experiments are self-contained, never import the eval, are measured cold by the caller, and get an adversarial review.
- Log each full-6-issue run to `mausoleobench_log.jsonl` (`{n, exp, score, description, sec_per_page, budget_ok}`, n = last+1) → live graph `scripts/progress_server.py` (:8078 + cloudflare, per-experiment prediction viewer). Commit + push; update `registry.md`.

## Board
- **RECORD: exp_017 = 0.4275 @ 181.98 sec/page** — Qwen3-VL-8B column route via direct operator calls (Ray-free), 3 columns, max_tokens 8192.
- Prior frontier: exp_009 = 0.4071 @ 8.66 (article-level PaddleOCR-VL + fill-ratio guard) — still by far the cheapest good route.
- Reference ceiling: oracle ensembles 0.5941 / 0.5622, both far over cap.
- Ruled out (see `registry.md`): CHURRO and general VLMs below specialized PaddleOCR-VL at low cost; CLAHE preprocessing; length-ratio blob guard; naive LoRA on real GT; synthetic-augmented LoRA.

## Current
- exp_017 is the record but fragile and expensive (182 sec/page, 73% of cap). exp_015 (parse variant of the same route) scored only 0.252 @ 191.97 — the route is sensitive to how column output is parsed.

## Queue
1. **Cheaper or more uniform route holding ≥0.4275** — the stated open lever in `GOAL.md`. Attack the cost side of the exp_017 column route (fewer columns, smaller max_model_len, batching) before adding anything new.
2. **exp_168 revival** (8B per-region + grouper, archived 0.4342 @ ~150) rebuilt self-contained and caller-measured.
3. **Complementary merge** of the column route ⊕ the grouper route (archived oracle-selected 0.525; the naive two-8B merge was ~286 sec/page, over cap) — needs a cheaper second source to fit ≤250.
4. **Oracle-ensemble distillation** on non-eval corpus pages — the only untried high-ceiling lever.
