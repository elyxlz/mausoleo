# Loop state — live queue for the autoresearch loop

Rewritten every iteration. Rules in `program.md` (+ `registry.md`); this file is ONLY the current state + queue. Do not accumulate history here — results go to `mausoleobench_log.jsonl`, mechanisms to `registry.md`.

## Standing context
- Metric = **MausoleoBench** (`src/mausoleo/eval/evaluate.py`). Budget cap = **200.0 sec/page**, caller-measured (`scripts/time_experiment.sh`); `BUDGET_CAP` in `src/mausoleo/paths.py` is the single source of truth.
- Run on ripperred (`.venv/bin/python`), GPU1 (`CUDA_VISIBLE_DEVICES=1`). Experiments are self-contained, never import the eval, are measured cold by the caller, and get an adversarial review.
- Log each full-6-issue run to `mausoleobench_log.jsonl` (`{n, exp, score, description, sec_per_page, budget_ok}`, n = last+1) → live graph `scripts/progress_server.py` (:8078 + cloudflare, per-experiment prediction viewer). Commit + push; update `registry.md`.

## Board
- **RECORD: exp_018 = 0.6247 @ 8.95 sec/page** — Gemini 3.7-flash, full page, JSON array, thinking_budget=128. HIGH VARIANCE: five runs gave 0.6258 / 0.6045 / 0.5768 / 0.5583 / 0.6247 — quote ~0.62 as a mean, never a single draw. `sec_per_page` is API latency, NOT GPU time; corpus cost ~$5.7k.
- **Best LOCAL: exp_009 = 0.4051 @ 12.94 sec/page** — now above exp_017 (0.4029 @ 181.98) at 14x less compute, so the 8B column route has no remaining justification.
- exp_019 flash-lite = 0.3775 — REJECTED, the small tier cannot read this material.
- Reference: oracle ensembles 0.6500 / 0.5792, both far over cap.
- Ruled out (`registry.md`): CHURRO and general VLMs below PaddleOCR-VL at low cost; CLAHE; length-ratio blob guard; naive and synthetic-augmented LoRA; flash-lite.

## Current
- **Metric** now scores paragraphs (0.05, taken from page accuracy) and weights recall 4x precision (RECALL_BETA=2). Both adversarially tested; removing ordering was tried and REJECTED (scrambled output then scored 1.000).
- **GT**: 37 image-confirmed fixes to 1885/1895/1910; 65 blob articles re-split into real paragraphs; 1925/1935/1952 audited and found clean (67 NO_DEFECT, 1 confirmed). Convention: an article is the most sensible SEMANTIC grouping — a classified rubric is ONE article, adverts are paragraphs within it.
- **Board flattened**: 6 live rows, 13 superseded archived. `scripts/` is entrypoints only (108 lines); all logic in `src/mausoleo/`.
- exp_020 (column transcribe + regroup) scored 0.6621 on 1895 vs exp_018's 0.7600 — grouping worked (precision 0.953) but single-sample column text is worse than full-page (wCER 0.361 vs 0.204). The 0.8262 "ceiling" quoted for it was overstated: that oracle also DROPPED units matching no GT article, so it measured grouping plus a perfect junk filter.

## Queue
1. **True-column local route** — every column experiment (exp_003/015/017) used `num_columns=3`; the papers are 4-9 columns (1895 is 5, 1952 varies 4-9). That family has never been fairly tested. Layout bands already computed for all 42 pages. No API cost.
2. **Reduce exp_018's variance** — a spread of 0.068 across five runs is worth more than chasing points. Two-sample consensus per page, or a segmentation guard that rejects anomalous article counts.
3. **Decide the corpus route economically** — exp_018 ~$5.7k in API spend vs a local route at ~500 GPU-days. This is a product decision, not a score decision, and it gates phase 2.
4. **Fix exp_018's over-segmentation on 1935** (419 preds vs 257 GT) and its weakness on 1952 (0.423).
5. **Oracle-ensemble distillation** on non-eval corpus pages — still the untried high-ceiling lever for a local route.
