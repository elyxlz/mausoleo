# Goal

**Maximize MausoleoBench — the OCR quality score on the 6-issue eval — subject to the compute budget.** The objective is the best score achievable *within budget*; score alone does not count.

MausoleoBench is defined in `src/mausoleo/eval/evaluate.py` (quality-gated F1 over article segmentation + text; dominated by character error rate). It is scored over 6 human-verified issues (1885/1895/1910/1925/1935/1952-06-15). Higher is better; the oracle ceiling is 0.5941.

## Success criterion
Produce a pipeline that scores **higher than the current budget-compliant record (exp_009 = 0.4071 @ 8.66 sec/page)** on the 6-issue average, runs **within the budget cap**, verified by the caller and passing adversarial review. Ship the best budget-compliant pipeline.

## The hard constraint — budget
- **Hard cap: 250 sec/page** (caller-measured wall time via `scripts/time_experiment.sh`, over the 6 eval issues). Anything over the cap is a research artifact, not a candidate.
- `BUDGET_CAP` in `scripts/progress_server.py` (+ `scripts/seed_progress.py`) is the single source of truth.
- Corpus: 172,600 pages (1880–1959), run on ripperred (2×3090).

## Rules (never violate — full detail in `eval/autoresearch/program.md`)
- Never modify the eval metric or ground truth to move a score.
- Experiments never import/run the eval; scoring is external (`scripts/research.py eval`).
- Every experiment: self-contained + independent, caller-measured budget, and an adversarial review (overgeneration / giant blobs / holdout regression / GT-free probes on the 1943-07 issues).
- Any real gain counts (no effect-size floor); strict leave-one-issue-out for anything trained on GT.

## Current state
- **Record: exp_009 = 0.4071** — PP-DocLayout regions → trained boundary grouper → each article's union-bbox crop OCR'd by PaddleOCR-VL-1.6, with a geometric fill-ratio guard.
- **Top opportunity (at the raised budget):** the Qwen3-VL-8B column-structured route (archived **exp_045**, 0.4615 @ ~136 sec/page) and exp_168 (0.4342) were disqualified only on budget and are now in-budget — revive/re-measure them caller-side as the path above 0.4071. Then the two-8B merge idea (0.5266, ~286 sec/page — just over the cap) motivates a cheaper complementary-source merge.
- Ruled out with data: general VLMs and CHURRO at budget granularity lose to PaddleOCR-VL; LoRA fine-tuning is fragile (naive overfits, synthetic augmentation hallucinates). Oracle-ensemble distillation is the untried high-ceiling lever.

Operating manual + approach registry: `eval/autoresearch/program.md`, `eval/autoresearch/registry.md`. Live state/queue: `eval/autoresearch/LOOP.md`. Live dashboard: `scripts/progress_server.py` (:8078 + cloudflare).
