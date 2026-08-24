# Goal

Maximize **MausoleoBench** — the OCR quality score (`src/mausoleo/eval/evaluate.py`) on the 6-issue eval — **subject to the compute budget**. The objective is the best score achievable *within budget*; score alone does not count.

## Success
Beat the current budget-compliant record — **exp_018 = 0.6115 @ 8.55 sec/page** (hosted Gemini 3.7-flash; best LOCAL route is exp_017 = 0.4263 @ 181.98) — on the 6-issue average, within the budget cap, verified by the caller and passing adversarial review. Then ship the best budget-compliant pipeline. (Reference: oracle ceiling 0.5941 — exp_018 now exceeds it.)

Two open levers: (a) exp_018 is a HOSTED route, so its `sec_per_page` is API latency, not GPU time, and the real corpus constraint is ~$5.7k in API spend — a local route that approaches 0.61 is worth more than the score gap suggests; (b) exp_018 has high run-to-run variance (0.6258 vs 0.6045 on identical runs) — a more stable route at the same score is a genuine improvement.

## The one hard constraint
**Budget cap = 200 sec/page**, caller-measured (`scripts/time_experiment.sh`). Over the cap → not a candidate. The live value is `BUDGET_CAP` in `scripts/progress_server.py` (single source of truth).

## Everything else lives elsewhere — don't restate it here
- **How to operate** (metric definition, experiment contract, adversarial-review checklist, generalization protocol, budget mechanics): `eval/autoresearch/program.md`.
- **Approach families + what's ruled out**: `eval/autoresearch/registry.md`.
- **Live state, current queue, next levers**: `eval/autoresearch/LOOP.md`.
