# Goal

Maximize **MausoleoBench** — the OCR quality score (`src/mausoleo/eval/evaluate.py`) on the 6-issue eval — **subject to the compute budget**. The objective is the best score achievable *within budget*; score alone does not count.

## Success
Beat the current budget-compliant record — **exp_017 = 0.4275 @ 181.98 sec/page** (prior: exp_009 = 0.4071 @ 8.66) — on the 6-issue average, within the budget cap, verified by the caller and passing adversarial review. Then ship the best budget-compliant pipeline. (Reference: oracle ceiling 0.5941.) Open lever: exp_017's gain is fragile/expensive — a cheaper or more uniform route that keeps ≥0.4275 is the next target.

## The one hard constraint
**Budget cap = 200 sec/page**, caller-measured (`scripts/time_experiment.sh`). Over the cap → not a candidate. The live value is `BUDGET_CAP` in `scripts/progress_server.py` (single source of truth).

## Everything else lives elsewhere — don't restate it here
- **How to operate** (metric definition, experiment contract, adversarial-review checklist, generalization protocol, budget mechanics): `eval/autoresearch/program.md`.
- **Approach families + what's ruled out**: `eval/autoresearch/registry.md`.
- **Live state, current queue, next levers**: `eval/autoresearch/LOOP.md`.
