# Goal

Maximize **MausoleoBench** — the OCR quality score (`src/mausoleo/eval/evaluate.py`) on the 6-issue eval — **subject to the compute budget**. The objective is the best score achievable *within budget*; score alone does not count.

## Success
Beat the current budget-compliant record — **exp_018 = 0.6115 @ 8.55 sec/page** (hosted Gemini 3.7-flash; best LOCAL route is exp_017 = 0.4263 @ 181.98) — on the 6-issue average, within the budget cap, verified by the caller and passing adversarial review. Then ship the best budget-compliant pipeline. (Reference: oracle ceiling 0.5941 — exp_018 now exceeds it.)

Two open levers: (a) exp_018 is a HOSTED route, so its `sec_per_page` is API latency, not GPU time, and the real corpus constraint is ~$5.7k in API spend — a local route that approaches 0.61 is worth more than the score gap suggests; (b) exp_018 has high run-to-run variance (0.6258 vs 0.6045 on identical runs) — a more stable route at the same score is a genuine improvement.

## The one hard constraint
**Budget cap = 200 sec/page**, caller-measured (`scripts/time_experiment.sh`). Over the cap → not a candidate. The live value is `BUDGET_CAP` in `src/mausoleo/paths.py` (single source of truth).

## Strategy (per Elio, 2026-08-24)
Two stages, and they have DIFFERENT objectives:
1. **Find the best possible Gemini-based configuration.** It is the TEACHER, not the production route, so its budget and latency barely matter — it runs once over a training set, not 172,600 pages. Optimise purely for quality: multi-sample consensus, expensive prompting, ensembling are all fair game.
2. **Distil that teacher into a local model** (PaddleOCR-VL or similar). The distilled student is what meets the sec/page budget and runs the corpus.

This supersedes the earlier framing where the hosted route's API cost was the blocker. The corpus economics question now applies to the STUDENT; the teacher only needs to be good.

## Everything else lives elsewhere — don't restate it here
- **How to operate** (metric definition, experiment contract, adversarial-review checklist, generalization protocol, budget mechanics): `eval/autoresearch/program.md`.
- **Approach families + what's ruled out**: `eval/autoresearch/registry.md`.
- **Live state, current queue, next levers**: `eval/autoresearch/LOOP.md`.
