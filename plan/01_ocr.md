# Phase 1: OCR Quality Research

> **The live program is `eval/autoresearch/program.md`** (objective, budget, metrics, generalization protocol, orchestration discipline) **and `eval/autoresearch/registry.md`** (approach families and statuses); every experiment is in `log.jsonl`. This file states only what phase 1 is, when it is done, and the open bets. The pre-2026-07 design doc (Ray operator framework, model sweeps, bootstrap GT, 30-min-per-issue constraint) is superseded — see git history.

## What phase 1 is now

An autoresearch loop hillclimbing composite_v2 on human-verified GT, under a hard corpus budget of **6.9–13.9 GPU-s/page** steady-state on 2×3090:

- **Experiments** are self-contained scripts: `experiments/<name>.py <date...>` → `eval/predictions/<name>_<date>.json` in the Issue schema (`experiments/README.md`). No framework.
- **Cycle**: `uv run python scripts/research.py run <config>` → eval + adversarial audit + holdout + probe; one variable per experiment; every result logged with a mechanism line.
- **Legacy Ray harness** (`configs/ocr/` + `run_real_ocr.py`) survives only for the oracle ensembles (`ensemble_30min` recall oracle, `ensemble_prune5` v2 leader) and the production candidate `exp_157`.
- **Eval integrity**: metrics and GT are never modified to improve a score; metric changes only via a documented reward-hacking audit (`eval_review.md`).

## Where we are (2026-07-17)

| Config | v2 avg | GPU-s/page | Role |
|---|---|---|---|
| `ensemble_prune5` | 0.7776 | oracle-tier | quality upper bound |
| `ensemble_30min` | 0.7514 | ~600 | recall oracle (1.0/0.98), GT building |
| `exp_157` PaddleOCR-VL-1.6 + YOLO titles + squeeze | **0.4284** | **5.13** | production candidate, in budget |
| `exp_045` Qwen3-VL-8B col3 | 0.6305 | ~136 | 5–26× over budget, ensemble source only |

The production–oracle gap is **recall/segmentation** (exp_157 recall 0.36–0.49; wCER_all 0.74–0.77 is dominated by unmatched articles), not the character accuracy of matched text. Every phase-1 point from here comes from finding more articles within budget.

## Ground truth

- Human-verified: 1885-06-15, 1910-06-15 (`eval/ground_truth/<date>/ground_truth.json`).
- In flight: 1895/1925/1935/1952-06-15 machine-drafted via the oracle stack (`eval/tentative_gt/` workflow, `scripts/build_tentative_gt.py`) awaiting human review, then promotion. The ship bar below is defined over this expanded 6-era set.
- Unsupervised probes: 1943-07-03/15/25 (no GT; lexicon validity, repetition, length distributions via `scripts/research.py probe`).

## Ship bar for corpus v1 (proposed — needs Elio's sign-off)

Phase 1 is "good enough to ship phase 2 v1" when a single in-budget config achieves, on the promoted 6-era GT set:

| Criterion | Bar |
|---|---|
| composite_v2 avg | ≥ 0.60 |
| composite_v2 per issue | ≥ 0.50 (no era collapse) |
| Recall avg | ≥ 0.70 |
| Throughput | ≤ 13.9 GPU-s/page steady-state |
| Probes | no degradation vs current exp_157 baselines on the 1943 set |

Rationale: phase 3 can summarize noisy text but cannot recover articles OCR never found — recall losses are permanent, matched-text errors are largely survivable. Corpus **v0** runs before this bar (see 02_corpus_run.md); the bar gates the corpus phase 3 ships on. If the bar proves unreachable within budget, that is a product decision for Elio (relax the bar vs relax the budget), not something to paper over.

## Open technical bets (details + statuses in registry.md)

- **F3 layout/reading-order (top bet for recall):** PP-DocLayoutV3 (31M, newspaper class, predicts reading order) replacing DocLayout-YOLO + heuristics; untested, needs its own paddle env.
- **F1 sub-1B OCR refinements:** abandon-class filtering, horizontal-overlap handling on Paddle regions; other small models mostly BLOCKED (GLM-OCR repetition loops, HunyuanOCR gibberish).
- **F7 segmentation adapters:** embedding-similarity article grouping over region outputs; MergeMarkdownPages exists.
- **F4 oracle improvements (GT quality, not production):** precision-filtering prune5 survivors; Paddle as a cheap diversity source.
- **BLOCKED until new releases:** F2 long-horizon multi-page parsing (cross-page continuity) — both Unlimited-OCR routes dead; cross-page articles remain the top failure category and currently must be attacked via F3/F6 instead.
