# Phase 1: OCR Quality Research

> **The live program is `eval/autoresearch/program.md`** (objective, budget, metrics, generalization protocol) **and `registry.md`** (approach families); every experiment is in `log.jsonl`. This file only states what phase 1 is, when it is done, and the open bets. The pre-2026-07 design doc (Ray operator framework, model sweeps, bootstrap GT, 30-min constraint) is superseded — git history.

## What phase 1 is now

The autoresearch loop hillclimbing composite_v2 under the hard corpus budget (6.9–13.9 GPU-s/page steady-state on 2×3090). Experiments are self-contained scripts (`experiments/README.md`); the cycle runs via `scripts/research.py` (eval + audit + holdout + probe), one variable per experiment, everything logged. The legacy Ray harness survives only for the oracle ensembles and `exp_157`. Metrics and GT are never touched to improve a score.

## Status (2026-07-17, composite_v2)

| Config | v2 avg | GPU-s/page | Role |
|---|---|---|---|
| `ensemble_prune5` | 0.7776 | oracle-tier | quality upper bound |
| `ensemble_30min` | 0.7514 | ~600 | recall oracle (1.0/0.98), GT building |
| `exp_157` Paddle-VL + YOLO titles | **0.4284** | **5.13** | production candidate, in budget |

The production–oracle gap is **recall/segmentation** (exp_157 recall 0.36–0.49), not the accuracy of matched text. GT: 2 issues human-verified (1885/1910-06-15); 4 era-diverse drafts (1895/1925/1935/1952-06-15) in human review via `eval/tentative_gt/`.

## Ship bar for corpus v1 (proposed — needs Elio's sign-off)

On the promoted 6-era GT set, one in-budget config with: composite_v2 ≥ 0.60 avg, ≥ 0.50 on every issue (no era collapse), recall ≥ 0.70 avg, ≤ 13.9 GPU-s/page, no probe degradation on the 1943 set. Rationale: phase 3 can summarize noisy text but never recovers articles OCR missed — recall losses are permanent. Corpus v0 may run before the bar (02_corpus_run.md). If the bar proves unreachable within budget, that's a product decision (relax bar vs budget), not something to paper over.

## Open bets (statuses in registry.md)

- **F3 layout/reading-order** — PP-DocLayoutV3 regions; top bet for recall.
- **F1 sub-1B refinements** — Paddle abandon-class filtering, region overlap handling.
- **F7 segmentation adapters** — embedding-similarity article grouping.
- **F4 oracle-only** — precision-filtering prune5; Paddle as diversity source (GT quality, not production).
- **BLOCKED**: F2 long-horizon parsing (cross-page continuity — top failure category, waiting on a new model release), HunyuanOCR, GLM-OCR, olmOCR.
