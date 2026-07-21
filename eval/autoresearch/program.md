# OCR Pipeline Auto-Research Program

## Objective
Maximize **MausoleoBench** on the 6-issue eval with changes that generalize to any historical Italian newspaper issue (1880–1959), subject to the Resource Budget. The optimization target is the best score achievable **within budget** — score alone is not enough.

## Resource Budget
The corpus is **172,600 pages** (1880–1959) on endeavour. Full-corpus OCR runs on ripperred (2×3090).

| | sec/page | wall on 2×3090 |
|---|---|---|
| target | **20.7** | ~21 days |
| hard cap | **50.0** | ~50 days |

- **Disqualification line = 50.0 sec/page.** A config that beats the score but exceeds the cap is a research artifact, not a production candidate.
- **Budget is measured by the caller, not the experiment.** `scripts/time_experiment.sh experiments/<name>.py` times the whole run over the 6 eval issues end-to-end (model load + layout + OCR + grouping + IO) and reports `sec_per_page = wall_seconds / 42`. That number is `gpu_s_per_page` in the log.
- `BUDGET_CAP` in `scripts/progress_server.py` + `scripts/seed_progress.py` is the single source of truth; the dashboard recomputes `budget_ok` from `gpu_s_per_page` vs the cap.
- The measure is a conservative upper bound (it includes one-time load that amortizes at corpus scale), so passing here means passing on the corpus.

## Eval Metric — MausoleoBench
`evaluate_issue()` in `src/mausoleo/eval/evaluate.py`. Quality-gated F1, never modified to move a score:

`MausoleoBench = 0.40·(1−wCER) + 0.35·gated_F1 + 0.05·ordering + 0.10·(1−hCER) + 0.10·gated_page`

- **Matching**: best-first global-greedy over (GT, pred) word-overlap pairs, one pred per GT.
- **wCER**: length-weighted char error over ALL GT articles (unmatched → 1.0), per-article cap 1.0.
- **Quality gate**: a match earns structure credit `q = max(0, 1 − 1.5·cer)` — text with >66% CER earns nothing, so garbage text can't buy F1. `gated_F1` and `gated_page` use Σq.
- **ordering**: Spearman squared-displacement over matches with cer ≤ 0.5.
- **hCER**: headline CER over all GT-with-headline (unmatched → 1.0).

Adversarial invariants pinned in `scripts/eval_probes.py` (scramble / page-blob / full-blob / clean) — run after ANY metric change. Always evaluate all 6 issues and report the average.

## Experiment Contract
- **Self-contained script** `experiments/<name>.py <date...>` writing `eval/predictions/<name>_<date>.json`.
- **Never import or run the eval.** No `evaluate_issue`, no reading `ground_truth.json` to score or to pick a hyperparameter. Fixed a-priori choices only. Scoring happens outside via `scripts/research.py eval`. (Training a model on GT labels then applying it is fine; scoring/selecting against the eval is not.)
- **Independence + cold.** Reproducible from scratch, not sped up by or dependent on another experiment's cached artifacts. Caches (vllm prefix/KV, offline-trained models) are fine as long as they start cold. Running on a fresh machine reproduces the same predictions and cost.
- **Budget measured by the caller** (`scripts/time_experiment.sh`), not self-reported.

## Adversarial Review (every experiment, before its number is trusted)
Check for matcher-gaming and state a real-vs-gamed verdict + caveats in the log entry:
1. Overgeneration — preds ≫ GT count.
2. Giant blobs / cer>1 matches.
3. Holdout-half regression (`scripts/research.py holdout`).
4. GT-free probe degradation (`scripts/research.py probe`: lexicon validity, repetition, tiny/huge share) vs the baseline.

A high score riding on degraded probes, spam, or overgeneration is rejected, not logged as a win.

## Generalization Protocol
1. **Effect-size floor.** Accept a change only if the 6-issue average improves by ≥ 0.005 AND the delta is non-negative on a majority of issues. Below-floor deltas are ties — prefer the simpler variant.
2. **Holdout halves.** For filter/threshold changes, verify on the odd-indexed GT half; a change that wins on tune but regresses on holdout is overfit.
3. **Unsupervised probes.** Before promoting a structural change, run it on a probe issue and confirm `scripts/research.py probe` numbers don't degrade.
4. **Mechanism rule.** Every accepted change gets a one-line "why this generalizes" note in the log. No per-issue hyperparameters.

## Baseline
Budget-compliant **record: exp_167 = 0.3815** (trained per-region boundary grouper over PP-DocLayout regions + PaddleOCR-VL text). Oracle references (not production, cost far over budget): `ensemble_30min` 0.5941, `ensemble_prune5` 0.5622.

Segmentation is solved cheaply by the trained grouper (`experiments/grouper_features.py`). The open bottleneck is **budget-fit OCR text quality**.

## Running the Loop
Follow `LOOP.md` (current state + queue). Each iteration: advance ONE experiment, measure budget via the caller, adversarial-review, log a line to `mausoleobench_log.jsonl` (`{n, config, exp, score, description, gpu_s_per_page, budget_ok}`, n = last+1) so the live graph (`scripts/progress_server.py`, :8078 + cloudflare, with per-experiment prediction viewer) updates, commit + push, update `registry.md` and `LOOP.md`. Run compute on ripperred GPU1.
