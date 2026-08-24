# OCR Pipeline Auto-Research Program

## Objective
The goal (maximize MausoleoBench within the budget cap) and success criterion live in `GOAL.md`. This file is the operating manual: budget mechanics, the metric, the experiment contract, adversarial review, the generalization protocol, and the current baseline. Changes must generalize to any 1880–1959 issue.

## Resource Budget
The corpus is **172,600 pages** (1880–1959) on endeavour. Full-corpus OCR runs on ripperred (2×3090).

| | sec/page | wall on 2×3090 |
|---|---|---|
| target | **103.5** | ~15 weeks |
| hard cap | **200.0** | ~30 weeks |

- **Disqualification line = 200.0 sec/page** (raised 5× to 250.0 on 2026-07-22, tightened to 200.0 on 2026-08-24, per Elio; originally 50). A config that beats the score but exceeds the cap is a research artifact, not a production candidate. At this cap the Qwen3-VL-8B routes are now IN budget — archived **exp_045** (column-structured 8B, MausoleoBench 0.4615 @ ~136 sec/page) and **exp_168** (8B per-region, 0.4342 @ ~150) were DQ'd only on budget and are now the top opportunity to beat the record (0.4071). (The 0.5266 two-8B merge at ~286 is still just over the cap.)
- **GT segmentation convention (per Elio, 2026-08-24): ONE ARTICLE PER PRINTED UNIT.** Every distinct advert, classified entry, notice and filler is its own article — never lumped under a rubric heading. Current GT violates this: 1952 `VILLINI CASE TERRENI` is a single 33,119-char article covering a whole classified block, and 1935/1925 have the same pattern. A model that correctly emits individual adverts is penalised twice (the GT block counts as a miss, each advert as a false positive), which is most of why 1952 scores 0.370 against 1895's 0.760. Detect over-merges by counting how many printed content units a GT article's text spans in a unit-marked transcription.

- **Budget is measured by the caller, not the experiment.** `scripts/time_experiment.sh experiments/<name>.py` times the whole run over the 6 eval issues end-to-end (model load + layout + OCR + grouping + IO) and reports `sec_per_page = wall_seconds / 42`. That number is `sec_per_page` in the log.
- `BUDGET_CAP` in `scripts/progress_server.py` is the single source of truth; the dashboard recomputes `budget_ok` from `sec_per_page` vs the cap.
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
1. **Any gain counts.** If the 6-issue average improves, it's a gain — accept it. No minimum effect size. On an exact tie, prefer the simpler variant.
2. **Holdout halves.** For filter/threshold changes, verify on the odd-indexed GT half; a change that wins on tune but regresses on holdout is overfit.
3. **Unsupervised probes.** Before promoting a structural change, run it on a probe issue and confirm `scripts/research.py probe` numbers don't degrade.
4. **Mechanism rule.** Every accepted change gets a one-line "why this generalizes" note in the log. No per-issue hyperparameters.

## Research Toolbox
The goal is the best MausoleoBench **within budget**; the constraint is speed, not creativity. Beyond model/prompt swaps, these are all in scope (still: never touch the eval, self-contained + independent, caller-measured, adversarial-reviewed):
- **Speed-engineering** to make a strong approach fit ≤50 sec/page: quantization (awq/fp8/int8), batching/`max_num_seqs`, crop strategy (fewer/larger crops, resolution), persistent models, vllm flags.
- **Train / distill** budget-fit models (the trained boundary grouper is one; distilling a strong OCR/segmentation model into a cheap one is on the table).
- **Consult Fable** (subagent) for strategy, design, or hard implementation.
- **Research online** (WebSearch/WebFetch) for approaches, model releases, techniques.

## Baseline / Production pipeline
Budget-compliant **record: exp_017 = 0.4275** @ 181.98 sec/page (`experiments/exp_017_column8b_direct.py`): Ray-free column-structured 8B — 3-column split → Qwen3-VL-8B structured-JSON → MergePages → ParseIssue, operators called directly (VlmOcr loaded once). Beats the prior record by +0.0204 within the 250 cap, but is FRAGILE: carried by 1925 (0.675) / 1910 (0.566), collapses on dense layouts (1895 = 0.128, blob/repetition) — a real but uneven, expensive (21×) gain. Reproduces the archived exp_045 config (0.4615 on stale GT → 0.4275 on current GT).

Prior record (cheap + uniform): **exp_009 = 0.4071** @ 8.66 sec/page (`experiments/exp_009_article_fillguard.py`): PP-DocLayoutV3 regions → trained boundary grouper (`experiments/grouper_features.py`) → each article's union-bbox crop OCR'd by PaddleOCR-VL-1.6, with a geometric fill-ratio guard. Climb: 0.3826 → 0.3946 → 0.4071. Oracle references (far over budget, GT-building only): `ensemble_30min` 0.5941, `ensemble_prune5` 0.5622.

**Solution space mapped (this slate):** segmentation is solved cheaply by the grouper; text quality is the bottleneck. Winning recipe = specialized PaddleOCR-VL-0.9B + article-level context + geometry. Ruled out with data: all general VLMs (Qwen 2B/4B/8B, AWQ) and historical-tuned CHURRO-3B lose at any crop size; column-structured JSON blobs; CLAHE preprocessing hurts; **LoRA fine-tuning is fragile** — naive real-GT overfits (net −0.003), synthetic-augmented hallucinates (−0.130, the OCR model drifts to generation under SFT). The only untried high-ceiling lever is **oracle-ensemble distillation** on non-eval corpus pages (real-domain transcription labels, which — unlike literary synthetic — did NOT cause hallucination in the naive run; ~1 day teacher labeling; uncertain). 0.4071 was the practical ceiling under the old 50 sec/page budget; the 5x raise to 250 (2026-07-22) reopened the expensive 8B column routes, and exp_017 now holds the record at 0.4275 @ 181.98 sec/page.

## Running the Loop
Follow `LOOP.md` (current state + queue). Each iteration: advance ONE experiment, measure budget via the caller, adversarial-review, log a line to `mausoleobench_log.jsonl` (`{n, exp, score, description, sec_per_page, budget_ok}`, n = last+1) so the live graph (`scripts/progress_server.py`, :8078 + cloudflare, with per-experiment prediction viewer) updates, commit + push, update `registry.md` and `LOOP.md`. Run compute on ripperred GPU1.
