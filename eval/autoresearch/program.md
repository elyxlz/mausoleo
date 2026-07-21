# OCR Pipeline Auto-Research Program

## Objective
Maximize average composite score on the two GT issues (1885-06-15, 1910-06-15) with changes that **generalize to any historical Italian newspaper issue (1880–1959)** — subject to the corpus-scale Resource Budget below. The optimization target is **best composite achievable within the per-page GPU budget**, not composite alone.

## Resource Budget (2026-07-16, per Elio — supersedes the old 30-min/issue constraint)
Goal: OCR the ENTIRE corpus (1880 → 1959, ~175K pages ≈ 29K issues; ~1.07M pages exist in total through 1996) on ripperred in ~1 week, 2 weeks absolute max.

| Target | GPU-s/page | GPU-s/issue (avg 6pp) | wall/issue on 2 GPUs |
|---|---|---|---|
| 1 week | **6.9** | ~42 | ~21 s |
| 2 weeks (hard cap) | **13.9** | ~84 | ~42 s |

- Budget accounting is **steady-state throughput** (continuous batching across issues, model loaded once), not cold-start single-issue latency. Measure with ≥2 issues back-to-back, exclude model load.
- Every experiment/run MUST record measured GPU-s/page; a config that beats the score but exceeds ~14 GPU-s/page is a research artifact, not a production candidate.
- Reference points (MEASURED 2026-07-16): the 8-source ensemble_30min ≈ 600 GPU-s/page (~40–90× over budget — dead as production). One Qwen3-VL-8B col3 vllm pass = **74 GPU-s/page (1885) / 178 (1910), avg ~136** — 5–26× over budget, NOT near the 1-week line as previously assumed; decode volume (~13K chars/crop on dense 1910 pages at ~63 tok/s eager) is the binding cost. Sub-1B specialized models are the only budget-feasible family measured so far: PaddleOCR-VL-1.6 ≈ 10 GPU-s/page (exp_149).
- Old 30-min framing and its ensembles remain useful as **oracle/reference predictions** for GT work and quality upper bounds.

Every experiment ever run is in `log.jsonl` (prior program versions archived outside the repo: `~/mausoleo_archive/`).

## Eval Metrics
`evaluate_issue()` in `src/mausoleo/eval/evaluate.py` (never modify to improve a score; metric changes only via a documented reward-hacking audit like eval_review.md):
- **CER / wCER**: character error rate per matched article / length-weighted (lower better)
- **hCER**: headline CER (lower better)
- **Recall / F1**: GT article match rate (higher better)
- **Ordering**: Spearman squared-displacement (higher better)
- **Page accuracy**: correct page_span fraction (higher better)
- **Composite (v2, 2026-07-16)** = 0.40·(1−wCER_all) + 0.25·F1 + 0.15·ordering + 0.10·(1−hCER) + 0.10·page_accuracy — wCER over ALL GT articles with per-article cap 1.0; ordering=0 when <2 matches; hCER per-article capped. See `eval_review.md` for the reward-hacking audit that motivated v2. Pre-2026-07-16 log numbers are v1 (`composite_v1_score` field still computed for comparison).

Always evaluate BOTH dates and report the average. Report precision/F1 alongside composite; a change that drops precision >5pts needs explicit justification. The holdout rule covers structural changes that filter/drop articles, not just hyperparameters. Pipeline code must never read GT at inference nor re-emit another config's prediction file as its own.

## Current Baselines (MausoleoBench, 6-issue eval — fresh slate 2026-07-21)
Metric is **MausoleoBench** (`src/mausoleo/eval/evaluate.py`; quality-gated F1, see §Metric). composite_v2 and all 2-issue numbers are archived (`eval/autoresearch/archive/`). Old experiment scripts live in `experiments/archive/`. Eval = 6 human-verified all-article issues: 1885, 1895, 1910, 1925, 1935, 1952 (June 15).

| Config | 6-issue MausoleoBench | Role | GPU cost |
|---|---|---|---|
| `configs/ocr/ensemble_30min.py` | **0.5941** | recall oracle, ceiling reference | ~600 GPU-s/page |
| `configs/ocr/ensemble_prune5.py` | 0.5622 | pruned oracle reference | oracle-tier |
| `exp_045_qwen3vl_vllm` | **0.4615** | **current best real pipeline** | ~136 GPU-s/page (over budget) |
| `exp_140_yolo_smallregion_vllm` | 0.4170 | YOLO-region + vllm | budget-range |
| `exp_160_ppdoclayout_headblocks` | 0.3576 | PP-DocLayout head-blocks (ex-composite_v2 leader) | ~6.2 GPU-s/page |

**MausoleoBench reshuffled the board vs composite_v2**: the quality gate exposed that exp_160's high recall was bought with low-quality (high-CER) matched text. The structured-JSON Qwen3-VL route (exp_045) now leads on *text quality × segmentation jointly*. The climb: lift a budget-fittable pipeline from 0.46 toward the 0.59 oracle ceiling — either raise exp_045's segmentation, or graft PP-DocLayout recall onto Qwen3-VL text quality, or train the grouper. exp_045 at 136 GPU-s/page is over the 6.9–13.9 budget, so a production win must also be cheap.

Reproduce: `uv run --no-project python scripts/run_real_ocr.py ensemble_30min 1885-06-15 1910-06-15` → `eval/predictions/ensemble_30min_<date>.json` (sub-pipeline predictions cached as `<name>_<date>.json`).

The 8-source/30-min architecture is at its local optimum. Further gains need **new sources or new architecture**, not retuning.

## Generalization Protocol (anti-overfitting — MANDATORY)
All tuning so far used the same two issues that produce the headline score; ±0.0001 "wins" at saturation are noise-fitting. Rules:

1. **Effect-size floor.** Accept a change only if avg composite improves by ≥ 0.002 (ensemble/merge changes) or ≥ 0.005 (single-source/config changes), AND the delta is non-negative on both dates. Below-floor deltas are ties — prefer the simpler variant.
2. **Holdout halves.** For merge/ensemble hyperparameter tuning: tune on even-indexed GT articles, verify on odd-indexed (`scripts/research.py holdout`). A change that wins on the tune half but regresses on the holdout half is overfit — reject.
3. **Unsupervised probes.** Fixed probe set of unlabeled issues from a different era: **1943-07-03, 1943-07-15, 1943-07-25** (images in `eval/ground_truth/<date>/`, no GT). Before promoting any structural change to the production config, run it on ≥1 probe issue and check `scripts/research.py probe` outputs (lexicon validity rate, repetition rate, article count/length distribution) don't degrade vs the baseline's probe numbers. Catches era-specific overfitting the 1885/1910 pair can't.
4. **Mechanism rule.** Every accepted change gets a one-line generalization argument in `log.jsonl` ("why this helps any issue, not just these two"). No per-date hyperparameters, ever.
5. **Known noise floors.** Single Qwen3-VL transformers runs: ±0.15 composite run-to-run. vllm single-source runs: ~±0.01. Full 8-source ensemble: ~±0.002. Only trust deltas well above the relevant floor; when in doubt re-run once.

## Research Directions (July 2026 refresh)

### TIER 1: New-generation OCR models (downloads now ALLOWED — see policy)
Ripperred: vllm 0.19.1, torch 2.10, transformers 5.5.4. Disk was 99% full (30GB free) — `df -h ~` before downloading; prune stale non-OCR HF cache entries (Audiogen mocks, whisper, CLAP, MERT) if needed.

- **1A. Baidu Unlimited-OCR** (3B MoE, 500M active, ~6GB, MIT, vllm+transformers). One-shot long-horizon parsing: 40+ pages in a single forward pass via Reference Sliding-Window Attention. Directly attacks the **cross-page article truncation** failure category (biggest quality problem). Two integration modes to try: (a) whole issue, multi-page mode (`<image>Multi page parsing.`, 1024px/page — may be too low-res for broadsheet body text); (b) **all column crops of the issue in reading order in one pass** — model keeps context across crops, so articles continue naturally across columns AND pages. Referenced in README by Elio.
- **1B. GLM-OCR** (zai-org, 0.9B dense, ~2GB, 128K ctx, MTP speculative decoding). Tops OmniDocBench ~94.6. Tiny → cheap extra ensemble family; check vllm 0.19.1 has `glm_ocr` arch, else pin newer vllm in a Ray runtime_env or standalone venv.
- **1C. PaddleOCR-VL-1.6** (0.9B, vendor-reported 96.33 OmniDocBench v1.6). Layout+reading order+OCR in one pass.
- **1D. olmOCR-2-7B** — ALREADY CACHED on ripperred (`allenai/olmOCR-2-7B-1025`). Previously dismissed because it ignores the V2 JSON prompt — that was misuse. Use its native prompt → markdown, then the markdown→articles adapter (Tier 2).
- Note all these output **markdown/text, not our article JSON** — they need Tier 2 to become ensemble sources.

### TIER 2: Markdown→articles adapter (unlocks Tier 1) — DONE 2026-07-16
`MergeMarkdownPages` (src/mausoleo/ocr/operators/merge_markdown.py): headings/bold lines → headlines, blank-line blocks → paragraphs, crop provenance → page_span. Every markdown-native OCR model is now a candidate source. Reality check from exp_147/150: specialized models often emit NO headings on newspaper content — region-level detection (YOLO) provides the segmentation instead.

### TIER 3: Layout & cross-page
- **PP-DocLayoutV3** (RT-DETR 31M, Apache-2.0, newspaper class, predicts reading order): replaces DocLayout-YOLO + heuristics. Needs paddle runtime — keep in own venv/runtime_env.
- Cross-page stitching via Unlimited-OCR sequential context (see 1A-b) — likely supersedes the regex/LLM stitching ideas.
- Ar-Q-Former (ICDAR 2025) article separation — check for released weights before investing.

### TIER 4: Ensemble improvements (marginal at saturation)
Only worthwhile with NEW diverse sources from Tiers 1–2. Cross-model-family diversity is the proven lever (+0.026 last session from Qwen2.5+Qwen3 stacking). A genuinely different architecture family (MoE long-horizon, 0.9B specialized) should add more than another Qwen variant.

## Policy Changes (2026-07-16, per Elio)
- **Model downloads ALLOWED** on ripperred (was forbidden). Justify each download in the log; check disk first.
- Still forbidden: modifying eval metric code, ground truth, hardware assumptions (2× RTX 3090 24GB).
- Quality remains paramount; vllm preferred; max_model_len high enough to never truncate (32768 default).

## Orchestration Discipline (long-horizon research rules)
1. **Approach registry.** `eval/autoresearch/registry.md` groups all work into approach FAMILIES by underlying mechanism, not surface wording (the same model tried on col4 vs col5 is one family). Read it at the start of every iteration; update it at the end of every iteration. If recent experiments cluster in one family, deliberately redirect the next ones toward underexplored ACTIVE families.

2. **Blocked-route rule.** When an approach stalls on a hard failure (model loops, dependency wall, systematic quality floor), mark the route BLOCKED in the registry with the exact failure and a concrete UNBLOCK CONDITION. Never re-attempt a BLOCKED route without a materially new mechanism — a prompt rewording or hyperparameter nudge does not qualify.

3. **No false progress.** Re-tuning ensemble weights at a saturated optimum, or adding a source highly correlated with existing sources, is not progress even if the composite ticks up within noise. Verify genuine diversity (pairwise text distance, LOO contribution) before crediting a new source. Respect the per-issue time budget in program.md — a "win" that violates the budget is not a win.

4. **Parallel incompatible routes.** Keep ≥2 structurally different directions alive across rounds (e.g. fast small-model sources AND layout upgrades AND long-horizon parsing). Do not let one route monopolize iterations because its next step is easiest. Cross-pollinate only after each side works standalone.

5. **Adversarial audit before accepting ANY change** (all must pass):
   - both eval dates same-sign delta, above the noise floor for the change class
   - tune/holdout halves: no holdout regression (`scripts/research.py holdout`)
   - probe issues: no degradation in lexicon validity / repetition / length distribution (`scripts/research.py probe`)
   - matcher-gaming check: no giant blob articles inflating text_overlap; article length distribution sane
   - silent-truncation check: outputs not bumping against max_tokens/max_model_len
   - per-issue time budget respected (program.md Resource Budget section)

6. **Concrete artifacts only.** Every log entry names specific articles/failure cases that improved or worsened (from actually reading predictions vs GT). "Score went up, looks good" is a status report, not evidence — reject it.

7. **Persistence.** Do not conclude "saturated" or end the session because a wave of experiments failed. Failed waves are information: update the registry, pick the next-best family, launch the next round. Stop only when the user stops you or every ACTIVE family is blocked pending user input. Report intermediate state honestly: strongest verified gain + exact remaining gaps, never vague optimism.

8. **Search policy.** Web search is for integration knowledge (model usage, APIs, versions) — never to import benchmark numbers as our results.

## Running the Loop
Start with "continue the autoresearch" (or `/loop` with a prompt pointing here). The loop is self-paced: long GPU runs go in background tasks, ScheduleWakeup provides the heartbeat, and each iteration ends by rewriting `LOOP.md` (the live state + queue file next to this one) and scheduling the next wakeup with a prompt that just says to read `LOOP.md` and continue. All compute on ripperred; never endeavour (corpus storage only). Commit and push accepted work as you go.

## The Loop
1. **Read state** — this file + last entries of `log.jsonl`
2. **Propose** — ONE change (one variable at a time)
3. **Write the experiment** — `experiments/exp_NNN_desc.py` (next NNN from log), self-contained per `experiments/README.md`: `<date...>` argv in, `eval/predictions/<name>_<date>.json` out, implementation free (Ray not required). Oracle ensembles stay on the legacy `configs/ocr` harness.
4. **Run** — on ripperred, BOTH GT dates (plus probe dates when relevant)
5. **Evaluate** — `scripts/research.py eval <name>` (audit + holdout); apply the Generalization Protocol
6. **Inspect** — read actual predictions vs GT and the run log (never judge from the score alone)
7. **Decide** — accept (update baselines here) or revert; log to `log.jsonl` either way, with mechanism line
8. **Report + schedule** — one-line result; ScheduleWakeup for next iteration

## Infrastructure
Sync (never sync predictions/GT):
```
rsync -avz --exclude='.venv' --exclude='.git' --exclude='__pycache__' --exclude='eval/predictions' --exclude='eval/ground_truth' -e 'ssh -p 62022' ./ audiogen@81.105.49.222:~/mausoleo_di_roma/
```
Run (sequential, both dates; clean pycache first; `--force` to overwrite predictions after code changes):
```
ssh audiogen@81.105.49.222 -p 62022 "cd mausoleo_di_roma && find src/ -name __pycache__ -exec rm -rf {} + 2>/dev/null; .venv/bin/python scripts/run_real_ocr.py <config> 1885-06-15 && .venv/bin/python scripts/run_real_ocr.py <config> 1910-06-15"
```
Fetch:
```
scp -P 62022 audiogen@81.105.49.222:~/mausoleo_di_roma/eval/predictions/<config>_<date>.json eval/predictions/
```
Evaluate:
```python
from mausoleo.eval.evaluate import evaluate_issue
import json
scores = []
for date in ["1885-06-15", "1910-06-15"]:
    gt = json.loads(open(f"eval/ground_truth/{date}/ground_truth.json").read())
    pred = json.loads(open(f"eval/predictions/<config>_{date}.json").read())
    scores.append(evaluate_issue(gt, pred, config="<config>", date=date).composite_score)
print(sum(scores) / len(scores))
```
Ripperred quirks: SSH may time out transiently — retry. Check `nvidia-smi` before runs. Run experiments sequentially (parallel runs on separate GPUs have stalled/died silently).

## Key Learnings (condensed — full history in `history/` + `log.jsonl`)
- **Diversity beats tuning**: cross-model-family sources (Qwen2.5-VL + Qwen3-VL, yolo vs column splits) are where ensemble gains come from; column-split sources of the same model are nearly identical (pairwise distance <0.15), yolo is different (~0.45).
- **JSON-blob/repetition trimming was the single biggest win (+0.0165)** — VLM emits raw JSON blobs that poison the matcher; `trim_predictions` drops them.
- **Coverage vs text-quality decoupled**: additive-only merge (`replace_ratio=100.0`) for noisy-high-recall sources; replacement chain for quality sources.
- Keep prompts minimal (V2); complex prompts (V3, content-type lists) cause hallucination. Do NOT disable Qwen3-VL thinking.
- vllm ≫ transformers for speed at equal quality; Qwen3-VL registered in vllm 0.19.1.
- MergePages must override page_span from layout_json unconditionally (VLM always guesses [1]).
- max_model_len only affects vllm; too-low values silently truncate output.
- 4B-and-under Qwen models fail the structured-JSON prompt; 8B works. (Specialized 0.9B OCR models with native prompts are a different story — untested.)
- 1885 page_accuracy ceiling ≈0.68 is probably GT annotation error (all sources independently agree on the "wrong" pages for 12 articles).

## Dead Ends (do not retry without new information)
Full-page OCR without splits (catastrophic on 1910, single-model) · V3/ads-complex prompts · /no_think · grayscale/resize preprocessing · LLM post-correction with ≤7B (empty/repetitive) · length-aware body replacement (−0.29) · headline length-bonus / consensus voting · heuristic cross-page stitching (zero stitches) · Italian accent restoration · col7/col8 splits · overlap 6% (CER 2.35) · max_tokens>8192 (no gain, 2× slower) · InternVL3-8B (hallucinates placeholder text) · Nanonets OCR-s/OCR2-3B (vllm tied lm_head) · Qwen3-VL-32B-AWQ (OOM) · Qwen3-VL-30B-MoE-AWQ (scrambled) · Qwen3-VL-8B-Thinking (too slow) · T=0.3 sampling diversity (−0.02) · stochastic YOLO param tweaks (defaults near-optimal).

## Failure Categories (where remaining points are)
1. **Cross-page articles truncated/missed** (CER 0.7–1.7 each) — biggest open problem → Tier 1A.
2. **Non-standard-layout ads missed** on back pages — partially solved by col6+ads prompt; specialized models may do better.
3. **Interleaved fiction strips** (serialized novels along page bottom) confuse column crops.
4. Very long articles (>2000 chars) truncate or hallucinate.
