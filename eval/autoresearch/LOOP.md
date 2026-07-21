# Loop state — live queue for the autoresearch loop

Rewritten every iteration. Rules in `program.md` (+ `registry.md`); this is only CURRENT state + queue.

## Standing context
- Metric is **MausoleoBench** (`src/mausoleo/eval/evaluate.py`); quality-gated F1. Adversarial invariants pinned in `scripts/eval_probes.py` — run after ANY metric change. Never modify eval/GT to move a score.
- Run everything on ripperred (`.venv/bin/python`); GPU1 preferred (`CUDA_VISIBLE_DEVICES=1`; eve holds ~350MB GPU0). PP-DocLayoutV3 at `~/paddle_env`.
- Fable spend limit may bite → spawn subagents with `model: "opus"` on limit errors.
- Eval = 6 human-verified all-article issues (1885,1895,1910,1925,1935,1952-06-15). Region dumps ready: `semgroup/regions_<date>.json` (all 6).
- **Fresh slate (2026-07-21)**: composite_v2 board + exp_158–166 scripts archived. New experiments start at exp_167. Every full-6-issue run logs one line to `eval/autoresearch/mausoleobench_log.jsonl` (schema: `{n,config,exp,score,description,reference}`) → live graph `scripts/progress_server.py` (:8078, cloudflare tunnel). Append n = last n + 1; re-seed historical points with `scripts/seed_progress.py`.
- Commit + push as you go; update `registry.md` every iteration.

## State (2026-07-21) — fresh slate seeded, climb resumed
- MausoleoBench re-scored all prior full-coverage configs. **Board: oracle 0.5941 · best real exp_045 (Qwen3-VL structured JSON) 0.4615 · exp_140 0.4170 · exp_167 0.3815 · exp_160 0.3576.**
- Key finding: MausoleoBench reshuffled the board — the quality gate discounts exp_160's high-recall/high-CER matches; text quality (F5) is the lever that pays.
- **exp_167 (attempt 11, trained boundary grouper) = 0.3815** — beats ex-production exp_160; recall jumped to 0.57–0.89 (grouping ~solved). Bottleneck moved grouping → text quality (region text is cheap PaddleOCR-VL, wCER 0.42–0.70).
- **exp_168 (attempt 12, F3×F5 graft) = 0.4342** — Qwen3-VL text through the trained grouper, +0.053 over exp_167's PaddleOCR text. **Text quality confirmed as the lever.** Still just under exp_045 (0.4615): per-region OCR < column-crop structured OCR, and 1952 (dense classifieds) stays the killer (0.267).
- **Complementarity finding**: exp_168 BEATS exp_045 on 1885/1895/1935 (1895: 0.427 vs 0.171) while exp_045 wins 1910/1925/1952. The structured route collapses on some issues; the grouper route is consistent but text-capped. An oracle-select / merge of the two, or exp_045-quality text fed through the grouper, is the next high-EV move.
- Progress dashboard live (`scripts/progress_server.py` :8078 + cloudflare). **Fable agent `ae2f8dc36b3a7fe15` is rewriting the dashboard front-end** (interactive graph, sleek, record/all filter); restart the :8078 server to pick up its edit when it reports. Records so far: exp_045 (tops the set); climb is 0.46 → 0.59 ceiling under budget.

## Queue delta (2026-07-21, post exp_169 MERGE — NEW RECORD 0.5266)
- **exp_169 (attempt 13) = 0.5266 — NEW RECORD** (exp_045 0.4615 → +0.065; above the 0.525 oracle-select ceiling). Article merge of exp_045 (primary) + exp_168, fixed 0.50 overlap dedup, GT-free quality_score replace. Per-issue: 1885 .473 / 1895 .435 / 1910 .630 / 1925 .645 / 1935 .533 / 1952 .445.
- **Adversarial review**: gain is REAL, not gaming — the naive union keeps exp_045's blobs + exp_168's clean articles (total_chars ≈ sum of both, ~2× preds), BUT the quality gate zeroes blobs (cer>0.667→q=0), precision penalizes overgen, lexicon_validity (~0.93) & repetition don't degrade vs parents, holdout +0.017. Gain = genuine recall(union) + lower wCER(best-match text). **CAVEAT**: over budget (runs both parents), overgenerates ~2×, ordering degraded (0.46–0.99). It's a ceiling diagnostic, NOT production.
- **Contract enforced (per Elio)**: experiments must NEVER import/run the eval; every experiment gets an adversarial review. Codified in program.md §"Experiment Contract & Adversarial Review"; exp_169 was caught sweeping its threshold against evaluate_issue and rewritten to a fixed threshold.
- Dashboard: oracle ceiling (purple) lines removed per user; interactive graph + Record/All filter live on :8078.

## BUDGET DISQUALIFICATION (per Elio 2026-07-21) — reframes the whole climb
- **An experiment over the 6.9–13.9 GPU-s/page corpus budget DOES NOT COUNT.** Every log entry now carries `gpu_s_per_page` + `budget_ok`; the dashboard frontier/records only step on budget-compliant attempts. Budget cap = 13.9 GPU-s/page.
- **All Qwen3-VL-8B routes are DISQUALIFIED** (~74–286 GPU-s/page): exp_045, 055, 097, 102, 107, 138, 140, 142, 168, 169, 170. The 0.5266 merge does NOT count.
- **Budget-compliant attempts (the only ones that count)**: exp_157 0.1718 (5.1), exp_160 0.3576 (6.2), **exp_167 0.3815 (6.2) = CURRENT RECORD.** All use PaddleOCR-VL-1.6 (0.9B) / PP-DocLayout.
- **The real game**: maximize MausoleoBench WITHIN budget. Segmentation is ~solved cheaply (trained grouper, exp_167). The bottleneck is **text quality from the budget-fit OCR model** — PaddleOCR-VL-0.9B text (wCER 0.42–0.70) vs the disqualified Qwen-8B. Record every experiment's GPU-s/page; disqualify > 13.9.

## State (2026-07-21, later) — budget UI + viewer shipped, exp_172 running
- exp_171 (attempt 15) = 0.3815 NULL (text cleanup no-op; PaddleOCR-VL text already clean → wCER gap is raw recognition, not formatting).
- Dashboard: "Exp N" identity, over-budget = disqualified hollow rings off the frontier, GPU-s/page in tooltips, **each experiment click-opens a read-only prediction viewer (`/viewer?config=`) with marquee-zoom** in a new tab. Live on :8078 + cloudflare (35/35 puppeteer).
- **Budget measured COLD, no cache (per Elio)**: no vllm prefix caching, no reusing cached dumps/preds as free; else DQ. In program.md.
- **exp_172 RUNNING cold on GPU1** (waiter+eval task b7x431qar): full cold pipeline = fresh PP-DocLayout + fresh **Qwen3-VL-2B** region OCR (prefix caching off) + trained grouper, measuring steady-state GPU-s/page. Tests whether a budget-SIZE model beats PaddleOCR-VL-0.9B text while staying ≤13.9. If cold GPU-s/page >13.9 → DQ (informative either way). Cached budget-fit candidates also available: Qwen3-VL-4B, InternVL3-2B, Nanonets-OCR2-3B, GOT-OCR-2.0.

## Next (in order) — budget-compliant only
1. **exp_172 verdict** — if Qwen3-VL-2B is ≤13.9 GPU-s/page AND beats exp_167 0.3815, it's a budget-compliant record. If over budget, try a smaller/faster config (fewer max_tokens, smaller crops) or InternVL3-2B / GOT-OCR-2.0.
2. **Mid-size budget-fit OCR screen** — whichever cached 2–4B model gives the best text within 13.9 cold GPU-s/page + trained grouper.
3. **1952 dense-classifieds** — common weak point across every route (0.19–0.45).
- Both exp_045 & exp_168 are DQ (over budget), so their 0.5266 merge is DQ; a budget-compliant analog needs two budget-fit sources.

## Queue (in order)
1. **exp_167 — trained boundary grouper (top F3 unblock).** Frame grouping as per-region "does region i START a new article?" sequence labeling. (a) `experiments/grouper_features.py` already builds features+labels by aligning the 6 GTs to `regions_<date>.json`. (b) Train a small classifier (gradient-boosted trees / tiny MLP — NOT an 8B LLM; inference ~0 cost). By-issue cross-validation (train 5, test held-out) to avoid fitting 6. (c) Wire as `experiments/exp_167_grouped.py`: PP-DocLayout regions + classifier boundaries + head-block merge; but feed the **Qwen3-VL region text** (F5 quality) not the low-quality path, since MausoleoBench pays for text. Evaluate 6 issues + eval_probes; log to the graph; compare vs exp_045 0.4615.
2. **F3×F5 graft (parallel idea).** PP-DocLayout regions/reading-order as the crop plan feeding Qwen3-VL OCR — combine recall + text quality. Cheapest budget-fitting version wins.
3. **Cost check.** exp_045 is 136 GPU-s/page (over 6.9–13.9 budget). Any production candidate must hit the budget; record GPU-s/page every run.
4. Fold remaining eval-review protocol items (whole-issue holdout, 1943 probe, effect-size sign rule) into program.md before heavy climbing.
