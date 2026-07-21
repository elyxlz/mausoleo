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
- **exp_168 RUNNING (GPU1)**: same trained-grouper boundaries but each region re-OCR'd with Qwen3-VL (F3×F5 graft) to test the text-quality headroom. Waiter+eval task `bxoe4ydl0` will report on completion. If it lifts toward 0.46+, text quality is confirmed as the lever and a cheap high-quality-text path is the next target (exp_168 itself is over budget at 8B/region).
- Progress dashboard live (`scripts/progress_server.py` :8078 + cloudflare). Records so far: exp_045 (tops the historical set); climb is 0.46 → 0.59 ceiling under budget.

## Queue (in order)
1. **exp_167 — trained boundary grouper (top F3 unblock).** Frame grouping as per-region "does region i START a new article?" sequence labeling. (a) `experiments/grouper_features.py` already builds features+labels by aligning the 6 GTs to `regions_<date>.json`. (b) Train a small classifier (gradient-boosted trees / tiny MLP — NOT an 8B LLM; inference ~0 cost). By-issue cross-validation (train 5, test held-out) to avoid fitting 6. (c) Wire as `experiments/exp_167_grouped.py`: PP-DocLayout regions + classifier boundaries + head-block merge; but feed the **Qwen3-VL region text** (F5 quality) not the low-quality path, since MausoleoBench pays for text. Evaluate 6 issues + eval_probes; log to the graph; compare vs exp_045 0.4615.
2. **F3×F5 graft (parallel idea).** PP-DocLayout regions/reading-order as the crop plan feeding Qwen3-VL OCR — combine recall + text quality. Cheapest budget-fitting version wins.
3. **Cost check.** exp_045 is 136 GPU-s/page (over 6.9–13.9 budget). Any production candidate must hit the budget; record GPU-s/page every run.
4. Fold remaining eval-review protocol items (whole-issue holdout, 1943 probe, effect-size sign rule) into program.md before heavy climbing.
