# Loop state — live queue for the autoresearch loop

Rewritten every iteration. Rules in `program.md` (+ `registry.md`); this file is ONLY the current state + queue. Do not accumulate history here — results go to `mausoleobench_log.jsonl`, mechanisms to `registry.md`.

## Standing context
- Metric = **MausoleoBench** (`src/mausoleo/eval/evaluate.py`). Budget cap = **200.0 sec/page**, caller-measured (`scripts/time_experiment.sh`); `BUDGET_CAP` in `scripts/progress_server.py` is the single source of truth.
- Run on ripperred (`.venv/bin/python`), GPU1 (`CUDA_VISIBLE_DEVICES=1`). Experiments are self-contained, never import the eval, are measured cold by the caller, and get an adversarial review.
- Log each full-6-issue run to `mausoleobench_log.jsonl` (`{n, exp, score, description, sec_per_page, budget_ok}`, n = last+1) → live graph `scripts/progress_server.py` (:8078 + cloudflare, per-experiment prediction viewer). Commit + push; update `registry.md`.

## Board
- **RECORD: exp_018 = 0.6115 @ 8.55 sec/page** — hosted Gemini 3.7-flash, full page in, JSON array out, thinking_budget=128, de-hyphenation. HIGH VARIANCE: two identical runs gave 0.6258 and 0.6045; treat ~0.615 as the mean, not a point estimate. `sec_per_page` here is API latency, NOT GPU time — not comparable to local routes. Corpus cost ~$5.7k (intro pricing, doubles 2027).
- Local frontier: exp_017 = 0.4263 @ 181.98 (Qwen3-VL-8B, 3-column split — note the split was WRONG: 1895 is 5 columns, 1952 varies 4-9).
- Cheapest good local route: exp_009 = 0.4071 @ 8.66 — but it hardcodes `~/paddle_env/bin/python`, which no longer exists on ripperred; it cannot run as-is.
- Reference ceiling: oracle ensembles 0.5941 / 0.5622, both far over cap. exp_018 exceeds the 0.5941 reference.
- Ruled out (see `registry.md`): CHURRO and general VLMs below specialized PaddleOCR-VL at low cost; CLAHE preprocessing; length-ratio blob guard; naive LoRA on real GT; synthetic-augmented LoRA.

## Current
- GT corrected (fdd6d8a): 37 image-confirmed fixes to 1885/1895/1910 from a Gemini consensus + deterministic anchoring pipeline. 1885 had a systematic off-by-one page shift; 1910 gt138's railway timetable was fabricated. 1925/1935/1952 returned mostly NO_DEFECT and are NOT genuinely audited — 1952 still shows 51 derived order moves and 63 uncovered gaps. Anchoring artifacts live in `/home/elyx/gt_build`.
- Gemini credits are DEPLETED (429 RESOURCE_EXHAUSTED) — all hosted work is blocked until topped up.
- Paragraph structure: predictions carried ~1 paragraph/article (GT has 7,191 across 924). Fixed in 0e4d383 but NOT yet re-run. 8 corrected GT articles remain single-paragraph.

## Queue
1. **Validate the paragraph prompt, then re-run exp_018** (~$1.50). Gives paragraph-structured predictions and a third variance sample. Needs credits.
2. **True-column local route** — PP-DocLayoutV3 column bands (already computed for all 42 pages) feeding a local VLM. The 3-column split every previous column experiment used was simply wrong, so that family was never fairly tested. No API cost.
3. **Finish the GT audit on 1925/1935/1952** — anchoring evidence already computed; only the adjudication pass is missing.
4. **Fix exp_009's dead `~/paddle_env` path** so the cheap local baseline is runnable again.
5. **Oracle-ensemble distillation** on non-eval corpus pages — still the untried high-ceiling lever for a local route.
