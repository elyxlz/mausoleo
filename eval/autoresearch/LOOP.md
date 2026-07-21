# Loop state — live queue for the autoresearch loop

Rewritten every iteration. Rules in `program.md` (+ `registry.md`); this file is only the CURRENT state + queue.

## Standing context
- Metric = **MausoleoBench** (`src/mausoleo/eval/evaluate.py`). Budget = **50.0 sec/page** caller-measured (`scripts/time_experiment.sh`). Full rules in `program.md`.
- Run on ripperred (`.venv/bin/python`), GPU1 (`CUDA_VISIBLE_DEVICES=1`). Experiments are self-contained, never import the eval, measured cold by the caller, and get an adversarial review.
- Experiments numbered from **exp_001**. Log each full-6-issue run to `mausoleobench_log.jsonl` (`{n, config, exp, score, description, gpu_s_per_page, budget_ok}`, n = last+1) → live graph `scripts/progress_server.py` (:8078 + cloudflare, per-experiment prediction viewer). Commit + push; update `registry.md`.

## Board
- **RECORD: exp_009 = 0.4071** @ 8.66 sec/page (article-level OCR + PaddleOCR-VL + fill-ratio guard). Frontier: 0.3802/0.3826/0.3946/0.4071.
- Confirmed dead ends: general VLMs (2B/4B/8B/AWQ) + CHURRO all < specialized PaddleOCR-VL at any crop size; preprocessing (CLAHE) hurts; **naive LoRA fine-tune on real GT overfits (exp_012 0.4041, net -0.003)**.
- Structure (article context + geometry) + specialized PaddleOCR-VL is the winning recipe; text quality saturated ~0.41.


## Done this slate
- exp_001 0.3802, **exp_002 0.3826 (record)** — grouper + PaddleOCR-VL per-region.
- **exp_003 FAILED** (n=3): Qwen3-VL-8B column-structured JSON = 0.1934 @ 65.65 sec/page (DQ). Structured JSON merges columns into giant blobs (recall 0.16–0.68); did NOT reproduce archived exp_045 0.46. JSON is the wrong delivery for context.

- **exp_003 FAILED** (n=3): column-structured JSON 8B = 0.1934 @ 65.65 (DQ) — JSON blobs.
- **exp_004 FAILED** (n=4): CHURRO-3B per-region = ~0.15/cer 6.0 (DQ, hallucinates on crops + slow). Page-trained → region crops OOD.

- **exp_005 = 0.3946 RECORD** (n=5): article-level OCR (grouper boundaries + article-union-bbox crop OCR, PaddleOCR-VL) @ 8.74 sec/page. Context helps: 1885 +0.025, 1895 +0.083, 1910 +0.017, 1925 +0.027; but 1935 -0.063, 1952 -0.017 (dense pages: mis-grouped regions → huge multi-column crops → blobs).

- **exp_006 = 0.3931** (n=6): length-ratio blob guard, NOT a record — too blunt (article text longer than region text is usually the GAIN, so it reverts wins). exp_005 stays record.

- **exp_007 FAILED** (n=7): full-precision 8B on article crops OOM'd (big multi-region crops blow past 24GB even at max_num_seqs=32).

## Done this slate
- exp_005 = 0.3946 RECORD (article-level OCR + PaddleOCR-VL, 8.74 sec/page). exp_002 0.3826, exp_001 0.3802.
- Dead ends confirmed: general VLMs < specialized PaddleOCR-VL at ANY crop size (region 8B 0.334, article AWQ-8B 0.367 @ 31.78 s/pg; all < PaddleOCR article 0.395). Column-JSON blobs (exp_003 0.19). CHURRO page-model hallucinates on crops (exp_004). Length-guard reverts gains (exp_006).

## Board
- **RECORD: exp_009 = 0.4071** @ 8.66 sec/page (article-level OCR + PaddleOCR-VL + fill-ratio guard). Frontier: 0.3802/0.3826/0.3946/0.4071.
- Confirmed dead ends: general VLMs (2B/4B/8B/AWQ) + CHURRO all < specialized PaddleOCR-VL at any crop size; preprocessing (CLAHE) hurts; **naive LoRA fine-tune on real GT overfits (exp_012 0.4041, net -0.003)**.
- Structure (article context + geometry) + specialized PaddleOCR-VL is the winning recipe; text quality saturated ~0.41.


## Status: REOPENED by the 5x budget raise (cap 250 sec/page, 2026-07-22)
- **RECORD: exp_009 = 0.4071** @ 8.66 sec/page (article-level OCR + PaddleOCR-VL + fill-guard).
- **The 5x budget unlocks the Qwen3-VL-8B routes** that were DQ'd only on cost. Top opportunity to beat 0.4071:
  1. **Revive exp_045** (Qwen3-VL-8B column-structured, archived MausoleoBench 0.4615 @ ~136 sec/page) — rebuild self-contained + caller-measure to confirm ≤250, log as the new budget-compliant record.
  2. **exp_168** (8B per-region + grouper, 0.4342 @ ~150) — likewise.
  3. **Complementary merge**: exp_045 ⊕ exp_168 oracle-selected to 0.525; the two-8B merge (0.5266) is ~286 (just over cap) — build a cheaper complementary-source merge that fits ≤250.
- GOAL.md now states the objective (maximize MausoleoBench within budget). Elio is running /goal on it — /goal may drive from here; keep this queue in sync.
- Still-untried ceiling lever: oracle-ensemble distillation. Ruled out: general VLMs/CHURRO lose; LoRA fine-tune fragile.

## If the loop continues
1. Revive exp_045 (8B column-structured) self-contained, caller-measured; if ≤250 and ~0.46 -> NEW RECORD (n=14). Then exp_168; then the complementary merge.
2. Anti-overfit/adversarial review as always; budget via caller.