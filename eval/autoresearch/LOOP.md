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


## Status: CONSOLIDATED at the budget-compliant ceiling
- **RECORD / PRODUCTION: exp_009 = 0.4071** @ 8.66 sec/page (article-level OCR + PaddleOCR-VL + fill-ratio guard). Climb this slate: 0.3826 -> 0.3946 -> 0.4071.
- **Solution space fully mapped.** Winning recipe: specialized PaddleOCR-VL + article context + geometry. Ruled out with data: all general VLMs + CHURRO (lose at any crop size); column-JSON (blobs); CLAHE (hurts); LoRA fine-tuning FRAGILE (naive overfits -0.003; synthetic hallucinates -0.130).
- **Only untried high-ceiling lever: oracle-ensemble distillation** (real newspaper teacher labels on non-eval corpus pages, ~1 day teacher labeling; uncertain but real-domain SFT didn't hallucinate). This is a big resource decision — do NOT auto-launch; surface for Elio. Otherwise 0.4071 is the practical budget-compliant ceiling and exp_009 is the production pipeline.

## If continuing (Elio's call)
1. **Oracle distillation** (the big lever): run ensemble_30min on ~200-300 decade-stratified NON-eval corpus pages (endeavour images, exclude 6 eval + 31 1943-07 probe dates) -> article-crop->oracle-text pairs -> LoRA PaddleOCR-VL (real-domain, more data). Watch for hallucination (meanCER, overgeneration) + 1943-07 probes.
2. Else: consolidate/ship exp_009; no further cheap levers remain.