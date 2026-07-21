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


## Current
- **exp_010 RUNNING** (waiter b523hv5y3): exp_009 + CLAHE contrast enhancement on page images, targeting the degraded dense 1952 scans (weakest at 0.272). Cheap test; ship only if >=4/6 improve (anti-overfit).
- **Fable domain-adaptation plan IN** (registry §F7): PaddleOCR-VL-1.6 is LoRA-fine-tunable (ms-swift); ranked plan + LOIO anti-overfit protocol recorded.


## Current
- **exp_013 augmented-fold go/no-go RUNNING** (waiter bj8xhdrjj, ~1-2h): synthetic period-Italian augmentation to fix exp_012's small-data overfit. Render 4000 degraded broadsheet-style crops from pre-1962 Gutenberg Italian (integrity-safe) + real GT (oversampled 25%), LoRA 1 epoch, test held-out 1935 vs base 0.430 AND un-augmented FT 0.436. Pipeline: synth_fetch_text.py / synth_render.py / ft_prepare_data_aug.py.

## Queue
1. **exp_013 verdict**: if augmented FT beats base 0.430 AND un-augmented 0.436 on held-out 1935 -> run full 6-fold augmented (likely NEW RECORD > 0.4071); scale synth 4K->20K if wCER moves. If no lift -> small-model fine-tuning near ceiling.
2. If fine-tuning plateaus: oracle-ensemble distillation (bigger data, ~1.7 days teacher labeling on non-eval corpus) OR accept 0.4071 as practical ceiling + consolidate/ship exp_009.
- Anti-overfit: strict LOIO; GT-free probe on the 31 image-only 1943-07 issues on any FT record; watch over-degradation->hallucination in predictions.