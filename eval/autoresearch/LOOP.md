# Loop state — live queue for the autoresearch loop

Rewritten every iteration. Rules in `program.md` (+ `registry.md`); this file is only the CURRENT state + queue.

## Standing context
- Metric = **MausoleoBench** (`src/mausoleo/eval/evaluate.py`). Budget = **50.0 sec/page** caller-measured (`scripts/time_experiment.sh`). Full rules in `program.md`.
- Run on ripperred (`.venv/bin/python`), GPU1 (`CUDA_VISIBLE_DEVICES=1`). Experiments are self-contained, never import the eval, measured cold by the caller, and get an adversarial review.
- Experiments numbered from **exp_001**. Log each full-6-issue run to `mausoleobench_log.jsonl` (`{n, config, exp, score, description, gpu_s_per_page, budget_ok}`, n = last+1) → live graph `scripts/progress_server.py` (:8078 + cloudflare, per-experiment prediction viewer). Commit + push; update `registry.md`.

## Board
- **RECORD: exp_002 = 0.3826** (n=2) — grouper + PaddleOCR-VL with hi-res crops, 5.91 sec/page. exp_001 baseline 0.3802 (n=1).
- Oracle references (not production): `ensemble_30min` 0.5941, `ensemble_prune5` 0.5622.
- Segmentation is solved cheaply by the trained grouper. Open bottleneck: **budget-fit OCR text quality** (`registry.md` §F5).
- Note: **any gain counts** (no effect-size floor). 1952 dense-classifieds still the weak point (~0.27).

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
- **RECORD: exp_009 = 0.4071** @ 8.66 sec/page (article-level OCR + PaddleOCR-VL + geometric fill-ratio guard). Frontier: 0.3802 -> 0.3826 -> 0.3946 -> 0.4071. Oracle reference 0.5941.
- Confirmed dead ends: general VLMs (2B/4B/8B/AWQ) < specialized PaddleOCR-VL at any crop size; column-JSON blobs; CHURRO hallucinates on region crops; length-guard reverts gains.

## Current
- **exp_010 RUNNING** (waiter b523hv5y3): exp_009 + CLAHE contrast enhancement on page images, targeting the degraded dense 1952 scans (weakest at 0.272). Cheap test; ship only if >=4/6 improve (anti-overfit).
- **Fable domain-adaptation plan IN** (registry §F7): PaddleOCR-VL-1.6 is LoRA-fine-tunable (ms-swift); ranked plan + LOIO anti-overfit protocol recorded.


## Queue (domain adaptation — registry §F7; the real ceiling lever)
1. **exp_011 CHURRO-articles = LOSES** (finishing 1952; early read 5/6 all worse than exp_009: -0.02 to -0.08, wCER worse). Historical fine-tuning (CHURRO) does NOT beat specialized PaddleOCR-VL even on in-distribution article crops. Decision: adapt the WINNING model -> LoRA fine-tune PaddleOCR-VL-1.6.
2. **exp_012 LoRA fine-tune PaddleOCR-VL — WORKS; 6-FOLD RUNNING** (waiter bgw6cbazd, ~3-4h). Single-fold proof: held-out 1935 base 0.430 -> finetuned 0.436 (+0.006, wCER 0.513->0.500, meanCER 0.423->0.395) @ 10.85 sec/page from 261 pairs, no augmentation. Now running full LOIO 6-fold (per date: ft_prepare_data -> ft_train_lora -> fix_merged.py -> exp_012_paddleft) -> assembled exp_012 predictions across all 6, likely a NEW RECORD. Then adversarial-review (GT-free probe on 1943-07) + log n=12.
   - Checkpoint fix (fix_merged.py): ms-swift merged dir needs custom .py code copied, base config.json (empty text_config) swapped in, weights remapped (model.visual.->visual.) + 12 frozen vision-head weights filled from base. transformers pinned 5.5.4; USE_HF=1; GPU=1.
   - AMPLIFIER (next if 6-fold positive): synthetic period-Italian degraded-text augmentation (Fable Option 1) and/or oracle-ensemble distillation on non-eval corpus pages — more/better training data is the path to bigger gains.
2. **LoRA PaddleOCR-VL-1.6** single fold (train 5 issues' GT article-crops→GT-text, test 1935) via ms-swift/trl — the go/no-go on domain adaptation. If a fold shows real wCER drop → full 6-fold LOIO + synthetic augmentation.
3. **Distill oracle → PaddleOCR-VL** on non-eval corpus pages (teacher labels), or ByT5 post-OCR corrector (stacks).
- exp_010 (CLAHE) REJECTED (0.4024 < 0.4071, preprocessing hurts).
- Anti-overfit: strict LOIO, cross-decade split, GT-free probes on the 31 image-only 1943-07 issues.