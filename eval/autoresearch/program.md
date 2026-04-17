# OCR Pipeline Auto-Research Program

## Objective
Maximize composite score across both eval issues (1885-06-15 and 1910-06-15) while maintaining high article recall (>70%).

## Eval Metrics
- **CER**: character error rate per matched article (lower = better)
- **wCER**: length-weighted CER — long articles count more than short ads (lower = better)
- **hCER**: headline CER — measures headline transcription quality (lower = better)
- **Recall**: fraction of GT articles matched to a prediction (higher = better)
- **F1**: harmonic mean of precision and recall (higher = better)
- **Ordering**: Spearman squared-displacement score, 1.0 = perfect order. Adjacent swaps barely penalized, large displacements heavily penalized (higher = better)
- **Page Accuracy**: fraction of matched articles with correct page_span (higher = better)
- **Composite**: weighted combination: 0.40*(1-wCER) + 0.25*recall + 0.15*ordering + 0.10*(1-hCER) + 0.10*page_accuracy

Eval on BOTH dates: `evaluate_issue()` on 1885-06-15 (41 articles, 60K chars) and 1910-06-15 (193 articles, 185K chars). Average the composite scores.

## Current Best
- Config: `2-config ensemble` (col3_qwen3_8b primary + YOLO secondary, both with page_span fix)
- 1885: Recall 95.1%, Page ~52%, wCER 0.310, Composite 0.782
- 1910: Recall 79.3%, Page ~72%, wCER 0.266, Composite 0.795
- **Average Composite: 0.799** (+0.061 over old 2-config 0.738, +0.050 over old multi_ensemble 0.749)
- Merge threshold: 0.50 (Jaccard word overlap — higher = less aggressive duplicate filtering)
- Text replacement: when YOLO version of a matched article is >5% longer, replace col3 version (catches truncation)
- Page_span fix in MergePages: uses layout_json to set correct page numbers, overriding VLM guesses
- Single-config baseline `col3_qwen3_8b_v2_structured` at 0.698 avg (with page_span fix)
- Tractable: only 2 pipeline runs needed (col3 + YOLO)

## Failure Analysis

### Category 1: Cross-page articles get truncated (biggest quality problem)
Articles spanning page boundaries are severely degraded or missed entirely:
- **L'usciere accoltellato** (pages 2→3) — completely missed (CER=1.0)
- **Ne succedono delle graziose** (pages 3→4) — truncated, 426 of 1615 chars (CER=0.744)
- **I vetturini hanno sempre torto** (pages 3→4) — truncated, 459 of 1340 chars (CER=0.694)
- **I MAESTRI ELEMENTARI** (pages 1→2) — hallucinated extra text, 5578 vs 2040 chars (CER=1.738)

Root cause: column-split processes each page independently. No mechanism to detect or stitch article continuations across pages.

### Category 2: Page 4 advertisements completely missed (8 articles, all CER=1.0)
Every ad on page 4 missed: Bagni delle Acque Albule, Tabacco MIL, Balsamo, Motori a Gas, etc. Ads have different layouts (smaller text, borders, mixed fonts) that the structured JSON prompt doesn't handle. Other configs (col4, yolo) DO capture some of these.

### Category 3: MAGNETIZZATA fiction interleaved with news (CER=0.557)
Serialized novel runs along bottom strip of pages 1-3. Column-split captures fragments mixed with news from above. The model sees a column slice with news at top and fiction at bottom — can't cleanly separate them.

### Category 4: Easy articles (CER<0.1) — already solved
Short dispatches and single-column articles on page 3 are near-perfect: La Russia in Italia, L'italiano liberato, La questione afgana, Bastimenti in moto. These are self-contained within one column on one page.

### What works well
- Single-page, single-column articles: CER 0.0-0.05
- Mid-length news articles within one page: CER 0.15-0.25
- Articles with clear headlines: easier to segment

### What fails
- Cross-page continuations: CER 0.7-1.7 or missed entirely
- Ads with non-standard layouts: missed entirely
- Fiction interleaved with news: confused segmentation
- Very long articles (>2000 chars): tend to truncate or hallucinate

## Research Directions (prioritized)

### TIER 1: High-impact model swaps (try first)

#### 1A. PP-DocLayoutV3 for layout detection + reading order
- RT-DETR architecture, 31M params, 90.4% mAP@0.5, Apache 2.0
- Detects 23 layout categories, specifically supports newspapers
- **Predicts reading order in same forward pass** via Global Pointer Mechanism
- Replaces DocLayout-YOLO + ColumnSplit + reading order heuristics in one model
- HuggingFace: `PaddlePaddle/PP-DocLayoutV3`
- Need to: install paddlepaddle, write a new operator `PPDocLayoutCrop`, integrate
- Could run on GPU 1 while VLM runs on GPU 0

#### 1B. Better VLM for OCR (swap Qwen3-VL-8B)
- ~~Chandra-OCR-2 as VLM~~ — Dead end as OCR. But outputs layout bboxes — potential YOLO alternative (see TIER 1A)
- **InternVL3-8B** — transformers 5.5.4 incompatible (missing all_tied_weights_keys). Fix: use Ray runtime_env to pin older transformers version for InternVL actor.
- **olmOCR 2** (Allen AI, 7B): 82.4 olmOCR-Bench, specifically improved on multi-column layouts. GitHub: allenai/olmocr. Potential drop-in replacement.
- **Qwen3-VL-4B / Qwen3-VL-2B**: Smaller Qwen3 variants cached on ripperred. Worth testing if quality holds — faster inference.
- Remaining VLM options limited without downloading new models.

#### 1C. YOLO parameter deep tuning
- Current: `conf_threshold=0.2, min_region_area=3000, merge_vertical_gap=50, merge_horizontal_overlap=0.5`
- Only 2 configs ever tried (default + exp_019 sensitive). Systematic sweep needed:
  - `conf_threshold`: 0.15, 0.20, 0.25, 0.30
  - `min_region_area`: 1000, 2000, 3000, 5000
  - `merge_vertical_gap`: 30, 50, 80, 120
  - `merge_horizontal_overlap`: 0.3, 0.5, 0.7
  - `text_classes`: add "table", "list" — might catch classified ads
  - `padding`: 5, 10, 20, 30
- Can test many combos since YOLO detection is fast — the VLM step is the bottleneck

#### 1D. Fine-tune DocLayout-YOLO on historical newspaper data
- Available datasets: Chronicling Germany (693 pages, 1852-1924), DocBed (3000 US newspaper pages)
- Could also bootstrap from our own 2 eval issues (manually annotate bboxes for ~10 pages)
- DocLayout-YOLO supports standard YOLO-format fine-tuning

### TIER 2: Article segmentation / stitching

#### 2A. Boundary stitching with small LLM
- Use bbox positions to identify adjacent crops on same page
- Feed last ~200 chars of crop N + first ~200 chars of crop N+1 to small LLM
- Ask: "Are these the same article? If yes, merge." Much simpler than full LLM cleanup.
- Can use Qwen2.5-7B on GPU 1 for this lightweight task
- Validated by STRAS paper (2025) which uses text embedding similarity for same purpose

#### 2B. STRAS-style embedding similarity for article grouping
- Extract text embeddings per detected region (FastText, SpaCy, or sentence-transformers)
- Group regions with similar embeddings + spatial proximity → same article
- Lightweight, no GPU needed for grouping step (just embeddings)
- Paper: link.springer.com/article/10.1007/s00799-025-00437-5
- 0.83-0.86 accuracy on 19th-century French/Finnish newspapers

#### 2C. Ar-Q-Former for article segmentation (ICDAR 2025)
- Multimodal transformer: image + bboxes + text → article grouping
- +19% improvement on Finnish newspapers, +22% on French
- Specifically designed for historical newspapers
- May need training data — check if pre-trained weights available

#### 2D. Cross-page continuation detection
- Regex for "continua a pagina X", "segue da pagina X" patterns
- Topic similarity matching between truncated articles and page-start articles
- Unsolved in literature — regex + similarity is the practical approach

### TIER 3: Reading order improvements

#### 3A. FocalOrder (Jan 2026, SOTA)
- LayoutLMv3-large backbone, 0.4B params, 97.1% accuracy, 12.3ms inference
- arxiv:2601.07483
- Could run as post-processing after layout detection
- Only needed if PP-DocLayoutV3 doesn't give good enough reading order

#### 3B. Column-aware spatial heuristic (no model needed)
- Current: sort by (page_span, position_in_issue)
- Better: detect columns via x-coordinate clustering, sort within columns top-to-bottom, then columns left-to-right
- Can implement using existing bbox data from layout_json
- Zero compute cost, just better sorting logic in MergePages

### TIER 4: Prompt and config tuning

#### 4A. Prompt refinements for medium-CER articles
- Broken words at column boundaries — add "rejoin words split by hyphens at line endings"
- Keep changes minimal (V3 proved complex prompts hurt)

#### 4B. Column geometry tuning
- num_columns (3 vs 4), overlap_pct (0.03 vs 0.06)
- Less important if we move to YOLO-primary pipeline

#### 4C. Ensemble tuning (PROVEN — current best)
- Merge strategy: col3 primary + YOLO secondary, threshold=0.50
- Current best: 0.789 average composite
- Could try: use YOLO as primary (better layout detection) + col3 as secondary (better text quality)
- Could try: weighted text selection — for articles found by both, use whichever has lower CER estimate

### TIER 5: Full pipeline rearchitecture (ambitious)

#### 5A. PaddleOCR-VL-1.5 end-to-end (0.9B)
- Layout + reading order + OCR in single forward pass
- 94.5% accuracy on OmniDocBench, Apache 2.0
- Could replace entire pipeline for a baseline comparison
- arxiv:2601.21957

#### 5B. Two-pass hybrid pipeline
1. PP-DocLayoutV3 for layout + reading order
2. Crop detected regions using layout boxes (not blind column splits)
3. Qwen3-VL-8B or Chandra-OCR on each region
4. STRAS embedding similarity for article grouping
5. Regex "continua a pagina" for cross-page linking

### Completed / Dead Ends
- ~~Ad-aware prompting (V3)~~ — CONFIRMED BAD. CER=2.014, hallucination. Complex prompts hurt.
- ~~LLM cleanup (Qwen2.5-3B/7B)~~ — FAILED. Both models produce empty/repetitive output.
- ~~Disable thinking (/no_think)~~ — CONFIRMED BAD. -0.205 composite on 1910.
- ~~Preprocessing (grayscale+resize)~~ — CONFIRMED BAD. Destroys quality.
- ~~Full-page OCR (no column split)~~ — CONFIRMED BAD. Catastrophic on 1910.
- ~~Chandra-OCR-2 as VLM OCR~~ — Not an OCR model. Outputs layout bboxes+labels, not text. BUT could be useful as YOLO alternative for layout detection (see TIER 1A).
- **InternVL3-8B** — needs Ray runtime_env with older transformers. Not yet tested.
- ~~YOLO param tuning (vgap80, smallregion)~~ — Both directions worse. Default YOLO params are near-optimal.
- ~~ColumnSplit overlap 6%~~ — Catastrophic (CER=2.351). 3% overlap is the sweet spot.
- ~~max_tokens 10k~~ — No effect on quality, 2x slower. 8192 is not the bottleneck.

## Priorities
- **Quality is paramount** — maximize composite score above all else. Compute time and inference speed are secondary concerns.
- **Prefer vllm backend** — vllm is much faster than transformers and should be the default. Use it for Qwen2.5 models. Only fall back to transformers when vllm is unavailable for a model.
- **No overfitting** — all changes must generalize to any historical Italian newspaper issue, not just the two eval dates.
- **Timeout is 3600s** — generous limit, don't optimize for speed at the cost of quality.
- **Avoid hitting max_model_len** — set max_model_len high enough that input image tokens + prompt + max_tokens never hits the ceiling. Truncated outputs lose article text silently. Use max_model_len=32768 unless memory-constrained.
- **Inference time budget** — the full pipeline for one issue must complete in ≤2 pipeline runs on 2 GPUs (i.e., max 2 configs run sequentially or in parallel). Ensembles using 60 existing predictions are not tractable for production. Target: col3 primary + 1 YOLO secondary = 2 runs.

## The Loop

Each iteration:

1. **Read state** — this file for priorities, `eval/autoresearch/log.jsonl` for past experiments
2. **Propose** — ONE targeted change (one variable at a time)
3. **Write config** — `configs/ocr/exp_NNN_description.py` (if new prompt, add to `src/mausoleo/ocr/prompts.py`)
4. **Run** — sync to ripperred, execute, copy prediction back
5. **Evaluate** — run `evaluate_issue()` locally, log to `log.jsonl`
6. **Decide** — if improved update baseline here, if not revert and try different direction
7. **Report** — one-line: `EXP NNN: [description] CER=X.XXX (baseline 0.373) [IMPROVED/WORSE]`
8. **Schedule next** — use ScheduleWakeup

## Infrastructure

Sync code to remote:
```
rsync -avz --exclude='.venv' --exclude='.git' --exclude='__pycache__' --exclude='eval/predictions' --exclude='eval/ground_truth' -e 'ssh -p 62022' ./ audiogen@81.105.49.222:~/mausoleo_di_roma/
```

Clean pycache + run a config (run on BOTH dates):
```
ssh audiogen@81.105.49.222 -p 62022 "cd mausoleo_di_roma && find src/ -name __pycache__ -exec rm -rf {} + 2>/dev/null && .venv/bin/python scripts/run_real_ocr.py <config_name> 1885-06-15 && .venv/bin/python scripts/run_real_ocr.py <config_name> 1910-06-15"
```

Copy predictions back:
```
scp -P 62022 audiogen@81.105.49.222:~/mausoleo_di_roma/eval/predictions/<config>_1885-06-15.json eval/predictions/
scp -P 62022 audiogen@81.105.49.222:~/mausoleo_di_roma/eval/predictions/<config>_1910-06-15.json eval/predictions/
```

Evaluate locally (BOTH dates):
```python
from mausoleo.eval.evaluate import evaluate_issue
import json

scores = []
for date in ["1885-06-15", "1910-06-15"]:
    gt = json.loads(open(f"eval/ground_truth/{date}/ground_truth.json").read())
    pred = json.loads(open(f"eval/predictions/<config>_{date}.json").read())
    r = evaluate_issue(gt, pred, config="<config>", date=date)
    scores.append(r.composite_score)
    # r.mean_cer, r.weighted_cer, r.headline_cer, r.article_recall, r.ordering_score, r.composite_score
avg_composite = sum(scores) / len(scores)
```

## Config Format

```python
from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, Preprocess, VlmOcr

config = OcrPipelineConfig(
    name="exp_NNN_description",
    operators=[
        ColumnSplit(num_columns=3, overlap_pct=0.03),
        VlmOcr(model="Qwen/Qwen3-VL-8B-Instruct", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192),
        MergePages(),
        ParseIssue(),
    ],
)
```

## What You Can Change
1. **Prompt** — edit `src/mausoleo/ocr/prompts.py` to add new prompt variants
2. **Column splitting** — num_columns (1-6), overlap_pct (0.0-0.10)
3. **Preprocessing** — `Preprocess(grayscale=False, max_dimension=1024)` before other operators
4. **Model** — Qwen2.5-VL-7B, Qwen2.5-VL-3B, Qwen3-VL-8B, Qwen3-VL-4B, Qwen3-VL-2B (all cached on ripperred)
5. **Backend** — "transformers" or "vllm" (vllm only for Qwen2.5)
6. **max_tokens** — 4096 to 16384
7. **Operator composition** — combine operators in new ways

## What You Cannot Change
- The eval metric code or ground truth
- The models available (don't download new ones)
- The hardware (2x RTX 3090 24GB)

## Learnings
- Qwen3-VL-8B + transformers is very slow (~10 min for 4-page, ~20 min for 6-page issue single GPU)
- Running two Qwen3-VL-8B transformers instances in parallel causes ~3-4x slowdown from CPU/memory bus contention
- max_tokens 8192→12288 improves quality (1885 composite 0.650→0.672) but too slow even at 3600s for 6-page issues
- Longer/more complex prompts (V3) confused model output format — keep prompt changes minimal
- Qwen2.5-VL-7B handles column crops poorly (composite 0.264 vs 0.712 baseline) — stick with Qwen3-VL-8B for OCR
- Qwen2.5-3B is too weak for LLM cleanup — degenerates into repetition. Need 7B+ for cleanup
- Use both GPUs for multi-operator pipelines: VlmOcr on GPU 0, LlmCleanup on GPU 1 (don't set CUDA_VISIBLE_DEVICES)
- Always ensure GPU is free before starting a new run (check nvidia-smi)
- vllm backend should be preferred for speed; it supports Qwen2.5 models
- **Run-to-run variance is significant** — Qwen3-VL-8B + transformers gives different outputs each run (GPU non-determinism). EXP 012 showed ±0.15 composite swing between runs with identical config. Only trust large, consistent improvements across both dates.
- max_model_len only affects vllm backend, not transformers — changing it for transformers configs has no effect
- **Do NOT disable Qwen3-VL thinking** — /no_think degrades quality. The thinking step helps the model reason about newspaper layout before generating structured output. EXP 013 showed -0.205 composite drop on 1910.
- Parallel runs on separate GPUs can stall/die silently — always run experiments sequentially (one at a time) for reliability
- **page_span fix**: MergePages now uses layout_json to set correct page numbers instead of crop indices. VLM always guesses page_span=[1], so MergePages must override unconditionally. Page accuracy improved from ~10% to ~55-65%. Both ColumnSplit and YoloCrop populate layout_json with "page" field.
- Use `--force` flag with run_real_ocr.py to overwrite existing predictions when re-running with code changes
- **V3 prompt confirmed bad** — CER=2.014 on 1885 (vs 0.341 with V2). Complex content type lists cause hallucination. Stick with V2.
- **Merge threshold tuning**: higher threshold (0.50) better than 0.30. Less aggressive duplicate filtering lets more YOLO articles through, improving both recall and wCER.

## Rules
- One change per experiment
- Always eval on BOTH dates (1885-06-15 and 1910-06-15), report average composite
- Name configs `exp_NNN_shortdesc.py` with incrementing numbers
- Log EVERY result to `eval/autoresearch/log.jsonl`, even failures
- Don't modify ground truth or eval code
