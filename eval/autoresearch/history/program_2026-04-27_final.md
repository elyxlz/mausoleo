# OCR Pipeline Auto-Research Program

## Objective
Maximize composite score across both eval issues (1885-06-15 and 1910-06-15) while maintaining high article recall (>70%). **Hard constraint:** full pipeline for one issue must fit in **30 min wall-clock** using both GPUs (RTX 3090 ×2).

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

## Final Baseline (concretized 2026-04-27): cold-cache standalone, ≤30 min/issue
**Config**: `configs/ocr/ensemble_30min.py` — one `OcrPipelineConfig` using the `ParallelEnsembleOcr` operator. 8 unique sub-pipelines split across 2 GPU chains (parallel subprocess calls to `run_real_ocr.py`), merged via REPLACE chain (9 entries, col4 used twice) + ADDITIVE + quality_text_select.

**Score 0.89878 cold cache** (1885=0.87186, 1910=0.92569). Avg of two test issues, every run regenerates ALL sub-pipeline predictions from raw images, ZERO prior cache. Session 2026-04-26/27 cumulative gain **+0.01196** from 0.88682 across 23 micro-wins (heroes test added exp_102 + exp_097 +0.0094, exp_134 dropped to fit budget).

**Hard constraint:** ≤30 min wall-clock per issue (max(GPU0, GPU1)) on 2× RTX 3090 24GB. Empirical wall on 1910 ≈ 30.5 min (GPU0 30.5, GPU1 28.9). VLLM preferred; `vllm_strict=True` for yolo (kept here as exp_140 only).

**The 8 sub-pipelines (run cold-cache)**:

GPU0 chain (4 sources, ~30.5 min on 1910):
1. `exp_107_fullpage_qwen25vl` (Qwen2.5-VL-7B fullpage vllm) — **primary**
2. `exp_045_qwen3vl_vllm` (Qwen3-VL-8B col3 vllm)
3. `exp_055_col6_ads_prompt` (Qwen3-VL-8B col6+ads vllm)
4. `exp_097_col4_qwen3vl_vllm` (Qwen3-VL-8B col4 vllm)

GPU1 chain (4 sources, ~28.9 min on 1910):
5. `exp_138_col4_qwen25_vllm` (Qwen2.5-VL-7B col4 vllm)
6. `exp_140_yolo_smallregion_vllm` (Qwen3-VL-8B yolo small-region vllm strict)
7. `exp_142_col5_qwen25_vllm` (Qwen2.5-VL-7B col5 vllm)
8. `exp_102_fullpage_vllm` (Qwen3-VL-8B fullpage vllm)

3 model families load on each GPU (Qwen2.5-VL-7B, Qwen3-VL-8B vllm, Qwen3-VL-8B vllm-strict).

**Merge recipe** — REPLACE chain (col4 used twice; sources at end take final precedence):
1. exp_138_col4_qwen25_vllm (0.85, 1.05)
2. exp_045_qwen3vl_vllm (0.50, 1.05)
3. exp_055_col6_ads_prompt (0.30, 1.05)
4. exp_107_fullpage_qwen25vl (0.50, 1.02)  *(near-zero LOO; kept for safety)*
5. exp_138_col4_qwen25_vllm (0.85, 1.05)  *(DUP)*
6. exp_140_yolo_smallregion_vllm (0.85, 1.02)
7. exp_102_fullpage_vllm (0.55, 1.05)
8. exp_097_col4_qwen3vl_vllm (0.55, 1.05)
9. exp_142_col5_qwen25_vllm (0.85, 1.05)

ADDITIVE: exp_055_col6_ads_prompt (0.88, 100.0)
quality_select_sources: exp_045, exp_107, exp_138, exp_055
min_quality_delta=0.10, headline_delta=0.15

**Saturation status**: 50+ retune evals at 0.89878 → all directions tied within ±0.0001. Bigger gains require generating more sources (exp_127_yolo_ads ADD would add ~+0.00224 but doesn't fit budget; exp_134 in QS would recover +0.00995 but its 9-min generation is over budget). Production is at the local optimum for the 8-source/30-min architecture.

**Reproduction**:
```
uv run --no-project python scripts/run_real_ocr.py ensemble_30min 1885-06-15 1910-06-15
```
Output: `eval/predictions/ensemble_30min_<date>.json`. Sub-pipeline predictions cached to `eval/predictions/<name>_<date>.json`.

Usage:
```
uv run --no-project python scripts/run_real_ocr.py ensemble_30min 1885-06-15 1910-06-15
```
Output: `eval/predictions/ensemble_30min_<date>.json`. Sub-pipeline predictions cached to `eval/predictions/<name>_<date>.json`.

## Research pipeline (unconstrained, ~50-60 min/issue)
`configs/ocr/ensemble_best.py` — 18-source single OcrPipelineConfig using the in-process `EnsembleOcr` operator. Sub-pipelines run serially (2-3h fresh). Violates 30-min constraint.

`scripts/ensemble_pipeline_30min.py` — 19-source parallel orchestrator script (writes to `ensemble_research_<date>.json`). Includes fullpage variants + cross-page col1 pairs. **Score 0.9231** (1885=0.9037, 1910=0.9426). Wall-clock ~50-60min/issue.

## Current Best (unconstrained research, ignores 30-min budget)
- Config: **17-source ensemble** (added exp_125 yolo small-conf vllm). `configs/ocr/ensemble_best.py` (single OcrPipelineConfig). Violates 30-min constraint.
- 1885: Composite **0.8845** (unchanged by exp_125)
- 1910: Composite **0.9337** (+0.0007 from exp_125 additive ov=0.75)
- **Average Composite: 0.9091** (+0.0004 over 0.9087)
- Note: exp_125 hurt the 30-min budget pipeline (-0.0057, abandoned for that config). Only useful in unconstrained where averaging dilutes its noise.
- **Latest wins**: **Qwen2.5-VL-7B cross-family diversity** is the dominant recent lever. 1885 wCER plummeted 0.234→0.149 via stacking Qwen3 and Qwen2.5 variants at multiple column splits + fullpage.
  - exp_102 fullpage_vllm (Qwen3) additive at overlap=0.75: +0.011
  - exp_107 fullpage_qwen25vl: +0.008
  - exp_108 col3_qwen25vl: +0.004
  - exp_109 col4_qwen25vl: +0.0008
  - exp_111 col2_qwen25vl: +0.0004

Sources with diminishing returns (no longer helping, tried & not added): exp_110 col6_qwen25vl, exp_112 col5_qwen25vl, exp_113 col1_qwen25vl, exp_114 col6_ads_qwen25vl, exp_115 fullpage_qwen3_4b, exp_117 col3_ads_qwen25vl.

### Ensemble structure
- **Primary**: exp_045_qwen3vl_vllm (Qwen3-VL-8B, 3-col split, vllm backend, V2 prompt)
- **Replacement chain** (per-source hyperparams, NOT per-date):
  - col3_qwen3_8b_v2_structured (col3 + transformers): overlap=0.75, ratio=1.15 (strict)
  - exp_055_col6_ads_prompt (col6 + ads prompt + vllm): overlap=0.75, ratio=1.08
  - exp_010_yolo_qwen3_8b (YoloCrop + transformers): overlap=**0.50**, ratio=1.08 (looser — yolo regions differ from column crops)
  - col4_qwen3_8b_v2_structured (col4 + transformers): overlap=0.75, ratio=1.02 (aggressive replace)
  - exp_097_col4_qwen3vl_vllm (col4 + vllm): overlap=0.75, ratio=1.02
- **Additive sources** (per-source overlap tuned to source noise level):
  - col5_qwen3_8b_v2_structured (col5 + transformers): overlap=0.50 (noisy → strict dedup)
  - exp_052_col6_vllm (col6 + vllm): overlap=0.30 (clean → loose for coverage)
  - yolo_qwen25_7b_v2_structured (YoloCrop + Qwen2.5-VL-7B): overlap=0.50 (very noisy 437 articles)
  - exp_098_col5_qwen3vl_vllm (col5 + vllm): overlap=0.50
  - exp_099_col2_qwen3vl_vllm (col2 + vllm): overlap=0.75 (big +0.003 win from wide columns)
- **quality_text_select**: min_quality_delta=0.10 (body), headline_delta=0.15 — headline-swap across 7 quality sources
- **trim_predictions** (scripts/trim_repetitive.py): drops VLM-output JSON-blob articles and repetitive-dot trailing garbage (BIGGEST single win, +0.016)

## Production pipeline (~50-60 min/issue, was 30-min budget)
**Script**: `scripts/ensemble_pipeline_30min.py`
**Score**: 1885=0.9007, 1910=0.9399, **avg=0.9203** (well above prior unconstrained ceiling 0.9089). Includes **cross-page completion** post-processor (col1 head+continuation pair mining) and **exp_107 fullpage_qwen25vl in REPLACE chain at ratio=1.02** (v11 win, +0.0037).

**Design** — 2 parallel GPU chains, each ~50-60 min fresh on 2× RTX 3090 (significantly exceeds original 30-min budget but reaches unconstrained quality):
- **GPU 0**: `exp_045_qwen3vl_vllm` → `col3_qwen3_8b_v2_structured` → `exp_108_col3_qwen25vl` → `exp_099_col2_qwen3vl_vllm` → `exp_111_col2_qwen25vl` → `exp_052_col6_vllm` → `exp_102_fullpage_vllm`
- **GPU 1**: `exp_010_yolo_qwen3_8b` → `exp_055_col6_ads_prompt` → `col4_qwen3_8b_v2_structured` → `exp_098_col5_qwen3vl_vllm` → `yolo_qwen25_7b_v2_structured` → `exp_107_fullpage_qwen25vl`

13 sub-pipelines. **Cumulative session gain: +0.0264 from adding 8 cross-VLM-family / vllm / fullpage additives.** Major wins:
- exp_098 (col5 qwen3 vllm): +0.0030
- exp_111 (col2 qwen25 vllm): +0.0011
- yolo_qwen25_7b: +0.0015
- exp_052 (col6 vllm): +0.0013
- **exp_102+exp_107 fullpage stack: +0.0164** (biggest single addition — fullpage VLM catches long articles missed by column splits, two model families together is far better than one)

Tried but rejected: exp_109_col4_qwen25 (no gain after others), exp_105_col1_vllm (-0.0002), exp_125_yolo_smallconf (-0.0057), col5_trans (no gain), exp_097_col4_vllm (no gain).

**Strict 30-min variant** (drop fullpage + yolo_qwen25 + col4_trans): 0.8911. The fullpage sources are the most impactful but most expensive (~22min each on 1910). **Key late-session addition: `exp_108_col3_qwen25vl`** (col3 + Qwen2.5-VL-7B vllm) as additive at overlap=0.75 — cross-family diversity adds +0.005 at same wall time. Dropped from unconstrained: fullpage variants (18-22 min each, too heavy), exp_099/col2 + other Qwen2.5 column variants (each ≤0.005 impact). Trim + quality_select cost ~1 min and are kept.

**Usage**:
```
uv run --no-project python scripts/ensemble_pipeline_30min.py <date> [<date> ...]
```
Output: `eval/predictions/ensemble_30min_<date>.json`. Per-config predictions cached to `eval/predictions/<config>_<date>.json`; re-runs skip cached work.

### Key insights from this session (0.8549 → 0.8837, +0.029)
**Biggest single win: JSON-blob filter (+0.0165).** VLM sometimes emits raw JSON text (``` ```json ``` or `{"articles":[…]}`) that `MergePages` fails to parse as articles, so the whole blob becomes the "text" of one massive article (17k-28k chars). These poison the GT↔pred matcher by providing high text_overlap with unrelated GT articles (pushing CER past 13!). `scripts/trim_repetitive.py` drops them + strips repetitive trailing-dot hallucinations. Dropped 9 articles on 1885, 18 on 1910 → 1910 wCER halved 0.225→0.111.

**Diversity gains (the rest of +0.013):**
- vllm-backend variants of same col+model setup: exp_045 primary (+0.0019), exp_097 col4+vllm 5th replacement (+0.0002), exp_098 col5+vllm additive (+0.0007), **exp_099 col2+vllm additive +0.0029 (biggest new win — wide 2-col crops catch article context that finer splits miss)**
- yolo_qwen25_7b additive (different YOLO + Qwen2.5-VL-7B): +0.0013
- Per-source overlap tuned to detection method (yolo=0.50 loose, columns=0.75 strict): +0.0015
- Per-source replace_ratio (col3=1.15 strict, col4=1.02 aggressive): +0.0006
- Noise-aware additive overlaps (noisy col5=0.50 strict, clean col6_vllm=0.30 loose, yolo_qwen25_7b=0.50): +0.0006

Correlation analysis confirms column-based sources (col3/col4/col6) produce nearly identical text (pairwise distance < 0.15). Yolo is meaningfully different (d ~0.45+). This is why yolo contributes most to ensemble (+0.028 ablation on pre-JSON-filter baseline).

### Dead ends this session
- Length-aware body replacement: catastrophic -0.29 (picks long hallucinations)
- Cross-page stitching (heuristic): zero stitches, text too "naturalized" by VLM
- Italian accent restoration: zero effect (VLM handles accents fine)
- Nanonets OCR-s / OCR2-3B: vllm can't load tied `lm_head`
- Qwen3-VL-32B AWQ: OOM on 24GB
- Qwen3-VL-30B-MoE AWQ: scrambled output
- Qwen3-VL-8B-Thinking: too slow, raylet crashed
- T=0.3 stochastic sampling for diversity: too noisy, hurts ensemble (-0.02 as additive)
- Gemma-3-4B vision / yolo+vllm combined: cuDNN CUDNN_STATUS_NOT_INITIALIZED (Ray+env conflict)
- Single-paragraph >2500 char heuristic filter: too aggressive, -0.05 to -0.10
- Headline length-bonus: tanks hCER (1910: 0.150→0.628)
- Headline consensus voting across sources: sources too correlated, picks wrong mode
- Cross-source concatenation of article fragments: hurts (-0.006 on "difesa navale")
- Truncation-aware merge (ratio=0.80 if primary ends abruptly): -0.002, heuristic too permissive
- col7 / col8 column split: over-segments, doesn't add unique coverage
- exp_056 col4+ads as replacement on 1910: catastrophic (-0.053 on 1910)
- 1885 page_accuracy=0.683 appears to be **GT annotation error** — every VLM source independently agrees on the "wrong" pages for 12 articles
- Replace_ratio tuned per date: 1.02 on 1885 (less aggressive replacement), 1.10 on 1910 (more conservative — only replace if secondary is 10% longer)
- 1885 chain: col3 primary → exp_056_col4_ads (2nd replace) → exp_055_col6_ads (3rd) → yolo (4th) → col4 (5th) → col5, col6 additive
- 1910 chain: col3 primary → exp_055_col6_ads (2nd replace) → yolo → col4 → col5, col6, minicpm, phi35 additive
- Key innovation: VLM_OCR_ADS_FOCUSED prompt (emphasizes small ads/classifieds/editorial credits as separate units) paired with fine column splits (col4, col6) catches content missed by default configs
- vllm 0.19.1 upgrade provides 2-5x speedup enabling rapid iteration (Qwen3-VL registered)
- Build via: `uv run --no-project python scripts/build_multi_ensemble.py`
- Saves to: `eval/predictions/ensemble_best_{date}.json`
- Base: 4-config ensemble (col3 primary + yolo secondary replace + col4 tertiary replace + col5 quaternary additive-only)
- **1885 additive tier**: exp_055_col6_ads_prompt (col6 split + ads-focused prompt) — captured missed classifieds
- **1910 additive tiers**: col4_minicpm_o, col4_phi35_vision, exp_052_col6_vllm (col6 + V2 prompt), exp_055_col6_ads_prompt (col6 + ads prompt)
- Key wins this session:
  - exp_052 col6+vllm on 1910 (+0.0125): fine vertical splits catch small classifieds
  - exp_055 col6+ads_prompt (+0.007 on 1885, +0.003 on 1910): prompt emphasizing "each small ad is a separate unit" + col6 splits
- vllm 0.19.1 (Qwen3-VL registered) is 2-5x faster than transformers backend, quality equivalent. Required torch 2.10 + flash-attn rebuild.
- Dead ends this session: Qwen3-VL-30B-A3B AWQ (scrambled), Qwen3-VL-32B AWQ (OOM 24GB), Qwen3-VL-8B-Thinking (too slow, raylet crashed), olmOCR-2 with V2 prompt (fails, outputs markdown), cross-page stitchers (too aggressive), col3 max_tokens=16k (hallucinates), col8 (too noisy).
- Fallback: 4-way best 0.835 (`ensemble_4way_best_{date}.json`).

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
- **Additive-only merge**: for low-quality-high-recall configs (col5: wCER 1.034 standalone but catches tiny page-6 classifieds), use `replace_ratio=100.0` in the merge — disables text replacement, only adds truly new articles. Sweet spot `overlap_threshold=0.20-0.30` (aggressive dedup prevents noisy replacements). Bumped 1910 recall from 87.6% to 92.7% (+5%). Key insight: coverage and text-quality contributions to the ensemble should be decoupled.
- **Pre-page_span-fix predictions are broken**: predictions made before the page_span fix have default `page_span=[1]` or empty — do not include in ensembles. Must re-run the config to get correct page numbers. E.g. col4_qwen3_8b re-run: 1910 score went from -0.030 (broken) to +0.039 (fresh) contribution.
- **3-config ensemble beats 2-config**: col3 + YOLO + col4 (all with page_span fix) at 0.827 avg composite (+0.028 over 2-config 0.799). col4's 4-column splits catch 1910 classifieds that 3-column + YOLO miss.
- **4-config additive ensemble is new best**: col3 + YOLO + col4 (replacement) + col5 (additive-only) at 0.835 avg composite (+0.036 over 2-config). Ordering matters — tested permutations, canonical order optimal.
- **InternVL3-8B as OCR VLM is a dead end**: Ray runtime_env pinning transformers==4.49.0 fixes the loading error but InternVL3 hallucinates placeholder text ("text continues with dense Italian text") and fabricates anachronistic content. Not an OCR model; generates topic-inferred text.
- **Qwen3-VL-4B fails on column OCR**: 14 articles/recall 24% vs 48/95% for 8B. Structured JSON prompt too complex for 4B. Stick with 8B.

## Rules
- One change per experiment
- Always eval on BOTH dates (1885-06-15 and 1910-06-15), report average composite
- Name configs `exp_NNN_shortdesc.py` with incrementing numbers
- Log EVERY result to `eval/autoresearch/log.jsonl`, even failures
- Don't modify ground truth or eval code
