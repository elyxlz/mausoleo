# Phase 1–2: OCR Pipeline — Eval, Design, and Production

## Goal

Build and evaluate an OCR pipeline for 1880–1945 Italian newspaper archives (Il Messaggero). Find the best approach through systematic benchmarking, then process the full archive (~130K pages) at scale using Ray Data.

## 1. Ground Truth & Evaluation

### Eval Dataset

3 complete issues spanning eras (16 pages total):
- 1885-06-15 (4 pages) — dense 4-column, yellowed, earliest era
- 1910-06-15 (6 pages) — cleaner, pre-WWI, 5-6 columns
- 1940-04-01 (6 pages) — WWII era, photos, modern layout

Images at `eval/ground_truth/{date}/*.jpeg`, sourced from endeavour `/media/sdr/`.

### Ground Truth Creation

Bootstrap approach (no human transcriptions yet):
1. Run all OCR configs on the 3 eval issues
2. Select consensus prediction (lowest avg CER to all others) per issue
3. Dedup repeated text (VLM hallucination artifacts)
4. Manual review: correct obvious OCR errors by reading the page images

Bootstrapped GT at `eval/bootstrap_gt/{date}/ground_truth.json`.

Manual corrections applied:
- 1885: "frremetel"→"fremete!", "Il calcolo del buco"→"Il calcio del bue", "sefagurata"→"sciagurata"
- 1910: "teleggr."→"telegr."
- 1940: "accendo"→"avendo", "terra schiavita"→"terra schipetara", removed repeated race results

### Metrics

- **CER** (Character Error Rate) via jiwer
- **WER** (Word Error Rate) via jiwer
- **Kendall's tau** for reading order (implemented, not yet computed)
- Code: `src/mausoleo/eval/metrics.py`, `src/mausoleo/eval/evaluate.py`

## 2. Pipeline Framework

### Architecture

Ray Data operator pattern (from model-factory Apollo/data):
- `OcrPipelineConfig` has `operators: list[BaseOperatorConfig]` — the config IS the pipeline
- Each operator registered via `@register_operator`, dispatched by `apply_operator()`
- `mock=True` flag on any operator for testing without GPU
- Configs live in `configs/ocr/*.py`, loaded dynamically by `scripts/run_real_ocr.py`

### Operators

| Operator | Type | What it does |
|----------|------|-------------|
| `VlmOcr` | GPU | Per-page VLM OCR. vLLM or transformers backend. |
| `LlmCleanup` | GPU | Text-only LLM restructures raw OCR into articles |
| `LlmPostCorrect` | GPU | Text-only LLM fixes character-level OCR errors |
| `MergePages` | CPU | Merges per-page structured JSON into single issue |
| `ParseIssue` | CPU | Validates JSON, assigns deterministic IDs |
| `Preprocess` | CPU | Grayscale + contrast + resize |
| `ColumnSplit` | CPU | Fixed N-column split with configurable overlap |
| `YoloCrop` | GPU | DocLayout-YOLO detection + merge + crop columns |
| `WholeIssueVlm` | GPU | All pages at once (abandoned: image tokens exceed context) |
| `SuryaOcr` | GPU | Surya OCR engine (not yet tested on GPU) |

### Output Format

```json
{
  "date": "1885-06-15",
  "source": "il_messaggero",
  "page_count": 4,
  "articles": [
    {
      "id": "1885-06-15_a00",
      "unit_type": "article",
      "headline": "I deputati-telegrafo",
      "paragraphs": [{"id": "1885-06-15_a00_p00", "text": "..."}],
      "page_span": [1],
      "position_in_issue": 0
    }
  ]
}
```

### Prompt Versions

- **v1**: basic "transcribe and identify articles" (4096 tokens)
- **v2**: column-aware, explicit reading order, "no markdown", "do not truncate" (8192 tokens)
- **VLM_OCR_COLUMN**: per-column raw transcription
- **LLM_CLEANUP_V2**: context-aware article assembly
- **LLM_POST_CORRECTION**: character-level error correction preserving archaic Italian

## 3. Approaches Benchmarked

### 10 Pipeline Approaches

1. **Full-page structured**: VlmOcr(structured prompt) → MergePages → Parse
2. **Full-page raw + cleanup**: VlmOcr(raw) → LlmCleanup → Parse
3. **Structured + post-correction**: VlmOcr(structured) → LlmPostCorrect → MergePages → Parse
4. **Preprocessing + structured**: Preprocess → VlmOcr → MergePages → Parse
5. **Column split + structured**: ColumnSplit(N) → VlmOcr(structured) → MergePages → Parse
6. **Column split + raw + cleanup**: ColumnSplit → VlmOcr(column) → LlmCleanup → Parse
7. **Column split + raw**: ColumnSplit → VlmOcr(raw) → MergePages → Parse
8. **Column split no overlap**: ColumnSplit(overlap=0) → VlmOcr → MergePages → Parse
9. **Preprocess + column split**: Preprocess → ColumnSplit → VlmOcr → MergePages → Parse
10. **YOLO crop + structured**: YoloCrop → VlmOcr(structured) → MergePages → Parse

### 9 VLMs Configured

| Model | Backend | VRAM | Status |
|-------|---------|------|--------|
| Qwen2.5-VL-7B | vLLM | ~17GB | ✅ Best, fast |
| Qwen2.5-VL-3B | vLLM | ~7GB | ✅ Good |
| Qwen3-VL-8B | transformers | ~17GB | queued |
| Qwen3-VL-4B | transformers | ~9GB | queued |
| Qwen3-VL-2B | transformers | ~5GB | queued |
| Gemma-3-9B | transformers | ~19GB | queued |
| Gemma-3-2B | transformers | ~5GB | queued |
| InternVL3-6B | transformers | ~13GB | queued |
| InternVL3-2B | transformers | ~5GB | queued |

42 configs total (10 approaches × varying models). Config generator: `scripts/generate_configs.py`.

### Models That Failed

| Model | Reason |
|-------|--------|
| InternVL2.5-8B | cuDNN init error in Ray actors |
| Phi-3.5-Vision | Chat template incompatible with vLLM |
| Llama-3.2-11B | Gated (needs HF approval) |
| MiniCPM-V-2.6 | Gated |
| Florence-2 | transformers version mismatch |
| HunyuanOCR | Needs unreleased transformers for hunyuan_vl |
| GOT-OCR2 | Works but heavy hallucination on 1880s text (CER 1.1) |
| InternVL2.5-4B | Works but poor on Italian (CER 2.0) |

## 4. Results

### Leaderboard (bootstrap eval, 2026-04-10)

| Rank | Config | Avg CER | Approach |
|------|--------|---------|----------|
| 1 | qwen_vl_7b_structured | 0.139 | Full-page, v1 prompt, 4K tokens |
| 2 | qwen7b_structured | 0.313 | Same model, different run |
| 3 | qwen7b_v2_structured_postcorrect | 0.388 | v2 + post-correction |
| 4 | qwen7b_v2_structured | 0.397 | v2 prompt, 8K tokens |
| 5 | qwen3b_structured | 0.583 | 3B model |

### Key Findings

**What works**:
- Qwen2.5-VL-7B with structured JSON prompt is the clear winner
- v2 prompt + 8K tokens extracts 2.3× more text (117K vs 54K on 1940 issue)
- Column-split (4 cols) extracts 2.3× more text than full-page (124K vs 54K on 1885)
- Simple fixed column split outperforms learned YOLO layout detection
- Single-model pipelines through Ray + vLLM work reliably

**What doesn't work**:
- Whole-issue VLM: image tokens exceed context for multi-page newspapers
- Two-model pipelines: Ray creates both GPU actors simultaneously → OOM
- YOLO DocLayout: 90+ detections per page, too granular; misses 1940s layouts
- Preprocessing (grayscale + contrast): didn't help, sometimes hurt
- GOT-OCR2: hallucination loops on degraded 1880s text
- InternVL: poor on Italian historical text

**Open questions**:
- Column-split extracts more text but bootstrap CER isn't better — real GT needed to know if it's actually higher quality
- Hybrid approach (Tesseract + VLM correction) not yet tested
- Cross-page article stitching not implemented
- Newer models (Qwen3-VL, Gemma-3, InternVL3) not yet benchmarked

## 5. Infrastructure

### Hardware
- **Ripperred** (audiogen@81.105.49.222:62022): 2× RTX 3090 24GB, CUDA 12.4, PyTorch 2.5.1+cu124, vLLM 0.7.3
- **Endeavour** (elio@81.105.49.222:62420): GTX 1080 8GB — too weak for VLMs, used for data storage only

### Ray + vLLM Integration
- Ray packages working dir → fix: `runtime_env={"excludes": ["pyproject.toml", "uv.lock", ".venv", ".git"]}`
- Two GPU operators OOM → partial fix: use 3B for cleanup model
- `ray.shutdown()` between pipeline runs to release GPU memory

### Scripts
- `scripts/run_real_ocr.py` — main runner, loads configs from `configs/ocr/`, supports `all` mode
- `scripts/run_single_hf.py` — alternative runner bypassing Ray (for models that don't work with Ray)
- `scripts/bootstrap_and_eval.py` — consensus GT selection + CER/WER evaluation
- `scripts/repair_predictions.py` — fixes JSON-in-text formatting artifacts
- `scripts/generate_configs.py` — generates all 42 config files systematically

## 6. Production Pipeline (Not Yet Started)

Once winning config is finalized:
1. Process full archive (1880–1945, ~130K pages, ~22K daily issues)
2. Output: one JSON per date in `ocr_output/{year}/{date}.json`
3. Checkpointing: skip already-processed dates
4. Quality spot-check across eras

## Implementation Log

### 2026-04-10: New VLM models + two-GPU Ray fix

**Two-GPU Ray pipeline fix:**
- Bug: `apply_operator` gave each GPU operator `max_actors = n_gpu / gpu_fraction = 2`, so VlmOcr + LlmPostCorrect both tried to claim all 2 GPUs simultaneously → OOM
- Fix: Added `n_gpu_operators` parameter, divides GPU budget per operator. Also set `ray.data.ExecutionResources(gpu=n_gpu)` limit for multi-operator pipelines
- Files: `base.py:apply_operator`, `pipeline.py:setup_ray + run_pipeline`

**New model support added to VlmOcr operator:**
- Model-type detection via `_detect_model_type()` with dispatch in `_transformers_call()`
- **Florence-2** (`microsoft/Florence-2-large`): Uses `<OCR>` task prompt, `AutoModelForCausalLM` with `attn_implementation="eager"`, `post_process_generation()`. Small (0.77B).
- **GOT-OCR2** (`stepfun-ai/GOT-OCR-2.0-hf`): Native HF transformers support via `AutoModelForImageTextToText`. 580M params. Uses `processor(image, return_tensors="pt")` directly (no chat template).
- **Phi-3.5-Vision** (`microsoft/Phi-3.5-vision-instruct`): `AutoModelForCausalLM` + `trust_remote_code`. Uses `<|image_1|>` placeholder. Processor needs `num_crops=16`. 4.2B params.
- **MiniCPM-o-2.6** (`openbmb/MiniCPM-o-2_6`): Ungated successor to MiniCPM-V-2.6. Uses `.chat()` API (different from InternVL). 8B params.
- **Llama-3.2-11B-Vision** (`unsloth/Llama-3.2-11B-Vision-Instruct`): Ungated via unsloth. Native `mllama` type, standard chat template. 11B params, needs 4bit for single 3090.
- **HunyuanOCR**: NOT added — needs vLLM >= 0.12.0 (ripperred has 0.7.3) or pinned transformers commit that would break other models. CUDA 12.9 required vs our 12.4.

**New configs (19 total):**
- Florence-2: `florence2_large_ocr`, `florence2_base_ocr`, `col4_florence2_large_ocr`
- GOT-OCR2: `got_ocr2_hf`, `col4_got_ocr2_hf`
- Phi-3.5: `phi35_vision_v1_structured`, `phi35_vision_v2_structured`, `col4_phi35_vision_v2_structured`, `phi35_vision_v2_raw_cleanup`
- MiniCPM: `minicpm_o_v1_structured`, `minicpm_o_v2_structured`, `col4_minicpm_o_v2_structured`, `minicpm_o_4bit_v2_structured`, `minicpm_o_v2_raw_cleanup`
- Llama-3.2: `llama32_11b_v1_structured`, `llama32_11b_v2_structured`, `col4_llama32_11b_v2_structured`, `llama32_11b_4bit_v2_structured`, `llama32_11b_v2_raw_cleanup`

**Issues encountered during testing:**
- Florence-2 `AutoModelForVision2Seq` fails with transformers 4.57.6 (not in auto mapping). Fix: use `AutoModelForCausalLM` with `trust_remote_code=True`
- Florence-2 `_supports_sdpa` error. Fix: `attn_implementation="eager"`
- 4bit configs need `bitsandbytes` — installed on ripperred
- Models landing on GPU 0 (where batch was running) → OOM. Fix: `CUDA_VISIBLE_DEVICES=1` for test runs

**Total configs:** 61 (42 old + 19 new)

### 2026-04-14: Overnight batch run + ground truth refinement

**New model support:**
- **Gemma-3-12B** (`google/gemma-3-12b-it`): Multimodal, uses `AutoModelForImageTextToText`. ~24GB bf16, needs `gpu_fraction=2.0` to spread across both 3090s. Even at 512px and 2 GPUs, too slow for multi-page newspaper OCR (>15 min per 4-page issue). The 4B variant (`google/gemma-3-4b-it`) works but is borderline on 6-page issues (~12 min).
- **Chandra-OCR-2** (`datalab-to/chandra-ocr-2`): Based on Qwen3.5-VL (~8B). Requires transformers >= 5.x (`Qwen3_5ForConditionalGeneration`). Too slow even at 768px with no runtime_env — needs 512px. Dedicated `_init_chandra` with `padding_side="left"`.
- **InternVL3-8B** (`OpenGVLab/InternVL3-8B`): Original config referenced non-existent `InternVL3-6B`. Needs dedicated `_init_internvl` using `AutoModel` (not `AutoModelForImageTextToText`) because InternVL uses a custom `.chat()` API that only exists on its own model class.

**Code quality (via /simplify):**
- Merged 4 duplicate call methods (`_gemma_call`, `_chandra_call`, `_hunyuan_call`, `_generate_api_call`) into single `_chat_template_call(do_sample)` 
- Fixed `dtype` bug in `_init_gemma` (was passing `dtype=` which is ignored, should be `torch_dtype=`)
- Added `_init_image_text_model` shared helper for gemma/chandra init
- Cached InternVL torchvision transform (was rebuilt per image)
- Added `PipelineTimeoutError` + `finally: signal.alarm(0)` cleanup in batch runner
- Added `ray.shutdown()` on timeout/error to prevent stale GPU actors blocking subsequent runs

**Runtime_env lessons:**
- `git+https://github.com/huggingface/transformers` in pip list causes Ray to clone entire repo + build from source on every actor spawn — takes 10+ minutes, eats most of the timeout budget. Use pinned release versions instead (e.g. `transformers==4.49.0`).
- Upgrading main venv transformers from 4.57.6 to 5.x breaks vllm 0.7.3 (rope_scaling conflict). Must keep 4.57.6 in main venv and use runtime_env for models needing newer versions.
- Runtime_env pip caches accumulate ~13GB per unique env per Ray session. Old sessions must be cleaned periodically or `/tmp` fills up.
- `PIP_FIND_LINKS` in `env_vars` does NOT affect pip install inside runtime_env (Ray manages its own pip invocation).

**Ground truth refinement — cross-column reading order challenges:**

The 1885-06-15 issue revealed several systematic challenges for OCR pipeline evaluation:

1. **Articles spanning multiple columns**: "I deputati-telegrafo" spans two columns on page 1. Column-split configs capture each column separately but don't know the article continues. Full-page configs sometimes merge them correctly, sometimes not. The reading order is: left column down, then right column down — but within a single article.

2. **Articles spanning multiple pages**: "I MAESTRI ELEMENTARI NON SONO PAGATI" starts on page 1 column 4 and continues on page 2 column 1, with the bottom of page 1 containing unrelated content (the serialized fiction "MAGNETIZZATA"). This is a newspaper convention where articles jump across pages — the physical reading order (top-to-bottom, page-by-page) differs from the logical article order.

3. **Subheadings within articles**: "COSA NE PENSANO I MURATORI / Le processioni della morte" appears as a subheading within the "deputati-telegrafo" article, not a separate article. VLM configs frequently split these into separate articles, inflating the article count.

4. **Serialized fiction interleaved with news**: The APPENDICE "MAGNETIZZATA" runs along the bottom of multiple pages, interleaved with news articles. Column-split configs capture fragments of fiction mixed with news text.

5. **Best configs for ground truth bootstrapping**: `col4_qwen25_7b_v2_raw` captures the most faithful raw text per column. `col5_qwen25_7b_v2_structured` captures the best structured article segmentation. `col4_qwen3_8b_v2_structured` captures the most complete cross-column text. No single config handles all challenges — manual cross-referencing across configs is needed for accurate ground truth.

These findings suggest that the evaluation metric should account for reading order separately from character accuracy, and that cross-page article stitching is a necessary post-processing step for production use.

### 2026-04-14: Hand-built ground truth + article-level evaluation

**Hand-built ground truth for 1885-06-15:**
- 41 articles across 4 pages, 60,001 characters total
- Built article-by-article: user reads original page images, provides first/last words and column break locations; Claude cross-references all OCR config outputs to assemble the best transcription
- Content types: 29 news articles, 2 serialized fiction (MAGNETIZZATA appendice + Angelo Pitou by Dumas), 1 entertainment (Sciarada), 1 masthead, 7 advertisements
- Historical grave accents preserved (perchè not perché — 1885 Italian convention)
- Stored at `eval/ground_truth/1885-06-15/ground_truth.json`

**New article-level evaluation metrics** (`src/mausoleo/eval/article_metrics.py`):

Old approach: concatenate all text, compute CER/WER. Problem: penalises different article ordering even when text is correct.

New approach:
1. **Article matching**: For each GT article, find best matching prediction by Jaccard word overlap (threshold 0.15)
2. **Per-article CER/WER**: Compute CER/WER per matched article pair, average across matches
3. **Article detection F1**: Precision (what fraction of predicted articles match GT) × Recall (what fraction of GT articles found)
4. **Page span accuracy**: Does the prediction assign the correct pages to each article?

**Results on 1885-06-15 with article-level eval:**

| Rank | Config | CER | WER | Recall | F1 |
|------|--------|-----|-----|--------|-----|
| 1 | col3_qwen3_8b_v2_structured | 0.373 | 0.454 | 78.0% | 70.3% |
| 2 | qwen7b_structured | 0.556 | 0.635 | 56.1% | 68.7% |
| 3 | yolo_qwen7b_structured | 0.573 | 0.632 | 97.6% | 16.7% |
| 4 | qwen25_7b_v1_structured | 0.606 | 0.689 | 58.5% | 71.6% |
| 5 | col4_qwen7b_v2_structured | 0.767 | 0.747 | 75.6% | 72.9% |

**Key insight**: Column-split (col3) with Qwen3-8B is the best overall — 3-column splitting matches the 1885 newspaper layout well. YOLO detection gets the highest recall (97.6%) but massively over-segments (437 predicted articles vs 41 GT). Many configs score 0% because their outputs contain JSON artifacts or broken text that doesn't match any GT article.

**Page span accuracy is universally poor** (~10%) — no config currently tracks which page an article comes from, which is critical for cross-page article stitching in production.

**Detailed failure analysis** (per-article breakdown of col3_qwen3_8b, the best config):

| Category | Count | CER Range | Root Cause |
|----------|-------|-----------|------------|
| Easy (near-perfect) | 6 | 0.000-0.023 | Short dispatches, single column, single page |
| Medium (good w/ errors) | 17 | 0.14-0.49 | Hyphenation artifacts, archaic spelling, minor truncation |
| Hard (significant errors) | 8 | 0.50-0.98 | Cross-page truncation, fiction interleaving, non-standard layout |
| Failed (missed/broken) | 10 | 1.0+ | Page 4 ads entirely missed, cross-page articles lost |

Three systematic failure modes identified:
1. **Cross-page articles truncated**: 4 articles spanning page boundaries either missed entirely or captured <30% of text. Column-split processes pages independently with no continuation detection.
2. **Page 4 advertisements missed**: All 8 ads on page 4 scored CER=1.0. Non-standard ad layouts (borders, small text, mixed fonts) not handled by structured JSON prompt.
3. **Serialized fiction interleaving**: MAGNETIZZATA runs along bottom of pages 1-3, gets mixed with news in column splits.

Research directions prioritized and documented in `eval/autoresearch/program.md` for automated hillclimbing via `/ocr-autoresearch` skill.

### 2026-04-22/23: Composite-score hillclimb (0.799 → 0.8837)

Extended autoresearch loop across ~100 experiments. Full log: `eval/autoresearch/log.jsonl`. Full state: `eval/autoresearch/program.md`.

**Composite score progression (avg across 1885-06-15 + 1910-06-15):**
- 0.799 — 2-config ensemble (col3 primary + yolo secondary)
- 0.835 — 4-config with col5 additive (EXP 038)
- 0.8549 — per-date replace_ratio tuning (EXP 058) — starting point of this session
- 0.8567 — dropped phi35/minicpm noise + quality_delta=0.10 (EXP 069)
- 0.8586 — exp_045 (vllm variant of col3) as primary (EXP 072)
- 0.8599 — yolo_qwen25_7b additive (EXP 074)
- 0.8603–0.8635 — per-source overlap/ratio tuning (EXP 077–083)
- **0.8800 — JSON-blob filter (`scripts/trim_repetitive.py`) (EXP 089)** — biggest single win, +0.0165
- 0.8808 — exp_098 col5+vllm additive (EXP 098)
- **0.8837 — exp_099 col2+vllm additive at overlap=0.75 (EXP 099)** — second biggest, +0.0029

**Key architectural learnings:**

1. **VLMs occasionally leak raw JSON into article text** when `MergePages` fails to parse their output. Always post-filter predictions by detecting text starting with ` ```json `, `{"articles":` or containing `"unit_type"+"paragraphs"` patterns, plus strip repetitive trailing-dot hallucinations. `scripts/trim_repetitive.py`.

2. **Backend diversity is a free lunch.** Same model + same prompt under vllm vs transformers produces meaningfully different text (sampling params differ); both in the ensemble is better than either alone. Applies at col3, col4, col5, col6.

3. **Column granularity diversity matters.** col2 (wide), col3/4/5/6 (progressively finer), plus yolo region detection all contribute unique coverage. col7 over-segments without new coverage.

4. **Per-source hyperparameters beat uniform.** Different sources detect articles with different boundaries, so:
   - Yolo-based sources: `overlap_threshold=0.50` (regions don't align to columns)
   - Column-based sources: `overlap_threshold=0.75` (strict dedup)
   - Noisy additives (col5, yolo_qwen25_7b): strict overlap to filter bad matches
   - Clean additives (col6_vllm): loose overlap for coverage
   - col3 primary-sibling: `replace_ratio=1.15` (strict)
   - col4 : `replace_ratio=1.02` (aggressive — 4-col splits often genuinely better)

5. **Reliable dead ends**: cross-page article stitching (VLM naturalizes text), accent restoration, stochastic sampling as diversity, length-aware body replacement, headline consensus voting, most 30B+ quantized models, Nanonets/Gemma3 in vllm (tied lm_head / cuDNN conflicts).

**Unconstrained pipeline**: `scripts/build_multi_ensemble.py` — 16 sub-pipelines + ensemble, reproduces **0.9087**. Runtime: ~2-3 hours for one issue on 2× RTX 3090 — violates the 30 min/issue production constraint. Kept as research upper bound.

**Late-session breakthroughs (2026-04-23)**:
1. **Whole-page VLM** additives at overlap=0.75: +0.011. Fullpage captures long multi-column articles (like "deputati-telegrafo" spanning 2 columns) that all column-split configs truncate. 1885 wCER dropped 0.234→0.182.
2. **Qwen2.5-VL-7B cross-family diversity** at col2/col3/col4/fullpage: +0.013 stacked. Different pre-training, different error patterns → orthogonal coverage. 1885 wCER 0.182→0.149.

**Production pipeline (30 min/issue)**: `scripts/ensemble_pipeline_30min.py` — **0.8777 avg** (only -0.006 from unconstrained). 6 configs in 2 parallel GPU chains: GPU 0 runs `exp_045 → col3_trans → exp_099_col2_vllm` (~24-28 min); GPU 1 runs `exp_010_yolo → exp_055_col6_ads → col4_trans` (~25-28 min). Wall-clock max ≤ 30 min.

### 2026-04-14: Eval framework consolidation

Consolidated scattered eval code into single module `src/mausoleo/eval/evaluate.py`:
- `compute_cer()`, `compute_wer()` — character/word error rates via jiwer
- `match_articles()` — Jaccard word overlap matching between GT and predicted articles
- `evaluate_issue()` — per-article CER/WER + full-text CER/WER + article F1 + page accuracy
- `evaluate_all()` — batch eval across all configs and dates
- `print_results()` — formatted leaderboard

Un-gitignored `eval/predictions/` — 194 prediction files now tracked in git.
Removed obsolete: `eval/compare.py`, old `evaluate.py`, bootstrap scripts, one-off GT builders.
Added `/ocr-autoresearch` skill for automated benchmark hillclimbing (Karpathy autoresearch pattern).

### 2026-04-26: Lean5 cold-cache rebaseline + constraint reformalization

**The "warm-cache cheating" problem**: Discovered that the 0.92717 score in earlier sessions used 24 sources in REPLACE/ADDITIVE/crosspage chains, but only 5 of those (the gpu_chains) are actually generated cold-cache. The other 19 were silently used from previous experiment cache. A fresh-machine cold-cache run with the same config would silently skip 19 sources and score much lower.

Stripped production config to ONLY reference sources actually run cold-cache. Real cold-cache score with that config: **0.88682** (1885=0.86297, 1910=0.91066) — well below the inflated 0.92717.

**Constraints (formalized)**:
- **HARD**: ≤30 min wall-clock per issue from cold cache (max(GPU0 chain, GPU1 chain) on 2× RTX 3090 24GB)
- **HARD**: Fully standalone — every run regenerates all sub-pipeline predictions from raw images, zero prior computation
- ~~Max 5 sub-pipelines~~ — **dropped 2026-04-26**, only 30-min budget matters now

**Lean5 wins (0.88682 → 0.89198, +0.00516)**:
1. exp_107 fullpage_qwen25vl as REPLACE pos 0 (+0.00007)
2. col4_qwen3_8b at REPLACE pos 1 (+0.00033)
3. quality_select sources = all 5 sub-pipelines (+0.00316 cumulative)
4. exp_055 ADDITIVE at ov=0.88 (+0.00020)
5. primary_name = exp_107 (+0.00006)
6. exp_010_yolo ov 0.45→0.25 (+0.00122)
7. exp_055 in REPLACE ov=0.30 r=1.02 (+0.00012)

**Verified essential**:
- Qwen2.5-VL-7B fullpage (exp_107) — replacing with any Qwen3 fullpage drops 1885 by 0.02-0.04
- Transformers backend for col4 — vllm version closes gap with `vllm_strict=True` (dtype=bf16, prefix_caching=False, seed=0) added to `VlmOcr` config

**vllm strict mode investigation**: `enable_prefix_caching=False` is critical for output match with transformers (~30-50% slower but score within -0.0002). Even at temperature=0, prefix caching's KV reuse introduces numerical precision differences that change generated tokens.

**Cold-cache timings (1885, 4 pages)**:
| Pipeline | Time | Backend |
|---|---|---|
| exp_107 fullpage Qwen2.5 | 4m 20s | vllm |
| exp_045 col3 Qwen3 | 5m 27s | vllm |
| exp_055 col6+ads Qwen3 | 7m 7s | vllm |
| exp_134 yolo Qwen3 strict | 6m 21s | vllm (prefix_caching=True in this run; ~9 min with prefix=False) |
| col4_qwen3_8b | **20m 45s** | transformers ← bottleneck |
| **exp_136 col4 Qwen3 vllm** | **5m 9s** | vllm (4× faster than transformers) |

**Production budget**: max(GPU0 chain 16m54s, GPU1 chain 27m6s) = ~27 min on 1885; extrapolates to ~40 min on 1910 (over budget). col4 transformers is THE bottleneck.

**Current path**: switch col4 to vllm (~5 min); use saved budget headroom to add more sub-pipelines (col1+crosspage, exp_028 yolo_smallregion, exp_098 col5, etc.) targeting score recovery toward warm-cache 0.927.
