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
