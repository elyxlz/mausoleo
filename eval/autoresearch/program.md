# OCR Pipeline Auto-Research Program

## Objective
Minimize article-level CER on the 1885-06-15 eval issue while maintaining high article recall (>70%).

## Current Best
- Config: `col3_qwen3_8b_v2_structured`
- Article CER: 0.373
- Article Recall: 78.0%
- F1: 70.3%

## What You Can Change
1. **Prompt** (`src/mausoleo/ocr/prompts.py`) — the VLM_OCR_STRUCTURED_V2 prompt or create new variants
2. **Column splitting** — number of columns (2,3,4,5), overlap_pct (0.0 to 0.10)
3. **Preprocessing** — max_dimension (512-2048), grayscale (true/false)
4. **Model** — any model already cached on ripperred (Qwen2.5-VL-7B, Qwen3-VL-8B, Qwen2.5-VL-3B, Qwen3-VL-4B)
5. **Backend** — transformers or vllm
6. **max_tokens** — 4096 to 16384
7. **Config composition** — combine operators in new ways

## What You Cannot Change
- The eval metric code
- The ground truth
- The models available (don't try to download new ones)
- The hardware (2x RTX 3090 24GB)

## Research Priorities (in order)
1. **Prompt engineering** — the biggest lever. Try: explicit column reading order instructions, Italian-specific guidance, different output formats, fewer/more constraints
2. **Column count tuning** — 3 columns works well for 1885 but maybe 2 or 4 with different overlap is better
3. **Preprocessing** — does grayscale help? Different resolutions?
4. **Two-stage pipelines** — raw OCR + structured cleanup might beat single-stage

## Constraints
- Each experiment must complete in <15 minutes (900s timeout)
- Use only 1 GPU unless the model requires 2
- Name experimental configs as `exp_NNN_description.py` in `configs/ocr/`
- After each run, record the result in `eval/autoresearch/log.jsonl`

## Strategy
- Start with prompt variations on the winning config (col3 + qwen3_8b)
- Small targeted changes — one variable at a time
- If a change helps, keep it and build on it
- If unclear, try the opposite direction to confirm
