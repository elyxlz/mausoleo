# OCR Pipeline Auto-Research Program

## Objective
Minimize article-level CER on the 1885-06-15 eval issue while maintaining high article recall (>70%).

## Current Best
- Config: `col3_qwen3_8b_v2_structured`
- Article CER: 0.373
- Article Recall: 78.0%
- F1: 70.3%

## The Loop

Each iteration:

1. **Read state** — this file for priorities, `eval/autoresearch/log.jsonl` for past experiments
2. **Propose** — ONE targeted change (one variable at a time)
3. **Write config** — `configs/ocr/exp_NNN_description.py` (if new prompt, add to `src/mausoleo/ocr/prompts.py`)
4. **Run** — sync to ripperred, execute, copy prediction back
5. **Evaluate** — run `evaluate_issue()` locally, log to `log.jsonl`
6. **Decide** — if improved update baseline here, if not revert and try different direction
7. **Report** — one-line: `EXP NNN: [description] CER=X.XXX (baseline 0.373) [IMPROVED/WORSE]`
8. **Schedule next** — use ScheduleWakeup, each run takes ~5-10 min

## Infrastructure

Sync code to remote:
```
rsync -avz --exclude='.venv' --exclude='.git' --exclude='__pycache__' --exclude='eval/predictions' --exclude='eval/ground_truth' -e 'ssh -p 62022' ./ audiogen@81.105.49.222:~/mausoleo_di_roma/
```

Clean pycache + run a config:
```
ssh audiogen@81.105.49.222 -p 62022 "cd mausoleo_di_roma && find src/ -name __pycache__ -exec rm -rf {} + 2>/dev/null && .venv/bin/python scripts/run_real_ocr.py <config_name> 1885-06-15"
```

Copy prediction back:
```
scp -P 62022 audiogen@81.105.49.222:~/mausoleo_di_roma/eval/predictions/<config>_1885-06-15.json eval/predictions/
```

Evaluate locally:
```python
from mausoleo.eval.evaluate import evaluate_issue
import json
gt = json.loads(open("eval/ground_truth/1885-06-15/ground_truth.json").read())
pred = json.loads(open("eval/predictions/<config>_1885-06-15.json").read())
result = evaluate_issue(gt, pred, config="<config>", date="1885-06-15")
# result.mean_cer, result.article_recall, result.article_f1
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

## Research Priorities (try in this order)
1. **Prompt engineering** — biggest lever. Try Italian-specific instructions, different output formats, reading order hints, fewer/more constraints
2. **Column count** — 3 works but maybe 2 with overlap, or 4 with less, is better
3. **Preprocessing** — resolution, grayscale
4. **Model switching** — Qwen2.5-VL-7B via vllm might be faster and comparable
5. **Two-stage** — raw OCR + LLM cleanup

## Rules
- One change per experiment
- Always eval on 1885-06-15
- Name configs `exp_NNN_shortdesc.py` with incrementing numbers
- Log EVERY result to `eval/autoresearch/log.jsonl`, even failures
- Don't modify ground truth or eval code
