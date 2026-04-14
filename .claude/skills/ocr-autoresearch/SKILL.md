---
name: ocr-autoresearch
description: Hillclimb the OCR benchmark by iterating on pipeline configs, running on ripperred, and evaluating against ground truth.
when_to_use: When user says "do the ocr autoresearch", "hillclimb the benchmark", "run ocr experiments", or "autoresearch"
allowed-tools: Bash Read Write Edit Grep Glob
---

# OCR Autoresearch

Run a self-paced loop that hillclimbs the OCR benchmark. Each iteration: propose a config change, run it on the remote GPU server, evaluate, log, decide next step.

## How It Works

1. Read `eval/autoresearch/program.md` for current baseline, priorities, and learnings
2. Read `eval/autoresearch/log.jsonl` for past experiment results
3. Propose ONE targeted change (one variable at a time)
4. Write config to `configs/ocr/exp_NNN_description.py` (if new prompt, add to `src/mausoleo/ocr/prompts.py`)
5. Sync to ripperred, run on `1885-06-15`, copy prediction back
6. Evaluate with `evaluate_issue()`, log to `log.jsonl`
7. Update `program.md` if baseline improved
8. Report one-line summary, schedule next iteration

## Key Commands

Sync code:
```bash
rsync -avz --exclude='.venv' --exclude='.git' --exclude='__pycache__' --exclude='eval/predictions' --exclude='eval/ground_truth' -e 'ssh -p 62022' ./ audiogen@81.105.49.222:~/mausoleo_di_roma/
```

Clean pycache + run config:
```bash
ssh audiogen@81.105.49.222 -p 62022 "cd mausoleo_di_roma && find src/ -name __pycache__ -exec rm -rf {} + 2>/dev/null && export HF_TOKEN=$HF_TOKEN && .venv/bin/python scripts/run_real_ocr.py <config_name> 1885-06-15"
```

Copy prediction back:
```bash
scp -P 62022 audiogen@81.105.49.222:~/mausoleo_di_roma/eval/predictions/<config>_1885-06-15.json eval/predictions/
```

Evaluate:
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
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_NNN_description",
    operators=[
        ColumnSplit(num_columns=3, overlap_pct=0.03),
        VlmOcr(model="Qwen/Qwen3-VL-8B-Instruct", prompt=prompts.YOUR_PROMPT, backend="transformers", max_tokens=8192),
        MergePages(),
        ParseIssue(),
    ],
)
```

## What You Can Change

- **Prompt** in `src/mausoleo/ocr/prompts.py`
- **Column count** (1-6) and overlap (0.0-0.10)
- **Preprocessing**: `Preprocess(grayscale=False, max_dimension=1024)` before other operators
- **Model**: Qwen2.5-VL-7B, Qwen2.5-VL-3B, Qwen3-VL-8B, Qwen3-VL-4B, Qwen3-VL-2B
- **Backend**: "transformers" or "vllm" (vllm only for Qwen2.5)
- **max_tokens**: 4096-16384

## Rules

- One change per experiment
- Always eval on `1885-06-15`
- Log every result in `log.jsonl`, even failures
- Update `program.md` baseline when you beat it
- Use `/loop` dynamic pacing — each run takes ~5-10 min
