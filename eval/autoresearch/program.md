# OCR Pipeline Auto-Research Program

## Objective
Minimize article-level CER on the 1885-06-15 eval issue while maintaining high article recall (>70%).

## Current Best
- Config: `col3_qwen3_8b_v2_structured`
- Article CER: 0.373
- Article Recall: 78.0%
- F1: 70.3%

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

### 1. Cross-page article stitching (biggest potential gain)
- Process pages separately, then post-process to detect continuations
- Text similarity at page boundaries: if last sentence of page N is incomplete, search page N+1
- Or: process pages in overlapping pairs (page 1+2, page 2+3) to capture cross-page articles
- Or: whole-issue context — give the model all pages and ask it to identify article continuations

### 2. Ad-aware prompting
- Current prompt says "separate distinct content units" but doesn't specifically mention ads
- Try explicit content type listing: "articles, advertisements, notices, serialized fiction (APPENDICE)"
- Two-pass approach: first pass for articles, second pass specifically for ads
- Page 4 specifically needs different handling — it's mostly novel + ads, not news

### 3. MAGNETIZZATA / APPENDICE separation
- The bottom strip of each page is serialized fiction — column-split mixes it with news
- Try: crop bottom 20% separately as APPENDICE, process with different prompt
- Or: add to prompt "The bottom strip contains serialized fiction (APPENDICE) — separate from news above"
- Or: use YOLO layout detection to identify the APPENDICE strip

### 4. Prompt refinements for medium-CER articles
- 17 articles at CER 0.1-0.5 have systematic errors:
  - Broken words at column boundaries (hyphenation artifacts)
  - Archaic spelling confusion
  - Truncation near max_tokens limit
- Try: increase max_tokens from 8192 to 12288
- Try: "preserve hyphenated words across line breaks" in prompt
- Try: "this is 1885 Italian — preserve archaic forms like perchè, poichè"

### 5. Per-page column count
- col3 works for news pages (1-3) but page 4 has different layout (novel + ads)
- Try: 3 columns for pages 1-3, full-page (no split) for page 4
- This would require a config that varies column count per page (not currently supported — may need operator changes)

### 6. Ensemble / multi-config merge
- No single config captures everything:
  - col3_qwen3_8b: best CER (0.373) but misses all ads
  - yolo_qwen7b: best recall (97.6%) but terrible CER
  - full-page qwen25_7b: captures cross-page text but lower recall
- Merge strategy: use col3 for text quality, yolo for article detection, combine the best of both
- This is a post-processing approach, not a single-config solution

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

## Rules
- One change per experiment
- Always eval on 1885-06-15
- Name configs `exp_NNN_shortdesc.py` with incrementing numbers
- Log EVERY result to `eval/autoresearch/log.jsonl`, even failures
- Don't modify ground truth or eval code
