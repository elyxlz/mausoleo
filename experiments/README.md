# Experiments — the script IS the experiment

Every experiment is one self-contained Python script. No framework, no config registry, no Ray requirement — full freedom of implementation. The only fixed things are the input and output contracts.

## Contract

```
.venv/bin/python experiments/<exp_name>.py <date> [<date> ...]
```

- **Input**: page images at `eval/ground_truth/<date>/<N>.jpeg` (N from 1, reading order).
- **Output**: `eval/predictions/<exp_name>_<date>.json` — one file per date, Issue schema:

```json
{
  "date": "1885-06-15",
  "source": "il_messaggero",
  "page_count": 4,
  "articles": [
    {
      "id": "1885-06-15_a00",
      "unit_type": "article | advertisement | obituary | notice | editorial | other",
      "headline": "text or null",
      "paragraphs": [{"id": "1885-06-15_a00_p00", "text": "..."}],
      "page_span": [1],
      "position_in_issue": 0
    }
  ]
}
```

- `<exp_name>` = the script's filename stem; it must match the prediction filename prefix.
- The script must be idempotent per date and print per-date wall time (loads reported separately when feasible).

## Rules (unchanged from program.md)

- One variable per experiment relative to a named baseline.
- Never read `eval/ground_truth/*/ground_truth.json` at inference; never re-emit another experiment's prediction file.
- Evaluate with `scripts/research.py eval <exp_name>` (audit + holdout; also `holdout`/`probe`/`board` subcommands); throughput with `scripts/bench_throughput.py` where applicable — production candidates must report steady-state GPU-s/page vs the 6.9–13.9 budget.
- Log every result to `eval/autoresearch/log.jsonl` with a mechanism line; update `registry.md`.

## Reusable pieces (optional)

`src/mausoleo/ocr/` is importable when convenient (`sys.path.insert(0, "src")`): `mausoleo.ocr.merge` (trim_predictions, merge_with_replacement, select_best_text), `mausoleo.ocr.prompts`, operators like `YoloCropOperator`, `MergePages`, `MergeMarkdownPages` — see `experiments/_template.py` for the direct-call pattern (no Ray). Using none of it is equally fine.

## Legacy harness

`configs/ocr/` + `scripts/run_real_ocr.py` remain ONLY for the verified oracle ensembles (`ensemble_30min`, `ensemble_prune5`) and the production candidate (`exp_157`). New work goes here.
