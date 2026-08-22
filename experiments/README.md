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
- Never read `eval/ground_truth/*/ground_truth.json` at inference; never re-emit another experiment's prediction file; never import `mausoleo.eval`.
- Evaluate with `scripts/research.py eval <exp_name>` (audit + holdout; also `holdout`/`probe`/`board` subcommands).
- Budget is measured by the caller: `scripts/time_experiment.sh experiments/<name>.py` reports sec/page over the 6 eval issues. Cap = `BUDGET_CAP` in `scripts/progress_server.py` (250.0).
- Log every result to `eval/autoresearch/mausoleobench_log.jsonl` with a mechanism line; update `registry.md` and `LOOP.md`.

## Reusable pieces (optional)

`src/mausoleo/ocr/` is importable when convenient (`sys.path.insert(0, "src")`): `mausoleo.ocr.prompts`, `mausoleo.ocr.models`, and the operators (`column_split`, `merge_pages`, `parse_issue`, `VlmOcrOperator`) — all plain functions/classes called directly, no framework. See `experiments/exp_017_column8b_direct.py` for the direct-call pattern, or `_template.py` to start from scratch. Using none of it is equally fine.

## Archive

`experiments/archive/` holds superseded experiment scripts, kept as the reproducible record behind the claims in `registry.md`. Some of them import modules that no longer exist; they are history, not runnable code.
