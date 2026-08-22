# Mausoleo di Roma — OCR Autoresearch

High-quality OCR of the Il Messaggero historical newspaper corpus (1880–1959, ~175K pages scraped as JPEGs on endeavour). The repo contains phase 1 (the OCR research loop) only; later phases (hierarchical index, search, agent CLI) are planned in `plan/roadmap.md` and will be rebuilt on top of the finished OCR corpus.

## Layout

- `GOAL.md` — the objective and the one hard constraint (budget cap).
- `experiments/` — **each experiment is one self-contained script.** Contract: `python experiments/<name>.py <date> [<date>...]` reads page images from `eval/ground_truth/<date>/*.jpeg` and writes `eval/predictions/<name>_<date>.json` in the Issue schema. Implementation is entirely free. See `experiments/README.md`; superseded scripts live in `experiments/archive/`.
- `src/mausoleo/eval/evaluate.py` — MausoleoBench. **Never modified to improve a score.**
- `src/mausoleo/ocr/` — reusable pieces importable from experiments: `models`, `prompts`, and the `operators` (`column_split`, `merge`, `parse`, `vlm_ocr`) called directly as plain functions.
- `eval/autoresearch/` — the research program: `program.md` (protocol, budget mechanics, review checklist), `registry.md` (approach families and what is ruled out), `LOOP.md` (live state + queue), `mausoleobench_log.jsonl` (every scored run).
- `eval/ground_truth/<date>/` — page images per issue, plus `ground_truth.json` for the 6 human-verified eval issues (1885-06-15, 1895-06-15, 1910-06-15, 1925-06-15, 1935-06-15, 1952-06-15). The 1943-07-* issues are images only.
- `eval/predictions/` — one JSON per (experiment, date); `archive/` holds the outputs still cited by `registry.md`.
- `scripts/` — `research.py` (eval / board / holdout / probe), `eval_probes.py` (pinned anti-gaming invariants), `time_experiment.sh` (caller-side budget measurement), `progress_server.py` (live board + prediction viewer, owns `BUDGET_CAP`), `review_server.py` (human GT review UI), `scrape_messaggero.py` (corpus acquisition).
- `plan/roadmap.md` — the whole roadmap: product, all five phases, standing decisions.

## Rules

- All compute on ripperred; endeavour is corpus storage only.
- Budget cap **250.0 sec/page**, caller-measured by `scripts/time_experiment.sh`; `BUDGET_CAP` in `scripts/progress_server.py` is the single source of truth.
- One variable per experiment; adversarial audit + holdout + probe checks before accepting anything; every run logged to `eval/autoresearch/mausoleobench_log.jsonl` with a mechanism line.
