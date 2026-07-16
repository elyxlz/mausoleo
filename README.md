# Mausoleo di Roma — OCR Autoresearch

High-quality OCR of the Il Messaggero historical newspaper corpus (1880–1959, ~175K pages scraped as JPEGs on endeavour). The repo currently contains phase 1 (OCR research loop) only; later phases (hierarchical index, search, agent CLI) are planned in `plan/` and will be rebuilt on top of the finished OCR corpus.

## Layout

- `experiments/` — **each experiment is one self-contained script.** Contract: `python experiments/<name>.py <date> [<date>...]` reads page images from `eval/ground_truth/<date>/*.jpeg` and writes `eval/predictions/<name>_<date>.json` in the Issue schema. Implementation is entirely free (no framework required). See `experiments/README.md`.
- `configs/ocr/` + `scripts/run_real_ocr.py` — legacy Ray harness, kept only for the verified oracle ensembles (`ensemble_30min` recall-oracle, `ensemble_prune5` v2 leader) and current baselines.
- `src/mausoleo/ocr/` — pipeline operators used by the legacy harness; reusable pieces (merge, trim, prompts) importable from experiments.
- `src/mausoleo/eval/` — metrics (`evaluate_issue`, composite_v2). **Never modified to improve a score.**
- `eval/autoresearch/` — the research program: `program.md` (objective, budget, protocol), `registry.md` (approach families), `log.jsonl` (every experiment), `eval_review.md` (metric audit).
- `eval/ground_truth/<date>/` — page images per issue; `ground_truth.json` where human-verified GT exists (1885-06-15, 1910-06-15).
- `eval/tentative_gt/<date>/` — machine-reconstructed draft GT awaiting human review (`scripts/build_tentative_gt.py`).
- `eval/predictions/` — one JSON per (config|experiment, date).
- `scripts/` — `research.py` (eval / board / holdout / probe), `bench_throughput.py` (steady-state GPU-s/page), `build_tentative_gt.py`, `run_real_ocr.py` + `run_sub_pipeline.py` (legacy oracle harness), `scrape_messaggero.py`.
- `plan/` — roadmap including the later phases (hierarchical index, search & CLI, packaging).

## Rules

- All compute on ripperred; endeavour is corpus storage only.
- Corpus budget: 6.9–13.9 GPU-s/page steady-state (`bench_throughput.py` is the standard measurement).
- One variable per experiment; adversarial audit + holdout + probe checks before accepting anything; everything logged to `eval/autoresearch/log.jsonl` with a mechanism line.
