# Phase 2: Corpus-Scale Production OCR Run

> **STATUS 2026-07-17: not started, no driver yet.** Runs the production config (exp_157 lineage) over all ~29K issues (~175K pages). At 5.13 GPU-s/page a full pass is ~5.2 days on 2×3090 (budget cap 13.9 → ~2 weeks) — cheap enough to rerun, so plan for corpus versions: **v0** early to unblock phase 3, **v1+** as phase-1 quality improves.

## Known concerns (solve when building, not before)

- Persistent engine: models loaded once per GPU worker, steady-state batching; same code path as eval runs, never a production fork.
- Resumability across ~29K issues: one output file per issue is the checkpoint; restart = manifest diff; per-issue failure isolation + quarantine list.
- Staging: corpus JPEGs don't fit on ripperred — bounded prefetch spool from endeavour, delete after commit; versioned outputs pushed back to endeavour.
- Quality monitoring during the run, GT-free: per-issue probe metrics (lexicon validity, repetition, articles/page) vs rolling per-decade baselines; GT issues flow through as sentinels scored with composite_v2.

## When to trigger

v0: as soon as expanded GT shows no era collapse for the production config and a pilot slice (~a month per decade) survives a mid-run kill/resume. Do not wait for the phase-1 ship bar — a rerun costs ~a week. v1 (what phase 3 ships on) waits for the bar in 01_ocr.md.

## Exit criterion

≥99% of issues with valid Issue JSON, versioned + manifested on endeavour; per-decade probe report clean; residual failures listed explicitly.

## Open questions

- Partial-issue policy for corrupt/missing pages.
- Where corpus copies live besides endeavour (single-copy risk).
