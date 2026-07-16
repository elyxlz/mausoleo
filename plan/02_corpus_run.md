# Phase 2: Corpus-Scale Production OCR Run

> **STATUS 2026-07-17: designed, driver not built.** Runs the production config over all ~29K issues (1880–1959, ~175K pages). Rerunnable by design: one full pass costs ~5–13 days on 2×3090, so corpus versions are expected (v0 with the best in-budget config, v1+ as phase-1 quality improves).

## Scope and budget

| | |
|---|---|
| Input | endeavour `/media/sdr/<year>/<MonthName>/<day>/<N>.jpeg` |
| Output | one Issue-schema JSON per issue, versioned corpus on endeavour |
| Config | single production config (exp_157 lineage), one engine load, steady-state |
| Cost at exp_157 (5.13 GPU-s/page) | ~5.2 days on 2×3090 |
| Hard cap (13.9 GPU-s/page) | ~2 weeks |

One config → one run: the corpus driver invokes the same code path as the eval runs (`run_real_ocr.py <config> <date>` semantics), never a divergent production fork. If the driver path and the eval path can drift, quality numbers stop meaning anything.

## Driver design

- **Manifest.** One-time scan of the endeavour tree → `corpus_manifest.jsonl`: `{date, page_paths, page_count}` per issue, ~29K rows. Malformed directories (missing days, non-contiguous page numbers) logged at build time, not discovered mid-run.
- **Workers.** Two persistent worker processes, one per GPU (`CUDA_VISIBLE_DEVICES` pinning), each loading the full pipeline once (YOLO + PaddleOCR-VL engine, CUDA graphs). Sharding by issue-index parity — no coordination, no queue service.
- **Staging.** Ripperred disk is tight and the corpus (~hundreds of GB of JPEGs) never fits locally. A prefetch thread rsyncs the next K issues from endeavour into a bounded spool (cap ~20 GB); page images are deleted as soon as the issue's output is committed. Throughput math: at exp_157 speed the two workers consume ~1 issue per ~15 s aggregate (~30 GPU-s per 6-page issue per GPU), so fetch needs only ~1–2 MB/s sustained — verify in the pilot anyway.
- **Per-issue flow.** fetch → OCR pages → merge → Issue JSON → GT-free probe metrics → atomic write (tmp + rename) → append status line to run log → delete spooled images.

## Resumability

- **Done = valid output.** An issue is complete iff its output file exists, parses, and `page_count` matches the manifest. Restart = diff manifest against output dir, process the remainder. No separate checkpoint state to corrupt.
- **Run log.** `run_v<N>.jsonl`: one line per issue — date, status, wall s, GPU-s/page, probe metrics, error. This is the monitoring surface and the post-run quality report input.
- **Supervisor.** Worker crash → restart the process (engine reload ~minutes, amortized over thousands of issues). An issue that kills the engine twice is quarantined, not retried forever.

## Failure handling

- Per-issue isolation, max 2 retries, then quarantine with the exact error.
- Corrupt/missing page image: OCR the remaining pages, record the gap in the run log (partial issue beats no issue); the issue is flagged, not failed.
- Final pass re-attempts the quarantine list; residual failures ship as an explicit known-gaps list in the corpus manifest. Target: ≥99% of issues with valid output.

## Output and storage

- `ocr_corpus/v<N>/<year>/<date>.json`, Issue schema plus a run-metadata header (config name, git hash, corpus version, timestamp).
- Total output is small (~29K JSONs, low GB) but ripperred disk is tight: push to endeavour in batches, verify, prune local copies.
- Final artifacts: corpus tree + `corpus_manifest.jsonl` + `run_v<N>.jsonl` + per-decade probe summary.

## Quality monitoring during the run (GT-free)

- Per issue, the driver computes the `scripts/research.py probe` metrics: lexicon validity rate, mean/high-share repetition, articles per page, chars per page.
- Rolling per-decade baselines; an issue deviating hard (e.g. lexicon validity ≥10 pts below decade median, or chars/page collapse) is flagged for inspection — the run does not block.
- **GT sentinels:** the 6 GT issues and the 1943-07 probe issues flow through the same driver; composite_v2 on the GT issues verifies end-to-end that driver output ≡ eval output.
- Post-run: per-decade probe report; a decade whose distribution looks broken (e.g. wartime paper stock, layout shifts) becomes a targeted phase-1 probe era before corpus v(N+1).

## When to trigger vs waiting for phase-1 quality

Corpus **v0** (best in-budget config, currently exp_157) runs as soon as:
1. Expanded GT (6 eras) is promoted and the production config shows no era collapse (per-issue composite floor per 01_ocr.md).
2. Driver pilot passes: one full month per decade (~250 issues), including a mid-run kill/resume test and spool/network throughput check.
3. Storage round-trip (push to endeavour, prune, verify) works.

Do **not** gate v0 on the ship bar: a ~1-week rerun cost means waiting buys nothing, while v0 unblocks phase-3 development on real full-corpus data and surfaces era-specific failures no 6-issue GT set can. Corpus **v1** — the version phase 3 ships on — requires the phase-1 ship bar (01_ocr.md).

## Open questions

- Partial-issue policy detail: minimum page fraction below which an issue is quarantined rather than shipped partial.
- Whether v0 output lives only on endeavour or also mirrored elsewhere (single-copy risk on scraped data that took weeks to collect).
- Exact spool sizing / prefetch depth once real per-issue fetch times are measured in the pilot.
