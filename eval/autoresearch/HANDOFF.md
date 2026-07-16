# Session Handoff — 2026-07-16 (evening update)

Continuation point for the autoresearch loop running on ripperred (tmux). Read `program.md` (objective, budget, protocol), `registry.md` (approach families), and the last entries of `log.jsonl` first. Work the loop per `.claude/skills/ocr-autoresearch/SKILL.md`; self-pace with ScheduleWakeup.

## Verification queue — ALL DONE (see log.jsonl 2026-07-16T20:3x entries)
1. **Ensemble rewrite equivalence VERIFIED**: fresh ParallelEnsembleOcr merges reproduce the documented v1 baselines exactly (1885 0.8719, 1910 0.9257). `/tmp/ens_1910_before.json` was a stale April artifact — ignore it. v2 baseline: 0.7514 avg (0.7111 / 0.7917).
2. **Failing test FIXED**: `crop_page()` now tolerates YoloLayout's per-page region-list shape; stale `/ocr` server tests removed. Full suite green (47 pipeline tests), pyright clean on touched files.
3. **Leaderboard re-based to v2** (program.md table): lean 3-source `ensemble_3way_textrep` leads at 0.7537 → F4 reopened for source pruning.
4. **Fresh sub-pipeline runs FIXED**: vllm 0.19.1/torch 2.10 upgrade broke cold-start of the 20480-len fullpage subs (KV 2.55 vs 2.81 GiB needed, both GPUs). `gpu_memory_utilization` field on VlmOcr; 0.94 for the two fullpage subs; verified end-to-end on GPU1.

## Environment notes
- **GITHUB_TOKEN on ripperred is EXPIRED** — commits land locally, `git push` fails. User must refresh auth; then push the pending commits.
- An unrelated Audiogen service (`eve`, ~350 MiB) sits on GPU0 — do not kill. GPU1 preferred for tight-memory engines (`CUDA_VISIBLE_DEVICES=1`).
- Disk: ~12 GB free on /home (100% full rounding) — prune before any model download.

## Experiment queue
- **exp_045 steady-state throughput** (RUNNING at handoff): both dates back-to-back on GPU1, `--force`, log in scratchpad `exp045_bench.log`; canonical exp_045 predictions backed up in scratchpad and MUST be restored after timing extraction (benchmark outputs are vllm-noise variants; keep the verified cache stack).
- **exp_151_olmocr_native_col3** (config READY): olmOCR-2-7B native front-matter prompt + MergeMarkdownPages (front-matter stripping added). One variable vs exp_047 (prompt/adapter).
- **exp_152_paddleocr_yolo_titles** (config + mechanism READY): exp_148 + YoloCrop separate title regions + MergePages nearest-below headline attachment. Unit-tested; flag-off paths byte-identical.
- **Unlimited-OCR column-crop sequence** (F2 unblock): col3 crops as page sequence via `scripts/run_unlimited_standalone.py` + `~/unlimited_env`; crops pre-generated in scratchpad if session survived.

## Constraints (already in CLAUDE.md/program.md — enforced)
- Budget 6.9–13.9 GPU-s/page; record timing every run.
- One config → one run → one result. All compute on ripperred. One variable per experiment. Log everything with mechanism lines; update registry.md every iteration; never stop the loop on failed waves.
