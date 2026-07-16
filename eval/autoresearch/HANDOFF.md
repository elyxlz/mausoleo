# Session Handoff — 2026-07-16 (night update)

Continuation point for the autoresearch loop running on ripperred (tmux). Read `program.md` (objective, budget, protocol), `registry.md` (approach families), and the last entries of `log.jsonl` first. Work the loop per `.claude/skills/ocr-autoresearch/SKILL.md`; self-pace with ScheduleWakeup.

## Session results (2026-07-16 evening/night — all logged in log.jsonl, all committed locally)
- **Verification queue DONE**: ensemble rewrite verified (reproduces v1 baselines exactly); layout_json shape fix + stale test cleanup (suite green); leaderboard re-based to v2; fresh sub-pipeline runs fixed (`gpu_memory_utilization` knob — vllm/torch upgrade had broken 20480-len cold starts).
- **Throughput ground truth** (`scripts/bench_throughput.py`, per Elio's directive: single engine, 5 era-diverse issues, generation-only): Qwen3-8B ≈ 136 GPU-s/page (dead for production); Paddle eager 14.54; **Paddle + CUDA graphs 5.13 GPU-s/page = 5.2-day corpus**.
- **Accepted**: exp_152 (YOLO title-class headlines, +0.025), exp_155 (CUDA graphs, 2.8×), **exp_157 (= production candidate: + char-run squeeze guard, 0.4284 v2, 5.13 GPU-s/page)**, **ensemble_prune5 (v2 leader 0.7776, +0.026 over 8-source; split-stable, holdout flat)**.
- **Rejected/blocked**: exp_153 (conf 0.15 worse), exp_154/156 (gap 30/40 split-overfit per holdout), exp_151 olmOCR (no headings, blocked), Unlimited-OCR tile sequence (F2 fully blocked — degenerates on legible tiles too).

## Environment notes
- **GITHUB_TOKEN on ripperred EXPIRED** — ~10 local commits await push (`git push origin master` after user refreshes auth).
- `eve` service holds ~350 MiB on GPU0 — do not kill; prefer GPU1 (`CUDA_VISIBLE_DEVICES=1`).
- Disk ~12 GB free — prune before model downloads.
- Paddle graphs mode: small run-to-run output variance on degraded scans despite seed=0; eval-date outputs stable.

## Next queue (registry-prioritized)
1. **F3 PP-DocLayoutV3** (31M, newspaper class + reading order): the recall bottleneck (0.36–0.49) is region segmentation, not OCR. Own venv/runtime_env (paddle runtime). Compare regions vs DocLayout-YOLO on the same Paddle OCR stage (one variable).
2. **F4 second pass**: precision-filter ensemble_prune5 survivors (dedup/confidence gating), or add exp_157-Paddle as diversity source to the oracle.
3. **F1**: abandon-class filtering; horizontal_overlap lever (NOT vertical gap — holdout-rejected twice).
4. Corpus production harness (separate from eval): persistent-engine driver streaming issues (bench_throughput.py already proves the shape).

## Constraints (already in CLAUDE.md/program.md — enforced)
- Budget 6.9–13.9 GPU-s/page; bench_throughput.py is the standard measurement for production candidates.
- One config → one run → one result. One variable per experiment. Adversarial audit + holdout + probe before accepting. Log with mechanism lines; update registry.md every iteration; never stop the loop on failed waves.
