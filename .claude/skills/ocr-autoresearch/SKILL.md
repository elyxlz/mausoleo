---
name: ocr-autoresearch
description: Hillclimb the OCR benchmark by iterating on pipeline configs, running on ripperred, and evaluating against ground truth.
when_to_use: When user says "do the ocr autoresearch", "hillclimb the benchmark", "run ocr experiments", or "autoresearch"
allowed-tools: Bash Read Write Edit Grep Glob
---

Read `eval/autoresearch/program.md` and follow it. Use `/loop` (ScheduleWakeup) to self-pace iterations. Long GPU runs go in background Bash tasks; schedule a fallback wakeup while they run. All compute runs on ripperred (never endeavour; endeavour only stores the scraped corpus).

## Orchestration discipline (long-horizon research rules)

1. **Approach registry.** `eval/autoresearch/registry.md` groups all work into approach FAMILIES by underlying mechanism, not surface wording (the same model tried on col4 vs col5 is one family). Read it at the start of every iteration; update it at the end of every iteration. If recent experiments cluster in one family, deliberately redirect the next ones toward underexplored ACTIVE families.

2. **Blocked-route rule.** When an approach stalls on a hard failure (model loops, dependency wall, systematic quality floor), mark the route BLOCKED in the registry with the exact failure and a concrete UNBLOCK CONDITION. Never re-attempt a BLOCKED route without a materially new mechanism — a prompt rewording or hyperparameter nudge does not qualify.

3. **No false progress.** Re-tuning ensemble weights at a saturated optimum, or adding a source highly correlated with existing sources, is not progress even if the composite ticks up within noise. Verify genuine diversity (pairwise text distance, LOO contribution) before crediting a new source. Respect the per-issue time budget in program.md — a "win" that violates the budget is not a win.

4. **Parallel incompatible routes.** Keep ≥2 structurally different directions alive across rounds (e.g. fast small-model sources AND layout upgrades AND long-horizon parsing). Do not let one route monopolize iterations because its next step is easiest. Cross-pollinate only after each side works standalone.

5. **Adversarial audit before accepting ANY change** (all must pass):
   - both eval dates same-sign delta, above the noise floor for the change class
   - tune/holdout halves: no holdout regression (`scripts/eval_holdout.py`)
   - probe issues: no degradation in lexicon validity / repetition / length distribution (`scripts/probe_metrics.py`)
   - matcher-gaming check: no giant blob articles inflating text_overlap; article length distribution sane
   - silent-truncation check: outputs not bumping against max_tokens/max_model_len
   - per-issue time budget respected (program.md Resource Budget section)

6. **Concrete artifacts only.** Every log entry names specific articles/failure cases that improved or worsened (from actually reading predictions vs GT). "Score went up, looks good" is a status report, not evidence — reject it.

7. **Persistence.** Do not conclude "saturated" or end the session because a wave of experiments failed. Failed waves are information: update the registry, pick the next-best family, launch the next round. Stop only when the user stops you or every ACTIVE family is blocked pending user input. Report intermediate state honestly: strongest verified gain + exact remaining gaps, never vague optimism.

8. **Search policy.** Web search is for integration knowledge (model usage, APIs, versions) — never to import benchmark numbers as our results.
