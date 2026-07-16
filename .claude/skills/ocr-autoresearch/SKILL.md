---
name: ocr-autoresearch
description: Hillclimb the OCR benchmark by iterating on experiments, running on ripperred, and evaluating against ground truth.
when_to_use: When user says "do the ocr autoresearch", "hillclimb the benchmark", "run ocr experiments", or "autoresearch"
allowed-tools: Bash Read Write Edit Grep Glob
---

Read `eval/autoresearch/program.md` and follow it fully — objective, budget, metrics, Generalization Protocol, Orchestration Discipline, and The Loop. Read `registry.md` at the start and update it at the end of every iteration; log every result to `log.jsonl` with a mechanism line.

Experiments are self-contained scripts per `experiments/README.md`. All compute runs on ripperred (never endeavour — corpus storage only). Long GPU runs go in background Bash tasks; self-pace with ScheduleWakeup and schedule a fallback wakeup while they run.
