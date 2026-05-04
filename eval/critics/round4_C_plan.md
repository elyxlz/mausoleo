# Stage C Round 4 plan — Strategy 1d (sentence-length variance)

## Strategy

1d = increase sentence-length variance by splitting long uniform sentences into short+long alternation.

## Pre-edit audit

Body sentence-length distribution (n=233 sentences, body lines 1-171):

| metric | value |
|---|---|
| mean | 25.9 words |
| median | 24 words |
| stdev | 13.9 words |
| min | 3w |
| max | 88w |
| <10w short | 25 (10.7%) |
| 10-20w | 53 (22.7%) |
| 20-30w | 76 (32.6%) |
| 30-40w | 42 (18.0%) |
| >40w long | 37 (15.9%) |

Already wide variance: σ=13.9w with both short fragments (3w) and long compound sentences (88w).

## SKILL warning

essay-iter SKILL says: "Burstiness alone is no longer a lever. GPTZero v4.4b (Aug 2025+) penalises short fragments at 1.0 generated_prob; the tactic was trained against."

This means:
1. Existing variance (σ=13.9, ~16% long, ~11% short) already generates whatever burstiness signal is achievable.
2. ADDING more short fragments via splitting may raise the score (short fragments now generated_prob=1.0 by default).

## Verdict on Strategy 1d

**Counter-indicated for R95 baseline.** Variance is already moderate-high; further splitting risks regression per SKILL. Documented and skipped without quota burn.

## Action

- Do NOT edit. Do NOT consume quota.
- Mark Strategy 1d complete (informed-skip, not no-op).
- Advance R5_C → Strategy 2a (vocabulary simplification).
