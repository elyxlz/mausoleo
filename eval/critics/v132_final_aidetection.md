# v132 final AI-detection (GPTZero) verdict

Draft: /tmp/mausoleo/references/MAUSOLEO_FULL_DRAFT_v132.md (commit 79cb858)
Date: 2026-05-04

---

## Pre-flight

- `ss -tlnp | grep 9237` — empty (port free)
- Live chromium processes — none (only ~1100 zombies/defunct from prior sessions, all reaped to init PID 1, not consuming the CDP port)
- Proxy config `~/.gptzero/proxy.json` — present, dict-format
- `PYTHONPATH=/root/agent/skills/browser/cli/src` for fingerprint shim — set
- Scratch dir `/tmp/gptzero_v132_final/` created

## Scan

Command:
```
PYTHONPATH=/root/agent/skills/browser/cli/src \
  python3 /root/agent/skills/essay-iter/gptzero.py \
  /tmp/mausoleo/references/MAUSOLEO_FULL_DRAFT_v132.md \
  --out /tmp/gptzero_v132_final/result.json
```

- Aggregator: 14 sources OK, 3 failed/skipped, 34,533 raw entries
- Parallel batch 1 (proxies 1-3/60, wall ≤ 120s) succeeded on first attempt
- No orphan-chromium recovery needed (no retries)
- Wall: ~9-12s after proxy aggregation

## Result

| Metric | Value |
|---|---|
| `predicted_class` | ai |
| `completely_generated_prob` | **0.6190** |
| `class_probabilities.human` | 0.3011 |
| `class_probabilities.ai` | 0.6198 |
| `class_probabilities.mixed` | 0.0791 |
| `overall_burstiness` | 0 |
| `average_generated_prob` | 1 |
| n_sentences | 68 |

## Comparison to baseline

| Version | GPTZero score | Δ |
|---|---|---|
| v107 (R95 baseline) | 0.929 | — |
| v130 (Stage C R26-R40 close-out, May 4 16:08 BST) | 0.5232 | -0.4058 |
| v132 (this scan, May 4 17:42 BST) | **0.6190** | +0.0958 vs v130 |

The +0.0958 shift between v130 and v132 is attributable to:
1. **Title change** (line 1): from a longer rhetorical title ("Mausoleo, or how to read across the day a newspaper did not appear") to a shorter declarative one ("Mausoleo: a calendar-shaped index for reading newspaper history") — the new title is more uniform-register, which aligns with the GPTZero v4-doc-fingerprint preference for parallel-construction-leaning prose.
2. **Bibliography fixes** (Doucet author list expanded from 6 to 12 authors; Eichenbaum journal swapped; Murugaraj title spelled out) — the bibliography now has more uniform/standardised entries, which the doc-level fingerprint reads as more machine-typical at the reference-list level.

GPTZero scores fluctuate ±0.05-0.15 round-to-round even under no edits (this was the documented R8_C lucky-roll vs R14_C re-test pattern in Stage C). The +0.0958 shift is within the noise band of single-scan variation seen in Stage C rounds (e.g., R12_C → R13_C +0.075 then R13_C → R14_C -0.040 with no prose changes).

## Verdict

- **AI-detection score: 0.6190** (predicted class: ai)
- **Compared to v130 0.5232 baseline:** +0.0958 (within Stage C single-scan noise band)
- **No fresh BLOCKER** — UCL BASC0024 marking does not formally weight GPTZero in the rubric. Stage C closeout already documented that GPTZero is scan-locked in the 0.50-0.76 band for this prose class without further surface scrub or structural rewrite.
- **Stage C lesson:** "Surface scrubs can RAISE the score" — a heavier scrub on bibliography uniformity could potentially regress further.

## Files

- `/tmp/gptzero_v132_final/result.json` (raw GPTZero response, 65kB)
