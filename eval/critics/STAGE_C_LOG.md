# Stage C Log — GPTZero AI-detection reduction with GAN Pareto

Stage C entry baseline = R95 (commit 65c74b1) + R109 sentence-8 edit (uncommitted in working tree at task start).

## Baselines

- **GPTZero baseline (R95 working-tree)**: 0.929 AI / pure_ai / AI_ONLY (full result `eval/critics/final_aidetection.json`).
  - Document-level `completely_generated_prob = 0.9286`, `confidence_score = 0.9289`.
  - Per-sentence: nearly every sentence in abstract + §1 sits in 0.95-0.99 range; only 3 sentences below 0.5 (the recent-NLP-literature footnote, the "Mausoleo borrows..." short pivot, the absent-day pay-off sentence — these are the human-readable anchors).
  - Hot zones: Abstract ¶1, §1 ¶1, §1 ¶3, §1 ¶6, Preface (all 99%+).

- **GAN baseline (R95)**: 2/3 PASS LEAN (BEST_GAN ceiling tied by R80, R94, R95, R101 TSI S2). MUST NOT regress below this.

- **Strategy ladder per dispatch**: 1a → 1b → 1c → 1d → 2a → 2b → 2c → 3a → 3b → 3c → 4 (LAST RESORT).

## Per-round summary

(rounds appended below)

---

### R1_C — Strategy 1a (em-dash + en-dash strip): NO-OP

Pre-edit audit found 0 em-dashes and 0 non-citation en-dashes in body (lines 1-171). References page-ranges preserved. Strategy already exhausted in earlier Stage A/B rounds.

- GPTZero: 0.929 (unchanged)
- GAN: 2/3 LEAN (unchanged)
- Quota burn: 0 (no edit, no scan needed; budget conserved for substantive rounds)

Plan: `round1_C_plan.md`. Advance R2_C → Strategy 1b.

### R2_C — Strategy 1b (parallel-triplet collapse, low-GAN-risk)

Edits: §3 OCR triplet → semicolon cascade; §4 results triplet → 3 short sentences.
Result: GPTZero **0.929 → 0.929** (Δ=0). Doc-level fingerprint unmoved by low-prominence-section surface edits.
Action: REVERTED (git checkout). No GAN check (Pareto irrelevant).
Quota: GPTZero 1/7 in 24h.
Plan: `round2_C_plan.md`. Verdict: `round2_C_gptzero.md`. Advance R3_C → Strategy 1c.

### R3_C — Strategy 1c (hedge cadence variation): NO-OP

Pre-edit grep: 0 matches for `Notably / Importantly / It is worth noting / Furthermore / Moreover / Additionally / Indeed / Crucially / Specifically / Interestingly / In conclusion / Overall / In summary / It should be noted` openers. Already scrubbed in Stage A/B.
Action: zero-cost confirmation.
Plan: `round3_C_plan.md`. Advance R4_C → Strategy 1d.

### R4_C — Strategy 1d (sentence-length variance): SKIP

Pre-edit measurement: mean 25.9w, median 24w, σ=13.9w, range 3-88w, 11% short (<10w), 16% long (>40w). Already moderate-high variance. Per SKILL: "Burstiness alone is no longer a lever. GPTZero v4.4b penalises short fragments at 1.0 generated_prob." Adding more short fragments risks regression.
Action: informed skip, no edit, no quota burn.
Plan: `round4_C_plan.md`. Advance R5_C → Strategy 2a (vocabulary simplification).


