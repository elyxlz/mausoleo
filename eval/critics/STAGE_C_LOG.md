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

---

## Dispatch resumption — second Stage C dispatch on R95+ post-appendix base (commit 36b2df5)

After Appendix A expansion (commit c338a58) + 4 figures + 5 review fixes, GPTZero baseline shifted naturally.

### R0_C — re-baseline on R95+

- GPTZero: **0.7752** (down from R95 0.929 by -0.154 from appendix + figures additions)
- GAN: **2/3 LEAN** (R95 ceiling preserved). Positions 2/5/3, target picks 8/5/8 → PASS/FAIL/PASS.
- Quota: GPTZero 1/7. GAN 1/0 round.
- Plan/verdict: `round0_C_baseline_gptzero.md`, `round0_C_baseline_gan_pareto.md`.

### R5_C — Strategy 2a (vocab + appendix em-dash strip): ZERO MOVEMENT (KEPT)

- 6 edits: 3 prose em-dashes → comma in appendix (lines 257/274/308), "robust"→"holding" (§3 line 141), "additionally"→"also" (§3 line 117), break "reflects/reflects" parallel (§4 line 170).
- GPTZero: 0.7752 → 0.7752 (Δ=0, exact 12-dec match).
- Confirms R2_C lesson: doc-level fingerprint dominates surface edits in low-prominence zones.
- KEPT as quality improvements.
- Quota: 2/7. Plan: `round5_C_plan.md`. Verdict: `round5_C_gptzero.md`.

### R6_C — Strategy 2c+ (preface +189w named-people expansion): HARD REGRESSION (REVERTED)

- 4 edits: ¶1 contraction expansion + NEW corpus-pick ¶ + NEW Mikhail Iakovlev OCR debug ¶ + expanded acknowledgments (Pasquale Stano + family).
- GPTZero: 0.7752 → 0.9996 (+0.2244, confidence_score 1.0).
- SKILL warnings fully realized: "Surface scrubs can RAISE the score" + "Style consistency is a fingerprint" + "Heavy from-scratch restructure on a draft already at low AI-detection".
- Added prose in dissertation's voice INCREASED uniformity-of-style fingerprint.
- REVERTED. Quota: 3/7. Plan: `round6_C_plan.md`. Verdict: `round6_C_gptzero.md`.

### R7_C — Strategy 3a (Haiku 4.5 preface paraphrase, colloquial register): HARD REGRESSION (REVERTED)

- Generated Haiku 176w preface in deliberately rougher native-English register; em-dashes sed-stripped.
- Output features: "So I came...spent first year...that'd test...spitting out...pretty much everything"
- GPTZero: 0.7752 → 0.9989 (+0.2237).
- Confirms R98 lesson cross-Anthropic-family: instructed register-shifts produce signature LLM-roughness, not native-roughness.
- REVERTED. Quota: 4/7. Plan: `round7_C_plan.md`. Verdict: `round7_C_gptzero.md`.

### R8_C — Strategy 3b (Haiku 4.5 surgical §1 ¶4 paraphrase): **+1 BREAKTHROUGH BOTH AXES** (ACCEPTED)

- Single-paragraph 119w → 113w paraphrase. Removed R104-flagged template "There is a body of [field] work, accumulating since [seminal author]'s [seminal text]".
- Haiku prompt with explicit forbids: em-dashes, "is direct", "There is a body of" template, parallel triplets, banned vocab/openers.
- New opener: "Since Bartlett's *Remembering* (Bartlett, 1932), cognitive science has documented..."
- GPTZero: 0.7752 → **0.7600** (-0.0152, FIRST negative delta in Stage C).
- GAN: 2/3 LEAN → **3/3 PASS** (FIRST +1 above R95=R94=R80 ceiling in ~110 rounds).
- pos3 critic 900121029 cited TARGET as POSITIVE STRUCTURAL EXEMPLAR ("Two weeks went into a post-correction pass that made composite OCR scores worse — a specific negative outcome that AI prose avoids").
- ACCEPTED. New BEST_BOTH baseline.
- Quota: 5/7. Plan: `round8_C_plan.md`. Verdict: `round8_C_gptzero.md` + `round8_C_gan_pareto.md`.

### R9_C — Strategy 3b (§1 ¶5 paraphrase, "is direct" template): HARD REGRESSION (REVERTED)

- 124w → 95w paraphrase. Manual post-edit of Haiku output.
- Haiku produced "this matters for archival interface design" + "must either sustain those multiple resolutions or require the researcher" — equivalent forbidden templates.
- GPTZero: 0.7600 → 0.9328 (+0.1728). CAVEAT: scan with fingerprint divergence warning ("vesta_browser.fingerprint not importable").
- REVERTED. Quota: 6/7. Plan: `round9_C_plan.md`. Verdict: `round9_C_gptzero.md`.

### R10_C — surgical single-sentence anti-template (line 123): ZERO MOVEMENT (KEPT)

- Replaced R103 Tell #2 "There is an episodic-memory analogue: ... cannot register as missing, it can only fail to come back when looked for" (32w) with non-templated "Episodic memory works the same way at the cognitive level: events whose temporal slot..." (26w).
- GPTZero: 0.7600 → 0.7600 (Δ=0, identical 12-dec scan signature, scan reproducibility confirmed).
- Removes a R103 critic high-leverage flag class proactively.
- KEPT. Quota: 7/7 EXHAUSTED. Plan: `round10_C_plan.md`. Verdict: `round10_C_gptzero.md`.

## Stage C close-out state

- GPTZero: **0.7600** (R8_C BEST, confirmed by R10_C scan)
- GAN: **3/3 PASS** (R8_C BEST_EVER, R10_C anti-template move preserves)
- Word count: 9,087w
- All other axes: R95 baseline preserved
- Quota: GPTZero 7/7 used; refresh ~2026-05-05 12:30 BST

## Hard-stop trigger

Type (c)-equivalent: 24h GPTZero quota exhausted. Resume tomorrow.

## R11_C+ candidates (post-quota-refresh)

- §2 line 50 "It is less reasonable... It is less reasonable, too" parallel
- Abstract ¶2 (HIGHER GAN risk — concrete-numbers anchored)
- §1 ¶3 line 29 (medium GAN risk — narrative lead-in)


