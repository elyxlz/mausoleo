# Round 98 plan — STRATEGY B1: Opus 4.7 full Abstract+§1 rewrite on R95 base

## Strategy

After R97 0/3 with all 3 critics flagging pre-existing R80 abstract+§1 features (sweeping-historical opener "A century of cognitive-science work", abstract-feature opener "with date as a facet on the side", parallel triplets, balanced antithesis closer "documented silence ... cannot provide", "the dissertation argues + builds" meta-commentary, hyphenated compound stacking), the actual high-leverage targets sit in Abstract + §1 — exactly what STRATEGY B1 attacks.

## Implementation

a. Source v10 = R95 baseline (commit 65c74b1).
b. Extract Abstract + Preface + Chapter 1 block (~1441w).
c. Submit to Opus 4.7 via claude CLI with explicit anti-tell prompt: ban balanced antithesis, abstract-noun grammatical subjects, meta-hedging, definitional openers, parallel triplets, sweeping-historical openers, em-dashes, banned vocab, hyphenated compound stacking, and specific R97-flagged phrase "with date as a facet on the side". Distancing prompt: "you are a doctoral supervisor rewriting a dissertation chapter for a final-year UCL undergraduate. Eliminate any sentence that smells of ChatGPT prose. Prefer concrete-particular over abstract-general at every choice."
d. Hard preserves: all citations, italicised proper names, specific numbers, historical sequence, substantive claim, §1 paragraph order.
e. Length target 1,400-1,500w.
f. Replace v10 abstract+ch1 with rewrite output.

## Pareto rule

If R98 ≥ 3/3, ship. If R98 = 2/3, save as BEST_AXIS_B1 + branch B2 (§3 rewrite). If R98 < 2/3, revert v10 to R95 baseline and proceed to B2 from R95 directly.

## Anti-fail discipline

R83 + R84 + R85 lessons confirmed: cross-model REPLACEMENT can produce same-class tells. R98 differs from R83/R84/R85 in three ways: (1) using Opus 4.7 not Haiku/SICO; (2) explicit anti-tell prompt with 11 specific bans, not just SICO style-anchor or distancing prompt alone; (3) full-block rewrite of Abstract+Ch1 not paragraph-level swaps — covers ALL pre-existing R80 features in this block at once.
