# Round 98 verdict — STRATEGY B1: Opus 4.7 full Abstract+§1 rewrite on R95 base

## Result

**0/3 PASS** (1 NEAR_CERTAIN + 1 LEAN + 1 NEAR_CERTAIN, all picked TARGET). Hard regression below R80=R94=R95 = 2/3 LEAN. Reverted v10 to R95 baseline (commit 65c74b1).

## Seeds + positions

- seed=514411643, pos=2: critic picked Essay 2 (TARGET) "near certain" → FAIL. Tells: (1) "Date-bounded questions get poor service from an article-ranked interface" — STRUCTURAL HIGH-LEVERAGE — B1 rewrite produced "no authorial presence" / "X does Y procedure-recital voice" — opposite of intent; (2) "The pay-off is clearest at 26 July" — SURFACE HIGH-LEVERAGE — NEW phrasing from B1 rewrite; (3) parallel triplet "The first asks... The second reads... The third tracks..." — SURFACE STRUCTURAL — NEW from B1 rewrite (intended to break R80 triplet, replaced with same-class triplet); (4) citation parenthetical stacking; (5) "documented silence" nominalisation (kept); (6) metrics "no interpretive warmth"; (7) "There is an episodic-memory analogue" template; (8) uniform sentence length 18-25 words.
- seed=90456486, pos=6: critic picked Essay 6 (TARGET) "lean toward" → FAIL. Tells: (1) "It is less reasonable for the historian... It is less reasonable, too" — UNTOUCHED §2 R94 A1 leftover; (2) "What I have not shown:" §5 untouched; (3) "[X] is direct" repetition (§1 ¶5 + §2 ¶4 unchanged); (4) Bartlett→Miller→Cowan chain (B1-rewritten but content intrinsic); (5) "minimal personal voice in preface" (B1 kept preface as-is); (6) clean IMRAD structure; (7) "with the qualification that" §5 untouched.
- seed=297011084, pos=8: critic picked Essay 8 (TARGET) "near certain" → FAIL. Tells: (1) parallel fragment triplet in NEW B1 abstract ("What the paper said... How the regime change... How the balance...") — SURFACE STRUCTURAL — B1-INTRODUCED tell; (2) cog-sci literature parade (intrinsic); (3) NEW B1 balanced antithesis "Either the interface does the compressing, or the reader's head does"; (4) enumerated limitations "What I have not shown:" §5 untouched; (5) "[X] is direct" repetition; (6) precise abstract metrics; (7) clean problem-solution-evaluation framing; (8) detached preface voice.

## Strategy applied

B1 single-axis on R95: Opus 4.7 full Abstract + Preface + Chapter 1 rewrite (~1441w) via claude CLI with explicit anti-tell prompt. 11 hard bans: balanced antithesis, abstract-noun subjects, meta-hedging, definitional openers, parallel triplets, sweeping-historical openers, em-dashes, banned vocab (delve/tapestry/etc), banned openers (Moreover/Furthermore), hyphenated compound stacking, R97-flagged "with date as a facet on the side". 9 hard preserves: citations, italicised proper names, specific numbers, historical sequence, substantive claim, §1 paragraph order. Distancing prompt: "doctoral supervisor rewriting a dissertation chapter for a final-year UCL undergraduate".

Word count: 7,108 → 7,110w (+2w). Original abstract+preface+ch1 1441w → rewritten 1440w.

Sanity-checks pre-test: NO em-dashes, NO banned vocab. The rewrite passed surface-level scrub validation.

## Insight — Opus 4.7 anti-tell prompt does NOT prevent same-class tell production

R98 confirms R83/R84/R85 lesson at scale: REPLACEMENT operations (whether SICO Haiku, cross-model Haiku, Opus 4.7 with anti-tell prompt) produce same-class tells in different surface phrasings. Even with 11 explicit bans, Opus 4.7:
- Replaced R80 abstract triplet with NEW triplet "The first asks... The second reads... The third tracks..." — SAME pattern, different words
- Replaced R80 abstract closer with NEW antithesis "Either the interface does the compressing, or the reader's head does"
- Replaced "with date as a facet on the side" with "Date-bounded questions get poor service from an article-ranked interface" — flagged as new "X does Y procedure-recital voice" tell
- Introduced NEW phrasing "The pay-off is clearest at 26 July" — flagged as classic AI-ism
- Stripped first-person voice from the abstract — paradoxically introduced "no authorial presence" tell that R95's A2 first-person edits had defused

The model has an LLM-cadence prior that defeats explicit anti-tell prompts. The output reads like LLM pretending to follow anti-LLM rules. R98 produces a DIFFERENT distribution of LLM tells from R80 — neither is uniformly better, but the new distribution is being read as MORE confident-AI by 2 of 3 critics.

## Pareto

R98 = 0/3 < R80=R94=R95 = 2/3 LEAN BEST. v10 reverted to R95 baseline (commit 65c74b1).

STRATEGY B1 exhausted. The dispatch said B1 is "full §1 rewrite via Opus 4.7 with explicit anti-tell prompt" — done. B2 (full §3 rewrite), B3 (full §4 rewrite), B4 (full §5 rewrite) are next per dispatch but the same R83/R84/R85/R98 lesson WILL apply: cross-model REPLACEMENT produces same-class tells. The dispatch's note "B1+B4 should still try aggressive Opus rewrites layered on the structural rewrite" assumes the anti-tell prompt would suppress the LLM cadence — R98 demonstrates it does not.

DECISION: B2-B4 unlikely to outperform R95. Per the strategy ladder rules, attempt B2 anyway as a discipline check; if B2 also regresses, skip B3+B4 and pivot to STRATEGY C (embodied first-person concrete-detail injection) which is an ADDITION operation not a replacement.

## R99 plan

STRATEGY B2 = full §3 rewrite via Opus 4.7 same protocol on R95 base. If B2 < 2/3 (likely per R98 lesson), pivot to C1 (OCR-segmentation-error footnote anecdote injection in §3 + new concrete first-person research-process anecdote elsewhere).
