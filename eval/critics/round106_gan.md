# Round 106 verdict — STRATEGY TSI Sentence 4: "the relevance / implication is direct" surface rewrite

## Result

**0/3 PASS** (all 3 critics picked TARGET, all LEAN). HARD REGRESSION below R80=R94=R95 = 2/3 LEAN ceiling. Same pattern as R104/R105: §5 ¶4 + ¶5 dominate the FAIL signal regardless of which other axis is rewritten. TSI Sentence 4 surface rewrite was NOT visible in any critic tells.

## Seeds + positions

- seed=771721572, pos=3: critic picked Essay 3 (TARGET) "lean toward" → FAIL. Tell #1 high-leverage: §5 "What I have not shown:" colon-list (Sentence 7, untouched in R106). Tell #2: calendar-shaped tree central conceit repetition.
- seed=561459047, pos=5: critic picked Essay 5 (TARGET) "lean toward" → FAIL. Tell #1: §5 "What I have not shown:" self-disclaiming closer (Sentence 7).
- seed=433375627, pos=6: critic picked Essay 6 (TARGET) "lean toward" → FAIL. Tell #1: §5 ¶4 cog-sci framing chain density "I read the results as consistent... with the qualification that..." (Sentence 6).

## Strategy applied

Two TSI handcrafted edits on R95 base (commit 65c74b1, freshly reverted after R105):

**§1 ¶5 sentence:**
Before: "The relevance for an archival interface is direct: the cognitive system already runs multi-resolution hierarchical structure for tasks of an analogous form."
After: "An archival interface that asks researchers to hold multi-resolution hierarchical structure mentally is asking them to do work the cognitive system already does for analogous spatial and conceptual problems."

**§2 ¶4 opener:**
Before: "The relevance to an archival interface is direct."
After: "An archival interface inherits this distinction."

Net change: +10w. Word count: 7,108 → 7,118.

## Insight — TSI ladder exhausted, §5 floor confirmed

Across R100-R106 single-sentence TSI on each of the 7 multi-flagged sentences:
- R100 S1 (abstract opener "A century"): 2/3, NOT +1, TSI flagged ("scripted/parallel-list")
- R101 S2 (abstract "with date as a facet"): 2/3, NOT +1, TSI flagged ("narrative scene-as-puzzle-frame")
- R102 S7 (§5 "What I have not shown:"): 1/3, TSI flagged ("pre-emptive limitations bracket")
- R103 S3 (abstract triplet): 1/3, TSI not flagged but escalated severity (LEAN→HIGH 95%) on §1 ¶6 balanced antithesis
- R104 S6 (§5 hedge-stack): 0/3, TSI flagged ("hypothesis-confirmation arc"); paradox confirmed
- R105 S5 (§2 "less reasonable"): 0/3, TSI not flagged; §5 ¶4+¶5 dominated
- R106 S4 (the relevance is direct): 0/3, TSI not flagged; §5 ¶4+¶5 dominated

Pattern across all 7 TSI rounds:
1. The §5 paragraph 4 (Sentence 6) AND §5 paragraph 5 (Sentence 7) are flagged in EVERY round regardless of which sentence is targeted.
2. R102 + R104 directly attempted to fix Sentence 6 + Sentence 7 — both got 0-1/3 with new TSI sentence directly flagged.
3. The "any-rewrite-introduces-same-class-tell" trap (R98+R100+R101+R102+R104) means single-sentence TSI on §5 is structurally trapped.
4. Non-§5 TSI either ties at 2/3 (R100, R101) or regresses (R103, R105, R106).

Zero +1 validated across S1-S7. The dispatch's R107 COMBINE move is moot (no validated +1s to combine).

## Pareto

R106 = 0/3 << R80=R94=R95 = 2/3 LEAN BEST_GAN. v10 reverted to R95 baseline (commit 65c74b1).

## R107 plan — pivot to STRATEGY D1

Per dispatch ladder: "R108+ = if combined still 2/3 LEAN ceiling, pivot to STRATEGY D1 (cohort-anchored mega-rewrite via Opus 4.7 with strongest cohort exemplars as POSITIVE style anchor). Last untried meta-strategy."

Since R107 combine is moot (no +1 saved), R107 IS R108: STRATEGY D1.

D1 = full §5 rewrite via Opus 4.7 with strongest cohort exemplar (2018SKYS9 Ghent Altarpiece — directly cited as positive cohort first-person template by R100 PASS critic 333131964) supplied as POSITIVE STYLE ANCHOR (not as anti-tell prompt). Different from R98 B1 which used anti-tell prompt; this provides the model with a SPECIFIC HUMAN VOICE to mimic.

Risk: same R98 lesson (REPLACEMENT introduces same-class tells). Mitigation: D1 supplies a 1500w cohort exemplar block to anchor on, vs B1 which used negative-feature constraints.

If D1 < 2/3: pivot to STRATEGY E1 (aggressive structural §3+§4 collapse — last untried).
