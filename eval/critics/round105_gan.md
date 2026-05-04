# Round 105 verdict — STRATEGY TSI Sentence 5: §2 ¶2 "less reasonable" triplet handcrafted rewrite

## Result

**0/3 PASS** (all 3 critics picked TARGET, all LEAN). HARD REGRESSION below R80=R94=R95 = 2/3 LEAN ceiling. The new Sentence 5 rewrite was NOT visible in any critic's tells — failures keyed exclusively on §5 paragraphs 4 + 5 (Sentence 6 + Sentence 7) and intrinsic content (calendar-tree repetition, cog-sci chain). Per dispatch rule: revert, try Sentence 4 (R106).

## Seeds + positions

- seed=242959354, pos=5: critic picked Essay 5 (TARGET) "lean toward" → FAIL. Tell #1 [high-leverage] flagged §5 "What I have not shown:" colon-list (Sentence 7, R95 baseline restored). Tell #2: calendar-shaped tree central conceit repetition (intrinsic).
- seed=807007102, pos=2: critic picked Essay 2 (TARGET) "lean toward" → FAIL. Tell #1: §5 "What I have not shown:" self-disclaiming closer (Sentence 7).
- seed=575870145, pos=6: critic picked Essay 6 (TARGET) "lean toward" → FAIL. Tell #1: §5 ¶4 cog-sci framing density "I read the results as consistent with the cognitive-science framing chapter two laid out, with the qualification that consistency at this scale is weak evidence: working-memory limits..." (Sentence 6).

## Strategy applied

Single TSI handcrafted edit on R95 base (commit 65c74b1, freshly reverted after R104). Replaced §2 ¶2 closing two sentences:

Before:
> It is less reasonable for the historian who wants to understand a corpus they cannot read in full at the article level. It is less reasonable, too, for one whose answer is not a list of articles, where the answer is a shape that moves across days, or an absence that might matter more than what was printed.

With (Candidate E', two short non-parallel sentences):
> For a historian who cannot read the corpus in full at the article level, the access template is structurally insufficient. The answer they need is a shape across the corpus, not a ranked list of articles.

Net change: -16w. Word count: 7,108 → 7,092.

## Insight — §5 dominance confirmed across all TSI attempts

R100 (Sentence 1 abstract opener): 2/3 — TSI flagged
R101 (Sentence 2 abstract feature-cadence): 2/3 — TSI flagged
R102 (Sentence 7 §5 "What I have not shown:"): 1/3 — TSI flagged
R103 (Sentence 3 abstract triplet): 1/3 — Sentence NOT flagged but other critic patterns dominated
R104 (Sentence 6 §5 hedge-stack): 0/3 — TSI flagged + paradox confirmed
R105 (Sentence 5 §2 triplet): 0/3 — TSI NOT flagged; §5 ¶4 + ¶5 still dominate

Pattern: §5 paragraphs 4 and 5 (Sentence 6 + Sentence 7) are flagged in EVERY round, regardless of which other sentence is targeted. The single-axis TSI lifts for Sentences 6 and 7 directly fail (R102, R104). The non-§5 TSI lifts (R100/R101/R103/R105) tie at 2/3 or regress.

This converges with the dispatch's expectation: "After all 7 individually validated... R107 = COMBINE all 7 BEST_TSI_X edits in one round on R95 base (per Pareto rule, only combine validated +1s)." But none of the 7 has hit +1. So R107 combine is moot.

Per dispatch ladder: "If R107 combined still 2/3 LEAN ceiling, pivot to STRATEGY D1 (cohort-anchored mega-rewrite via Opus 4.7 with strongest cohort exemplars as POSITIVE style anchor)."

The dispatch path: continue Sentences 4 (R106), then move to D1 (R107 since combine is moot) given no +1 saved across S1-S7. Sentence 4 is "the relevance / implication is direct" repetition (§1 ¶5 + §2 ¶4) — surface tic, simpler axis.

## Pareto

R105 = 0/3 << R80=R94=R95 = 2/3 LEAN BEST_GAN. v10 reverted to R95 baseline (commit 65c74b1).

## R106 plan

R106 = TSI Sentence 4: "the relevance / implication is direct" repetition (§1 ¶5 + §2 ¶4). This is a SURFACE rewrite (rephrase the bridging formula) rather than structural. Lower risk of triggering same-class tells than Sentences 1-3, 5-7.
