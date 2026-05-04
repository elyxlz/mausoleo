# Round 104 verdict — STRATEGY TSI Sentence 6: §5 "with the qualification that" full meta-frame removal

## Result

**0/3 PASS** (all 3 critics picked TARGET, all LEAN). HARD REGRESSION below R80=R94=R95 = 2/3 LEAN ceiling. Per dispatch rule: revert to R95, try Sentence 5 (R105).

## Seeds + positions

- seed=667655877, pos=3: critic picked Essay 3 (TARGET) "lean toward" → FAIL. Tells: "There is a body of cognitive-science work, accumulating since Bartlett's *Remembering*" template; tripartite OCR/summariser/CLI semicolon stack; "What I have not shown:" self-disclaiming closer; hyper-precise 11.3/28.3 numeric stacking; "I cannot put to the cases as run" self-disclaiming construction (still present in §5 ¶4).
- seed=728383915, pos=4: critic picked Essay 4 (TARGET) "lean toward" → FAIL. Tells: "What I have not shown:" pre-emptive limitations colon block; calendar-shaped tree central conceit repetition; "There is a body of cognitive-science work, accumulating since" template.
- seed=987561056, pos=6: critic picked Essay 6 (TARGET) "lean toward" → FAIL. Tell #1 [high-leverage]: my NEW TSI Sentence 6 rewrite "Working-memory limits (Miller, YEAR; Cowan, YEAR) predict a cost gap of the right sign, and the data show one" — flagged as "structurally signature AI: a clause + clause construction that mirrors a hypothesis-confirmation move ('X predicts Y, and Y is observed') without the connective tissue a human writer would add. Real dissertations either don't frame predictions this cleanly or surround them with hedges ('the data are at least consistent with this,' 'the cost gap could equally reflect...')."

## Strategy applied

Single TSI handcrafted edit on R95 base (commit 65c74b1). Replaced §5 ¶4 opener:

Before:
> I read the results as consistent with the cognitive-science framing chapter two laid out, with the qualification that consistency at this scale is weak evidence: working-memory limits (Miller, 1956; Cowan, 2001) predict a cost gap of the right sign and the data show one.

With (Candidate C, drop meta-frame entirely, lead with empirical claim):
> Working-memory limits (Miller, 1956; Cowan, 2001) predict a cost gap of the right sign, and the data show one.

Net change on this sentence: 67w → 19w, -48w. Word count: 7,108 → 7,060.

## Insight — paradoxical hedge vs anti-hedge trap

R104 critic explicitly described BOTH directions as AI-flagged:
1. WITH hedges ("with the qualification that consistency at this scale is weak evidence") = "performative humility / pre-emptive scope acknowledgment" (R102 lesson)
2. WITHOUT hedges ("Working-memory limits predict a cost gap... and the data show one") = "structurally signature AI: clause + clause construction that mirrors a hypothesis-confirmation move 'X predicts Y, and Y is observed'" (R104 lesson)

The cog-sci-framing-vs-data discussion is going to be flagged as AI either way at the surface level. The trap: any clean prediction-confirmation arc reads as AI; any meta-hedged version of the same arc reads as performative humility AI.

Critic suggested specific human-form hedges: "the data are at least consistent with this," "the cost gap could equally reflect..." These are MORE-hedged forms with epistemic-uncertainty markers. R96 lesson predicted: removing meta-hedging ("of course", "worth flagging") strips register-roughening features. R104 confirms the inverse: removing the meta-hedge "with the qualification that" exposed the underlying clean prediction-confirmation arc as AI-signature.

Combined R96+R104 lesson: §5 ¶4 cog-sci-framing-vs-data discussion is a STRUCTURAL trap — neither hedged nor unhedged version passes; the only way out is to DELETE the prediction-confirmation arc entirely and only retain the broader "I do not have the power to discriminate that explanation from a generic compression-helps-retrieval one" hedge. But that loses the substantive cog-sci framing the abstract relies on.

The all-3-FAIL result is the most decisive evidence yet that single-sentence TSI within §5 cannot lift the ceiling. The §5 paragraph 4 + 5 hedging shape is a multi-sentence structural feature, not a single-sentence one.

## Pareto

R104 = 0/3 << R80=R94=R95 = 2/3 LEAN BEST_GAN. v10 reverted to R95 baseline (commit 65c74b1).

## R105 plan

R105 = TSI Sentence 5: §2 ¶2 "less reasonable... less reasonable, too" R94 A1-introduced triplet rewrite. R96 attempted corrective ("The historian whose question is about... finds little to work with here, and the same goes for the historian whose answer should be...") and got 1/3. R105 will attempt a fresh Candidate E': two short sentences with NO parallel "less reasonable" structure and NO "and the same goes for" construction.

Push at R104 trigger met (5 rounds since R99): push after R104 commit.
