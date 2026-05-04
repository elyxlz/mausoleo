# Round 103 verdict — STRATEGY TSI Sentence 3: abstract closer parallel triplet handcrafted rewrite

## Result

**1/3 PASS** (1 PASS NEAR_CERTAIN + 1 LEAN FAIL + 1 NEAR_CERTAIN FAIL on TARGET). HARD REGRESSION below R80=R94=R95 = 2/3 LEAN ceiling. The new Sentence 3 rewrite was NOT directly flagged by any visible critic — failures keyed on UNTOUCHED higher-leverage sentences (§5 "What I have not shown:" colon-list and §5 "with the qualification that" hedge-stack), confirming Sentence 3 is not the highest-leverage current axis. Revert to R95; move to Sentence 6 next (R104).

## Seeds + positions

- seed=232746473, pos=3: critic picked Essay 3 (TARGET) "lean toward" → FAIL. Tell #1 flagged §5 "What I have not shown:" colon-list (R95 baseline restored after R102, untouched here). Tells #2-#5: calendar-tree central conceit (intrinsic), Bartlett→Craik&Lockhart→contemporary chain (intrinsic), "the relevance is direct" formula (Sentence 4 in TSI list), clean asymmetric three-case design (intrinsic).
- seed=715968007, pos=4: critic picked Essay 3 (= 2020GDKT1 Causal Algorithmic Fairness at slot 3, NOT TARGET) "near certain" → **PASS NEAR_CERTAIN**. 8 tells on Causal Algorithmic Fairness: recursive thesis-preview-delivery, "delicate balance / nuanced understanding" register tics, perfectly symmetric case treatment, parallel triplet stacking, em-dash parentheticals, "X increasingly shapes Y" abstract definitional opener, "It is worth noting / Importantly" filler hedges, no authorial stance asymmetry.
- seed=863295738, pos=6: critic picked Essay 6 (TARGET) "near certain" → FAIL. Tell #1 [high-leverage]: "Heavy use of meta-commentary and self-reference" — flagged §5 "I read the results as consistent with the cognitive-science framing chapter two laid out, with the qualification that consistency at this scale is weak evidence" (Sentence 6 in TSI list, untouched). Tell #2: templated literature engagement (intrinsic).

## Strategy applied

Single TSI handcrafted edit on R95 base (commit 65c74b1, freshly reverted after R102). Replaced abstract paragraph 2 sentence:

Before:
> Three questions are put to *Il Messaggero* in July 1943: what the paper said on the absent 26 July, how it covered the regime change of 25 to 27 July, and how the balance of war and domestic-politics coverage moved across the month.

With (Candidate D, single hierarchical sentence, no parallel triplet):
> The three case studies in chapter four turn on the absent 26 July, with the surrounding regime-change days and the wider July war-and-politics balance as the larger questions the system is asked to handle around it.

Net change: -13 words. Word count: 7,108 → 7,095.

## Insight — Sentence 3 not the leverage point

Critically, the TSI Sentence 3 rewrite was NOT visible in any FAIL critic's tell list. Both FAIL critics keyed on different sentences:
- Pos 3 critic: §5 "What I have not shown:" colon-list (Sentence 7) + intrinsic content tells
- Pos 6 critic: §5 "with the qualification that" hedge-stack (Sentence 6) + intrinsic content tells

This suggests the abstract closer triplet is NOT the dominant signal at this point — the §5 paragraph 4 (Sentence 6) and §5 paragraph 5 (Sentence 7) are. The cumulative intrinsic-content tells (calendar-tree repetition, cog-sci chain, three-case design symmetry) form the topic-fit floor that surface scrubs cannot defuse.

The PASS critic 715968007 picking Causal Algorithmic Fairness LEAN at pos 4 with TARGET not picked indicates that on some critic samples, TARGET reads cleanly enough relative to a particularly LLM-sounding cohort essay (Causal Algorithmic Fairness has the most flagged register/structural tells).

## Pareto

R103 = 1/3 < R80=R94=R95 = 2/3 LEAN BEST_GAN. v10 reverted to R95 baseline (commit 65c74b1).

## R104 plan

R104 = TSI Sentence 6: §5 "with the qualification that consistency at this scale is weak evidence" hedge-stack rewrite. R96 already attempted partial rewrite (split into "I read the results as consistent... laid out. Consistency at this scale is weak evidence:") and got 1/3. R104 will attempt a different rewrite: drop "with the qualification that" subclause AND drop colon-elaboration, keep cog-sci framing claim with a tighter shape.

## File-system observation

Mid-R103 file-system instability observed: verdict.md files repeatedly went from real size (6041, 4798, 6499 bytes) back to 0 bytes between Monitor invocations. Real verdict text was extractable in the first Monitor call after claude exited; subsequent stat calls returned 0. Hypothesis: tmpfs / overlay caching layer in the Bash sandbox that doesn't reflect immediate writes; Monitor sees fresh state on first arm but subsequent reads see snapshot. Workaround applied: extract verdict text via Monitor + stream the head -80 in single command. Did not impact correctness of result interpretation.
