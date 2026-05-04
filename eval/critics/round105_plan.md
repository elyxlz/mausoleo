# Round 105 plan — STRATEGY TSI Sentence 5: §2 ¶2 "less reasonable... less reasonable, too" R94 A1 triplet rewrite

## R105 target: Sentence 5

Current text (§2 ¶2 closer, two sentences):
> It is less reasonable for the historian who wants to understand a corpus they cannot read in full at the article level. It is less reasonable, too, for one whose answer is not a list of articles, where the answer is a shape that moves across days, or an absence that might matter more than what was printed.

Critic flag class: parallel "less reasonable... less reasonable, too" triplet (with the prior "is reasonable for most historical research" forming an implicit triplet). R94 A1 introduced the split that exposed this parallel structure. Multi-round flag (R95 NEAR_CERTAIN, R98 LEAN, R99 LEAN).

R96 corrective attempted: "The historian whose question is about a corpus they cannot read in full at the article level finds little to work with here, and the same goes for the historian whose answer should be a shape that moves across days, or an absence that might matter more than what was printed." — got 1/3 (regression).

R96 form has "and the same goes for" parallel construction. Same flag class.

## R96 + R104 lessons applied

R96: removing parallels can strip register-roughening (1/3).
R104: clean prediction-confirmation arcs read as AI; hedged versions read as performative humility (paradoxical trap).

R105 candidate must:
- NO "less reasonable" or "and the same goes for" parallel construction
- NO clean prediction-confirmation arc (avoid R104 trap)
- NOT introduce new parallel-list structure
- Shorter than original (minimize new attack surface)

## Candidate

**E' (two short sentences, no parallel structure):**

Replace:
> It is less reasonable for the historian who wants to understand a corpus they cannot read in full at the article level. It is less reasonable, too, for one whose answer is not a list of articles, where the answer is a shape that moves across days, or an absence that might matter more than what was printed.

With:
> For a historian who cannot read the corpus in full at the article level, the access template is structurally insufficient. The answer they need is a shape across the corpus, not a ranked list of articles.

Rationale:
- DROPS "less reasonable / less reasonable, too" parallel
- DROPS "and the same goes for" R96 corrective parallel
- TWO short sentences, both subject-first declarative
- First sentence: structural insufficiency claim (not parallel-symmetric)
- Second sentence: "answer they need" specifies what's missing (positive form, not "less than X" hedge)
- AVOIDS em-dash (banned per SKILL)
- AVOIDS hedged hypothesis-confirmation arc (R104 trap)
- Net change: -16w (75w → 59w)

## Edit plan

Single edit on R95 base (commit 65c74b1, freshly reverted after R104). Replace closing two sentences of §2 ¶2 as above.

## Test protocol

3 random positions per essay-iter SKILL: positions spread 2-6.

## Pareto rule

- If R105 ≥ 3/3: ship.
- If R105 = 2/3 LEAN with FAIL critic NOT keying on the new sentences or §2: save BEST_TSI_S5; branch R106 from R105.
- If R105 = 2/3 LEAN with FAIL critic flagging new sentences: NOT +1, revert to R95, try Sentence 4 (R106).
- If R105 < 2/3: hard regression, revert to R95, try Sentence 4 (R106).
