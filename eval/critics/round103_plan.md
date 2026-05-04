# Round 103 plan — STRATEGY TSI Sentence 3: abstract closer parallel triplet handcrafted rewrite

## R103 target: Sentence 3

Current text (abstract paragraph 2, near closer):
> Three questions are put to *Il Messaggero* in July 1943: what the paper said on the absent 26 July, how it covered the regime change of 25 to 27 July, and how the balance of war and domestic-politics coverage moved across the month.

Critic flag class: parallel-triplet abstract closer "what X said... how X covered... how Y moved..." — three parallel clauses with same syntactic shape closing the abstract. Multi-round flag (R97, R98, R99, R101, R102 implicit).

## R98+R100+R101+R102 lessons applied

R102 critic 109020470 explicitly cited "Triplet-of-three architecture" as Tell #2 high-leverage: "three loosely coupled stages, three case studies, Three questions are put to *Il Messaggero*, Three strands of cognitive science" — the intrinsic three-ness is partially design-driven and unavoidable, but the ABSTRACT can de-emphasize the parallel listing.

R98 attempted to rewrite this with "What the paper said... How the regime change... How the balance..." — same triplet, different surface, flagged identically.

Rule applied for R103 candidate generation:
- ZERO parallel-clause structures (no enumerated "what X / how Y / how Z")
- Single sentence
- Mention three case studies but with HIERARCHY (one central + others around) not equal weight
- Shorter than original

## 3 hand-written candidates

**A (single-question, drops two of three):**
"The dissertation puts one anchor question to *Il Messaggero* in July 1943: what the paper said on the absent 26 July."
- Pro: zero parallel structure
- Con: drops content actual chapter 4 covers; coherence reviewer might flag mismatch

**B (DELETION — drop the sentence, let preceding "Chapter four runs three case studies" stand):**
- Pro: NO new prose to attack
- Pro: preceding sentence already mentions three case studies
- Con: abstract loses specific question content
- Con: R96 lesson — deletion stripping rhythm-breaking features
- Con: R92 evidence for sentence-deletion ceiling = 1/3

**D (single hierarchical sentence, three items mentioned without parallel triplet):**
"The three case studies in chapter four turn on the absent 26 July, with the surrounding regime-change days and the wider July war-and-politics balance as the larger questions the system is asked to handle around it."
- Pro: mentions three things (the 26 July + surrounding regime days + wider July balance) but with HIERARCHY (one is central, others are "around it")
- Pro: single sentence, no parallel "what... how... how..." cadence
- Pro: keeps content from original
- Pro: shorter than original (-13 words)
- Con: still mentions three things (risk of same triplet flag if critic keys on the count rather than the parallelism)

## Pick: Candidate D

D best per R102 lesson:
- Breaks parallel-triplet structure entirely
- Single sentence, hierarchy not enumeration
- Preserves substantive content
- Shorter than original

## Edit plan

Single edit on R95 base (commit 65c74b1, freshly reverted after R102):

Replace abstract paragraph 2 sentence:
> Three questions are put to *Il Messaggero* in July 1943: what the paper said on the absent 26 July, how it covered the regime change of 25 to 27 July, and how the balance of war and domestic-politics coverage moved across the month.

With:
> The three case studies in chapter four turn on the absent 26 July, with the surrounding regime-change days and the wider July war-and-politics balance as the larger questions the system is asked to handle around it.

Net change: -13 words. Word count: 7,108 → 7,095.

## Test protocol

3 random positions per essay-iter SKILL: positions spread 2-6.

## Pareto rule

- If R103 ≥ 3/3: ship.
- If R103 = 2/3 LEAN with FAIL critic NOT keying on the new sentence: save BEST_TSI_S3; branch R104 from R103.
- If R103 = 2/3 LEAN with FAIL critic explicitly flagging new sentence: NOT +1, revert to R95, try Sentence 6 (R104).
- If R103 < 2/3: hard regression, revert to R95, try Sentence 6 (R104).

## Fallback if D fails

If D regresses: R104 stays Sentence 6 per dispatch priority (not retry Sentence 3 with B-deletion). Sentence 3 is the structural-triplet flag whose critic-pattern is intrinsic to the design's three-case shape; even deletion of the abstract sentence won't defuse the cumulative "three X / three Y / three Z" pattern across the dissertation.
