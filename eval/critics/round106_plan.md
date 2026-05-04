# Round 106 plan — STRATEGY TSI Sentence 4: "the relevance / implication is direct" repetition surface rewrite

## R106 target: Sentence 4 (two instances, repetition is the flag)

Current text:
- §1 ¶5 (Tolman/Eichenbaum/Whittington paragraph): "The relevance for an archival interface is direct: the cognitive system already runs multi-resolution hierarchical structure for tasks of an analogous form."
- §2 ¶4 (Memory section, ¶1): "The relevance to an archival interface is direct. A corpus carries material from individual articles..."

Critic flag class: "the relevance / the implication is direct" formula — "transitional padding LLMs insert to link theoretical claims to practical consequences" (R98 critic 90456486 + R103 critic 232746473 explicitly cited as Tell #4 high-leverage). The REPETITION across §1 and §2 amplifies the flag.

This is a SURFACE rewrite axis (replace the bridging formula in both places with different non-formulaic phrasings) rather than structural. R96 + R104 lessons less applicable since no parallel-structure or hedge-stack involved.

## R96 + R104 lessons applied

R96: removing surface tics can strip register-roughening. Mitigation here: REPLACE with different bridges in each location, don't just delete.
R104: paradox doesn't apply here (no hypothesis-confirmation arc).

## Candidates

**§1 ¶5 candidate:**
Replace "The relevance for an archival interface is direct: the cognitive system already runs multi-resolution hierarchical structure for tasks of an analogous form."
With: "An archival interface that asks researchers to hold multi-resolution hierarchical structure mentally is asking them to do work the cognitive system already does for analogous spatial and conceptual problems."
- DROPS "the relevance is direct" formula
- Restructures as a single observation about what the interface asks
- Avoids new triplet/parallel
- Net change: 22w → 32w (+10w)

**§2 ¶4 candidate:**
Replace "The relevance to an archival interface is direct. A corpus carries material from individual articles up through narrative arcs and aggregate patterns at the month or longer scale, and a researcher reading the corpus moves between those levels."
With: "A corpus carries material from individual articles up through narrative arcs and aggregate patterns at the month or longer scale, and a researcher reading the corpus moves between those levels — much as the cognitive systems just described do for analogous problems."

Wait — that introduces an em-dash (banned). Let me revise:

With: "An archival interface inherits this distinction. A corpus carries material from individual articles up through narrative arcs and aggregate patterns at the month or longer scale, and a researcher reading the corpus moves between those levels."
- DROPS "The relevance to an archival interface is direct"
- New opener "An archival interface inherits this distinction" is direct, not formulaic
- Net change: 6w → 6w (same length)

## Edit plan

Two edits on R95 base (commit 65c74b1, freshly reverted after R105):
1. Replace §1 ¶5 sentence as above
2. Replace §2 ¶4 opener as above

## Test protocol

3 random positions per essay-iter SKILL: positions spread 2-6.

## Pareto rule

- If R106 ≥ 3/3: ship.
- If R106 = 2/3 LEAN with FAIL critic NOT keying on either new sentence: save BEST_TSI_S4; advance to R107 STRATEGY D1 (since no +1 saved across S1-S7).
- If R106 = 2/3 LEAN with FAIL critic flagging new sentences: NOT +1, revert to R95, advance to R107 STRATEGY D1.
- If R106 < 2/3: hard regression, revert to R95, advance to R107 STRATEGY D1.
