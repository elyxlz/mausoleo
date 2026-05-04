# Round 108 plan — STRATEGY E1: handcrafted §3+§4 explicit-three framing reduction

## Strategy

E1 per dispatch = "aggressive structural §3+§4 collapse — method + concrete worked-example". Full collapse risks coherence breakage and the R98/R107 cross-model rewrite trap.

Modified E1 = HANDCRAFTED reduction of explicit "three" framings in §3 + §4 chapter openers and titles. Targets the structural pattern flagged by R102 critic 109020470 ("three loosely coupled stages, three case studies, Three questions are put to *Il Messaggero*, Three strands of cognitive science") and R107 D1 critics ("Scope: X / Y / Z parallel triplet"). The intrinsic three-ness of the design is preserved; only the EXPLICIT verbal markers are reduced.

Per R98 + R107 lesson: avoid Opus-mediated rewrites at scale. Stay handcrafted.

## Edits

**Edit 1: §3 chapter opener**
Before: "Three loosely coupled stages of Mausoleo connect through a single ClickHouse table called `nodes`."
After: "Mausoleo's components connect through a single ClickHouse table called `nodes`."

**Edit 2: §4 chapter title**
Before: "## Chapter 4: The missing 26 July, and two contrast cases"
After: "## Chapter 4: The missing 26 July, with contrast cases"

**Edit 3: §4 chapter opener**
Before: "The comparison in all three cases is to a BM25 baseline over the same hand-cleaned article transcriptions in the `documents` table, with no access to the `nodes` hierarchy."
After: "Each case study compares Mausoleo against a BM25 baseline over the same hand-cleaned article transcriptions in the `documents` table, with no access to the `nodes` hierarchy."

## Expected effect

Removes 3 explicit "three" / "Three" / "three case studies" markers in chapter openers without touching the underlying design. The cumulative "three X / three Y / three Z" pattern that critics consistently flag is partially defused.

## Test protocol

3 random positions per essay-iter SKILL: positions spread 2-6.

## Pareto rule

- If R108 ≥ 3/3: ship.
- If R108 = 2/3 LEAN with FAIL critic NOT keying on §3+§4: save BEST_E1; advance to R109 with new TSI sentences from R103/R107 critic surfacing ("There is an episodic-memory analogue:" balanced antithesis, "There is a body of cognitive-science work, accumulating since" template).
- If R108 = 2/3 LEAN with FAIL critic still flagging "three" pattern: NOT +1, revert.
- If R108 < 2/3: hard regression, revert to R95.

## After E1

Per dispatch hard-stop rule (b): "all 7 sentences + R107 combine + D1 + E1 ALL exhausted with no improvement". After R108, the dispatch ladder is exhausted. Per dispatch rule (7) NEVER declare "irreducible local optimum". Continue with NEW TSI candidates (R109+).
