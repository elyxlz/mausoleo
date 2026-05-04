# R40_C Plan + Verdict — chapter-title interrogative push (REVERTED)

## Plan

Try ambitious chapter-title interrogative recasting: §1 "Reading across the regime change of July 1943" → "How does an archive read across the day a regime fell?" (new interrogative form), §3 "How I built Mausoleo" → "Building Mausoleo, stage by stage" (drop first-person, gerund + adverbial), §4 "Reading the missing 26 July, with two contrast cases" → "Three case studies: the missing 26 July plus two contrasts" (declarative + colon).

## GPTZero result

- R38_C baseline: 0.5232
- R40_C: 0.5791 (+0.0559 hard regression)
- REVERTED.

## Lesson

Aggressive chapter-title rewrite regressed hard. Probable mechanism: §3 lost first-person voice anchor ("How I built" → "Building..." — voice-impersonalisation), and §1 interrogative paired with existing §5 interrogative may have created duplicate-shape signal that the classifier read as templated.

Confirms: heading-shape diversity is non-monotonic. There is an OPTIMAL diversity level at the current state (R38_C = 0.5232), and further perturbation in this direction regresses.

## Final state

R38_C = 0.5232 is the BEST and likely the architectural floor for this iteration of doc-shape changes. Further moves require risking the floor.
