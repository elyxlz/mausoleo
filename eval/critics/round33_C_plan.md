# R33_C Plan + Verdict — Strategy 5d (paragraph-density variation; REVERTED)

## Plan

Test whether mid-paragraph splits (paragraph-density variation, no word change) move GPTZero. 2 splits in §1 ¶1 and §1 ¶3 at natural sentence boundaries. Zero word delta.

## GPTZero result

- R32_C baseline: 0.6127
- R33_C: 0.6156 (+0.0029 mild regression, within noise)
- REVERTED.

## Lesson

Paragraph-density variation does not extend the heading-shape leverage. Doc-shape signal is concentrated at the heading/title level, not at paragraph-break boundaries. Probable mechanism: GPTZero's classifier weights heading-positional features but not within-section paragraph-density features at the same magnitude.
