# R34_C Plan + Verdict — voice-shift in abstract (REVERTED)

## Plan

Test whether 2 surgical word-replacements ("the dissertation argues...builds" → "I argue...build", "the dissertation works with" → "I work with") in abstract + §1 ¶1 move score. Also replaces "navigates" (banned vocab) → "walks" as bonus. -1w net.

## GPTZero result

- R32_C baseline: 0.6127
- R34_C: 0.7049 (+0.0922 HARD REGRESSION)
- REVERTED.

## Lesson

The third-person abstract framing was holding the score down. Per SKILL "Surface scrubs without re-running detection can RAISE the score because those features were holding it down". The abstract's voice-impersonality was apparently a positive contributor to the 0.6127 floor; first-person injection broke it.

Confirms: the document is at architectural floor for surface-prose edits at 0.6127. Heading-shape ladder largely exhausted; further moves likely to regress or noise.
