# R12_C plan — surgical single-sentence anti-template (line 72)

## Target

Line 72 of MAUSOLEO_FULL_DRAFT_v110.md (currently the §1 bridge sentence between cog-sci framing and Chapter 4 case studies):

> "The case studies in chapter four ask whether the prediction implied by these three strands of cognitive science shows up in the metrics."

R11_C R110 GAN pos2 critic flagged this as Tell #2 [SURFACE high-leverage]:
> "This framing-ahead move appears repeatedly. Human dissertations occasionally signpost; this one does so in nearly every paragraph transition, producing an overly tidy reading experience."

## Mechanism (per R10_C precedent)

Manual single-sentence rewrite. R10_C zero-movement edit was KEPT because it removed a known AI-template flag class. This R12_C move is structurally identical: remove a known signposting cliché, replace with something flatter.

## Constraints
- Preserve the bridge function (linking cog-sci §1 chapter to Chapter 4 case studies).
- Length-neutral: ±5w.
- No "is direct" / "shows up in" / "the prediction implied by" / "three strands" templates.
- Concrete > abstract noun-of-noun.

## Candidate replacement

Original (28w): "The case studies in chapter four ask whether the prediction implied by these three strands of cognitive science shows up in the metrics."

Replacement candidate (18w): "Chapter four tests the framing on three corpus questions and reports the tool-call counts and rubric scores."

This is more concrete (names the test instrument: tool-call counts, rubric scores), shorter, and breaks the "implied by these three strands" / "shows up in the metrics" templates. It also drops the "asks whether" hedge that pos2 critic flagged as "metronomic predictability".

## Pareto plan

1. Apply edit
2. GPTZero scan (fresh proxy)
3. If GPTZero unchanged or down, GAN check (3 positions, all in 2-6)
4. If GAN ≥ 2/3 LEAN, KEEP. Bump v110 → v111.
5. Else REVERT.
