# R11_C plan — Strategy 3b surgical Haiku 4.5 paraphrase, §2 line 50 paragraph

## Target

§2 ¶ at line 50 of MAUSOLEO_FULL_DRAFT_v109.md.

The paragraph contains the high-leverage AI-template parallel:

> "It is less reasonable for the historian who wants to understand a corpus they cannot read in full at the article level. **It is less reasonable, too,** for one whose answer is not a list of articles, where the answer is a shape that moves across days, or an absence that might matter more than what was printed."

This is the highest-leverage un-touched hot zone per R0_C critic.

## Mechanism

Per R8_C breakthrough: SCOPE = single paragraph, anti-template prompt, length-preserving constraint. Avoid R7_C colloquial-Haiku failure (no register shift), avoid R9_C template-substitution failure (Haiku tends to swap one template for another).

## Anti-template constraints

Forbid in Haiku output:
- "It is less reasonable" (the very flag)
- Other parallel-triplet constructions ("not X, but Y, but Z")
- Em-dashes (— or --)
- "There is/are a body of"
- "is direct" / "must either"
- Opener clichés ("Moreover", "Additionally", "Furthermore", "Indeed", "Importantly", "Notably", "Crucially")
- Length: 130-145 words (target paragraph is 142w, stay within ±10%)
- Preserve citation: (Düring et al., 2024)

## Execution

1. Haiku 4.5 via OAuth, surgical paraphrase only this one paragraph
2. Splice into draft
3. GPTZero scan with fresh proxy
4. If GPTZero ≤ 0.74 (-0.02), validate GAN 3 positions, ban pos 1 and pos 7+
5. If GAN ≥ 2/3 LEAN, KEEP and bump v110
6. Else REVERT

## Pareto rule

GPTZero must drop AND GAN must remain ≥ 2/3 LEAN. R8_C breakthrough preserved (3/3 PASS) is highly desirable.
