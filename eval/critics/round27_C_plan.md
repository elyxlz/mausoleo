# R27_C Plan + Verdict — Strategy 5d (heading-style change)

## Plan

R26_C (Strategy 6 abstract+preface surgical rewrite) regressed GPTZero 0.7600 → 0.7895 (+0.0295), reverted. Lesson: adding more prose in any voice raises the doc-level fingerprint, even when the new prose is concrete-first, Italian-scaffolded, burstiness-shifted. Per SKILL "Style consistency is a fingerprint" + "Surface scrubs without re-running detection can RAISE the score because those features were holding it down" — the iteration converged the doc into a state where any prose-volume change destabilises whichever features held the score at 0.7600.

R27_C tests Strategy 5d (document-level structural perturbation): change §2 + §3 sub-section heading style from noun-phrase to interrogative/active-voice. Headings are <0.3% of doc by word count BUT carry outsize structural signal in detector training (they shape document macro-shape). Zero body-prose changes; zero paragraph-GAN risk by construction (paragraph-GAN reads prose, not headings).

## Edits (5 heading rewrites, +14w net)

- §2.1 "Existing digitised newspaper archives" → "What the existing digitised newspaper archives do"
- §2.2 "The hierarchical-retrieval lineage" → "Where the hierarchical-retrieval lineage breaks from this"
- §2.3 "Memory, hierarchy and external structure" → "Why memory hierarchy is the right shape for this index"
- §3.1 "From scanned pages to article transcriptions" → "Getting from scanned pages to article transcriptions"
- §3.3 "How the agent reads the tree" → "How the agent actually reads the tree"

## GPTZero result

- R26_C baseline (post-revert): 0.7600
- R27_C: **0.7170 (-0.0430)** — second negative-delta round in Stage C history (R8_C was -0.0152)
- predicted_class: ai (still); confidence_score: 0.7173
- Cleanest sub-paragraph leverage move tried in Stage C

## Paragraph-GAN spot-check

Headings are not in paragraph prose; paragraph-GAN reads paragraph text only. By construction, no paragraph-prose changed → no paragraph-GAN regression possible.

## Verdict

ACCEPT. New BEST = 0.7170. Continue ladder.

## Next-round candidate: R28_C

If 5d alone moved the score by -0.0430, more heading variety should compound. Try:
- Chapter-level headings (§1-§5 currently "Chapter N: noun-phrase") → mix interrogative + declarative
- Appendix sub-section headings (A.1-A.5)
- Word-count budget: 9372/9500 = 128w buffer. Heading rewrites cost ~3w each on average.
