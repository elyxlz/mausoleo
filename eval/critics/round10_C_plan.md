# Stage C Round 10_C plan — Single-sentence rewrite of R103-flagged "There is an episodic-memory analogue:" template

## Strategy

R8_C-validated operative move scaled DOWN: single-sentence surgical edit to remove the R103 critic Tell #2 high-leverage flag class — "Balanced antithesis with colon-preamble".

Target sentence (line 123, end of §4 case-1 paragraph):

> "There is an episodic-memory analogue: an event whose temporal slot the memory system does not hold cannot register as missing, it can only fail to come back when looked for."

This sentence has TWO flagged structural patterns in the GAN history:
1. "There is an X analogue:" template (parallel to R104 "There is a body of [field] work" template that R8_C removed).
2. Balanced antithesis "cannot register as missing, it can only fail to come back when looked for" with colon-preamble.

R0_C critic 124273414 flagged the "is direct" template at line 33; the same template class shows up here. R103 critic explicitly cited this sentence as Tell #2 high-leverage.

## Edit

Replace the sentence with a non-templated single-sentence statement preserving the same semantic content:

OLD: "There is an episodic-memory analogue: an event whose temporal slot the memory system does not hold cannot register as missing, it can only fail to come back when looked for."

NEW: "Episodic memory works the same way at the cognitive level: events whose temporal slot the memory system does not hold cannot be flagged as missing, only as failures of recall."

Diff:
- Removes "There is an X analogue:" existential template
- Removes colon-preamble that introduces the antithesis
- Removes "cannot...it can only" balanced antithesis
- Preserves all factual content (episodic memory parallel, slot-not-held → not-flagged-as-missing relationship)
- Word count: 32w → 26w (-6w)

## Pareto rule

- GPTZero drops AND GAN ≥ 3/3: ACCEPT, advance to R11_C.
- GPTZero drops AND GAN ≥ 2/3 LEAN: ACCEPT, advance.
- GPTZero stays AT R8_C (0.7600): ACCEPT (single-sentence edit, addresses GAN-flagged template proactively, no harm).
- GPTZero rises significantly: REVERT.

## GAN risk

LOW — the original R103-flagged sentence is a STRUCTURAL TELL the critic surfaced. Removing it is GAN-positive.

## GPTZero risk

UNKNOWN — single-sentence edit in deep-§4 zone (paragraph 0.978-0.99 range per R0_C breakdown). May not move doc-level score either way.

## Quota

GPTZero 7/7 (final scan in 24h cycle).
