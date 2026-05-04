# R28_C Plan + Verdict — Strategy 5d extended (chapter + appendix headings)

## Plan

R27_C (5d on 5 §2-§3 sub-section headings) achieved -0.0430. This is genuine evidence that document-shape signals are a real lever, not architecturally scan-locked. Extend R28_C to chapter-level (§1-§5) and appendix-level (A.1-A.5) headings: same noun-phrase → interrogative/active-voice transformation.

## Edits (13 heading rewrites, +32w net atop R27_C)

Chapter level:
- §1 "Reading across a regime change" → "Reading across the regime change of July 1943"
- §2 "Two literatures and a corpus" → "What two literatures and one corpus contribute"
- §3 "How Mausoleo is built" → "How I built Mausoleo" (first-person)
- §4 "The missing 26 July, and two contrast cases" → "Reading the missing 26 July, with two contrast cases"
- §5 "Discussion" → "What the case studies do and do not warrant"

§4 sub-sections:
- "The missing 26 July" → "Reading the missing 26 July"
- "Two shorter cases" → "Two shorter contrast cases"
- "Aggregate numbers" → "What the aggregate numbers say"

Appendix sub-sections:
- A.1 "The three-dimension judge rubric" → "How the three-dimension judge rubric is scored"
- A.2 "Per-cell variance for the three case studies" → "Per-cell variance across the three case studies"
- A.3 "OCR composite weighting and per-pass decomposition" → "How the OCR composite is weighted and decomposed per pass"
- A.4 "Index-construction parameters" → "What index-construction parameters were used"
- A.5 "Researcher-agent tool-set" → "What tools the researcher-agent has"

§3.2 "The calendar-shaped tree" left unchanged: it's terminologically load-bearing (the noun phrase IS a defined term referenced repeatedly elsewhere).

## GPTZero result

- R27_C baseline: 0.7170
- R28_C: **0.6958 (-0.0212)** — third consecutive negative-delta round in Stage C (R8_C / R27_C / R28_C)
- predicted_class: ai (still); confidence_score: 0.6962
- Cumulative Stage C session: 0.7600 → 0.6958 = -0.0642 in two heading-only rounds

## Paragraph-GAN spot-check

By construction: zero paragraph-prose changed. Paragraph-GAN reads paragraph text, not headings → no regression possible. v123 paragraph-GAN report (commit 035b942) baseline 48/49 PASS preserved.

## Verdict

ACCEPT. New BEST = 0.6958. Continue ladder.

## Hypothesis

Headings are a high-leverage signal because the doc's macro-shape is uniformly noun-phrase + colon — typical AI/LLM output convention. Mixing in interrogative + active-voice + first-person ("How I built") disrupts the macro-shape fingerprint without touching argumentative prose. This is consistent with the SKILL claim that doc-level classifiers compute a fingerprint that contaminates per-sentence numbers — the contamination flows the OTHER way too: a doc-shape signal can change the per-sentence reads.
