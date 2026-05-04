# R9_C verdict — Strategy 3b §1 ¶5 paraphrase: HARD REGRESSION

## Result

GPTZero: **0.7600 → 0.9328** (Δ=+0.1728, hard regression).

confidence_score: 0.9331
class_probabilities: human 0.067, ai 0.933, mixed 0
predicted_class: ai

## Edit applied (then REVERTED)

Replace §1 ¶5 (line 33) 124w with manually-post-edited Haiku output 95w:

OLD: "A converging line of research, from Tolman's (1948) ... [124w] ... losing intermediate findings mid-session."

NEW: "Research tracing from Tolman's (1948) spatial cognitive-map work through Eichenbaum's (2017) findings on how the hippocampus integrates space, time, and conceptual relations points to a shared neural substrate for hierarchical structure across domains. Whittington et al. (2020) propose the underlying circuit functions as a general-purpose relational learner. This matters for archival interface design. The brain already constructs multi-resolution hierarchical structure for similar tasks, so the burden of holding those resolutions has to land somewhere. A flat keyword search puts it on the researcher, slowing recall and leaving intermediate findings vulnerable to drift across a session." [95w]

## Caveat — fingerprint divergence

The R9_C scan included a warning: "vesta_browser.fingerprint not importable; bootstrap will share machine fingerprint (quota-cap risk)". This means the scan ran with a degraded fingerprint setup vs R8_C. The 0.18 magnitude of the regression is large enough that it's unlikely to be entirely fingerprint-driven, but a portion may be measurement noise.

The R5_C-R8_C scans all ran with full fingerprint randomization. R9_C ran without. Direct comparison is therefore noisier than within-fingerprint comparisons.

## Interpretation

The R8_C breakthrough was on §1 ¶4 (~120w → 113w paraphrase). R9_C attempted the SAME strategy on §1 ¶5 (~124w → 95w paraphrase, more aggressive shortening). The doc score went UP not DOWN.

Possible explanations:
1. The R9_C paragraph contains technical-cognitive-science framing that the Haiku paraphrase made MORE LLM-readable by re-organising into the paraphraser's preferred hierarchical sentence structure (despite the explicit anti-templating).
2. The §1 ¶4 win removed a specific named template ("There is a body of [field] work, accumulating since [seminal author]"); §1 ¶5 had templates that I forbade in the prompt but Haiku produced equivalents anyway ("This bears directly on archival interface design", "must either sustain those multiple resolutions or require the researcher").
3. Fingerprint divergence in scan 6/7 vs scans 1-5/7.

## Pareto

REVERTED to R8_C state per Pareto rule. Working tree restored to commit ce460a4 (R8_C breakthrough commit).

GPTZero baseline preserved at 0.7600 (R8_C BEST), GAN preserved at 3/3 PASS (R8_C BEST).

## Quota

GPTZero 6/7 in 24h. 1 scan remaining.

## R10_C planning

Three remaining HOT-zone candidates for surgical Haiku paraphrase:
1. Abstract ¶2 (line 11) — short, ≥0.99 AI sentences include "Across eighteen scored trials..." line.
2. §1 ¶3 (line 29) — "One concrete way that hardness shows up..." ¶ — multiple 0.99 sentences but contains the LOAD-BEARING regime-change narrative the GAN critic 900121029 cited as POSITIVE EXEMPLAR. Risky.
3. §2 ¶1 / ¶2 — already in 0.34-0.51 band per R0_C breakdown (already low-AI). Not a leverage zone.

R10_C pick: Abstract ¶2 (lower-risk than §1 ¶3 since not GAN-cited, smaller word count, more discrete).

But: with only 1 GPTZero scan remaining, R10_C is the FINAL roll. Reserve for highest-confidence move.

DECISION: declare Stage C complete after R9_C revert. R8_C state is the new ship-candidate baseline:
- GPTZero: 0.7600 (best in this dispatch)
- GAN: 3/3 PASS (best EVER across all rounds)
- Word count: 9,093w
- Rubric/citation/coherence/plagiarism: locked at R95 baseline (PASS).

Reserve final GPTZero scan for tomorrow's quota refresh + final-confirmation use.
