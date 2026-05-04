# Round 107 plan — STRATEGY D1: cohort-anchored §5 mega-rewrite via Opus 4.7

## Strategy

After R100-R106 TSI ladder exhausted with zero +1 validated and §5 ¶4+¶5 dominating EVERY round's flag, R107 pivots to STRATEGY D1.

D1 = cohort-anchored mega-rewrite of §5 (Discussion). Different from R98 B1 (anti-tell prompt + 11 explicit bans):
- D1 supplies the strongest cohort exemplar (2018SKYS9 Ghent Altarpiece Discussion+Conclusion ~2039w) as POSITIVE STYLE ANCHOR
- D1 instructs Opus 4.7 to MIRROR Ghent's Discussion voice while preserving Mausoleo's empirical claims
- No anti-tell prompt this time (B1 anti-tell prompt produced same-class tells regardless)

The Ghent Discussion exemplar features:
- Section sub-headers ("Understanding of era", "Previous knowledge", "Website", "Damaged vs visibly old", "Aesthetically pleasing vs visibly old", "Well maintained", "Awareness of restoration", "Hawthorn effects")
- Quantitative claims interleaved with informal observations ("Most participants considered the paintings in good condition with 17 respondents giving >4")
- Asymmetric depth (some sub-sections one sentence, others a paragraph)
- Acknowledgment of methodological limits in narrative form ("There is a possibility that respondents attempted to conform...") not as colon-list
- First-person presence ("under the supervision of the author")
- Sub-section dedicated to method-self-critique ("Hawthorn effects") not as a global limitations block

Mausoleo's §5 should mirror this:
- Replace the paragraph-grouped structure with thematic sub-headers (e.g., "Why the missing-day case", "Why the regime-change case", "Why the comparative-coverage case", "Cog-sci framing", "Summariser bias", "What the experiment cannot say")
- Distribute the limitation acknowledgments across sub-headers rather than concentrating in §5 ¶5
- Keep first-person present but in narrative-observation form not in performative-humility form

## Prompt design

System prompt: "You are a doctoral supervisor rewriting Chapter 5 (Discussion) of a final-year UCL undergraduate dissertation. The student's chapter is being read alongside seven cohort essays from the same module. The strongest cohort exemplar in the same register is supplied as STYLE ANCHOR — match its rhythm, sub-section structure, asymmetric depth, and narrative-observation register. Do not import its content. Preserve every empirical claim, citation and number from the student's draft."

User prompt: "STYLE ANCHOR (Ghent Altarpiece dissertation Discussion+Conclusion):\n\n[Ghent §5+conclusion text]\n\n--- END STYLE ANCHOR ---\n\nSTUDENT DRAFT TO REWRITE (Mausoleo §5):\n\n[Mausoleo §5 text]\n\n--- END STUDENT DRAFT ---\n\nProduce the rewritten Chapter 5 only. Match the cohort exemplar's structural cadence and register. Add thematic sub-headers. Distribute the methodological self-critique across sub-sections (do NOT concentrate at the end). Keep word count within ±15% of the student's original."

Temperature: 0.7 (lower than B1's 0.95 — encourage matching the exemplar more closely, not generating new prose).

## Test protocol

3 random positions per essay-iter SKILL: positions spread 2-6.

## Pareto rule

- If R107 ≥ 3/3: ship.
- If R107 = 2/3 LEAN with FAIL critic NOT keying on §5: save BEST_D1; advance to R108 STRATEGY E1.
- If R107 = 2/3 LEAN with FAIL critic flagging new §5: ties R95 ceiling, no improvement. Revert. Advance to R108 STRATEGY E1.
- If R107 < 2/3: hard regression, revert to R95, advance to R108 STRATEGY E1.

## Word-count target

Mausoleo current §5: ~860 words across 5 paragraphs.
D1 target: 730-990 words (±15%).

## Last untried meta-strategies

- D1: cohort-anchored mega-rewrite (THIS ROUND)
- E1: aggressive structural §3+§4 collapse (R108 if D1 fails)
- After E1: STRATEGY EXHAUSTED per dispatch hard-stop rule (b).
