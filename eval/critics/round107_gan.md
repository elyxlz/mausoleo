# Round 107 verdict — STRATEGY D1: Opus 4.7 cohort-anchored §5 mega-rewrite (Ghent exemplar style anchor)

## Result

**0/3 PASS** (all 3 critics picked TARGET, all LEAN). HARD REGRESSION below R80=R94=R95 = 2/3 LEAN ceiling. Per dispatch rule: revert to R95, advance to R108 STRATEGY E1.

## Seeds + positions

- seed=570674257, pos=4: critic picked Essay 4 (TARGET) "lean toward" → FAIL. Tell #1 [high-leverage] STRUCTURAL: "Section sub-header proliferation in Chapter 5 — ten thematic sub-headers including three 'Scope:' sub-headers at the end. The 'Scope: X / Scope: Y / Scope: Z' parallel label sequence is especially diagnostic."
- seed=991825938, pos=5: critic picked Essay 5 (TARGET) "lean toward" → FAIL. Tell #1 [high-leverage] STRUCTURAL: "Heavy section sub-header decomposition in Chapter 5 — 10 sub-headers, each typically containing a single short paragraph... fragmented sub-section structure with parallel labelling — especially the 'Scope: X / Scope: Y / Scope: Z' triplet at the end — is a signature LLM organization pattern."
- seed=639175819, pos=6: critic picked Essay 6 (TARGET) "lean toward" → FAIL. Same Tell #1 flag pattern.

## Strategy applied

D1 = full §5 rewrite via Opus 4.7 with Ghent Altarpiece Discussion+Conclusion (~2039w) as POSITIVE STYLE ANCHOR.

Prompt design (per round107_plan.md):
- "Match cohort exemplar's structural cadence: thematic sub-headers, asymmetric depth, narrative-observation register"
- "Distribute methodological self-critique across sub-sections rather than concentrating in single closing block"
- "Do NOT import cohort exemplar's content"
- "Preserve every empirical claim, citation, number"
- Word count target: 730-990w
- 11 explicit format constraints (no em-dashes, no banned vocab, no banned openers)

Output: 737w, 10 sub-headers ("Regime-change case", "Comparative-coverage case", "Missing-day case", "Working-memory framing", "Hippocampal-mapping framing", "Summariser elaboration", "Named-entity check", "Scope: the temporal window", "Scope: question types", "Scope: the agent"). Pre-test sanity passed: zero em-dashes, zero banned vocab, zero banned openers.

Word count: 7,108 → 7,367 (+259w).

## Insight — D1 introduced NEW STRUCTURAL HIGH-LEVERAGE tell

R98 + R107 lessons confirmed at TWO different cross-model rewrite axes:
- B1 (R98): Opus 4.7 anti-tell prompt → introduced new triplet "What... How... How..." + new antithesis "Either X or Y"
- D1 (R107): Opus 4.7 positive-cohort-anchor prompt → introduced new "Scope: X / Scope: Y / Scope: Z" parallel triplet + 10-sub-header decomposition that critics read as "fragmented sub-section structure"

The Ghent exemplar features 8 sub-headers organically organised (not parallel-labelled). Opus 4.7 mirrored the FREQUENCY of sub-headers but introduced NEW parallel-label structure ("Scope: X / Y / Z" — exactly the kind of triplet the original "What I have not shown:" colon-list was). The model converted one structural AI-tell into another.

This is the strongest evidence yet that NO Opus-mediated rewrite (anti-tell prompt OR positive cohort anchor) can defuse §5 without introducing new same-class structural tells. The model's LLM-cadence prior produces parallel-list / colon-list / triplet structures regardless of operator instructions.

Multi-flagged cumulative tell signals after R107:
- Calendar-shaped tree central conceit repetition (intrinsic content)
- Cog-sci citation chain (intrinsic content)
- Three-case design (intrinsic structure)
- §5 ¶4 hedge-stack (Sentence 6) — survived R104 attack
- §5 ¶5 colon-list (Sentence 7) — survived R102 attack
- §5 ¶? sub-header proliferation (NEW from D1)

## Pareto

R107 = 0/3 << R80=R94=R95 = 2/3 LEAN BEST_GAN. v10 reverted to R95 baseline (commit 65c74b1).

## R108 plan — STRATEGY E1

Per dispatch ladder: "R109 = STRATEGY E1 (aggressive structural §3+§4 collapse) only if D1 also fails."

E1 = collapse §3 (How Mausoleo is built) and §4 (The missing 26 July, and two contrast cases) into a single "Method, with a worked example" chapter. Reduces the "three-stage method × three-case study" parallel structure that critics consistently flag as AI.

Risk: word count + coherence reviewer impact. E1 is the LAST untried meta-strategy per dispatch.

If E1 < 2/3: per dispatch hard-stop rule (b) "all 7 sentences + R107 combine + D1 + E1 ALL exhausted with no improvement", strategy ladder is exhausted. Per dispatch rule (7) NEVER declare "irreducible local optimum". So if E1 fails, return to TSI ladder with NEW sentence candidates from R107 critic flags (e.g., the §1 ¶6 balanced antithesis "There is an episodic-memory analogue:..." that R103 critic surfaced as Tell #2 — never targeted by S1-S7).
