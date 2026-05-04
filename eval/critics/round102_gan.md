# Round 102 verdict — STRATEGY TSI Sentence 7: §5 "What I have not shown:" handcrafted single-sentence rewrite

## Result

**1/3 PASS** (1 NEAR_CERTAIN PASS + 2 LEAN FAIL on TARGET). HARD REGRESSION below R80=R94=R95 = 2/3 LEAN ceiling. Per TSI rule "If 0/3 or 1/3: revert to last best, try next sentence". Revert to R95; try Sentence 3 (R103).

## Seeds + positions

- seed=109020470, pos=4: critic picked Essay 4 (TARGET) "lean toward" → FAIL — directly flagged TSI Sentence 7 as Tell #1 "Pre-emptive limitations bracket [high-leverage]"
- seed=156112917, pos=5: critic picked Essay 5 (TARGET) "lean toward" → FAIL — verdict text is malformed/looped (multiple repeated "1) Strongest Tells" lines suggesting Opus retry/loop) but identified TARGET as Essay 5 with confidence "lean toward"; the diagnostic content keys on "Bundled calendar-tree exposition" intrinsic-to-topic
- seed=253203852, pos=6: critic picked Essay 8 (= 2022SMMH0 Artificial Creativity at slot 8, NOT TARGET) "near certain" → **PASS NEAR_CERTAIN**

## Strategy applied

Single TSI handcrafted edit on R95 base (commit 65c74b1, freshly reverted after R101). Replaced last paragraph of §5:

Before:
> What I have not shown: whether the cost gaps generalise outside July 1943, whether they hold for question types I have not yet tested (a single-event close-reading without temporal-aggregation structure, or a long-arc comparative across years), and whether a researcher who is not an LLM agent shows the same pattern. None of these are within the dissertation's scope and none of them can be inferred from the cases as run.

With (Candidate D, single concrete-particular limitation, no list, no meta-frame):
> I cannot say from the cases as run whether the cost gaps generalise to corpora outside July 1943; that is the project's main empirical limit.

Net change: -50 words. Word count: 7,108 → 7,058.

## What the critics flagged

**FAIL critic 109020470 (lean, picked TARGET at pos 4)**:
- Tell #1 [high-leverage]: my TSI Sentence 7 rewrite "I cannot say from the cases as run whether the cost gaps generalise to corpora outside July YEAR; that is the project's main empirical limit" — flagged as "Pre-emptive limitations bracket. This single-sentence 'What I haven't shown' closer is the AI tell of the chapter. Real dissertations either don't broadcast their limits or develop them across paragraphs with specific reasoning about why each constraint matters; this terse one-line self-disclosure reads as performative humility designed to inoculate against critique."
- Tell #2 [high-leverage]: "Triplet-of-three architecture" — flagged "three loosely coupled stages, three case studies, Three questions are put to *Il Messaggero*, Three strands of cognitive science" cumulative pattern (intrinsic to design).
- Tell #3: "calendar-shape rhetorical bow... compound nominalization with 'over which a researcher agent navigates' is genuine AI cadence" (R80 abstract closer, untouched).
- Tell #4: "Templated comparison block" with three precise stats — intrinsic factual content.
- Tells #5-#8: not visible in extracted output but pattern continues per critic style.

**FAIL critic 156112917 (lean, picked TARGET at pos 5)**:
- Verdict is malformed (Opus apparent retry-loop emitted multiple "1) Strongest Tells" headers) but identified TARGET. Tell #1 visible: "STRUCTURAL — Bundled 'calendar-tree' exposition [high-leverage]" — keys on the calendar-tree topic intrinsic to the dissertation's central system.

**PASS critic 253203852 (near-certain, picked Artificial Creativity at pos 8, Mausoleo at slot 6 NOT picked)**:
- 8 tells on Artificial Creativity: enumerated four-criteria framework, "by no means perfect" pre-emptive hedging, IMRAD limitations + future work coda, balanced antithesis "where there is art there is also creativity, and vice versa", "It must first and foremost be noted that" register tic, "still in its infancy" cliché, symmetric disciplinary survey, "such as this one" self-congratulatory close.
- TARGET (Mausoleo at slot 6) NOT picked.

## Insight — TSI Sentence 7 single-sentence concession STILL flagged as pre-emptive

R98+R100+R101+R102 lessons confirmed at HANDCRAFTED scale: REPLACEMENT operations produce same-class tells. The specific case: any sentence that ends a chapter with self-disclosed scope-limitation gets read as "performative humility / pre-emptive hedging" by adversarial critics, regardless of length, framing, or specificity. Even a one-line concession was flagged with the same diagnostic class as the original colon-list-of-3.

The critic explicitly proposed alternatives: "Real dissertations either don't broadcast their limits or develop them across paragraphs with specific reasoning about why each constraint matters." This suggests two divergent fixes:
1. DELETION: don't broadcast limits at all — let the chapter end on the summariser/prolepsis paragraph.
2. EXPANSION: develop the limitations across multiple paragraphs with specific case-by-case reasoning (anti-TSI: would require adding 200+ words of specifically-reasoned limitation-discussion).

Per dispatch rule: when current sentence rewrite fails, move to next sentence. Both DELETION and EXPANSION moves can be tried as fallbacks if the rest of the TSI ladder fails.

The pos5 verdict's malformed/looped emission is concerning and may be another Opus-on-740KB-prompt failure mode (similar to R96/R97 hallucinations at pos 8). Worth noting but doesn't change the result — the verdict still picked TARGET.

## Pareto

R102 = 1/3 < R80=R94=R95 = 2/3 LEAN BEST_GAN. v10 reverted to R95 baseline (commit 65c74b1).

## R103 plan

R103 = TSI Sentence 3: handcrafted rewrite of abstract closer parallel triplet "Three questions are put to *Il Messaggero* in July 1943: what the paper said on the absent 26 July, how it covered the regime change of 25 to 27 July, and how the balance of war and domestic-politics coverage moved across the month."

R98+R100+R101+R102 lessons applied to candidate generation:
- Avoid parallel-triplet structure (the explicit flag on this sentence is already triplet)
- Avoid first-person within-project scene-setting
- Avoid pre-emptive meta-framing
- Single concrete claim > polished compound sentence
- Shorter than original to minimize new attack surface

Three Sentence 3 candidates pre-drafted in R102 plan; revisit with R102 lessons applied.
