# Round 108 verdict — STRATEGY E1 (handcrafted variant): §3+§4 explicit-three framing reduction

## Result

**0/3 PASS** (all 3 critics picked TARGET, all LEAN). HARD REGRESSION below R80=R94=R95 = 2/3 LEAN ceiling. The 3 explicit-"three" markers I removed did not visibly affect critic flags — failures still keyed on §5 ¶4 + ¶5 (Sentence 6 + 7) and intrinsic content (calendar-tree, cog-sci chain).

## Seeds + positions

- seed=152031438, pos=4: critic picked Essay 4 (TARGET) "lean toward" → FAIL. Tell #1: §5 "What I have not shown:" colon-list (Sentence 7).
- seed=802228154, pos=2: critic picked Essay 2 (TARGET) "lean toward" → FAIL. Same pattern.
- seed=979943256, pos=3: critic picked Essay 3 (TARGET) "lean toward" → FAIL. Same pattern.

## Strategy applied

E1 (handcrafted variant) = remove 3 explicit "three" framings:
1. §3 opener: "Three loosely coupled stages of Mausoleo connect" → "Mausoleo's components connect"
2. §4 chapter title: "## Chapter 4: The missing 26 July, and two contrast cases" → "## Chapter 4: The missing 26 July, with contrast cases"
3. §4 opener: "The comparison in all three cases is to a BM25 baseline" → "Each case study compares Mausoleo against a BM25 baseline"

Per R98 + R107 lesson: avoided full Opus-mediated structural collapse (would have introduced new same-class tells).

Word count: 7,108 → 7,108 (net 0).

## Insight — explicit-three reduction insufficient

Removing the 3 explicit verbal "three" markers in §3 + §4 chapter openers did not defuse the cumulative "three X / Y / Z" pattern critics flag. The pattern is INTRINSIC to the design:
- 3 OCR sub-pipelines (technical content)
- 3 case studies (the empirical core)
- 3 strands of cognitive-science evidence in §2 (the framing)
- 3 questions in abstract (Sentence 3, untouched in R108)

To genuinely defuse the cumulative "three" pattern would require dropping one case study or one cog-sci strand. Both destructive.

R108 also confirmed: §5 ¶4+¶5 dominate FAIL signals regardless of which other axis is touched. The strategy ladder's underlying assumption — that single-axis interventions can lift the 2/3 LEAN ceiling — does not hold for this cohort-fit configuration.

## Pareto

R108 = 0/3 << R80=R94=R95 = 2/3 LEAN BEST_GAN. v10 reverted to R95 baseline (commit 65c74b1).

## Dispatch ladder exhausted

After R108, the dispatch's full strategy ladder (TSI S1-S7 + R107 D1 + E1) is exhausted with ZERO +1 validated. Per dispatch hard-stop rule (b): "all 7 sentences + R107 combine + D1 + E1 ALL exhausted with no improvement" — TRIGGERED. Per dispatch rule (7): NEVER declare "irreducible local optimum. Just keep going."

## R109 plan — NEW TSI candidates surfaced by critic flags

Critic-surfaced HIGH-LEVERAGE flags NOT in original 7-sentence list:
1. **§1 ¶6 "There is an episodic-memory analogue: an event whose temporal slot the memory system does not hold cannot register as missing, it can only fail to come back when looked for."** — flagged by R103 critic 232746473 as Tell #2 high-leverage "Balanced antithesis with colon-preamble".
2. **§1 ¶4 "There is a body of cognitive-science work, accumulating since Bartlett's *Remembering* (Bartlett, 1932), that bears on the kind of reading the case studies in chapter four put to the corpus."** — flagged by R104 critic 667655877 as Tell #1 high-leverage "There is a body of [field] work, accumulating since [seminal author]'s [seminal text]" template.

R109 = TSI Sentence 8 (§1 ¶4 "There is a body of" template). This is a SURFACE pattern — replace with non-template opener for §1 ¶4.

Per R98+R100+R101+R102+R104+R107 lessons: REPLACEMENT introduces same-class tells. Mitigation: keep the rewrite SHORT (single concrete observation).
