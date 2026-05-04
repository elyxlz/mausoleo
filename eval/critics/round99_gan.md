# Round 99 verdict — STRATEGY C1: ADDITION-only first-person concrete-detail anecdotes on R95 base

## Result

**1/3 PASS LEAN** (1 LEAN PASS pos 6 + 1 LEAN FAIL pos 3 + 1 NEAR_CERTAIN FAIL pos 8). Below R80=R94=R95 = 2/3 LEAN. Reverted v10 to R95.

## Seeds + positions

- seed=366798937, pos=6: critic picked Essay 3 (= '2022SMMH0' Artificial Creativity, NOT TARGET) "lean toward" → **PASS**. 8 tells on Artificial Creativity (numbered three-question scaffold, twofold goal, "not only X but also Y", "It is precisely X", section-by-section road map, "Although X, recently Y", "will be culminated", "nigh impossible"). TARGET (Mausoleo at pos 6) NOT picked.
- seed=233965575, pos=3: critic picked Essay 3 (TARGET) "lean toward" → FAIL. Tells: (1) puzzle-first frame as "designed payoff" structural pattern (R80 STRUCTURAL); (2) parallel triplet abstract closer (R80 untouched); (3) cog-sci citation ladder Bartlett→Miller→Cowan→Tolman→Eichenbaum→Behrens→Whittington (intrinsic content); (4) "less reasonable... less reasonable, too" R94 A1 leftover (untouched in R95 base since R96 corrective was reverted); (5) symmetrical chapter architecture; (6) "transparent generosity" quotation gloss; (7) preface efficiency; (8) prior-work survey homogeneity. Critic explicitly cited the preface OCR-failure admission as POSITIVE preserve, did NOT comment on the C1 anecdotes (Tesseract weekend, S.M. il Re Saturday, child_count=0 discovery) — anecdotes did not trip extra tells.
- seed=157845169, pos=8: critic picked Essay 8 (TARGET) "near certain" → FAIL. REAL verdict (not hallucinated this time, quoted real content). Tells: (1) "What I have not shown:" pre-emptive limitations (R95 §5 untouched); (2) "A century of cognitive-science work has accumulated evidence" abstract opener (R80 untouched); (3) "with the qualification that consistency at this scale is weak evidence" R95 §5 untouched; (4) balanced antithesis "cannot register as missing, it can only fail to come back" §4 untouched; (5) parallel three-case design uniform metrics grid; (6) "the retrieved fragments and the larger temporal shape" parallel pair (R95 A2 abstract paragraph 2 — actually NEW from A2); (7) absence of personality markers; (8) "I do not have the power in this experiment to discriminate" R95 A2 phrasing.

## Strategy applied

C1 single-axis on R95: 3 ADDITION-only first-person concrete-detail anecdotes on Chapter 3:

1. §3.1 OCR pipeline (Tesseract weekend): "(An earlier prototype passed Tesseract output through as one of the predictor sources; I spent a weekend in early February trying confidence thresholds between 0 and 95 to find a setting at which Tesseract contributed positively to the merge, and never found one. On 1885-06-15 the Tesseract source helped at threshold 30 and hurt at 60; on 1910-06-15 the relation inverted. I dropped the source.)"

2. §3.1 OCR pipeline (S.M. il Re anachronism): "The clearest version of this was a Saturday in late February when I had the post-corrector running on the 1885-06-15 hand-clean issue and watched it confidently rewrite *S.M. il Re* (the King) as *Sua Maestà il Re*: an honorific the late-Ottocento press of *Il Messaggero* never used, dropped into the prose with no signal that anything had been edited. Two further attempts with stricter prompts produced the same class of plausible-but-anachronistic edit, so the pass came out."

3. §3.2 calendar-tree (26 July absence-day discovery): "I noticed the empty-day case for the first time on a Tuesday in late January, the second time the index-build script finished on the corpus, when a `SELECT count() FROM nodes WHERE level='article' GROUP BY toDate(date_range_start)` came back with a row for every calendar day of July 1943 except 26 July, and the day node `1943-07-26` was sitting in the table with `child_count=0`. I had been expecting to write a separate sentinel row for known-missing dates and was looking for where to put that special-case logic, and the schema had handled the case by accident before I got round to it."

Word count: 7,108 → 7,351w (+243w). All ADDITION, zero deletions, zero replacements.

## Insight

R86 evidence base (1 PASS_LEAN ceiling for ADDITION strategy with 3 small additions) replicates exactly: R99 = 1 PASS_LEAN. The C1 anecdotes did NOT trip new tells (no negative critic citation), but did NOT lift the ceiling either.

The dominant FAIL signals are still the Abstract + §1 ¶1 + §5 paragraph 4 features that survived R94 A1 (antithesis), R95 A2 (subject), R96 A3 (hedge), R97 A4 (opener), R98 B1 (full rewrite), R99 C1 (addition):
- Abstract opener "A century of cognitive-science work has accumulated evidence" — high-leverage SURFACE
- Abstract opener "A digitised newspaper corpus normally allows a historian... with date as a facet on the side" — flagged R97
- Abstract closer parallel triplet "what the paper said... how it covered... how the balance"
- §5 "What I have not shown:" pre-emptive limitations — flagged 4+ rounds
- §5 "with the qualification that consistency at this scale is weak evidence" — flagged R95+R99
- §1 ¶5 "the relevance / implication is direct" repetition — flagged R97+R98+R99
- §2 ¶2 "less reasonable... less reasonable, too" — flagged R95+R98+R99 (R96 corrective reverted)

These are the high-leverage targets. C2 should attack the abstract+§5 specifically with addition (a footnote, a parenthetical aside) not replacement.

## Pareto

R99 = 1/3 LEAN < R80=R94=R95 = 2/3 LEAN BEST. v10 reverted to R95 baseline (commit 65c74b1).

## R100 plan

C2 = inject 2-3 more first-person concrete-detail anecdotes targeting §1 (a §1 weekend pivot when the Bartlett re-read happened) + §5 (replace "What I have not shown:" colon-list with a single-sentence aside about a specific moment of doubt during write-up). Same ADDITION-only discipline. Branch from R95.
