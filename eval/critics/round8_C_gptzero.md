# R8_C verdict — Strategy 3b (HOT-zone §1 ¶4 heavy paraphrase via Haiku 4.5): +1 IMPROVEMENT

## Result

GPTZero: **0.7752 → 0.7600** (Δ=-0.0152, first non-zero NEGATIVE delta in Stage C).
GAN: **2/3 LEAN → 3/3 PASS** (BREAKS R95=R94=R80 ceiling for the first time across ~110 rounds).

## Strategy

Strategy 3b applied to a SURGICAL paragraph: §1 ¶4 (the "There is a body of cognitive-science work, accumulating since Bartlett's *Remembering*..." paragraph, GPTZero per-sentence flag 0.973+).

Haiku 4.5 prompt with explicit constraints:
- preserve all citations
- preserve all factual claims (Miller 7±2, Cowan 4 chunks, "compression along chain")
- aim 100-130 words (matched: 113w vs original 119w)
- vary sentence lengths (8-12w short + 30+w compound + medium)
- NO em-dashes
- NO banned openers
- NO parallel triplets
- NO "is direct" template (this was the R0_C critic flag at line 33)
- NO "There is a body of [field] work" template (this was the R104 critic flag)

## Edit

Replace lines 31 (§1 ¶4):

OLD: "There is a body of cognitive-science work, accumulating since Bartlett's *Remembering* (Bartlett, 1932), that bears on the kind of reading the case studies in chapter four put to the corpus. Bartlett's *War of the Ghosts* studies described how recall reconstructs a generic schema in place of the verbatim event. Subsequent working-memory research established a small functional capacity for the active workspace: roughly seven plus or minus two items in Miller's original framing (Miller, 1956), revised downwards to about four chunks in Cowan (2001). For an archive question that ranges across a month, this constraint requires compression somewhere along the chain, either in the interface or, failing that, in the reader's head, and the gap widens as the temporal scope expands."

NEW: "Since Bartlett's *Remembering* (Bartlett, 1932), cognitive science has documented how recall reconstructs generic schemas rather than verbatim events; his *War of the Ghosts* studies made this pattern visible. Working-memory research subsequently mapped a narrow functional capacity for active processing: Miller proposed roughly seven plus or minus two items (Miller, 1956), though Cowan's later work suggested about four chunks as the realistic ceiling (Cowan, 2001). When an archive question spans a month or longer, this constraint forces compression somewhere in the chain, whether at the interface level or within the reader's cognitive space, and the pressure intensifies as temporal scope widens. This matters directly for how the corpus-based case studies in chapter four operate."

Word count: 9099 → 9093 (−6w).

## GAN verdicts (round 8_C, master_seed=508001)

| seed | pos | TARGET slot | critic picked | result |
|---|---|---|---|---|
| 216457536 | 5 | Essay 5 | Essay 7 (Spanish Transition Films, LEAN) | **PASS** |
| 371987843 | 4 | Essay 4 | Essay 6 (Artificial Creativity, NEAR_CERTAIN) | **PASS** |
| 900121029 | 3 | Essay 3 | Essay 4 (Artificial Creativity, LEAN) | **PASS** |

**3/3 PASS** — first +1 above R95=R94=R80=R0_C ceiling in ~110 rounds.

Notably, pos3 critic cited TARGET (Essay 3 = Mausoleo) as POSITIVE STRUCTURAL EXEMPLAR: "Essay 3 (Mausoleo) serves as the exemplar: its preface admits 'Two weeks went into a post-correction pass that made composite OCR scores worse' — a specific negative outcome that AI prose avoids."

The §1 ¶4 paraphrase REMOVED two GPTZero-flagged template patterns:
1. "There is a body of [field] work, accumulating since [seminal author]'s [seminal text]" template (R104 critic Tell #1)
2. "the relevance for X is direct" template (R0_C pos5 critic flagged "is direct" at line 33; this paragraph is not the same line but shares the cog-sci framing chain)

## Pareto

ACCEPT. Both axes improved. Save as new BEST_BOTH baseline.

- BEST_GPTZero: 0.7600 (R8_C)
- BEST_GAN: 3/3 PASS (R8_C)

R5_C edits also kept (em-dash → comma in prose × 3, vocab swaps × 3).

## Quota

GPTZero 5/7 in 24h. 2 scans remaining.

## Next round plan

R9_C = continue Strategy 3b: heavy paraphrase NEXT highest-AI HOT-zone paragraph.

Candidates (per R0_C top-AI sentences):
1. §1 ¶5 (line 33): "A converging line of research, from Tolman's (1948) ... The relevance for an archival interface is direct: the cognitive system already runs multi-resolution hierarchical structure ... When a researcher reads an archive at several resolutions ..."
   — contains the "is direct" template that R0_C critic flagged
   — multiple 0.99+ sentences
2. Preface (lines 17-19) — already tested R6_C+R7_C, hard regression.
3. Abstract ¶2 (line 11) — short, high-AI.

Pick: §1 ¶5. Largest GPTZero leverage zone NOT yet tested with surgical Haiku paraphrase under R8_C constraints.

If R9_C also +1 improves: ship-ready close-out approaching.
If R9_C regresses: revert + advance to R10_C with abstract ¶2.
