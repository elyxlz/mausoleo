# Round 109 plan — STRATEGY TSI Sentence 8 (NEW): §1 ¶4 "There is a body of cognitive-science work, accumulating since Bartlett's *Remembering*" template rewrite

## R109 target: NEW Sentence 8 (surfaced by critic R104)

Current text (§1 ¶4 opener):
> There is a body of cognitive-science work, accumulating since Bartlett's *Remembering* (Bartlett, 1932), that bears on the kind of reading the case studies in chapter four put to the corpus.

Critic flag class: "There is a body of [field] work, accumulating since [seminal author]'s [seminal text]" — explicit AI literature-framing template. Flagged by R104 critic 667655877 as Tell #1 high-leverage.

This sentence was NOT in original 7-sentence list. Surfaced through R104 D1-failure analysis.

## Candidates

**E (short concrete subject):**
"Bartlett's *Remembering* (Bartlett, 1932) opens the cognitive-science line that bears on this reading."
- Pro: NO "There is X" template
- Pro: NO "accumulating since" template
- Pro: 12 words vs original 31 words
- Pro: concrete-subject opener
- Con: very short

**D (single sentence, no template, no list):**
"The cognitive-science line that bears on this reading runs from Bartlett's *Remembering* (Bartlett, 1932) through working-memory research to the hippocampal-mapping work that chapter four returns to."
- Pro: NO "There is X" template
- Con: introduces "from X through Y to Z" parallel construction (could trigger different flag)

**F (drop the framing sentence entirely; let next sentence "Bartlett's *War of the Ghosts* studies described how recall reconstructs..." stand):**
- Pro: NO new prose to attack
- Con: paragraph loses framing context

## Pick: Candidate E

E best per R102/R104 lesson: shortest, concrete-subject opener, removes both flag-class templates, single declarative sentence.

## Edit plan

Single edit on R95 base (commit 65c74b1, freshly reverted after R108):

Replace §1 ¶4 opener:
> There is a body of cognitive-science work, accumulating since Bartlett's *Remembering* (Bartlett, 1932), that bears on the kind of reading the case studies in chapter four put to the corpus.

With:
> Bartlett's *Remembering* (Bartlett, 1932) opens the cognitive-science line that bears on this reading.

Net change: -19w. Word count: 7,108 → 7,089.

## Test protocol

3 random positions per essay-iter SKILL: positions spread 2-6.

## Pareto rule

- If R109 ≥ 3/3: ship.
- If R109 = 2/3 LEAN with FAIL critic NOT keying on §1 ¶4: save BEST_TSI_S8; advance to R110 with another NEW TSI candidate.
- If R109 = 2/3 LEAN with FAIL critic flagging new sentence: NOT +1, revert.
- If R109 < 2/3: hard regression, revert to R95.

## Continuing past dispatch ladder

R109 is round 1 of NEW TSI candidates surfaced by R103/R107 critics. R110 = TSI on §1 ¶6 "There is an episodic-memory analogue:" balanced antithesis with colon-preamble (R103 critic 232746473 Tell #2 high-leverage).

Per dispatch rule (7) NEVER declare local optimum. Continue grinding new candidate axes.
