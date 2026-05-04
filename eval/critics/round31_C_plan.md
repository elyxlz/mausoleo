# R31_C Plan + Verdict — Strategy 7 epigraph (REVERTED)

## Plan

R29_C/R30_C established headings + title work. Try epigraph (Italian primary-source quote with citation, 22w block) below the title. Per Strategy 7 native-language scaffolding hypothesis: non-author-voice block dilutes doc-level fingerprint.

## Edit (1 epigraph block, +22w net)

Added below title:
```
> *Il giorno 26 luglio 1943 «Il Messaggero» non uscì.*
> — note in the *Emeroteca digitale* fund record for July 1943.
```

## GPTZero result

- R30_C baseline: 0.6185
- R31_C: **0.7677 (+0.1492 HARD REGRESSION)**
- REVERTED.

## Lesson

Epigraph blocks regress GPTZero, even Italian non-author-voice quoted material. Possible mechanism:
1. Short fragments at the document head are flagged by burstiness penalty (per SKILL: "GPTZero v4.4b penalises short fragments at 1.0 generated_prob; the tactic was trained against").
2. The epigraph + attribution form is a high-frequency LLM-generated dissertation template.
3. Adding any content of any voice contaminates per-sentence reads via doc-level fingerprint.

R26_C (abstract+preface rewrite +33w +0.029) and R31_C (epigraph +22w +0.149) together are evidence that ANY prose-volume addition risks regression. Doc-shape changes that DON'T add prose volume (heading rewrites, title rewrites, subtitle rewrites) work. Prose-volume changes regress.
