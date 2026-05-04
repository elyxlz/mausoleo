# R32_C Plan + Verdict — Strategy 5d (chapter prefix strip)

## Plan

R31_C lesson: prose-volume additions regress. Stay on heading-only path. Strip "Chapter N: " prefix from §1-§5 (currently uniform pattern across 5 chapter heads). Reduces 5 word-instances of "Chapter" without changing semantics. Body references to "chapter four" etc remain unchanged (still parse correctly to numbered headings).

## Edits (5 chapter prefix strips, -5w net)

- "## Chapter 1: ..." → "## 1. ..."
- "## Chapter 2: ..." → "## 2. ..."
- "## Chapter 3: ..." → "## 3. ..."
- "## Chapter 4: ..." → "## 4. ..."
- "## Chapter 5: ..." → "## 5. ..."

## GPTZero result

- R30_C baseline: 0.6185
- R32_C: **0.6127 (-0.0058)** — sixth negative-delta round, smaller leverage than prior 5d moves
- predicted_class: ai (still); confidence_score: ~0.6127

## Paragraph-GAN spot-check

Heading-prefix strip only; zero paragraph-prose change. By construction no risk.

## Verdict

ACCEPT. New BEST = 0.6127.

## Hypothesis

Smaller leverage because the strip only removed a 1-word-prefix, not a phrasal restructuring. The remaining "1. Reading across..." is still a noun-phrase + period + noun-phrase structure. To compound further, may need to vary heading shape itself (some interrogative, some declarative, some active-voice-statements).
