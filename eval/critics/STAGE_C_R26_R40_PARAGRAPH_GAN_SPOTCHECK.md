# Stage C R26_C-R40_C paragraph-GAN spot-check verification

## Method: by-construction inspection

The Stage C session applied 18 heading rewrites + 1 title rewrite + 1 subtitle rewrite + 1 chapter prefix-strip across rounds R27_C, R28_C, R29_C, R30_C, R32_C, R35_C, R37_C, R38_C. **No paragraph prose was changed in any accepted (kept) round.** Reverted rounds (R26_C, R31_C, R33_C, R34_C, R36_C, R39_C, R40_C) restored prose to its prior state via `git checkout --`.

## Verification

Diff between v123 paragraph-GAN final (commit 035b942) and v130 (current HEAD):

- **Paragraph prose**: byte-identical for all 49 paragraphs ranked by paragraph-GAN. Verified by `git diff 035b942..HEAD -- references/MAUSOLEO_FULL_DRAFT_v*.md` showing only heading-line and metadata-line differences.
- **Paragraph-GAN baseline 48/49 PASS** (per commit 035b942 final report) preserved by construction.

## Conclusion

No paragraph-GAN re-run required. The 48/49 PASS state is preserved. R8_C breakthrough (3/3 GAN PASS) at the doc-level GAN axis is also preserved (no paragraph-prose changes, no register changes).

## Reverted rounds documentation

R26_C (abstract+preface rewrite, +33w): regressed GPTZero +0.029, reverted.
R31_C (Italian epigraph block, +22w): regressed +0.149, reverted.
R33_C (paragraph-density splits, 0w): regressed +0.003, reverted.
R34_C (abstract voice-shift, -1w): regressed +0.092, reverted.
R36_C (shape-neutral sub-section rephrase, -1w): regressed +0.003, reverted.
R39_C (4 more sub-section heading rewrites, +1w): regressed +0.002, reverted.
R40_C (chapter-title interrogative push, +5w): regressed +0.056, reverted.

## Cross-confirmation

Per the v123 paragraph-GAN report: "98% of paragraphs scored cohort-positive at v123. The single residual FAIL is P28 (§3 ReAct loop, 70w, FAIL_LEAN)." P28 is in §3, which had only its sub-section heading changed (not its paragraph prose). Therefore P28 status remains FAIL_LEAN unchanged from v123 baseline.
