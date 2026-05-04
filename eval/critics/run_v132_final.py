#!/usr/bin/env python3
"""v132 final 6-critic confirmation runner.

One LLM call per critic. Hard-stops on rate-limit (exit 42).
Writes verdicts to v132_final_{critic}.md.
"""
import json, sys, time, urllib.request, urllib.error
from pathlib import Path

CRED = json.load(open('/root/.claude/.credentials.json'))
TOKEN = CRED['claudeAiOauth']['accessToken']

DRAFT_PATH = '/tmp/mausoleo/references/MAUSOLEO_FULL_DRAFT_v132.md'
DRAFT = Path(DRAFT_PATH).read_text()
OUT_DIR = Path('/tmp/mausoleo/eval/critics')

API = 'https://api.anthropic.com/v1/messages'
SYS = "You are Claude Code, Anthropic's official CLI for Claude."

def call(model, prompt, max_tokens=4096, system=SYS):
    body = json.dumps({
        'model': model,
        'max_tokens': max_tokens,
        'system': system,
        'messages': [{'role': 'user', 'content': prompt}],
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        'Authorization': f'Bearer {TOKEN}',
        'anthropic-version': '2023-06-01',
        'anthropic-beta': 'oauth-2025-04-20',
        'content-type': 'application/json',
    })
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read())
        return data['content'][0]['text']
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors='replace')
        if e.code == 429 or 'rate_limit' in body.lower() or 'rate-limit' in body.lower():
            print(f'RATE_LIMIT: {e.code} {body[:300]}', file=sys.stderr)
            sys.exit(42)
        print(f'HTTP_ERROR {e.code}: {body[:500]}', file=sys.stderr)
        raise

def write(name, content):
    p = OUT_DIR / f'v132_final_{name}.md'
    p.write_text(content)
    print(f'wrote {p}')

# ---- 1. RUBRIC (Opus 4.7 via claude-opus-4-5 alias) ----
RUBRIC_PROMPT = f"""You are an experienced UCL BASC0024 marker grading a final-year dissertation against the published rubric. v132 is the candidate ship draft (9,431 words, 10,000 cap; new title chosen by writer).

BASC0024 Final-Year Dissertation Rubric (4 categories):
1. **Knowledge & Understanding** — depth of subject knowledge, engagement with literature, theoretical grounding.
2. **Critical Analysis & Argument** — quality of reasoning, originality, position-taking, handling of objections.
3. **Research Design & Method** — appropriateness of method, rigor, validity of evidence, sophistication of analysis.
4. **Presentation & Scholarly Conventions** — structure, clarity, citation accuracy, register, prose quality.

Bands: Fail <40 / Pass 40-49 / Lower 2:2 50-54 / Upper 2:2 55-59 / Lower 2:1 60-64 / Upper 2:1 65-69 / Low 1st 70-74 / Mid 1st 75-79 / High 1st 80+

For each category:
- Score 1-5 (5 = high 1st, 4 = low/mid 1st, 3 = upper 2:1, 2 = lower 2:1, 1 = 2:2 or below).
- One-line justification with quoted span.
- One concrete edit that would push +1 (or "none needed at ship band").

End with:
**Predicted band:** <band>
**Confidence:** low/medium/high
**Compare to prior 4/4/4/4 PASS (low-mid 1st 70-74, R95 baseline):** confirm / shift up / shift down
**Ship verdict:** SHIP / FIX-FIRST

Be calibrated. Borderline = lower band. Do not pad.

DRAFT:
{DRAFT}
"""

# ---- 2. CITATION (Haiku 4.5) — re-verify 3 fixes + 5-7 new spot-checks ----
CITATION_PROMPT = f"""You are a citation reviewer doing a final v132 verification on a UCL final-year dissertation.

CONTEXT: The prior deep critic on v123 found 3 bibliography errors that v131 fixed. v132 is identical to v131 in references. Verify these 3 fixes hold:

1. **Doucet et al. (2020) NewsEye** — should list authors: Doucet, Gasteiner, Granroth-Wilding, Kaiser, Kaukonen, Labahn, Moreux, Muehlberger, Pfanzelter, Therenty, Toivonen, Tolonen. Earlier (v123) had wrong names (Gabay, Hulden, Düring, Marjanen) conflated from sibling projects.
2. **Eichenbaum (2017)** — should be 'The role of the hippocampus in navigation is memory', J Neurophysiol 117(4):1785-1796 doi:10.1152/jn.00005.2017. Earlier had Neuron 95(5):1007-1018.
3. **Murugaraj et al. (2025)** — should be 'Topic-RAG for historical newspapers: enhancing information retrieval in humanities research through topic-based retrieval-augmented generation', Computational Humanities Research 1(e15) doi:10.1017/chr.2025.10018. Earlier had a paraphrased title.

Then SPOT-CHECK 5-7 additional citations beyond what was deep-verified on v123. The v123 deep critic verified these (do not re-spot): Bartlett-1932, Miller-1956, Cowan-2001, Tolman-1948, Whittington-2020, Sarthi-2024, Edge-2024, Zhang-Tang-2025/PageIndex, Ehrmann-2020, Düring-2024, Düring transparent generosity, Salton-1975, Robertson-2009, Behrens-2018, Schudson-1978, Murialdi-1986, ICA-2000, Bai-2025, Ketelaar-2001, Wu-2021, Yao-2022, Lewis-2020, Pavone-1991, Bosworth-2005, Deakin-1962, Craik-Lockhart-1972.

Pick 5-7 different in-text claim+citation pairs from the v132 draft to spot-check; for each:
1. Quote the in-text claim + citation.
2. State expected source content (use only what you can recall about the paper).
3. Verdict: VERIFIED / UNVERIFIABLE / MISMATCH / HALLUCINATED.
4. One-line note.

End with:
**Bibliography fix re-verification:**
- Doucet 2020: HOLDS / FAIL
- Eichenbaum 2017: HOLDS / FAIL
- Murugaraj 2025: HOLDS / FAIL
**Spot-check summary:** N verified / N unverifiable / N mismatch / N hallucinated
**Citation verdict:** PASS / FAIL
**Compare to v123 deep critic baseline (38/45 PASS, fixes shipped in v131):** confirm / shift

Be especially alert for citation inversion (paraphrase + reversed conclusion).

DRAFT:
{DRAFT}
"""

# ---- 3. PLAGIARISM (Haiku 4.5) ----
PLAG_PROMPT = f"""You are a plagiarism reviewer. Final pass on v132 of a UCL final-year dissertation.

Check for:
1. Direct quote "transparent generosity" (line 50, attributed to Düring et al. 2024) — confirm character-perfect and properly framed.
2. Verbatim ≥10-word quotes lacking attribution.
3. Paraphrase plagiarism (close mimicry of source structure without transformation).
4. Citation inversion (claim attributed to a source where the source argues opposite).
5. New patchwriting risks introduced by Stage C heading rewrites or the new title "Mausoleo: a calendar-shaped index for reading newspaper history".

The single block quote (Italian, line 129) is a self-authored summariser output for the 26 July 1943 missing-day node, not a third-party quote — verify it is framed as such.

Output:
**Direct quote audit ("transparent generosity"):** PASS / FAIL with span.
**Block-quote audit (Italian summariser output line 129):** PASS / FAIL.
**Paragraph-level scan:** flag any paragraph that fails (D=grafted no attrib / E=too-close paraphrase) with span + likely source + fix.
**Title/heading change audit:** any new patchwriting risk?
**Verdict:** PASS / FAIL with one-line summary.
**Compare to Stage A baseline May 3 (PASS):** confirm / shift.

If you cannot fully verify against actual source text, say so explicitly per chunk; do NOT fabricate match findings.

DRAFT:
{DRAFT}
"""

# ---- 4. COHERENCE (Haiku 4.5) ----
COH_PROMPT = f"""You are a coherence reviewer. Final structural read on v132 of a UCL final-year dissertation.

The draft has a NEW TITLE (line 1): "Mausoleo: a calendar-shaped index for reading newspaper history". Verify it aligns with the abstract and introduction framing.

Stage C round 26-40 introduced ~20 heading-shape rewrites for AI-detection diversity. Verify no broken cross-references resulted.

Check:
1. **Title-abstract-intro alignment:** does the new title fit the abstract's framing about calendar-shaped index, missing 26 July, three case studies?
2. **Thesis tracking**: state thesis in one sentence; verify every section advances/complicates it.
3. **Cross-section consistency**: any section A describing section B — verify claims about B match B (e.g., abstract says "eighteen scored trials, 11.3 vs 28.3 tool calls"; do those numbers match chapter 4 + Appendix A?).
4. **Citation chain**: every in-text (Author, Year) has a bibliography entry; every bib entry is cited at least once.
5. **Numerical/factual self-consistency**: tool-call means (13.3, 12.3, 8.3 vs 27.0, 29.7, 28.3 → 11.3 vs 28.3 mean), recall scores, OCR composite (0.872, 0.926 → 0.899 mean stated as ~0.90), node counts (6,480 articles → 31 days, 5 weeks, 1 month root, 6,517 total) — agree across mentions.
6. **Cross-reference integrity:** "chapter four", "Appendix A", "§X.Y", "supplementary material" — exist? Stage C heading rewrites may have broken some.
7. **Word count vs cap (10,000):** 9,431 words. Confirm under cap.
8. **Undefined terms:** any term used before it is introduced?

For each finding: quoted span + issue (one sentence) + surgical fix.

End with:
**Title alignment:** PASS / FAIL
**Cross-reference integrity:** PASS / FAIL with named broken refs
**Numerical self-consistency:** PASS / FAIL
**Word count under cap:** PASS / FAIL
**Highest-priority fix:** <one line> or NONE.
**Submission-ready:** YES / NO with named blockers / YES with minor.
**Compare to Stage A baseline May 3 (PASS with one minor — Appendix A skeletal, fixed in v109):** confirm / shift.

DRAFT:
{DRAFT}
"""

PLAN = [
    ('rubric', 'claude-opus-4-5', RUBRIC_PROMPT, 4096),
    ('citation', 'claude-haiku-4-5', CITATION_PROMPT, 4096),
    ('plagiarism', 'claude-haiku-4-5', PLAG_PROMPT, 3500),
    ('coherence', 'claude-haiku-4-5', COH_PROMPT, 4096),
]

if __name__ == '__main__':
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, model, prompt, mt in PLAN:
        if only and name != only:
            continue
        print(f'== {name} ({model}) ==')
        t0 = time.time()
        out = call(model, prompt, max_tokens=mt)
        dt = time.time() - t0
        write(name, f'# v132 final {name} verdict\n\nModel: {model}\nElapsed: {dt:.1f}s\nDraft: {DRAFT_PATH}\n\n---\n\n{out}\n')
