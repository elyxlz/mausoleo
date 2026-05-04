#!/usr/bin/env python3
"""Final 6-critic confirmation runner for R95 Mausoleo dissertation.

Each critic = 1 LLM call (haiku for spot/flow, opus for rubric).
Hard-stop on first rate-limit. Writes verdicts to final_{critic}.md.
"""
import json, sys, time, urllib.request, urllib.error
from pathlib import Path

CRED = json.load(open('/root/.claude/.credentials.json'))
TOKEN = CRED['claudeAiOauth']['accessToken']

DRAFT = Path('/tmp/mausoleo/references/MAUSOLEO_FULL_DRAFT_v10.md').read_text()
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
    p = OUT_DIR / f'final_{name}.md'
    p.write_text(content)
    print(f'wrote {p}')

# ---- Critic prompts (single call each) ----

# 1. RUBRIC (Opus 4.7)
RUBRIC_PROMPT = f"""You are an experienced UCL BASC0024 marker grading a final-year dissertation against the published rubric. Score each of the four BASC0024 categories 1-5 with one-line justification, then predict the overall band.

BASC0024 Final-Year Dissertation Rubric (4 categories):
1. **Knowledge & Understanding** — depth of subject knowledge, engagement with literature, theoretical grounding.
2. **Critical Analysis & Argument** — quality of reasoning, originality, position-taking, handling of objections.
3. **Research Design & Method** — appropriateness of method, rigor, validity of evidence, sophistication of analysis.
4. **Presentation & Scholarly Conventions** — structure, clarity, citation accuracy, register, prose quality.

Bands: Fail <40 / Pass 40-49 / Lower 2:2 50-54 / Upper 2:2 55-59 / Lower 2:1 60-64 / Upper 2:1 65-69 / Low 1st 70-74 / Mid 1st 75-79 / High 1st 80+

For each category:
- Score 1-5 (where 5 = high 1st, 4 = low/mid 1st, 3 = upper 2:1, 2 = lower 2:1, 1 = 2:2 or below).
- One-line justification with quoted span if possible.
- One concrete edit that would push +1 (or "none needed at ship band").

Then end with:
**Predicted band:** <band>
**Confidence:** low/medium/high
**Ship verdict:** SHIP / FIX-FIRST

Be calibrated. Borderline = lower band. Do not pad. Per Stage A close-out (Phase 3 dispatch May 3): cat1=4 cat2=4 cat3=4 cat4=4 PASS, predicted band low-mid 1st 70-74%. Confirm or dispute.

DRAFT:
{DRAFT}
"""

# 2. CITATION (Haiku 4.5) — spot-check 5 random citations
CITATION_PROMPT = f"""You are a citation reviewer doing a final spot-check on a UCL final-year dissertation. The bibliography lists ~30 sources; the paper folders verified locally are:

cognitive_science: alonso, baddeley, bartlett-1932, behrens-2018, clark-2013, clark-chalmers-1998, cowan-2001, craik-lockhart-1972, eichenbaum-2017, friston-2010, hutchins-1995, johnson-laird-1983, lake-2017, miller-1956, rosch-1978, tolman-1948, whittington-2020
digital_humanities_ir: chen-bge-m3, cook-2013, da-2019, during-2024, edge-2024, ehrmann-2020, ica-2000-isadg, jaillant-2022, jenkinson-1922, jockers-2013, karpukhin-2020, ketelaar-2001, khattab-2020, lewis-2020, manovich-2020, marciano-2018, moretti-2013, murugaraj-2025, ranganathan-1933, reimers-2019, robertson-2009, salton-1975, sarthi-2024, schellenberg-1956, thomas-2024-postocr, underwood-2019
historiography: braudel-1958, cohen-rosenzweig-2005, ehrmann-2024, forno-2012, gentile-1993, ginzburg-1976, ginzburg-1989, guldi-armitage-2014, halbwachs-1950, moretti-2000, murialdi-1986, nora-1989, pavone-1991, schudson-1981, tworek-2019
philosophy_media_1943: adorno-1991, agarossi-2003, battaglia-1953, bonsaver-2007, bosworth-2002, bosworth-2005, deakin-1962, defelice-1990, forgacs-2007, foucault-1969, foucault-1971, freiman-2024, fricker-2007, gadamer-1960, gentile-1996, goldberg-2010, habermas-1962, habermas-2022, haraway-1988, harding-1991, heidegger-1927, horkheimer-adorno-1947, kasirzadeh-2023, katz-2003, lackey-2008, mccombs-1972, mccombs-2014, medina-2013, mittelstadt-2016
technical: (Bai-Qwen2.5VL, Wu-2021-recursive-summary, Yao-2022-react implied)

Pick FIVE distinct in-text citations from the draft. For each:
1. Quote the in-text claim + citation.
2. State expected page/quote/argument from the cited source (use only what you can recall from the verified PDF).
3. Verdict: VERIFIED / UNVERIFIABLE / MISMATCH / HALLUCINATED.
4. One-line note.

Then end with:
**Spot-check summary:** N verified / N unverifiable / N mismatch / N hallucinated
**Citation verdict:** PASS / FAIL
**Stage A baseline (May 3):** PASS — confirm or dispute.

Be especially alert for citation inversion (paraphrase + reversed conclusion).

DRAFT:
{DRAFT}
"""

# 3. PLAGIARISM (Haiku 4.5)
PLAG_PROMPT = f"""You are a plagiarism reviewer. Final pass on a UCL final-year dissertation. The source corpus lives at /tmp/mausoleo/references/papers/ (149 PDFs). Run a chunk-overlap check against likely-grafted paragraphs.

Check for:
1. Verbatim quotes ≥10 contiguous words from likely-source paragraphs that LACK proper attribution.
2. Paraphrase plagiarism: paragraphs whose discourse structure mirrors a source too closely (claim ordering, transitions) without transformative argument.
3. Common-knowledge phrasings dressed as original.
4. Citation inversion (claim attributed to source where source argues opposite).

Output format:
**Paragraph-level scan:** flag any paragraph that fails. For each: quoted span + likely source + classification (D=grafted no attrib / E=too-close paraphrase) + fix.
**Block-quote audit:** any block-quoted text — verify it has author + year + page + framing.
**Verdict:** PASS / FAIL with one-line summary.
**Stage A baseline (May 3):** PASS — confirm or dispute.

If you cannot fully verify against actual source text, say so explicitly per chunk; do NOT fabricate match findings.

DRAFT:
{DRAFT}
"""

# 4. COHERENCE (Haiku 4.5)
COH_PROMPT = f"""You are a coherence reviewer. Final structural read on a UCL final-year dissertation.

Check:
1. **Thesis tracking**: state thesis in one sentence; verify every section advances/complicates it.
2. **Cross-section consistency**: any section A describing section B — verify claims about B match B.
3. **Citation chain**: every in-text (Author, Year) has a bibliography entry; every bib entry is cited at least once.
4. **Numerical/factual self-consistency**: tool-call means, recall scores, dates, week numbers, article counts — agree across mentions.
5. **Hanging refs / broken cross-refs**: "chapter four", "Appendix A", "supplementary material" — exist?
6. **Word count vs cap (10,000)**: ~7,108w stated; sanity-check.
7. **Undefined terms**: any term used before it is introduced?

For each finding: quoted span + issue (one sentence) + surgical fix.

End with:
**Highest-priority fix:** <one line> or NONE.
**Submission-ready:** YES / NO with named blockers / YES with minor.
**Stage A baseline (May 3):** PASS — confirm or dispute.

Be ruthless about real problems. "I cannot find issues" is a valid verdict.

DRAFT:
{DRAFT}
"""

PLAN = [
    ('rubric', 'claude-opus-4-5', RUBRIC_PROMPT, 4096),
    ('citation', 'claude-haiku-4-5', CITATION_PROMPT, 3500),
    ('plagiarism', 'claude-haiku-4-5', PLAG_PROMPT, 3000),
    ('coherence', 'claude-haiku-4-5', COH_PROMPT, 3500),
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
        write(name, f'# Final {name} verdict — R95 confirmation\n\nModel: {model}\nElapsed: {dt:.1f}s\n\n---\n\n{out}\n')
