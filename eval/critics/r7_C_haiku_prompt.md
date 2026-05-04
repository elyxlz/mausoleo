You are paraphrasing a preface for a UCL final-year dissertation. The current preface reads as too uniform in academic register; an AI-detection classifier scores every sentence at 0.99+ AI.

Rewrite the preface in a deliberately ROUGHER native-English-undergraduate register: shorter sentences mixed with longer, occasional sentence fragments where natural, one or two contractions, named people, named months, concrete bug-discovery moments. The replacement should read like a CS-leaning student who writes papers fast and clean rather than slow and polished. Aim for 200-280 words total across two paragraphs (the body paragraph + the acknowledgments paragraph).

Preserve these factual claims:
- Project background: came from RAG pipeline work at year-zero/year-one
- Why Italian newspapers: chosen because of language access (the writer reads Italian)
- Corpus pick: Il Messaggero 1943, missing 26 July is the key feature
- OCR weeks: hard, especially the post-correction pass that made composite scores worse
- Bartlett's *Remembering* came in via second-year psychology, hippocampal-mapping work in Eichenbaum/Whittington came later
- Supervisor: Dr Yi Gong (single thanks acceptable for register-shorter version)

Output ONLY the rewritten preface text in two paragraphs separated by a blank line, no prefacing meta-commentary, no markdown headers, no quotes around the output.

Original preface for reference:

I came to this project from the engineering side, after a year working on retrieval-augmented generation pipelines for technical documentation. I wanted a final-year project that would put the same techniques in front of source material that does not chunk well, and Italian regime-period newspapers were the natural pick from the languages I read. The 1943 issue list of *Il Messaggero* was the most complete in the available scans, which is how I ended up with a corpus that contained one missing day. Two weeks went into a post-correction pass that made composite OCR scores worse, and another into the realisation that the date-bounded queries the system most needed to answer well were exactly those the keyword baseline was structurally incapable of resolving. Bartlett's *Remembering* was a paper I had read for second-year psychology and went back to once the OCR work began producing day-summary nodes; the hippocampal-mapping work in Eichenbaum and Whittington came in much later, after the calendar-shaped index had already been built.

I thank Dr Yi Gong for supervising. Her comments on the cognitive-science framing, and on what the system can and cannot warrant, shaped the chapters that follow.
