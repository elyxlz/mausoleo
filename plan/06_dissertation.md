# Phase 6: Dissertation

## Goal

Write the BASc (Arts and Sciences, UCL) dissertation around this system. The dissertation must be interdisciplinary — the technical system is the vehicle, but the argument must engage with theories and frameworks from multiple disciplines beyond CS/AI.

## 6.1 Core Thesis

Something along the lines of:

*"Hierarchical summarization structures, inspired by how human cognition organizes knowledge at multiple levels of abstraction, enable qualitatively superior access to historical newspaper archives compared to traditional bag-of-words retrieval — transforming archives from searchable databases into navigable knowledge structures."*

Refine this. The key claim is that your system doesn't just retrieve better — it enables a fundamentally different mode of interaction with archival material.

## 6.2 Interdisciplinary Framing

The technical work needs to be situated within and supported by multiple non-CS disciplines. Here are strong candidates:

### Cognitive Science / Neuroscience

- **Hierarchical predictive processing**: The brain organizes perception and knowledge hierarchically, with higher levels encoding more abstract representations. Your tree mirrors this — decade summaries are abstract, paragraphs are concrete. Cite: Karl Friston's free energy principle, predictive coding frameworks.
- **Levels of processing (Craik & Lockhart, 1972)**: deeper/more abstract processing leads to better memory and understanding. Your system enables agents to process at whichever depth is appropriate.
- **Chunking (Miller, 1956)**: working memory is limited, so we chunk information hierarchically. Your summaries are chunks at different granularities.
- **Cognitive maps (Tolman, 1948; O'Keefe & Nadel, 1978)**: the hippocampus builds spatial maps for navigation. Your tree is a "cognitive map" of a knowledge space that agents navigate.

### Information Science / Library Science

- **Faceted classification (Ranganathan)**: traditional library classification uses multiple facets (subject, time, place). Your hierarchy is primarily temporal but summaries capture topical facets. Discuss how this compares.
- **Aboutness and relevance theory**: what makes a summary "about" the right things? How do you decide what survives compression? Engage with information retrieval theory.
- **Archival science**: the principle of provenance, original order, respect des fonds. Your system preserves temporal order and newspaper structure while adding navigational layers. How does this relate to archival ethics?
- **Digital humanities**: the "distant reading" tradition (Franco Moretti) — using computation to study literature at scale. Your system enables "distant reading" of newspaper archives.

### Philosophy of Knowledge / Epistemology

- **Abstraction as knowledge**: what is lost and gained when you compress a day's news into a paragraph? Philosophical questions about summarization as a form of knowledge production.
- **Hermeneutic circle**: understanding parts requires understanding the whole, and vice versa. Your tree literally implements this — navigate from whole (archive) to parts (paragraphs) and back.
- **Situated knowledge (Donna Haraway)**: all knowledge is partial and situated. Your summaries are produced by an LLM with its own biases. Discuss the epistemological implications.

### History / Historiography

- **Annales school (Braudel)**: history operates at multiple timescales — the longue durée (centuries), conjonctures (decades), and événements (events). Your hierarchy maps directly onto this: decades capture slow trends, days capture events.
- **Microhistory (Carlo Ginzburg)**: reconstructing the lives of ordinary people from fragmentary records. Your example queries ("tell me about the Pichinon family") are literally microhistory enabled by technology.
- **Public history and collective memory**: how newspapers shape and reflect public consciousness. Your system enables studying this at scale.

### Media Studies

- **Agenda-setting theory (McCombs & Shaw)**: newspapers don't tell people what to think, but what to think about. Your summaries at different levels capture what the newspaper considered important. Analyzing how the summarization hierarchy reflects editorial priorities.
- **The archaeology of knowledge (Foucault)**: discourse analysis, how knowledge is produced and organized. Your system creates a new discursive structure on top of historical material.

### Linguistics

- **Italian language change**: 60 years of newspaper text captures orthographic and stylistic evolution. Potential for incidental linguistic findings.
- **Computational linguistics / NLP**: situate your OCR and summarization work within the NLP tradition for historical languages.

## 6.3 Evaluation / Case Studies

### Experimental Setup

Compare your hierarchical navigation system against a baseline:
- **Baseline**: keyword search over bag-of-words OCR (simulating Google Books / traditional digital archives)
- **Your system**: hierarchical tree + semantic search + LLM agent navigation

### Case Studies

Design 3-5 case studies based on real research questions. For each:

1. **Define the research question** (e.g., "How did Il Messaggero cover the Ethiopian War?")
2. **Run the baseline**: keyword search for relevant terms, read through results, compile findings
3. **Run your system**: agent navigates the tree, drills into relevant branches, compiles findings
4. **Evaluate both**:
   - Completeness: did the approach find all relevant information?
   - Efficiency: how many steps / how much text did the agent/user have to read?
   - Quality: LLM-as-judge evaluates the final compiled answer
   - Discoverability: did the approach surface unexpected/serendipitous findings?

### Case Study Candidates

1. **Specific family/person search** ("Pichinon" family across decades)
2. **Broad cultural question** (collective consciousness during fascism)
3. **Geographic query** (events on a specific street or neighborhood)
4. **Event tracking** (coverage of a specific historical event across days/weeks)
5. **Thematic exploration** (restaurants, food culture, daily life)

### Evaluation Metrics

- **Precision/Recall of relevant sources found** (requires human-annotated "relevant article" ground truth for each case study)
- **Steps to answer** (number of tool calls / search queries needed)
- **Text processed** (total characters the agent had to read to reach an answer)
- **Answer quality** (LLM-as-judge scoring on comprehensiveness, accuracy, insight)
- **Serendipity score** (did the system surface relevant information the researcher didn't explicitly search for?)

### LLM-as-Judge Protocol

For each case study answer:
1. Have a strong LLM (Claude) evaluate the answer on a rubric
2. Multiple evaluation dimensions: factual accuracy, completeness, insight, source quality
3. Blind evaluation — judge doesn't know which system produced which answer
4. Report inter-rater reliability if using multiple judge runs

## 6.4 Dissertation Structure (Draft)

1. **Introduction**
   - The problem: millions of digitized newspaper pages, practically inaccessible
   - The gap: keyword search is insufficient for meaningful historical research
   - The proposal: hierarchical knowledge structures for archival navigation

2. **Literature Review**
   - Current state of digital archives and their limitations
   - Hierarchical knowledge representations (cognitive science)
   - Levels of abstraction in historiography (Braudel, Annales school)
   - Distant reading and computational humanities (Moretti)
   - Information retrieval theory and its limits

3. **System Design**
   - Architecture overview
   - OCR pipeline: challenges of historical Italian newspaper digitization
   - Recursive hierarchical summarization: theory and implementation
   - The navigation paradigm: tree traversal + search

4. **OCR Evaluation**
   - Ground truth creation methodology
   - Model comparison results
   - Discussion of OCR quality across eras

5. **The Knowledge Index**
   - Schema design and rationale
   - Summarization quality analysis
   - Information loss across levels — what survives abstraction?

6. **Evaluation**
   - Experimental setup (your system vs. baseline)
   - Case studies (3-5, detailed)
   - Quantitative results
   - Qualitative analysis

7. **Discussion**
   - What this means for digital humanities
   - Epistemological implications of LLM-mediated archival access
   - Limitations and biases
   - Generalizability beyond newspapers

8. **Conclusion & Future Work**
   - Generic system for any unstructured data
   - Multi-archive support
   - Incremental index updates
   - Community-contributed corrections to OCR/summaries

## 6.5 Implementation Steps

1. Finalize case study questions with dissertation supervisor
2. Build the baseline system (keyword search over OCR text in ClickHouse)
3. Run all case studies with both systems
4. Design and run LLM-as-judge evaluation
5. Compile quantitative results
6. Write the dissertation following the structure above
7. Iterate with supervisor feedback

### Definition of Done

- Dissertation submitted
- All case studies executed with both systems
- Evaluation results compiled and analyzed
- Interdisciplinary framing reviewed and strengthened
- Supervisor approved
