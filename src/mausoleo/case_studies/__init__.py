"""Case-study runner for §6 of the Mausoleo dissertation.

Three case studies x two systems (Mausoleo + BM25 baseline) x three trials,
researcher agent identical (Claude Sonnet 4.5 over OAuth), only the toolset
differs. See ``runner.py`` for orchestration, ``tools.py`` for the tool
shapes, ``judges.py`` for LLM-as-judge scoring, and ``stats.py`` for the
sign tests.
"""
