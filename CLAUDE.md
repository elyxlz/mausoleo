## CRITICAL INSTRUCTIONS - READ FIRST

- ⚠️ **ABSOLUTE REQUIREMENT**: Thoroughly review ALL guidelines in this document BEFORE modifying code
- 🚫 **ZERO COMMENTS POLICY**: DO NOT add comments to code - the code must be self-documenting
- ⛔ **FORBIDDEN**: Do not add explanatory, descriptive, or purpose comments to code under ANY circumstances
- 🔴 **MANDATORY**: Apply ALL style guidelines from this document to your work without exception
- ⚠️ **ZERO TOLERANCE**: The user will not accept violations of these guidelines
- ❌ **REPEATED MISTAKES**: Will result in degraded user trust and experience

## Research & Infrastructure Constraints (2026-07-16, per Elio)

- **All compute on ripperred** (`ssh -p 62022 audiogen@81.105.49.222`): tests, inference, experiments. Never run compute locally (laptop OOMs) or on endeavour (corpus storage only, /media/sdr).
- **Corpus-scale budget** (5× raise 2026-07-22): full-corpus OCR (1880–1959, 172,600 pages) on 2×3090 → **target 103.5, hard cap 200.0 sec/page**. Budget is measured by the caller (`scripts/time_experiment.sh`), not the experiment; >200.0 is a research artifact, not production. `BUDGET_CAP` in src/mausoleo/paths.py is the single source of truth. The goal is stated in `GOAL.md`; operating manual in `eval/autoresearch/program.md`.
- **One script → one run → one result**: every experiment is a self-contained script `experiments/<name>.py <date...>` writing `eval/predictions/<name>_<date>.json` (contract in `experiments/README.md`). Experiments never import/run the eval; scoring is external via `scripts/research.py eval`. Reusable operators in `src/mausoleo/ocr/` are plain functions called directly — there is no pipeline framework.
- **Autoresearch loop**: `eval/autoresearch/program.md` (budget, MausoleoBench metric, contract, generalization) + `registry.md` (approach families) + `LOOP.md` (current state). Metric is MausoleoBench (`src/mausoleo/eval/evaluate.py`).
- **Eval integrity**: never modify eval metrics or GT to improve scores; audit every result for matcher gaming (overgeneration, giant blobs, holdout regression, probe degradation).
- **Simplicity mandate**: keep abstractions and interfaces minimal and low-cognitive-load; remove dead code aggressively; isolate complexity (e.g. model-specific quirks live in their own operator/env, never leak into shared paths).

## Package Management

- **uv**: This project uses `uv` for Python package management
- **Run Commands**: Use `uv run script.py` to run scripts
- **Install Packages**: Use `uv add package` to add dependencies

## Memory & Learning

- Update this file whenever user corrects or provides specific instructions
- Record user's command preferences and workflow patterns
- Proactively remember past corrections and apply them consistently
- Ask if unclear whether a correction should be recorded here

## Programming Paradigm

- **Purely Functional Core**: Implement core logic as pure functions with immutable data models
- **Avoid OOP**: No classes with methods, inheritance, or complex object hierarchies
- **Dataclasses Only**: Use frozen dataclasses for data structures, never mutable classes
- **State Flow Pattern**: State changes flow through function returns, never as side effects
- **Function Composition**: Build complex operations by composing smaller pure functions

## Function Design

- **Pure Functions**: No side effects, same output for same input (see `generate_job_id`, `build_job_env`)
- **Function Naming**: Use verb_noun format for function names (`create_job`, `update_job`)
- **Function Size**: Keep functions under 20 lines, extract helpers for logical parts
- **Private Helpers**: Use underscore prefix (`_parse_exit_code`, `_build_script_content`)
- **Function Replacement**: Prefer `dc.replace()` over mutation to modify dataclass instances

## Type System

- **Full Type Annotations**: Use complete type hints for all parameters and return values
- **Union Types**: Use pipe syntax for union types (`str | None`, not `Optional[str]`)
- **Literal Types**: Use `tp.Literal` for constrained string values (`JobStatus = tp.Literal["queued", "running", "completed", "failed"]`)
- **Type Aliases**: Define type aliases for complex types at the module level
- **Return Type Clarity**: Always specify return types, including `None` when appropriate
- **Pyright**: This project uses strict type checking with Pyright
- **Verification**: Always run `uv run pyright` (typically in src directory) before submitting changes
- **No Type Errors**: All code must satisfy Pyright's type checker without errors or warnings

## Import Style

- **Fixed Abbreviations**:
  - `import dataclasses as dc`
  - `import pathlib as pl`
  - `import datetime as dt`
  - `import fastapi as fa`
- **Explicit Module Imports**: Import from specific modules, not packages
- **Local Import Format**: `from package.service.core import exceptions, logger, models`

## Utils & Helpers

- **Composable Utils**: Small, reusable utility functions
- **Consistent Naming**: Similar operations use similar naming patterns
- **Parameter Order**: Context/logger parameters first, optional params last
- **Default Values**: Use sensible defaults for optional parameters

## Code Documentation - NO COMMENTS POLICY

- 🔴 **NEVER ADD COMMENTS**: Code should be clear enough without them - NO EXCEPTIONS
- 🔴 **ZERO EXPLANATORY COMMENTS**: Do not explain what the code does - the code itself is the explanation
- 🔴 **NO INLINE COMMENTS**: Never add comments next to or above code lines
- 🔴 **NO DESCRIPTIVE COMMENTS**: Never add comments describing what a section does
- 🔴 **NO CODE WALKTHROUGH COMMENTS**: Don't add comments explaining the algorithm or approach
- 🔴 **NO FUNCTION PURPOSE COMMENTS**: Function names and signatures must communicate purpose
- 🔴 **NO FILE PURPOSE COMMENTS**: File organization should make purpose obvious

- ✅ **Self-Documenting Code**: Use clear, descriptive variable and function names
- ✅ **Type-Based Documentation**: Rely on type signatures to document interfaces
- ✅ **Clean Interfaces**: Function names and signatures must be clear enough without comments
- ✅ **Docstrings Restricted**: Only add docstrings when function purpose isn't obvious from name/types

