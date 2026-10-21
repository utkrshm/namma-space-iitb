# AGENTS.md

Operating instructions for AI coding agents working in this repository. Human documentation lives in `README.md` and `docs/ARCHITECTURE.md`; this file is the executable subset — what to run, what to do, and what not to do.

## What this repository is

This repository is a submission to the Namma Space challenge for the IIT Bombay Techfest '26. The requirement is a pipeline that converts a smartphone-captured video into a digital twin with navigation. The project requirements and scope are documented in `docs/GOALS.md`, based on the [official challenge PDF](docs/Namma%20Space.pdf).

## Development environment
- Use `uv` for Python environment and package management. Never use pip directly. Maintain dependencies in `pyproject.toml`.
- Use `uv run` for running scripts, tests, formatting, linting and type checking. Document common workflows as exact `uv run` commands. Use `[project.scripts]` for application CLI entry points when appropriate. Use `uvx` for standalone tools that do not require the project environment.
- Run `uv run ruff check` to idenitfy linting errors, and correct them using `uv run ruff check --fix --fixable <codes>`.

## General instructions
- Make the code modular. Have clear separation boundaries, and clear contracts for handoff between different modules and strictly adhere to the contracts defined. For example, the input and output of the preprocessing step should be well documented and clearly defined, during the execution as well as the planning stages.
- Each pipeline stage must live in an aptly named package under `src/`. Define public import aliases/re-exports in each stage package's `__init__.py` and in `src/__init__.py`.
- Keep tests under `tests/`. Keep runnable demonstration scripts for the full pipeline or working parts of it at the repository root; these scripts should call the modules under `src/`.
- `data/` contains original captures; write derived outputs to `outputs/`.
- Maintain `docs/STATE.md` for current implementation progress and blockers, following a clear format. Create it when implementation progress is first recorded, and update it when implementation status changes.
- Before implementing a substantive design change, record the adopted decision and rationale in `docs/PLANS.md` with relevant timestamps. Create the file when the first such decision is recorded. Read-only reviews and exploratory discussions do not require file changes.
- `docs/ARCHITECTURE.md` contains the overall architecture of the planned pipeline. Update it when an adopted decision changes system structure or module contracts.
- Keep affected documentation consistent when scope, commands, or module contracts change. Update references when planned documents are created.
- For code comments, use common/existing terms for easy human understanding; avoid obsecure terms or coining unnecessary new concepts.
- Preserve original captures and retain frame IDs, timestamps and image-transformation metadata. Document coordinate frames, units and transform conventions at module boundaries. Keep generated artifacts separate from source inputs and curated test fixtures.
- Work in separate branches for separate components. Do not work in the `master` branch.
- Add blank lines between setup, validation, data preparation, computation, I/O and output handling whenever those phases are present and distinct. 
- Do not write long uninterrupted sequences of statements. Do not insert a blank line after every statement. Ensure Ruff formatting preserves the intended separation.

## Validation and Verification
- Test real implementations. Do not create mock or fake pipeline/model implementations merely to make tests pass. If required hardware or dependencies are unavailable, report the affected checks as not run instead of substituting a fake.
- Visually verify preprocessing outputs on a representative capture at least once. Repeat this check when changes affect the visual results, and record what was inspected.
