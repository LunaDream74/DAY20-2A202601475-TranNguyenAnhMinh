# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.11 project using a `src/` layout. Application code lives in `src/multi_agent_research_lab/`: agent roles are in `agents/`, shared models and configuration in `core/`, orchestration in `graph/`, provider integrations in `services/`, and benchmarking and tracing in `evaluation/` and `observability/`. The Typer entry point is `cli.py`. Tests mirror behavior under `tests/`. Keep runtime defaults in `configs/`, teaching material in `docs/` and `notebooks/`, helper automation in `scripts/`, and generated benchmark write-ups in `reports/`.

## Build, Test, and Development Commands

- `python -m venv .venv` creates a local environment; activate it before installing packages.
- `make install` installs the package in editable mode with development and LLM extras.
- `make test` runs the Pytest suite configured in `pyproject.toml`.
- `make lint` checks `src/` and `tests/` with Ruff; `make format` applies Ruff formatting.
- `make typecheck` runs strict MyPy checks over production code.
- `make run-baseline` and `make run-multi` exercise the two CLI workflow modes.

On Windows without `make`, run the underlying commands directly, such as `pytest` or `ruff check src tests`.

## Coding Style & Naming Conventions

Use four-space indentation, Python 3.11 syntax, complete type annotations, and a 100-character line limit. Ruff enforces imports and common correctness rules; MyPy runs in strict mode. Use `snake_case` for modules, functions, and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep orchestration in `graph/` and agent-specific behavior in `agents/`; avoid mixing provider calls into domain models.

## Testing Guidelines

Write Pytest tests as `tests/test_<feature>.py` with functions named `test_<behavior>`. Add focused tests for state transitions, validation, error paths, and report output. Pytest enforces at least 80% total coverage. Before submitting, run `ruff format --check src tests`, `make lint`, `make typecheck`, and `make test`.

## Commit & Pull Request Guidelines

Recent history generally uses Conventional Commit prefixes such as `feat:`, `fix:`, `docs:`, and `style:` followed by a concise imperative summary. Keep commits small and scoped. Pull requests should explain the change, motivation, and verification commands; link relevant issues and include screenshots or trace links for UI, slide, or workflow-observability changes. Add or update `reports/benchmark_report.md` when results change.

## Security & Configuration

Copy `.env.example` to `.env` for local secrets. Never commit `.env`, API keys, tokens, or provider credentials. Prefer configuration through environment variables and `configs/lab_default.yaml` rather than hard-coded values.
