# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`gs_gmlip` (Global Searching by Machine Learning Interatomic Potential) — a Python package for global searching of stable/metastable structures on catalyst surfaces under working conditions, accelerated by MLIPs (MACE, etc.). Builds on top of ASE.

Requires Python ≥ 3.12. Package name on disk is `gs_gmlip` (underscore); distribution name is `gs-gmlip`.

## Common Commands

The repo uses `uv` (see `uv.lock`, `.python-version`). The local virtualenv lives at `.venv/`.

```bash
# Install (core only — GA, GCMC, GOFEE, SSW, analysis)
uv venv && uv sync
uv pip install -e .

# Install with optional extras
uv pip install -e ".[mlip]"     # MACE
uv pip install -e ".[plumed]"   # PLUMED metadynamics
uv pip install -e ".[dev]"      # black, isort, ruff, pytest, sphinx
uv pip install -e ".[all]"      # everything

# Run all tests
uv run pytest tests/ -v

# Run a single test file or test
uv run pytest tests/test_gcmc.py -v
uv run pytest tests/test_gcmc.py::test_insert_move -v

# Lint / format (configured in pyproject.toml — line length 120, py312)
uv run ruff format check gs_gmlip/ tests/
uv run ruff check gs_gmlip/ tests/

# Interactive app (needs the `app` extra: uv pip install -e ".[app]")
uv run gs_gmlip app --project my_project        # http://127.0.0.1:8080
# Run one job directory headless (what every executor calls)
uv run gs_gmlip run-job my_project/jobs/<job_id>
```

The package lives under `src/gs_gmlip/` (src layout).

`deprecated/` is gitignored archived code — do not edit it. `examples/*/search/` directories are search outputs and are also gitignored; treat them as regenerable.

## Architecture

Five global-search methods share a single abstract interface; an evaluator abstraction decouples search from energy backend; structure objects extend ASE `Atoms` with a surface/adsorbate awareness scheme.

### Two abstract bases (the spine)

- [`BaseSearcher`](gs_gmlip/search/base.py) — every search method (`GARunner`, `GCMCRunner`, `GOFEERunner`, `SSWRunner`, `MetadynamicsRunner`) subclasses this. Required: `setup()` and `step()`. Inherited: `run(n_steps)`, `get_results()`, `get_best(n)`. The default `run` loop swallows per-step exceptions and logs them — a failing step does not abort the search.
- [`BaseEvaluator`](gs_gmlip/evaluate/base.py) — wraps any ASE `Calculator` behind `get_calculator()`, plus `evaluate(atoms, relax=...)`, `relax(...)` (BFGS by default), and `batch_evaluate(...)`. Concrete: `MACEEvaluator`, `ASECalculatorEvaluator` (any ASE calc, e.g. EMT), `VASPEvaluator` (stub).

A searcher is constructed with an evaluator instance — they are independently composable. The CLI ([`gs_gmlip/cli.py`](gs_gmlip/cli.py)) wires both from a YAML config via `build_evaluator(...)` and `build_searcher(...)`; each runner also has a `from_config(...)` classmethod.

### Interactive app, store, and jobs

The app never calls a searcher directly. The flow is: builder edits an `Interface` -> the structure is written to the store -> a `JobSpec` is handed to an executor -> a worker process runs the task and writes every intermediate structure back to the store -> the app polls the store.

- [`interface/operations.py`](src/gs_gmlip/interface/operations.py) -- all structure manipulation, done with pymatgen (`SlabGenerator`, `AdsorbateSiteFinder`, `SymmOp`). Functions are pure (return a new `Interface`); the app's undo relies on that. Slabs are oriented a along x, normal along z, c perpendicular.
- [`store.py`](src/gs_gmlip/store.py) -- `StructureStore` (ase.db SQLite + provenance: `stage`, `method`, `calc_name`, `parent_id`, `job_id`, `step`) and `Project` (`<root>/structures.db`, `<root>/jobs/<job_id>/`). Energies and forces are frozen into rows.
- [`evaluator/calculators.py`](src/gs_gmlip/evaluator/calculators.py) -- calculator registry (`register_calculator`, `get_calculator`; built in: `emt`, `matgl`, `mace`, `vasp`) and `CalculatorEvaluator`. Backends import lazily.
- [`jobs/`](src/gs_gmlip/jobs) -- `JobSpec`/`JobStatus` (JSON files in the job directory), tasks registered with `@register_task` (`singlepoint`, `relax`, `md`, `ga`, `gcmc`), `worker.py` (`python -m gs_gmlip.jobs.worker <job_dir>`), and `LocalExecutor`. New execution back ends (HPC managers) subclass `BaseExecutor`; new methods (SSW, GOFEE, minima hopping, Sella) are added as tasks.
- [`app/`](src/gs_gmlip/app) -- NiceGUI app. `scene.py` (plain Python, tested) turns an `Interface` into a JSON scene; `viewer.py` is the 3Dmol.js surface viewer (loaded from a CDN); `main.py` is the page.

### Surface/adsorbate identification scheme

### Data layer

[`CandidateDatabase`](gs_gmlip/data/database.py) wraps `ase.db` SQLite with a candidate lifecycle (unrelaxed → queued → relaxed) and is mainly used by the GA. The GCMC runner writes to a plain `ase.db` directly (not via `CandidateDatabase`).

## Conventions to preserve

- `from __future__ import annotations` is enforced by `ruff isort` (see `lint.isort.required-imports`) -- donot remove it.
- Google-style docstrings; ruff `pydocstyle` is enabled with selected ignores. Tests, docs, and examples have docstrings required waived in `pyproject.toml`.
- Line length 120; ruff handles formatting (donot run black even though config exists).
- New search methods must subclass `BaseSearcher` and implement `setup()` + `step()`; new evaluators must subclass `BaseEvaluator` and implement `get_calculator()`.
- Custom adsorbates: define a factory and call `register_block(name, factory)` rather than instantiating `MolecularBlock` ad hoc — the YAML CLI path looks blocks up by name through `get_block(...)`.
