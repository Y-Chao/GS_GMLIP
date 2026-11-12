# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GM2-Surf, package `gm2_surf` (Global Minimum search with Global Machine-Learning interatomic potentials, for Surfaces; formerly `gs_gmlip`) — a Python package for global searching of stable/metastable structures on catalyst surfaces under working conditions, accelerated by MLIPs (MACE, etc.). Builds on top of ASE.

Requires Python ≥ 3.12. Package name on disk is `gm2_surf` (underscore); distribution name is `gm2-surf`.

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
uv run ruff format check gm2_surf/ tests/
uv run ruff check gm2_surf/ tests/

# Interactive app (needs the `app` extra: uv pip install -e ".[app]"; optional: `solvent`, `matgl`)
uv run gm2_surf app --project my_project        # http://127.0.0.1:8080
# Run one job directory headless (what every executor calls)
uv run gm2_surf run-job my_project/jobs/<job_id>
```

The package lives under `src/gm2_surf/` (src layout).

`deprecated/` is gitignored archived code — do not edit it. `examples/*/search/` directories are search outputs and are also gitignored; treat them as regenerable.

## Architecture

Five global-search methods share a single abstract interface; an evaluator abstraction decouples search from energy backend; structure objects extend ASE `Atoms` with a surface/adsorbate awareness scheme.

### Two abstract bases (the spine)

- [`BaseSearcher`](gm2_surf/search/base.py) — every search method (`GARunner`, `GCMCRunner`, `GOFEERunner`, `SSWRunner`, `MetadynamicsRunner`) subclasses this. Required: `setup()` and `step()`. Inherited: `run(n_steps)`, `get_results()`, `get_best(n)`. The default `run` loop swallows per-step exceptions and logs them — a failing step does not abort the search.
- [`BaseEvaluator`](gm2_surf/evaluate/base.py) — wraps any ASE `Calculator` behind `get_calculator()`, plus `evaluate(atoms, relax=...)`, `relax(...)` (BFGS by default), and `batch_evaluate(...)`. Concrete: `MACEEvaluator`, `ASECalculatorEvaluator` (any ASE calc, e.g. EMT), `VASPEvaluator` (stub).

A searcher is constructed with an evaluator instance — they are independently composable. The CLI ([`gm2_surf/cli.py`](gm2_surf/cli.py)) wires both from a YAML config via `build_evaluator(...)` and `build_searcher(...)`; each runner also has a `from_config(...)` classmethod.

### Interactive app, store, and jobs

Storage rule: every structure, job and workflow lives in the project folder. The app opens no project by default (`create_app(None)` answers 409 until `/api/project` is called), and the global store (`~/.gm2_surf/structures.db`) is only written by an explicit action: "Add current row to global database" (`StructureStore.copy_to`) or `gm2_surf import --global`. `~/.gm2_surf` otherwise holds settings only (hosts, recent projects, global molecule library).

Built-in references: [`reference.py`](src/gm2_surf/reference.py) defines a light JSON Lines form (one structure per line: symbols, positions, cell, energy, role, level, a few tags) with `write_jsonl` / `import_jsonl` (`gm2_surf export`, `gm2_surf import file.jsonl`) and the read-only `ReferenceLibrary`. The package ships `src/gm2_surf/database/bulk.jsonl` (Cu, Ag, Au, Pd, Pt fcc lattices at 14 levels of theory, GPAW and VASP) and `molecule.jsonl` (7 gas molecules, VASP PBE-D3BJ) as the third reference tier, scope `builtin`, after project and global.

Cluster presets: [`jobs/presets.py`](src/gm2_surf/jobs/presets.py) holds the predefined Hosts-dialog values per cluster (Vanda, ASPIRE 2A, ASPIRE 2B, Hopper; PBS only), following the `hpc` and `install` docs of `~/repo/cytools`: scratch directory, queue, a small default resource slice, 2 h wall time, worker in `~/apps/gm2_surf/.venv`, and the environment for the cytools-registered VASP module (`module use ~/apps/modulefiles`, `module load vasp/<version>-<build>`, `ASE_VASP_COMMAND`, `VASP_PP_PATH`). `host_presets()` matches presets to `~/.ssh/config` aliases with `ssh -G`. VASP dispersion tags are matched to the functional (`DISPERSION_FUNCTIONAL_TAGS`: `DFTD4_XC`, `SDFTD3_XC`).

Naming: the project is GM2-Surf (package and command `gm2_surf`, also `gm2-surf`; formerly `gs_gmlip`). The former `~/.gs_gmlip` folder is moved to `~/.gm2_surf` on first use, `GS_GMLIP_HOME` is still honoured, and files carrying the old `gs_gmlip_interface` metadata key still load. Icons live in [`assets/`](assets); `assets/build_icons.py` regenerates the PNGs, the `.icns`, and the copies in `app/static/`.

Split of responsibilities: **JavaScript does interaction and rendering, Python does all chemistry.** The browser sends intents ("cleave this slab", "rotate these atoms", "run this task") and redraws from the JSON it gets back. The app never calls a searcher directly: an edit produces an `Interface`, the structure goes to the store, a `JobSpec` goes to an executor, a worker process runs the task and writes every intermediate structure back, and the frontend polls.

- [`interface/operations.py`](src/gm2_surf/interface/operations.py) -- all structure manipulation, done with pymatgen (`SlabGenerator`, `AdsorbateSiteFinder`, `SymmOp`): bulk, slab, sites, adsorbates, delete/substitute/move/rotate/fix, and ASE constraints (`FixBondLengths`, `FixSubsetCom`, `Hookean` wall). Functions are pure (return a new `Interface`); undo relies on that. Slabs are oriented a along x, normal along z, c perpendicular.
- [`interface/solvation.py`](src/gm2_surf/interface/solvation.py) -- Packmol water/solvent filling (`solvent` extra).
- [`store.py`](src/gm2_surf/store.py) -- `StructureStore` (ase.db SQLite + provenance: `stage`, `method`, `calc_name`, `parent_id`, `job_id`, `step`, `score`) and `Project` (`<root>/structures.db`, `<root>/jobs/<job_id>/`, `<root>/plugins/*.py`).
- [`evaluator/calculators.py`](src/gm2_surf/evaluator/calculators.py) -- calculator registry (`register_calculator`; built in: `emt`, `matgl`, `mace`, `vasp`). Default MatGL model: `TensorNet-PES-MatPES-r2SCAN-2025.2`.
- [`evaluator/vasp.py`](src/gm2_surf/evaluator/vasp.py) -- VASP inputs from pymatgen input sets (`VASP_PRESETS`), with tags layered set < precision < surface < dispersion < user settings.
- [`jobs/`](src/gm2_surf/jobs) -- `JobSpec`/`JobStatus`, tasks (`singlepoint`, `relax`, `cell_relax`, `nvt`, `npt`, `ga`, `gcmc`, `vasp_inputs`), `worker.py`, `LocalExecutor`. [`jobs/remote.py`](src/gm2_surf/jobs/remote.py): `RemoteHost` (stored in `~/.gm2_surf/hosts.json`), `RemoteExecutor` (SSH + scheduler: `pbs`, `shell`; each remote job is a self-contained mini project whose rows are merged back with remapped `parent_id`), `ProjectExecutor` (routes each job to local or its remote host; the session default). Tests use `LocalTransport`, so no SSH is needed. New schedulers are entries in `SCHEDULERS`; new execution back ends subclass `BaseExecutor`.
- [`search/asop.py`](src/gm2_surf/search/asop.py), [`search/minima_hopping.py`](src/gm2_surf/search/minima_hopping.py) -- `ASOPRunner` (Chen, Shang, Liu, JCP 156, 094104 (2022): supercell grids, per-composition search, Monte Carlo selection; the inner search is minima hopping, not SSW, and there is no potential retraining; input is the clean primitive surface cell) and `MinimaHoppingRunner` (Goedecker 2004). Tasks `asop` and `minima_hopping`.
- [`jobs/workflow.py`](src/gm2_surf/jobs/workflow.py) -- `WorkflowEngine`: chains of steps with fan-out over input structures and parameter sweeps (`StepSpec.take` = last/best N/all, `sweep` with dotted parameter names). No daemon: state lives in `<project>/workflows/<id>/` and advances when `Session.jobs()` is polled or through `gm2_surf run-workflow <dir>`.
- [`app/bundle.py`](src/gm2_surf/app/bundle.py) -- `gm2_surf make-app` writes a macOS launcher bundle that runs the current environment; `--allow-quit` adds the in-app Quit button. `config.choose_folder` is the native macOS folder dialog used by the project list.
- [`plugins.py`](src/gm2_surf/plugins.py) -- plugin loading (entry-point group `gm2_surf.plugins`, or `<project>/plugins/*.py`). A new search algorithm is a `BaseSearcher` subclass registered with `register_searcher`; a new engine is `register_calculator`; anything else is `register_task`.
- [`interface/intermediates.py`](src/gm2_surf/interface/intermediates.py), [`interface/molecules.py`](src/gm2_surf/interface/molecules.py) -- seed electrocatalysis intermediates (HER/OER/ORR/CO2RR/CORR; initial-guess geometries derived from ASE G2, not literature structures) and the tiered `MoleculeLibrary`: temporary < project (`<project>/molecules.json`) < global (`~/.gm2_surf/molecules.json`) < seed < built-in, first hit wins.
- [`energetics.py`](src/gm2_surf/energetics.py), [`importers.py`](src/gm2_surf/importers.py) -- every stored structure carries a `role` (bulk, molecule, surface, adsorbed, other) and a `level` of theory (`code/xc/basis`, or `calculator/model`); `tags` hold the rest. Derived energies (formation, surface, adsorption, grand potential) are only computed between structures of the same level, with references looked up by role in the project store, then the global store (`~/.gm2_surf/structures.db`). `import_ase_db` / `gm2_surf import` normalizes foreign ASE databases into that schema.
- [`config.py`](src/gm2_surf/config.py) -- user-level directory (`~/.gm2_surf`, or `GM2_SURF_HOME`): global molecule library and recent projects. Tests isolate it through an autouse fixture.
- [`app/session.py`](src/gm2_surf/app/session.py) -- the server-side session: several `Workspace`s ("windows": bulk, surface, molecule), each with its own structure, undo, and store link; plus edits, jobs, ranking, provenance graph. No web framework, fully testable. Every API route takes the window id as the `ws` query parameter. All browser tabs on one server share this session.
- [`app/server.py`](src/gm2_surf/app/server.py) -- thin FastAPI JSON API over `Session` (`create_app`, `run_app`).
- [`app/static/`](src/gm2_surf/app/static) -- the frontend as plain ES modules, no build step: `viewer.js` (3Dmol.js surface viewer, drag move/rotate), `graph.js` (chart, provenance graph), `app.js` (tool column, panels). 3Dmol.js loads from a CDN.

Frontend changes need no build; reload the page. When adding an edit operation, add it to `operations.py`, to the `handlers` map in `Session.edit`, and to `TOOLS` in `app.js`.

### Surface/adsorbate identification scheme

### Data layer

[`CandidateDatabase`](gm2_surf/data/database.py) wraps `ase.db` SQLite with a candidate lifecycle (unrelaxed → queued → relaxed) and is mainly used by the GA. The GCMC runner writes to a plain `ase.db` directly (not via `CandidateDatabase`).

## Conventions to preserve

- `from __future__ import annotations` is enforced by `ruff isort` (see `lint.isort.required-imports`) -- donot remove it.
- Google-style docstrings; ruff `pydocstyle` is enabled with selected ignores. Tests, docs, and examples have docstrings required waived in `pyproject.toml`.
- Line length 120; ruff handles formatting (donot run black even though config exists).
- New search methods must subclass `BaseSearcher` and implement `setup()` + `step()`; new evaluators must subclass `BaseEvaluator` and implement `get_calculator()`.
- Custom adsorbates: define a factory and call `register_block(name, factory)` rather than instantiating `MolecularBlock` ad hoc — the YAML CLI path looks blocks up by name through `get_block(...)`.
