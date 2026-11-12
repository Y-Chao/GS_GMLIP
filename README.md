<p align="center"><img src="assets/gm2-surf_icon.svg" alt="GM2-Surf" width="180"></p>

# GM2-Surf

**Global Minimum search with Global Machine-Learning interatomic potentials, for Surfaces** — a general package for global searching of stable/metastable structures on catalyst surfaces under working conditions, accelerated by MLIP (MACE, etc.).

## Architecture

```
gm2_surf/
├── structure/          # Atoms, regions, bonds, molecular blocks
├── evaluate/           # MLIP (MACE, generic ASE) & DFT (VASP stub) calculators
│   ├── mlip/           #   MACEEvaluator, ASECalculatorEvaluator
│   └── dft/            #   VASPEvaluator (stub)
├── search/             # 5 global search methods
│   ├── ga/             #   Genetic Algorithm (population, crossover, mutation)
│   ├── gcmc/           #   Grand Canonical Monte Carlo (insert/delete/swap/displace)
│   ├── gofee/          #   GPR surrogate + acquisition functions (LCB, EI)
│   ├── ssw/            #   Stochastic Surface Walking + minima hopping
│   └── metadynamics/   #   PLUMED-based (well-tempered) metadynamics
├── analysis/           # Structure clustering, trajectory convergence, export
├── data/               # SQLite candidate database, I/O, workflow manager
└── cli.py              # YAML config CLI entry point
```

## Features

- **5 search methods**: GA, GCMC, GOFEE, SSW, MC, Metadynamics — all with a unified `BaseSearcher` interface
- **MLIP acceleration**: MACE-MP foundation models out of the box; any ASE Calculator via adapter
- **Surface-aware**: Tag-based slab/adsorbate separation, molecular block system (H₂O, CO₂, CO, SO₂, HCOOH, OH, H, O, S)
- **Grand canonical**: Finite temperature/pressure equilibrium via GCMC with Metropolis acceptance
- **Surrogate-assisted**: GOFEE uses GPR surrogate with LCB/EI acquisition for sample-efficient search
- **Enhanced sampling**: Metadynamics via PLUMED wrapper with collective variable definitions
- **Analysis**: Structure deduplication, fingerprint clustering, convergence checks, CSV/XYZ export

## Installation

```bash
# Core (GA, GCMC, GOFEE, SSW, analysis)
pip install -e .

# With MACE MLIP
pip install -e ".[mlip]"

# With PLUMED metadynamics
pip install -e ".[plumed]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### CLI
```bash
gm2_surf --config config.yaml
```

### Python API
```python
from ase.build import fcc111
from gm2_surf.evaluate.mlip.mace_eval import MACEEvaluator
from gm2_surf.search.ga.runner import GARunner
from gm2_surf.structure.composition import water_block, hydrogen_block

slab = fcc111("Cu", size=(3, 3, 4), vacuum=15.0, periodic=True)
evaluator = MACEEvaluator(model="small")
blocks = [water_block(chemical_potential=-14.22), hydrogen_block(chemical_potential=-3.39)]

runner = GARunner(slab=slab, blocks=blocks, evaluator=evaluator, population_size=20, seed=42)
runner.setup()
runner.run(n_steps=50)
best = runner.get_best(n=3)
```

### YAML Configuration
```yaml
method: ga
slab_file: cu111_slab.xyz
evaluator:
  backend: mace
  model: small
  device: cpu
blocks:
  - name: H2O
    chemical_potential: -14.22
  - name: CO
    chemical_potential: -14.78
ga:
  population_size: 20
  n_steps: 100
seed: 42
```

## Examples

| System | Methods | Directory |
|--------|---------|-----------|
| Cu(111)/Cu(100) + CO₂/H₂O/SO₂/HCOOH (CSHO) | GA, GCMC | `examples/cu_csho/` |
| RuO₂(110) + H₂O/OH/H/O | GA, GCMC | `examples/ruo2_h2o/` |

Each example includes: `prepare_slab.py`, `run_ga.py`, `run_gcmc.py`, `analyze.py`, and YAML configs.

## Package Structure

- `gm2_surf/structure/` — `SlabAtoms` (extends ASE Atoms), `BoxRegion`/`SphereRegion`, `BondData`, `MolecularBlock`, `CompositionConstraint`
- `gm2_surf/evaluate/` — `BaseEvaluator` ABC → `MACEEvaluator`, `ASECalculatorEvaluator`, `VASPEvaluator`
- `gm2_surf/search/` — `BaseSearcher` ABC → `GARunner`, `GCMCRunner`, `GOFEERunner`, `SSWRunner`, `MetadynamicsRunner`
- `gm2_surf/analysis/` — `StructureAnalyzer` (fingerprint clustering), `TrajectoryAnalyzer` (convergence), visualization export
- `gm2_surf/data/` — `CandidateDatabase` (ASE SQLite), `WorkflowManager` (checkpoints), I/O utilities

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| ASE | ≥ 3.23.0 | Core atomic simulation framework |
| NumPy | ≥ 1.26.0 | Numerical computing |
| SciPy | ≥ 1.12.0 | Optimization |
| scikit-learn | ≥ 1.4.0 | GPR surrogate (GOFEE) |
| PyYAML | ≥ 6.0 | Config parsing |
| mace-torch | ≥ 0.3.14 | MLIP (optional) |
| plumed | ≥ 2.10.0 | Metadynamics (optional) |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
