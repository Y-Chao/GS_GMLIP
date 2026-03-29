The directory contains the All files that are related to the real active sites of Cu under the working conditions feeding by the flue gas. Specifically, the GCGA search is performed to sample the stable/metastable structures of Cu(111) surface in equilibrium with some possible adsorbates (*COOH, *HCOH, *CO, *S, *H), the formation energy is used to evaulate the stability of the sampled structures, which is calculated from the H₂O, CO₂, SO₂, H₂ at 300 K and a series of potentials (0, -0.5, -1.0, -1.5, -2.0 V vs. RHE). The search is performed using the modified Grand Canonical Monte Carlo (m-GCMC) method implemented in the `gs_gmlip` package. Simplicity, the search is firstly choose one action (adsorb, desorb, swap) based on the predefined probabilities (0.4, 0.4, 0.2), then randomly select the target site and adsorbate to build the initial structure of the corresponding composition space, and finally two methods were compared: the first ons is that the structure is conducted under the Global Optimization method (GOFEE) with some operations (rattle, permutation, etc.) to find the local minimum structure. The second one is that the structure is local optimized. The acceptance of the new structure is determined by the Metropolis criterion based on the formation energy difference between the new and old structures. The evaluated method is EMT firstly, after the workflow is established, the MLIP (MACE-MP) is used in production stage.

The directory contains the following files:
- `README.md`: this file descrbes the documentation of the files in this example.
- `config.yaml`: the configuration file for running the GCMC search (evaluator, temperature, adsorbate blocks with chemical potentials, move weights, number of steps, potential scan range, bond constraint settings, and database recording).
- `structure.vasp`: the initial structure of the Cu(111) 4×4 slab (64 atoms, 4 layers, bottom 2 layers fixed).
- `run_search.py`: the main script to execute the GCMC search. Supports single-potential (`--potential`) and full potential-scan (`--scan`) modes. Handles EMT element filtering and electrochemical potential shift.
- `analyze.py`: post-processing script that generates trajectory diagnostics (energy evolution, acceptance rate, move statistics, adsorbate count) and potential-scan summary plots.
- `search/`: the directory containing the search results:
  - `U_*V/`: per-potential subdirectories, each with `best.xyz`, `current.xyz`, `trajectory.json`, `summary.json`, and `gcmc_search.db` (ASE SQLite database of all accepted configurations).
  - `potential_scan.json`: combined results across all potentials.
  - `potential_scan.png`: formation energy and surface coverage vs. potential plot.
  - `trajectory_*.png`: four-panel trajectory diagnostics for each potential.
  - `analysis_summary.json`: full analysis output from `analyze.py`.

## v1.1 Features

### Hookean Bond Constraints
Multi-atom adsorbate molecules (CO, CO₂, H₂O, HCOOH, OH, SO₂) can dissociate during relaxation if no intramolecular bond protection is applied. In v1.1, Hookean spring constraints (`ase.constraints.Hookean`) are automatically generated for all intramolecular bonds within each adsorbate before every relaxation step. The spring activates when the bond length exceeds `threshold_scale × current_length` (default: 1.3×), applying a restoring force with spring constant `k` (default: 15 eV/Å²). This prevents C-O, O-H, and other bonds from breaking while still allowing reasonable thermal fluctuation. See `gs_gmlip/structure/constraints.py`.

### ASE Database Recording
Every accepted configuration is recorded to an ASE SQLite database (`gcmc_search.db`) inside each potential directory. Each record stores the structure, step number, energy, move type, acceptance status, number of adsorbates, and chemical formula. This enables retrospective querying of the full search history using standard ASE database tools (`ase db`, Python API).