"""m-GCMC search for CuS surface reconstruction under flue gas conditions.

Searches for stable/metastable structures on Cu(111) in equilibrium with
gas-phase reservoirs (CO2, SO2, H2O, H2) at 300 K. The search uses a
modified GCMC protocol: adsorb/desorb/swap with probabilities 0.4/0.4/0.2.

Two evaluation modes:
  - EMT: fast development & workflow validation
  - MACE-MP: production-level MLIP (uncomment in config.yaml)

Usage:
    python run_search.py                    # single run at 0 V
    python run_search.py --potential -1.0   # single run at -1.0 V
    python run_search.py --scan             # scan all potentials
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml
from ase.io import read, write

from gs_gmlip.evaluate.mlip.ase_calc import ASECalculatorEvaluator
from gs_gmlip.search.gcmc.runner import GCMCRunner
from gs_gmlip.structure import SlabAtoms, BoxRegion
from gs_gmlip.structure.composition import (
    CompositionConstraint,
    MolecularBlock,
    get_block,
    register_block,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# To register a custom molecular block, define a factory and call register_block():
#
#   from ase import Atoms
#   def cho_block(mu=0.0):
#       atoms = Atoms("CHO", positions=[[0,0,0],[1.1,0,0],[0,1.2,0]])
#       return MolecularBlock(
#           name="CHO", atoms=atoms, chemical_potential=mu,
#           bonds=[(0,1),(0,2)], n_electrons=1,
#       )
#   register_block("CHO", cho_block)
#
# Then add {name: CHO, chemical_potential: 5.0} in config.yaml.


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_evaluator(eval_config: dict):
    """Build evaluator from config dict."""
    backend = eval_config.get("backend", "emt")
    if backend == "emt":
        from ase.calculators.emt import EMT

        return ASECalculatorEvaluator(EMT())
    elif backend == "mace":
        from gs_gmlip.evaluate.mlip.mace_eval import MACEEvaluator

        return MACEEvaluator(
            model=eval_config.get("model", "medium"),
            device=eval_config.get("device", "cpu"),
            default_dtype=eval_config.get("dtype", "float64"),
            dispersion=eval_config.get("dispersion", False),
        )
    else:
        raise ValueError(f"Unknown evaluator backend: {backend}")


def build_blocks(
    block_configs: list[dict], evaluator_backend: str = "emt"
) -> list[MolecularBlock]:
    """Build adsorbate blocks from config.

    Filters out blocks with elements unsupported by the evaluator.
    """
    # EMT only supports: Cu, Ag, Au, Ni, Pd, Pt, H, C, N, O
    EMT_SUPPORTED = {"Cu", "Ag", "Au", "Ni", "Pd", "Pt", "H", "C", "N", "O"}

    blocks = []
    for bc in block_configs:
        name = bc["name"]
        mu = bc.get("chemical_potential", 0.0)
        block = get_block(name, mu=mu)

        # Check if all elements in this block are supported
        if evaluator_backend == "emt":
            block_elements = set(block.atoms.get_chemical_symbols())
            if not block_elements.issubset(EMT_SUPPORTED):
                unsupported = block_elements - EMT_SUPPORTED
                logger.warning(
                    "Skipping block %s: elements %s not supported by EMT",
                    name,
                    unsupported,
                )
                continue

        blocks.append(block)
    return blocks


def prepare_slab(slab_file: str, z_threshold: tuple | float | None = None) -> SlabAtoms:
    """Load slab and set up 3-region tags/constraints for GCMC.

    Regions (controlled by z_threshold):
        tag=0: substrate — fixed, no mutation
        tag=1: buffer   — relaxed, no mutation / rattle
        tag=2: surface  — relaxed + mutable / rattleable

    Parameters
    ----------
    slab_file : str
        Path to the slab structure file (VASP, XYZ, etc.).
    z_threshold : float or tuple(float, float), optional
        Height thresholds for region boundaries.
        - float: atoms below are substrate (0), above are buffer (1).
        - tuple (z_low, z_high): 3-region split (0, 1, 2).
        - None: infer from existing FixAtoms constraints. Fixed atoms
          get tag=0; the remaining atoms are split into buffer (lower
          half) and surface (upper half).
    """
    atoms = read(slab_file)
    slab = SlabAtoms.from_atoms(atoms)

    if z_threshold is not None:
        # User-specified height boundaries
        slab.tag_slab_by_height(z_threshold)
    else:
        # Infer from FixAtoms constraints
        from ase.constraints import FixAtoms

        fixed_indices = set()
        for c in slab.constraints:
            if isinstance(c, FixAtoms):
                fixed_indices.update(c.index)

        z = slab.positions[:, 2]
        free_mask = np.array([i not in fixed_indices for i in range(len(slab))])
        tags = np.zeros(len(slab), dtype=int)

        if free_mask.any():
            free_z = z[free_mask]
            z_mid = (free_z.min() + free_z.max()) / 2.0
            for i in range(len(slab)):
                if i in fixed_indices:
                    tags[i] = 0  # substrate
                elif z[i] < z_mid:
                    tags[i] = 1  # buffer
                else:
                    tags[i] = 2  # surface (active)
        slab.set_tags(tags)

    slab.set_slab_constraints()
    return slab


def run_single(config: dict, potential: float = 0.0) -> dict:
    """Run a single GCMC search at a given electrochemical potential.

    Parameters
    ----------
    config : dict
        Configuration from YAML.
    potential : float
        Electrochemical potential (V vs RHE). Shifts chemical potentials
        by n_e * potential for each species.

    Returns
    -------
    results : dict
        Summary of the search results.
    """
    # Prepare slab
    z_threshold = config.get("z_threshold", None)
    slab = prepare_slab(config["slab_file"], z_threshold=z_threshold)
    logger.info("Slab loaded: %d atoms, cell=%s", len(slab), slab.cell.lengths())

    # Build blocks (filter unsupported for EMT)
    eval_backend = config.get("evaluator", {}).get("backend", "emt")
    blocks = build_blocks(config["blocks"], evaluator_backend=eval_backend)

    # Electrochemical potential shift (CHE model):
    # mu_eff = mu_gas + n_e * e * U
    # n_electrons is defined on each MolecularBlock.
    if abs(potential) > 1e-10:
        for block in blocks:
            ne = block.n_electrons
            block.chemical_potential += ne * potential
            logger.info(
                "Block %s: mu_eff = %.3f eV (shift: %d e × %.2f V)",
                block.name,
                block.chemical_potential,
                ne,
                potential,
            )

    # Build evaluator
    evaluator = build_evaluator(config.get("evaluator", {}))

    # Work directory
    work_dir = Path(config.get("work_dir", "search")) / f"U_{potential:.1f}V"

    # GCMC runner
    runner = GCMCRunner(
        slab=slab,
        blocks=blocks,
        evaluator=evaluator,
        temperature=config.get("temperature", 300.0),
        move_weights=config.get("move_weights"),
        max_displacement=config.get("max_displacement", 0.8),
        seed=config.get("seed", 42),
        work_dir=str(work_dir),
        db_file=config.get("db_file"),
    )

    # Run search
    n_steps = config.get("n_steps", 200)
    logger.info(
        "Starting GCMC: T=%.0f K, U=%.1f V, %d steps",
        config.get("temperature", 300.0),
        potential,
        n_steps,
    )

    runner.setup()
    runner.run(n_steps=n_steps)

    # Collect results
    summary = runner.ensemble.get_summary()
    best = runner.get_best()

    results = {
        "potential_V": potential,
        "n_steps": n_steps,
        "best_energy_eV": runner.best_energy,
        "acceptance_rate": summary["acceptance_rate"],
        "move_stats": summary["move_stats"],
        "n_adsorbates": runner._count_adsorbates(runner.current_atoms),
        "final_energy_eV": runner.current_energy,
    }

    # Save outputs
    work_dir.mkdir(parents=True, exist_ok=True)
    if best:
        best_clean = best[0].copy()
        best_clean.calc = None
        best_clean.constraints = []
        write(str(work_dir / "best.xyz"), best_clean, format="extxyz")
        results["best_formula"] = best[0].get_chemical_formula()
        logger.info(
            "Best structure: %s, E=%.4f eV",
            best[0].get_chemical_formula(),
            runner.best_energy,
        )

    # Save current (final) structure
    if runner.current_atoms is not None:
        current_clean = runner.current_atoms.copy()
        current_clean.calc = None
        current_clean.constraints = []
        write(str(work_dir / "current.xyz"), current_clean, format="extxyz")

    # Save trajectory info
    with open(str(work_dir / "trajectory.json"), "w") as f:
        json.dump(runner.trajectory, f, indent=2, default=str)

    with open(str(work_dir / "summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results saved to %s", work_dir)
    return results


def run_potential_scan(config: dict):
    """Run GCMC at multiple electrochemical potentials."""
    potentials = config.get("potentials", [0.0])
    all_results = []

    for U in potentials:
        logger.info("=" * 60)
        logger.info("Potential scan: U = %.1f V vs RHE", U)
        logger.info("=" * 60)
        result = run_single(config, potential=U)
        all_results.append(result)

    # Save combined summary
    out_dir = Path(config.get("work_dir", "search"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(str(out_dir / "potential_scan.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print table
    print("\n" + "=" * 70)
    print("Potential Scan Summary")
    print("=" * 70)
    print(
        f"{'U (V)':>8} {'Best E (eV)':>14} {'N_ads':>6} {'Acc. Rate':>10} {'Formula':>20}"
    )
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['potential_V']:>8.1f} "
            f"{r['best_energy_eV']:>14.4f} "
            f"{r['n_adsorbates']:>6d} "
            f"{r['acceptance_rate']:>10.3f} "
            f"{r.get('best_formula', 'N/A'):>20}"
        )
    print("=" * 70)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="m-GCMC search for Cu(111) surface reconstruction"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Configuration YAML file"
    )
    parser.add_argument(
        "--potential",
        type=float,
        default=None,
        help="Electrochemical potential (V vs RHE). Default: 0 V.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Run potential scan over all values in config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.scan:
        run_potential_scan(config)
    else:
        U = args.potential if args.potential is not None else 0.0
        run_single(config, potential=U)


if __name__ == "__main__":
    main()
