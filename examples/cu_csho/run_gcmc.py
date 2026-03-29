"""Run GCMC search on Cu(100)/CSHO system.

Grand Canonical Monte Carlo search for surface equilibrium
configurations on Cu(100) at finite temperature and pressure,
with CO2, H2O, SO2, HCOOH gas phase reservoirs.
"""

from ase.io import read

from gs_gmlip.evaluate.mlip.mace_eval import MACEEvaluator
from gs_gmlip.search.gcmc.runner import GCMCRunner
from gs_gmlip.structure.composition import (
    co2_block,
    co_block,
    formic_acid_block,
    hydrogen_block,
    oh_block,
    oxygen_block,
    so2_block,
    sulfur_block,
    water_block,
)


def main():
    # Load slab
    slab = read("cu100_slab.xyz")

    # Molecular blocks with chemical potentials (eV)
    # mu = E_gas + kT*ln(p/p0) approximately
    blocks = [
        water_block(chemical_potential=-14.22),
        co2_block(chemical_potential=-22.96),
        co_block(chemical_potential=-14.78),
        so2_block(chemical_potential=-12.50),
        hydrogen_block(chemical_potential=-3.39),
        oxygen_block(chemical_potential=-4.93),
        oh_block(chemical_potential=-7.77),
        sulfur_block(chemical_potential=-4.00),
    ]

    # Evaluator
    evaluator = MACEEvaluator(model="small", device="cpu", default_dtype="float64")

    # GCMC runner
    runner = GCMCRunner(
        slab=slab,
        blocks=blocks,
        evaluator=evaluator,
        temperature=600.0,  # Typical catalysis T
        move_weights={"insert": 0.35, "delete": 0.35, "displace": 0.25, "swap": 0.05},
        max_displacement=0.8,
        seed=42,
        work_dir="gcmc_cu100_csho",
    )

    # Run
    runner.setup()
    runner.run(n_steps=500)

    # Results
    print(f"Best energy: {runner.best_energy:.4f} eV")
    print(f"Acceptance rate: {runner.ensemble.acceptance_rate:.3f}")
    print(f"Move stats: {runner.ensemble.get_summary()['move_stats']}")

    best = runner.get_best()
    if best:
        best[0].write("gcmc_cu100_csho/best.xyz")


if __name__ == "__main__":
    main()
