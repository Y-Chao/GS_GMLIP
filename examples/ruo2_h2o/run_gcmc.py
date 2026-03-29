"""Run GCMC search on RuO2(110)/H2O system.

GCMC at finite temperature to determine equilibrium water coverage
on RuO2(110) under OER-relevant conditions.
"""

from ase.io import read

from gs_gmlip.evaluate.mlip.mace_eval import MACEEvaluator
from gs_gmlip.search.gcmc.runner import GCMCRunner
from gs_gmlip.structure.composition import (
    hydrogen_block,
    oh_block,
    oxygen_block,
    water_block,
)


def main():
    slab = read("ruo2_110_slab.xyz")

    blocks = [
        water_block(chemical_potential=-14.22),
        oh_block(chemical_potential=-7.77),
        hydrogen_block(chemical_potential=-3.39),
        oxygen_block(chemical_potential=-4.93),
    ]

    evaluator = MACEEvaluator(model="small", device="cpu", default_dtype="float64")

    runner = GCMCRunner(
        slab=slab,
        blocks=blocks,
        evaluator=evaluator,
        temperature=350.0,  # Near ambient water
        move_weights={"insert": 0.35, "delete": 0.35, "displace": 0.25, "swap": 0.05},
        max_displacement=0.6,
        seed=123,
        work_dir="gcmc_ruo2_h2o",
    )

    runner.setup()
    runner.run(n_steps=500)

    print(f"Best energy: {runner.best_energy:.4f} eV")
    print(f"Acceptance rate: {runner.ensemble.acceptance_rate:.3f}")

    best = runner.get_best()
    if best:
        best[0].write("gcmc_ruo2_h2o/best.xyz")


if __name__ == "__main__":
    main()
