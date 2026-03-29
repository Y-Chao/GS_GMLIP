"""Run GA search on RuO2(110)/H2O system.

GA search for stable water/hydroxyl configurations on RuO2(110),
relevant to oxygen evolution reaction conditions.
"""

from ase.io import read

from gs_gmlip.evaluate.mlip.mace_eval import MACEEvaluator
from gs_gmlip.search.ga.runner import GARunner
from gs_gmlip.structure.composition import (
    CompositionConstraint,
    hydrogen_block,
    oh_block,
    oxygen_block,
    water_block,
)
from gs_gmlip.structure.region import BoxRegion


def main():
    slab = read("ruo2_110_slab.xyz")

    # Water-related species
    blocks = [
        water_block(chemical_potential=-14.22),
        oh_block(chemical_potential=-7.77),
        hydrogen_block(chemical_potential=-3.39),
        oxygen_block(chemical_potential=-4.93),
    ]

    constraints = CompositionConstraint(
        blocks=blocks,
        min_count=2,
        max_count=8,
        element_pool={"H": (0, 16), "O": (0, 8)},
    )

    evaluator = MACEEvaluator(model="small", device="cpu", default_dtype="float64")

    z_top = slab.positions[:, 2].max()
    region = BoxRegion(
        cell=slab.cell,
        z_min=z_top + 1.2,
        z_max=z_top + 5.0,
        pbc=slab.pbc,
    )

    runner = GARunner(
        slab=slab,
        blocks=blocks,
        evaluator=evaluator,
        region=region,
        composition_constraints=constraints,
        population_size=20,
        n_to_optimize=5,
        seed=123,
        work_dir="ga_ruo2_h2o",
    )

    runner.setup()
    runner.run(n_steps=100)

    best = runner.get_best(n=5)
    for i, atoms in enumerate(best):
        atoms.write(f"ga_ruo2_h2o/best_{i}.xyz")
        print(f"Rank {i}: E = {atoms.get_potential_energy():.4f} eV")


if __name__ == "__main__":
    main()
