"""Run GA search on Cu(111)/CSHO system.

Genetic Algorithm search for stable adsorbate configurations on Cu(111),
considering CO2, H2O, SO2, HCOOH and their dissociation intermediates.
Surface Cu reconstruction by CO, H, and S is allowed.
"""

from ase.io import read

from gs_gmlip.evaluate.mlip.mace_eval import MACEEvaluator
from gs_gmlip.search.ga.runner import GARunner
from gs_gmlip.structure.composition import (
    CompositionConstraint,
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
from gs_gmlip.structure.region import BoxRegion


def main():
    # Load slab
    slab = read("cu111_slab.xyz")

    # Define molecular blocks with approximate chemical potentials (eV)
    blocks = [
        water_block(chemical_potential=-14.22),
        co2_block(chemical_potential=-22.96),
        co_block(chemical_potential=-14.78),
        so2_block(chemical_potential=-12.50),
        formic_acid_block(chemical_potential=-29.50),
        hydrogen_block(chemical_potential=-3.39),
        oxygen_block(chemical_potential=-4.93),
        oh_block(chemical_potential=-7.77),
        sulfur_block(chemical_potential=-4.00),
    ]

    # Composition constraints: 0-6 adsorbate molecules total
    constraints = CompositionConstraint(
        blocks=blocks,
        min_count=2,
        max_count=6,
        element_pool={"C": (0, 4), "H": (0, 12), "O": (0, 8), "S": (0, 2)},
    )

    # Evaluator (MACE-MP-0, small model for testing)
    evaluator = MACEEvaluator(model="small", device="cpu", default_dtype="float64")

    # Region for adsorbate placement
    z_top = slab.positions[:, 2].max()
    region = BoxRegion(
        cell=slab.cell,
        z_min=z_top + 1.5,
        z_max=z_top + 6.0,
        pbc=slab.pbc,
    )

    # GA runner
    runner = GARunner(
        slab=slab,
        blocks=blocks,
        evaluator=evaluator,
        region=region,
        composition_constraints=constraints,
        population_size=20,
        n_to_optimize=5,
        seed=42,
        work_dir="ga_cu111_csho",
    )

    # Run
    runner.setup()
    runner.run(n_steps=100)

    # Results
    best = runner.get_best(n=5)
    for i, atoms in enumerate(best):
        atoms.write(f"ga_cu111_csho/best_{i}.xyz")
        print(f"Rank {i}: E = {atoms.get_potential_energy():.4f} eV")


if __name__ == "__main__":
    main()
