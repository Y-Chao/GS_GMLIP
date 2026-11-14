"""Regression tests for defects found by the local benchmarks (benchmarks/run_local.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc100
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT

from gm2_surf.evaluator.base import BaseEvaluator
from gm2_surf.interface.core import Interface
from gm2_surf.search.ga.operators import CutAndSpliceCrossover
from gm2_surf.search.ga.runner import GARunner
from gm2_surf.search.gcmc.moves import DeleteMove, InsertMove
from gm2_surf.search.gcmc.runner import GCMCRunner
from gm2_surf.search.region import BoxRegion

KB = 8.617333262145e-5


class _Zero(Calculator):
    """Non-interacting particles."""

    implemented_properties = ("energy", "forces")

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(self.atoms), 3))}


class _Evaluator(BaseEvaluator):
    def __init__(self, calculator, **kwargs):
        super().__init__(**kwargs)
        self._calculator = calculator

    def get_calculator(self):
        return self._calculator()


@pytest.fixture
def slab() -> Interface:
    interface = Interface(substrate=fcc100("Cu", (3, 3, 2), a=3.61, vacuum=8.0))
    interface.fix_substrate(layers=1)
    return interface


def _with_adatoms(slab: Interface, positions) -> Interface:
    candidate = slab.copy()
    top = slab.interface.positions[:, 2].max()
    for x, y in positions:
        candidate.add_adsorbate(Atoms("Cu", positions=[[x, y, top + 1.8]]))
    return candidate


def test_crossover_conserves_composition(slab):
    first = _with_adatoms(slab, [(1, 1), (4, 1), (1, 4), (4, 4)])
    second = _with_adatoms(slab, [(2, 2), (5, 2), (2, 5), (6, 6)])
    crossover = CutAndSpliceCrossover(rng=np.random.default_rng(0))
    for _ in range(50):
        child = crossover.get_new_individual([first, second])
        assert child is not None
        assert len(child.interface) - len(child.substrate) == 4


def test_ga_keeps_distinct_candidates_and_the_requested_composition(slab):
    runner = GARunner(
        _Evaluator(EMT, fmax=0.2, relax_steps=30),
        slab,
        blocks=[Atoms("Cu")],
        counts=[3],
        population_size=6,
        n_initial=6,
        seed=1,
    )
    runner.setup()
    energies = runner.population.energies
    # Candidates are duplicates only if energy and fingerprint agree.
    distinct = sum(
        1
        for i, e in enumerate(energies)
        if all(abs(e - other) > runner.energy_tolerance for other in energies[:i])
    )
    assert runner.population.size >= distinct
    top = slab.interface.positions[:, 2].max()
    for _ in range(15):
        runner.step()
    for candidate in runner.population.individuals:
        assert runner._has_composition(candidate)
        # No atom is left hanging in the vacuum, out of reach of the surface.
        assert candidate.interface.positions[len(slab.substrate) :, 2].max() < top + 4.5
    assert not runner._has_composition(_with_adatoms(slab, [(1, 1)]))


def test_random_candidates_keep_blocks_apart(slab):
    runner = GARunner(_Evaluator(EMT), slab, blocks=[Atoms("Cu")], counts=[4], seed=3)
    for _ in range(10):
        candidate = runner._generate_random()
        atoms = candidate.interface
        new = range(len(slab.substrate), len(atoms))
        distances = [
            atoms.get_distance(i, j, mic=True) for i in new for j in new if i < j
        ]
        assert min(distances) >= runner.blmin[(29, 29)]


@pytest.mark.parametrize("species", ["O", "Cu", "He", "CO"])
def test_delete_finds_the_chemical_potential_of_any_species(slab, species):
    block = (
        Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.15]])
        if species == "CO"
        else Atoms(species)
    )
    candidate = slab.copy()
    block.translate([3.0, 3.0, slab.interface.positions[:, 2].max() + 2.0])
    candidate.add_adsorbate(block)
    for move in (
        DeleteMove({species: -1.5}),
        DeleteMove({species: -1.5}, [block], [species]),
    ):
        _, log_ratio, info = move.propose(candidate, np.random.default_rng(0))
        assert info["chemical_potential"] == -1.5
        assert log_ratio == 0.0  # one group, one species


def test_insert_counts_groups_like_delete(slab):
    rng = np.random.default_rng(0)
    region = BoxRegion(slab.interface.get_cell(), z_min=12.0, z_max=14.0)
    for block, name in (
        (Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.15]]), "CO"),
        (Atoms("Cu"), "Cu"),
    ):
        inserted, log_insert, _ = InsertMove(
            [block], [name], {name: 0.0}, region
        ).propose(slab, rng)
        _, log_delete, _ = DeleteMove({name: 0.0}, [block], [name]).propose(
            inserted, rng
        )
        assert log_insert == pytest.approx(
            -log_delete
        )  # forward and reverse ratios cancel
        assert log_insert == pytest.approx(
            0.0
        )  # one group after the insertion, whatever its size


@pytest.mark.parametrize("species", ["He", "Cu", "CO"])
def test_ideal_gas_occupation_is_poisson(slab, species):
    """Non-interacting particles: mean and variance of N equal exp(mu / kT)."""
    mean, temperature, n_steps = 2.0, 300.0, 12000
    block = (
        Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.15]])
        if species == "CO"
        else Atoms(species)
    )
    cell = slab.interface.get_cell()
    runner = GCMCRunner(
        _Evaluator(_Zero),
        slab,
        blocks=[block],
        block_names=[species],
        chemical_potentials={species: KB * temperature * math.log(mean)},
        temperature=temperature,
        # Far above the slab, so that inserted particles are never grouped with it or with each other.
        region=BoxRegion(cell * 4, z_min=30.0, z_max=200.0, margin=0.0),
        move_weights={"insert": 0.5, "delete": 0.5, "displace": 0.0, "swap": 0.0},
        seed=0,
    )
    runner.setup()
    counts = []
    for step in range(n_steps):
        runner.step()
        runner._results.clear()
        if step >= n_steps // 10:
            counts.append(len(runner.current.adsList) + len(runner.current.clusterList))
    assert np.mean(counts) == pytest.approx(mean, abs=0.15)
    assert np.var(counts) == pytest.approx(mean, abs=0.3)


def test_box_region_samples_inside_a_hexagonal_cell():
    cell = np.array([[5.0, 0.0, 0.0], [2.5, 4.33, 0.0], [0.0, 0.0, 20.0]])
    region = BoxRegion(cell, z_min=10.0, z_max=12.0, margin=0.5)
    rng = np.random.default_rng(0)
    scaled = np.array(
        [np.linalg.solve(cell.T, region.random_position(rng)) for _ in range(200)]
    )
    assert scaled[:, :2].min() >= 0.0 and scaled[:, :2].max() <= 1.0
