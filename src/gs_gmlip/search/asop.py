"""Automated search for optimal surface phases (ASOP) in the grand canonical ensemble.

Follows the algorithm of Chen, Shang, and Liu (``Chen_JCP_2022``: D. Chen,
C. Shang, Z.-P. Liu, J. Chem. Phys. 156, 094104 (2022),
doi:10.1063/5.0084545). The search looks for the surface phase (periodicity
and composition of the species added to a clean surface) with the lowest
surface free energy at fixed chemical potentials:

1. **Composition grids.** All non-redundant surface supercells of the input
   cell are enumerated from integer transformation matrices (Eqs. 4-5); a
   supercell of scaled area ``A`` holds up to ``N_j = A * sites_per_cell``
   atoms per species at one monolayer. Each supercell carries a grid whose
   elements are the compositions ``(N_1, ..., N_k)``. Grids are visited from
   the coarsest (smallest ``A``) to the finest.
2. **Structure exploration.** For each selected composition an initial
   structure is grown by adding atoms one by one at random positions that are
   neither closer than ``min_distance`` nor farther than ``max_distance`` from
   the atoms already there, relaxing after each addition; a composition that
   was searched before restarts from its best structure. A short global
   search follows, and the best structure gives the surface free energy

   ``gamma = (E - A * E_clean - sum_i N_i mu_i) / A``            (Eq. 6)

   in eV per input surface cell.
3. **Monte Carlo selection.** Every composition of the next grid gets the
   energy ``eps_m = min_n gamma_n exp(-|q_m - q_n|^2 / (2 sigma^2))`` from all
   phases found so far (``q`` are the fractional compositions ``N_i / N_j``),
   and is selected with a Boltzmann-type probability that sharpens with the
   cycle number (``beta = 0.5 * n_cycle``, Eqs. 7-9). A grid is searched
   ``max_cycles`` times before the next one is opened.

Differences from the paper, which uses the LASP code:

- The fixed-composition global search is minima hopping
  (``MinimaHoppingRunner``), not stochastic surface walking.
- There is no retraining of the potential during the search (stage C of the
  paper); the evaluator is used as given.
- A composition may contain zero atoms of a species (the paper starts at
  one), so single-species adlayers are part of every grid.
- The selection probabilities are divided by their maximum, so the most
  favourable composition is always kept; Eq. 7 normalises them to a sum of
  one and the paper does not spell out the scaling used for the selection.
- For ``eps_0 >= 0`` (nothing more stable than the clean surface yet) Eq. 7
  would favour unstable compositions; ``|eps_0|`` is used so that lower
  ``eps`` is always more probable.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
from ase import Atom, Atoms
from ase.build import make_supercell
from ase.constraints import FixAtoms
from ase.geometry import get_distances

from gs_gmlip.evaluator.base import BaseEvaluator
from gs_gmlip.interface.core import Interface
from gs_gmlip.search.base import BaseSearcher
from gs_gmlip.search.minima_hopping import MinimaHoppingRunner

logger = logging.getLogger(__name__)

_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Supercell:
    """A surface periodicity.

    Attributes:
        matrix: Integer transformation ``((a1, b1), (a2, b2))`` of the input
            in-plane vectors: ``u' = a1 u + b1 v``, ``v' = a2 u + b2 v``, with
            a positive determinant.
        area: Scaled area ``A = det(matrix)``, in input surface cells.
        lengths: Lengths of the reduced cell vectors in Å, longer first.
        angle: Angle between ``u'`` and ``v'`` in degrees.
    """

    matrix: tuple[tuple[int, int], tuple[int, int]]
    area: int
    lengths: tuple[float, float]
    angle: float

    @property
    def label(self) -> str:
        """Short name such as ``A4 [2 0; -1 2]``."""
        (a1, b1), (a2, b2) = self.matrix
        return f"A{self.area} [{a1} {b1}; {a2} {b2}]"


def _reduce(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lagrange-Gauss reduction of an integer basis ``(rows of TM)`` in the metric of the surface cell.

    Each argument is ``(integer coefficients, Cartesian vector)`` flattened to
    one array of length four, so both representations are transformed together.
    """
    while True:
        if np.dot(first[2:], first[2:]) < np.dot(second[2:], second[2:]):
            first, second = second, first
        shift = round(
            float(np.dot(first[2:], second[2:]) / np.dot(second[2:], second[2:]))
        )
        if shift == 0:
            return first, second
        first = first - shift * second


def enumerate_supercells(
    cell: np.ndarray,
    min_area: int = 1,
    max_area: int = 4,
    max_ratio: float = 3.0,
) -> list[Supercell]:
    """Non-redundant surface supercells, coarsest first.

    Every sublattice of scaled area ``A`` is generated once from its Hermite
    normal form and brought to its reduced cell: ``|u'| >= |v'|`` and an angle
    between 60° and 90°, the conditions of Eqs. 4-5 of the paper. Cells with
    ``|u'| / |v'| >= max_ratio`` are dropped as too narrow, and sublattices
    with the same reduced lengths and angle are counted once.

    Args:
        cell: Cell of the input surface; the first two rows are ``u`` and ``v``.
        min_area: Smallest scaled area.
        max_area: Largest scaled area.
        max_ratio: Upper bound (exclusive) on ``|u'| / |v'|``.
    """
    u, v = np.asarray(cell[0], dtype=float)[:2], np.asarray(cell[1], dtype=float)[:2]
    found: dict[tuple[int, float, float, float], Supercell] = {}
    for area in range(int(min_area), int(max_area) + 1):
        for a in (n for n in range(1, area + 1) if area % n == 0):
            d = area // a
            for b in range(d):
                rows = [np.array([a, b, *(a * u + b * v)]), np.array([0, d, *(d * v)])]
                first, second = _reduce(*rows)
                if np.dot(first[2:], second[2:]) < 0:
                    second = -second  # angle between 60 and 90 degrees
                if first[0] * second[1] - first[1] * second[0] < 0:
                    first, second = second, first  # right-handed
                if first[:2].sum() + second[:2].sum() < 0:
                    first, second = -first, -second  # same cell, more readable label
                lengths = sorted(
                    (
                        float(np.linalg.norm(first[2:])),
                        float(np.linalg.norm(second[2:])),
                    ),
                    reverse=True,
                )
                if lengths[0] / lengths[1] >= max_ratio - _TOLERANCE:
                    continue
                cosine = float(np.dot(first[2:], second[2:])) / (
                    lengths[0] * lengths[1]
                )
                angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
                matrix = (
                    (int(first[0]), int(first[1])),
                    (int(second[0]), int(second[1])),
                )
                key = (
                    area,
                    round(lengths[0], 3),
                    round(lengths[1], 3),
                    round(angle, 2),
                )
                candidate = Supercell(matrix, area, (lengths[0], lengths[1]), angle)
                if key not in found or _complexity(candidate) < _complexity(found[key]):
                    found[key] = candidate
    return sorted(
        found.values(), key=lambda item: (item.area, -item.angle, item.lengths)
    )


def _complexity(supercell: Supercell) -> tuple[int, int, tuple[int, ...]]:
    flat = [value for row in supercell.matrix for value in row]
    return (
        sum(abs(value) for value in flat),
        sum(value < 0 for value in flat),
        tuple(-value for value in flat),
    )


@dataclass
class Phase:
    """Best structure found for one composition in one supercell.

    Attributes:
        grid: Index of the supercell in ``ASOPRunner.grids``.
        counts: Atoms added per species, in the order of ``ASOPRunner.species``.
        gamma: Surface free energy in eV per input surface cell.
        structure: The structure.
        fraction: Fractional composition ``N_i / N_j``.
        visits: Number of searches spent on this phase.
    """

    grid: int
    counts: tuple[int, ...]
    gamma: float
    structure: Interface
    fraction: tuple[float, ...]
    visits: int = 1


@dataclass
class _Grid:
    supercell: Supercell
    slab: Atoms
    capacity: int
    elements: list[tuple[int, ...]] = field(default_factory=list)


class ASOPRunner(BaseSearcher):
    """Grand-canonical search over surface periodicity and composition.

    Args:
        evaluator: Energy evaluator.
        surface: Clean surface in the smallest cell of interest, ideally the
            primitive surface cell; it defines the unit of area and coverage.
        chemical_potentials: Species (element symbol) -> chemical potential in
            eV. The keys are the species that are added to the surface.
        workdir: Working directory.
        min_area: Smallest supercell, in input surface cells.
        max_area: Largest supercell, in input surface cells.
        max_coverage: Largest coverage of each species in monolayers.
        sites_per_cell: Atoms per monolayer in the input cell; the number of
            atoms in one substrate layer if omitted.
        max_cycles: Searches of each grid before moving to the next (3 in the paper).
        max_compositions: Upper limit of compositions searched per cycle (15 in the paper).
        search_steps: Minima-hopping steps per composition; 0 only relaxes.
        sigma: Width of the Gaussian in composition space (Eq. 9).
        min_distance: A new atom is not placed closer than this to any atom (Å).
        max_distance: A new atom is not placed farther than this from its nearest atom (Å).
        relax_additions: Relax after every added atom (as in the paper)
            instead of only once at the end.
        max_ratio: Upper bound on the aspect ratio of supercells.
        hopping: Further keyword arguments for ``MinimaHoppingRunner``.
        seed: Random seed.
    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        surface: Interface,
        chemical_potentials: dict[str, float],
        workdir: str = "asop",
        min_area: int = 1,
        max_area: int = 4,
        max_coverage: float = 1.0,
        sites_per_cell: int | None = None,
        max_cycles: int = 3,
        max_compositions: int = 15,
        search_steps: int = 5,
        sigma: float = 0.1,
        min_distance: float = 1.5,
        max_distance: float = 3.0,
        relax_additions: bool = True,
        max_ratio: float = 3.0,
        hopping: dict | None = None,
        seed: int | None = None,
    ):
        super().__init__(evaluator, workdir, seed)
        if not chemical_potentials:
            raise ValueError(
                "ASOP needs at least one species with a chemical potential"
            )
        if not 1 <= min_area <= max_area:
            raise ValueError("need 1 <= min_area <= max_area")
        if len(surface.interface) != len(surface.substrate):
            raise ValueError(
                "ASOP starts from a clean surface; remove the adsorbates first"
            )
        self.surface = surface
        self.species = list(chemical_potentials)
        self.chemical_potentials = {
            name: float(mu) for name, mu in chemical_potentials.items()
        }
        self.min_area, self.max_area = int(min_area), int(max_area)
        self.max_coverage = float(max_coverage)
        self.sites_per_cell = sites_per_cell
        self.max_cycles = int(max_cycles)
        self.max_compositions = int(max_compositions)
        self.search_steps = int(search_steps)
        self.sigma = float(sigma)
        self.min_distance, self.max_distance = float(min_distance), float(max_distance)
        self.relax_additions = bool(relax_additions)
        self.max_ratio = float(max_ratio)
        self.hopping = dict(hopping or {})
        self.rng = np.random.default_rng(seed)

        self.clean_energy = float("nan")
        self.grids: list[_Grid] = []
        self.phases: dict[tuple[int, tuple[int, ...]], Phase] = {}
        self.grid_index = 0
        self.cycle = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # BaseSearcher interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Relax the clean surface and generate the composition grids (step 1)."""
        clean = self.evaluator.evaluate(self.surface.copy(), relax=True)
        self.clean_energy = float(clean.interface.get_potential_energy())
        unit = clean.interface.copy()
        fixed = sorted(
            {
                int(i)
                for c in unit.constraints
                if isinstance(c, FixAtoms)
                for i in c.index
            }
        )
        fixed_top = max(unit.positions[fixed, 2]) if fixed else None
        unit.set_constraint()
        unit.calc = None
        if self.sites_per_cell is None:
            self.sites_per_cell = max(
                1, round(len(unit) / max(1, clean.get_substrate_layers()))
            )

        self.grids = []
        for supercell in enumerate_supercells(
            unit.cell.array, self.min_area, self.max_area, self.max_ratio
        ):
            (a1, b1), (a2, b2) = supercell.matrix
            slab = make_supercell(unit, [[a1, b1, 0], [a2, b2, 0], [0, 0, 1]])
            # Keep the package convention: first in-plane vector along x.
            slab.rotate(
                -np.degrees(np.arctan2(slab.cell[0, 1], slab.cell[0, 0])),
                "z",
                rotate_cell=True,
            )
            slab.wrap()
            if fixed_top is not None:
                slab.set_constraint(
                    FixAtoms(mask=slab.positions[:, 2] <= fixed_top + 1e-3)
                )
            capacity = supercell.area * int(self.sites_per_cell)
            largest = int(np.floor(self.max_coverage * capacity + _TOLERANCE))
            elements = [
                counts
                for counts in itertools.product(
                    range(largest + 1), repeat=len(self.species)
                )
                if sum(counts) > 0
            ]
            if elements:
                self.grids.append(_Grid(supercell, slab, capacity, elements))
        if not self.grids:
            raise ValueError("no composition fits: increase max_area or max_coverage")
        self.grid_index, self.cycle = 0, 0
        self._results.append(clean)
        self._is_setup = True

    @property
    def n_cycles(self) -> int:
        """Total number of search cycles (calls of ``step``) of a complete run."""
        return len(self.grids) * self.max_cycles

    @property
    def finished(self) -> bool:
        """Whether every grid has been searched ``max_cycles`` times."""
        return self._is_setup and self.grid_index >= len(self.grids)

    def step(self) -> dict:
        """One cycle: select compositions of the current grid and search each (steps 2 and 3)."""
        if not self._is_setup:
            return {"status": "not_initialized"}
        if self.finished:
            return {"status": "finished"}
        index = self.grid_index
        grid = self.grids[index]
        selected = self._select(index)
        searched = []
        for counts in selected:
            try:
                phase = self._search(index, counts)
            except Exception:  # noqa: BLE001 - one composition that cannot be built must not stop the search
                logger.warning(
                    "ASOP: composition %s in %s failed",
                    counts,
                    grid.supercell.label,
                    exc_info=True,
                )
                continue
            searched.append(phase)
        info = {
            "status": "searched",
            "grid": grid.supercell.label,
            "cycle": self.cycle + 1,
            "compositions": [self.formula(counts) for counts in selected],
            "best_gamma": min(
                (phase.gamma for phase in self.phases.values()), default=float("nan")
            ),
        }
        self.history.append(info)
        self.cycle += 1
        if self.cycle >= self.max_cycles:
            self.grid_index, self.cycle = index + 1, 0
        return info

    def get_best(self, n: int = 1) -> list[Interface]:
        """Structures of the ``n`` phases with the lowest surface free energy."""
        return [phase.structure for phase in self.best_phases(n)]

    # ------------------------------------------------------------------
    # Thermodynamics
    # ------------------------------------------------------------------

    def formula(self, counts: tuple[int, ...]) -> str:
        """Added atoms as a formula, e.g. ``Ag1O2``."""
        return "".join(
            f"{name}{count}"
            for name, count in zip(self.species, counts, strict=True)
            if count
        )

    def added_counts(self, interface: Interface) -> tuple[int, ...]:
        """Atoms of each species added on top of the substrate."""
        symbols = interface.interface.get_chemical_symbols()[len(interface.substrate) :]
        return tuple(symbols.count(name) for name in self.species)

    def surface_free_energy(self, interface: Interface) -> float:
        """Surface free energy of formation (Eq. 6) in eV per input surface cell.

        The clean surface scores 0; negative values are more stable than the
        clean surface at the given chemical potentials.
        """
        area = len(interface.substrate) / len(self.surface.substrate)
        energy = float(interface.interface.get_potential_energy())
        if np.isnan(self.clean_energy):  # the clean surface itself, during setup
            return 0.0
        exchange = sum(
            count * self.chemical_potentials[name]
            for name, count in zip(
                self.species, self.added_counts(interface), strict=True
            )
        )
        return (energy - area * self.clean_energy - exchange) / area

    def best_phases(self, n: int | None = None) -> list[Phase]:
        """Phases sorted by surface free energy, most stable first."""
        ranked = sorted(self.phases.values(), key=lambda phase: phase.gamma)
        return ranked if n is None else ranked[:n]

    def phase_table(self) -> list[dict]:
        """The phase database as plain rows, most stable first."""
        return [
            {
                "supercell": self.grids[phase.grid].supercell.label,
                "area": self.grids[phase.grid].supercell.area,
                "composition": self.formula(phase.counts),
                **dict(zip(self.species, phase.counts, strict=True)),
                "gamma": phase.gamma,
                "visits": phase.visits,
            }
            for phase in self.best_phases()
        ]

    # ------------------------------------------------------------------
    # Step 3: Monte Carlo selection
    # ------------------------------------------------------------------

    def probabilities(self, index: int, beta: float) -> np.ndarray:
        """Selection probability of every composition of grid ``index`` (Eqs. 7-9), summing to one."""
        grid = self.grids[index]
        if not self.phases or beta == 0:
            return np.full(len(grid.elements), 1.0 / len(grid.elements))
        targets = np.array(grid.elements, dtype=float) / grid.capacity
        known = np.array([phase.fraction for phase in self.phases.values()])
        gammas = np.array([phase.gamma for phase in self.phases.values()])
        distance2 = ((targets[:, None, :] - known[None, :, :]) ** 2).sum(axis=2)
        epsilon = (gammas[None, :] * np.exp(-distance2 / (2 * self.sigma**2))).min(
            axis=1
        )
        lowest = epsilon.min()
        scale = max(abs(lowest), 1e-3)
        weights = np.exp(-beta * (epsilon - lowest) / scale)
        return weights / weights.sum()

    def _select(self, index: int) -> list[tuple[int, ...]]:
        grid = self.grids[index]
        # beta = 0.5 * n_cycle, counted over the cycles already done on the grid the
        # probabilities come from; the first cycle of the first grid is uniform.
        done = self.cycle if self.cycle else (self.max_cycles if index else 0)
        probability = self.probabilities(index, 0.5 * done)
        relative = probability / probability.max()
        order = self.rng.permutation(len(grid.elements))
        chosen = [i for i in order if self.rng.random() < relative[i]]
        chosen.sort(key=lambda i: -relative[i])  # stable: ties stay in random order
        return [grid.elements[i] for i in chosen[: self.max_compositions]]

    # ------------------------------------------------------------------
    # Step 2: structure exploration
    # ------------------------------------------------------------------

    def _search(self, index: int, counts: tuple[int, ...]) -> Phase:
        grid = self.grids[index]
        key = (index, counts)
        previous = self.phases.get(key)
        if previous is not None:
            start = previous.structure.copy()
        else:
            start = self._grow(grid, counts)
        if self.search_steps > 0:
            hopper = MinimaHoppingRunner(
                self.evaluator,
                start,
                workdir=self.workdir,
                seed=int(self.rng.integers(2**31)),
                **self.hopping,
            )
            hopper.setup()
            for _ in range(self.search_steps):
                hopper.step()
            best = hopper.best
        else:
            best = self.evaluator.evaluate(start, relax=True)
        gamma = self.surface_free_energy(best)
        self._results.append(best)
        if previous is None or gamma < previous.gamma:
            fraction = tuple(count / grid.capacity for count in counts)
            visits = 1 if previous is None else previous.visits + 1
            self.phases[key] = Phase(index, counts, gamma, best, fraction, visits)
        else:
            previous.visits += 1
        return self.phases[key]

    def _grow(self, grid: _Grid, counts: tuple[int, ...]) -> Interface:
        """Stage A: add the atoms one by one at random valid positions."""
        atoms = grid.slab.copy()
        n_substrate = len(atoms)
        symbols = [
            name
            for name, count in zip(self.species, counts, strict=True)
            for _ in range(count)
        ]
        self.rng.shuffle(symbols)
        for position, symbol in enumerate(symbols, start=1):
            atoms.append(Atom(symbol, self._free_position(atoms)))
            if self.relax_additions and position < len(symbols):
                relaxed = self.evaluator.evaluate(
                    Interface.from_ase(atoms, n_substrate), relax=True
                )
                atoms = relaxed.interface.copy()
                atoms.calc = None
        return Interface.from_ase(atoms, n_substrate)

    def _free_position(self, atoms: Atoms, attempts: int = 500) -> np.ndarray:
        top = atoms.positions[:, 2].max()
        for _ in range(attempts):
            fractional = np.array([self.rng.random(), self.rng.random(), 0.0])
            trial = fractional @ atoms.cell.array
            trial[2] = self.rng.uniform(top - 1.0, top + self.max_distance)
            _, distances = get_distances(
                trial, atoms.positions, cell=atoms.cell, pbc=atoms.pbc
            )
            nearest = float(distances.min())
            if self.min_distance <= nearest <= self.max_distance:
                return trial
        raise RuntimeError(
            f"no position between {self.min_distance} and {self.max_distance} Å from the surface atoms "
            f"after {attempts} attempts; the cell is full"
        )
