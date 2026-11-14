"""Local benchmarks of the search methods on cheap potentials with known answers.

Three groups of jobs, each cached per run in ``benchmarks/results/<job>/<seed>.json``
so that an interrupted campaign resumes where it stopped:

- ``lj``: minima hopping on Lennard-Jones clusters (reference energies from the
  Cambridge Cluster Database, Wales and Doye, J. Phys. Chem. A 101, 5111 (1997)).
- ``surface``: GA and minima hopping for Cu adatoms on Cu(100) with EMT; the
  reference is the lowest relaxed structure of an exhaustive enumeration of
  hollow-site occupations.
- ``gcmc``: grand canonical Monte Carlo of non-interacting particles; the number
  of particles must follow a Poisson distribution with mean exp(mu / kT), which
  is what the insert and delete proposal ratios of the code imply.

Usage::

    uv run python benchmarks/run_local.py all --workers 12
    uv run python benchmarks/run_local.py summary

Energies in eV (Lennard-Jones: epsilon = 1 eV, sigma = 1 Å), lengths in Å.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc100
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones

from gm2_surf.evaluator.calculators import CalculatorEvaluator, register_calculator
from gm2_surf.interface.core import Interface
from gm2_surf.search.ga.runner import GARunner
from gm2_surf.search.gcmc.runner import GCMCRunner
from gm2_surf.search.minima_hopping import MinimaHoppingRunner
from gm2_surf.search.region import BoxRegion

RESULTS = Path(__file__).parent / "results"
KB = 8.617333262145e-5

# Global minima of Lennard-Jones clusters (epsilon units), Cambridge Cluster Database.
LJ_REFERENCE = {13: -44.326801, 19: -72.659782, 38: -173.928427}

CALLS = {"n": 0}


class ZeroCalculator(Calculator):
    """Non-interacting particles: zero energy and forces."""

    implemented_properties = ("energy", "forces")

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        """Set zero energy and forces."""
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(self.atoms), 3))}


def _counted(calculator: Calculator) -> Calculator:
    """Count every energy/force evaluation of ``calculator`` in ``CALLS``."""
    original = calculator.calculate

    def calculate(*args, **kwargs):
        CALLS["n"] += 1
        return original(*args, **kwargs)

    calculator.calculate = calculate
    return calculator


register_calculator(
    "bench_lj",
    lambda **_: _counted(LennardJones(sigma=1.0, epsilon=1.0, rc=1e3, smooth=False)),
)
register_calculator("bench_emt", lambda **_: _counted(EMT()))
register_calculator("bench_zero", lambda **_: _counted(ZeroCalculator()))


def _cached(job: str, key: str, compute) -> dict:
    """Result of ``compute()``, stored as ``results/<job>/<key>.json``."""
    path = RESULTS / job / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    started = time.time()
    CALLS["n"] = 0
    result = compute()
    result.update({"job": job, "key": key, "wall_s": round(time.time() - started, 1)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    return result


def _search(
    runner, best_energy, reference: float, tolerance: float, max_steps: int
) -> dict:
    """Step ``runner`` until its best energy reaches the reference; record the first hit."""
    runner.setup()
    trace = [(0, CALLS["n"], float(best_energy()))]
    hit = None
    for step in range(1, max_steps + 1):
        if trace[-1][2] <= reference + tolerance:
            hit = {"steps": trace[-1][0], "calls": trace[-1][1]}
            break
        try:
            runner.step()
        except Exception as error:  # noqa: BLE001 - a failing step is part of what is measured
            logging.getLogger(__name__).warning("step failed: %s", error)
        trace.append((step, CALLS["n"], float(best_energy())))
    else:
        if trace[-1][2] <= reference + tolerance:
            hit = {"steps": trace[-1][0], "calls": trace[-1][1]}
    return {
        "success": hit is not None,
        "hit": hit,
        "best": trace[-1][2],
        "reference": reference,
        "steps_run": trace[-1][0],
        "calls": CALLS["n"],
        "trace": trace[:: max(1, len(trace) // 200)],
    }


# ----------------------------------------------------------------------
# Lennard-Jones clusters
# ----------------------------------------------------------------------


def _random_cluster(n_atoms: int, rng: np.random.Generator) -> Atoms:
    radius = 0.75 * n_atoms ** (1 / 3) + 0.5
    positions: list[np.ndarray] = []
    while len(positions) < n_atoms:
        point = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(point) <= radius and all(
            np.linalg.norm(point - other) > 0.85 for other in positions
        ):
            positions.append(point)
    return Atoms(
        f"Ar{n_atoms}", positions=np.array(positions) + 20.0, cell=[40.0] * 3, pbc=False
    )


def run_lj(args: tuple[int, int, int]) -> dict:
    """Minima hopping on one random Lennard-Jones cluster."""
    n_atoms, seed, max_hops = args

    def compute() -> dict:
        rng = np.random.default_rng(seed)
        start = Interface.from_ase(_random_cluster(n_atoms, rng), n_substrate=0)
        evaluator = CalculatorEvaluator("bench_lj", fmax=0.01, relax_steps=1000)
        runner = MinimaHoppingRunner(
            evaluator,
            start,
            workdir=str(RESULTS / "work"),
            temperature=2000.0,
            timestep=2.0,
            seed=seed,
        )
        result = _search(
            runner, lambda: runner.best_energy, LJ_REFERENCE[n_atoms], 1e-3, max_hops
        )
        return {
            **result,
            "method": "minima_hopping",
            "system": f"LJ{n_atoms}",
            "seed": seed,
        }

    return _cached(f"lj{n_atoms}_minima_hopping", f"seed{seed}", compute)


# ----------------------------------------------------------------------
# Cu adatoms on Cu(100), EMT
# ----------------------------------------------------------------------

SIZE = 4


def _slab() -> Interface:
    slab = fcc100("Cu", (SIZE, SIZE, 3), a=3.61, vacuum=8.0)
    interface = Interface(substrate=slab)
    interface.fix_substrate(layers=2)
    return interface


def _canonical(sites: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    """Representative of an occupation pattern under the translations and point group of the square lattice."""
    images = []
    for swap, sign_i, sign_j in itertools.product((False, True), (1, -1), (1, -1)):
        mapped = [((j, i) if swap else (i, j)) for i, j in sites]
        mapped = [(sign_i * i, sign_j * j) for i, j in mapped]
        for shift_i, shift_j in itertools.product(range(SIZE), repeat=2):
            images.append(
                tuple(
                    sorted(
                        ((i + shift_i) % SIZE, (j + shift_j) % SIZE) for i, j in mapped
                    )
                )
            )
    return min(images)


def enumerate_adatoms(n_adatoms: int) -> dict:
    """Relax every symmetry-distinct hollow-site occupation; the lowest is the reference."""

    def compute() -> dict:
        patterns = sorted(
            {
                _canonical(sites)
                for sites in itertools.combinations(
                    itertools.product(range(SIZE), repeat=2), n_adatoms
                )
            }
        )
        evaluator = CalculatorEvaluator("bench_emt", fmax=0.02, relax_steps=500)
        energies = []
        for pattern in patterns:
            interface = _slab()
            atoms = interface.interface.copy()
            for i, j in pattern:
                add_adsorbate(atoms, "Cu", 1.8, "hollow", offset=(i, j))
            candidate = Interface.from_ase(atoms, n_substrate=len(interface.substrate))
            candidate.fix_substrate(layers=2)
            relaxed = evaluator.evaluate(candidate, relax=True)
            energies.append((float(relaxed.interface.get_potential_energy()), pattern))
        energies.sort()
        return {
            "n_patterns": len(patterns),
            "reference": energies[0][0],
            "pattern": energies[0][1],
            "gap_to_second": energies[1][0] - energies[0][0],
            "lowest": energies[:5],
        }

    return _cached("cu100_enumeration", f"cu{n_adatoms}", compute)


def run_surface(args: tuple[str, int, int, int]) -> dict:
    """GA or minima hopping for ``n_adatoms`` Cu atoms on Cu(100)."""
    method, n_adatoms, seed, max_steps = args

    def compute() -> dict:
        reference = enumerate_adatoms(n_adatoms)["reference"]
        evaluator = CalculatorEvaluator("bench_emt", fmax=0.05, relax_steps=200)
        workdir = str(RESULTS / "work" / f"{method}_{n_adatoms}_{seed}")
        ga = GARunner(
            evaluator,
            _slab(),
            blocks=[Atoms("Cu")],
            counts=[n_adatoms],
            workdir=workdir,
            population_size=10,
            n_initial=10,
            seed=seed,
        )
        if method == "ga":
            result = _search(
                ga, lambda: ga.population.best_energy(), reference, 0.01, max_steps
            )
        else:
            start = (
                ga._generate_random()
            )  # the same random start the GA draws its first candidate from
            runner = MinimaHoppingRunner(evaluator, start, workdir=workdir, seed=seed)
            result = _search(
                runner, lambda: runner.best_energy, reference, 0.01, max_steps
            )
        return {
            **result,
            "method": method,
            "system": f"Cu{n_adatoms}/Cu(100)",
            "seed": seed,
        }

    return _cached(f"cu{n_adatoms}_{method}", f"seed{seed}", compute)


# ----------------------------------------------------------------------
# GCMC of non-interacting particles
# ----------------------------------------------------------------------


def run_gcmc(args: tuple[str, float, int, int]) -> dict:
    """Particle-number distribution of an ideal gas against Poisson(exp(mu / kT))."""
    species, mean, n_steps, seed = args

    def compute() -> dict:
        temperature = 300.0
        block = {
            "He": Atoms("He"),
            "Cu": Atoms("Cu"),
            "CO": Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.15]]),
        }[species]
        runner = GCMCRunner(
            CalculatorEvaluator("bench_zero"),
            _slab(),
            blocks=[block],
            block_names=[species],
            chemical_potentials={species: KB * temperature * math.log(mean)},
            temperature=temperature,
            # Dilute and far above the slab: particles that come within bonding distance of each
            # other are merged into one group by the structure classification and stop being countable.
            region=BoxRegion(
                _slab().interface.get_cell() * 4, z_min=30.0, z_max=200.0, margin=0.0
            ),
            move_weights={"insert": 0.5, "delete": 0.5, "displace": 0.0, "swap": 0.0},
            workdir=str(RESULTS / "work"),
            seed=seed,
        )
        runner.setup()
        counts: dict[int, int] = {}
        for step in range(n_steps):
            runner.step()
            runner._results.clear()  # keep memory flat; only the particle number is sampled
            if step >= n_steps // 10:
                groups = len(runner.current.adsList) + len(runner.current.clusterList)
                counts[groups] = counts.get(groups, 0) + 1
        total = sum(counts.values())
        observed_mean = sum(n * c for n, c in counts.items()) / total
        variance = sum((n - observed_mean) ** 2 * c for n, c in counts.items()) / total
        poisson = {
            n: math.exp(-mean) * mean**n / math.factorial(n)
            for n in range(max(counts) + 1)
        }
        distance = 0.5 * sum(
            abs(counts.get(n, 0) / total - p) for n, p in poisson.items()
        )
        return {
            "method": "gcmc",
            "system": f"ideal {species}",
            "seed": seed,
            "expected_mean": mean,
            "mean": observed_mean,
            "variance": variance,
            "total_variation": distance,
            "histogram": {str(n): counts[n] / total for n in sorted(counts)},
            "poisson": {str(n): p for n, p in poisson.items()},
            "summary": runner.get_summary(),
        }

    return _cached(f"gcmc_ideal_{species}", f"seed{seed}", compute)


# ----------------------------------------------------------------------
# Campaign and summary
# ----------------------------------------------------------------------


def summary() -> list[dict]:
    """One row per job: success rate and cost to the first hit."""
    rows = []
    for directory in sorted(
        path for path in RESULTS.iterdir() if path.is_dir() and path.name != "work"
    ):
        runs = [
            json.loads(path.read_text())
            for path in sorted(directory.glob("seed*.json"))
        ]
        if not runs:
            continue
        if "expected_mean" in runs[0]:
            rows.extend(
                {
                    "job": directory.name,
                    "runs": 1,
                    "expected_mean": run["expected_mean"],
                    "mean": round(run["mean"], 3),
                    "variance": round(run["variance"], 3),
                    "total_variation": round(run["total_variation"], 3),
                }
                for run in runs
            )
            continue
        hits = [run["hit"] for run in runs if run["success"]]
        rows.append(
            {
                "job": directory.name,
                "runs": len(runs),
                "success": len(hits),
                "median_steps": float(np.median([hit["steps"] for hit in hits]))
                if hits
                else "",
                "median_calls": float(np.median([hit["calls"] for hit in hits]))
                if hits
                else "",
                "p90_calls": float(np.percentile([hit["calls"] for hit in hits], 90))
                if hits
                else "",
                "worst_gap_eV": round(
                    max(run["best"] - run["reference"] for run in runs), 4
                ),
                "median_wall_s": float(np.median([run["wall_s"] for run in runs])),
            }
        )
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (RESULTS / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    """Run the selected group of jobs and write ``results/summary.csv``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("group", choices=["lj", "surface", "gcmc", "all", "summary"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=12)
    options = parser.parse_args()
    seeds = range(options.seeds)
    with Pool(options.workers) as pool:
        if options.group in ("gcmc", "all"):
            jobs = [(species, 3.0, 40000, 0) for species in ("He", "Cu", "CO")]
            pool.map(run_gcmc, jobs, chunksize=1)
        if options.group in ("lj", "all"):
            jobs = [(13, seed, 300) for seed in seeds] + [
                (38, seed, 600) for seed in seeds
            ]
            pool.map(run_lj, jobs, chunksize=1)
        if options.group in ("surface", "all"):
            for n_adatoms in (4, 6):
                enumerate_adatoms(n_adatoms)
            jobs = [
                (method, n_adatoms, seed, 150 if method == "ga" else 80)
                for n_adatoms in (4, 6)
                for method in ("ga", "minima_hopping")
                for seed in seeds
            ]
            pool.map(run_surface, jobs, chunksize=1)
    for row in summary():
        print(row)


if __name__ == "__main__":
    main()
