"""Derived energies from labelled structures: formation, surface, adsorption, grand potential.

Every quantity here is a difference of total energies, so it is only
meaningful when all energies involved share one level of theory (same code,
functional, cutoff, k-point density; or the same foundation potential). The
pure functions take numbers; ``ReferenceSet`` finds matching reference
energies in one or more structure stores by role and level.

Units: eV, Å. Surface energies are returned in eV/Å² (multiply by
``EV_PER_A2_TO_J_PER_M2`` for J/m²).

Conventions:

- Bulk reference: energy per atom of the lowest-energy stored bulk of that
  element (elemental crystals only).
- Gas references follow the computational hydrogen electrode: chemical
  potentials of H, O, and C are built from H2, H2O, and CO,
  ``mu_H = E(H2)/2``, ``mu_O = E(H2O) - E(H2)``, ``mu_C = E(CO) - mu_O``.
  These are electronic energies; add zero-point, thermal, and potential
  corrections yourself where needed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from gs_gmlip.interface.core import Interface
from gs_gmlip.store import StructureRecord, StructureStore

EV_PER_A2_TO_J_PER_M2 = 16.02176634


def formation_energy(
    energy: float,
    composition: Mapping[str, int],
    chemical_potentials: Mapping[str, float],
) -> float:
    """Energy relative to elemental references: ``E - sum_i n_i mu_i``.

    Args:
        energy: Total energy of the structure (eV).
        composition: Element symbol -> number of atoms.
        chemical_potentials: Element symbol -> reference energy per atom (eV).

    Raises:
        KeyError: If an element has no chemical potential.
    """
    missing = [symbol for symbol in composition if symbol not in chemical_potentials]
    if missing:
        raise KeyError(f"no chemical potential for {sorted(missing)}")
    return float(energy) - sum(
        count * chemical_potentials[el] for el, count in composition.items()
    )


def surface_energy(
    slab_energy: float,
    composition: Mapping[str, int],
    bulk_energies_per_atom: Mapping[str, float],
    area: float,
    n_surfaces: int = 2,
) -> float:
    """Surface energy of a slab, ``(E_slab - sum_i n_i E_bulk,i) / (n_surfaces * A)`` in eV/Å².

    Valid for elemental slabs and for stoichiometric compounds whose
    per-element bulk references are well defined.

    Args:
        slab_energy: Total energy of the slab (eV).
        composition: Element symbol -> number of atoms in the slab.
        bulk_energies_per_atom: Element symbol -> bulk energy per atom (eV).
        area: In-plane cell area (Å²).
        n_surfaces: Number of free surfaces (2 for a symmetric slab in vacuum).
    """
    excess = formation_energy(slab_energy, composition, bulk_energies_per_atom)
    return excess / (n_surfaces * float(area))


def adsorption_energy(
    total_energy: float, slab_energy: float, adsorbate_reference: float
) -> float:
    """Adsorption energy ``E(slab + adsorbates) - E(slab) - E_ref(adsorbates)`` in eV.

    Negative values mean binding relative to the reference.
    """
    return float(total_energy) - float(slab_energy) - float(adsorbate_reference)


def grand_potential(
    energy: float,
    counts: Mapping[str, int],
    chemical_potentials: Mapping[str, float],
    charge: float = 0.0,
    electron_potential: float = 0.0,
) -> float:
    """Grand-canonical energy ``E - sum_i mu_i N_i + q * Phi`` in eV.

    Args:
        energy: Total energy (eV).
        counts: Species -> number exchanged with the reservoirs.
        chemical_potentials: Species -> chemical potential (eV).
        charge: Net electronic charge of the cell in units of e (excess
            electrons count as negative charge), for constant-potential DFT.
        electron_potential: Electron chemical potential expressed as the work
            function / absolute electrode potential (eV per e).
    """
    exchange = sum(chemical_potentials[name] * count for name, count in counts.items())
    return float(energy) - exchange + float(charge) * float(electron_potential)


def che_chemical_potentials(
    e_h2: float, e_h2o: float, e_co: float | None = None
) -> dict[str, float]:
    """Chemical potentials of H, O (and C) from H2, H2O (and CO) total energies."""
    mu_h = e_h2 / 2.0
    mu_o = e_h2o - e_h2
    potentials = {"H": mu_h, "O": mu_o}
    if e_co is not None:
        potentials["C"] = e_co - mu_o
    return potentials


def composition_of(symbols: Iterable[str]) -> dict[str, int]:
    """Element symbol -> count."""
    return dict(Counter(symbols))


@dataclass
class Energetics:
    """What could be derived for one structure, and what was missing.

    Attributes:
        role: Role label of the structure.
        level: Level of theory of its energy.
        energy: Total energy (eV), if evaluated.
        energy_per_atom: ``energy / natoms`` (eV).
        values: Derived quantities, name -> value.
        references: Reference used for each ingredient, name -> description.
        missing: Human-readable reasons for quantities that could not be computed.
    """

    role: str = ""
    level: str = ""
    energy: float | None = None
    energy_per_atom: float | None = None
    values: dict[str, float] = field(default_factory=dict)
    references: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "role": self.role,
            "level": self.level,
            "energy": self.energy,
            "energy_per_atom": self.energy_per_atom,
            "values": self.values,
            "references": self.references,
            "missing": self.missing,
        }


class ReferenceSet:
    """Reference energies at one level of theory, drawn from structure stores.

    Args:
        stores: Stores to search, in priority order (e.g. project, then global).
        level: Level-of-theory label the references must carry.
    """

    def __init__(self, stores: Iterable[tuple[str, StructureStore]], level: str):
        self.stores = list(stores)
        self.level = level
        self._cache: dict[str, list[tuple[str, StructureRecord]]] = {}

    def _records(self, role: str) -> list[tuple[str, StructureRecord]]:
        if role not in self._cache:
            self._cache[role] = [
                (name, rec)
                for name, store in self.stores
                for rec in store.records(role=role, level=self.level)
                if rec.energy is not None
            ]
        return self._cache[role]

    @staticmethod
    def _describe(source: str, rec: StructureRecord) -> str:
        return f"{source} #{rec.id} {rec.formula}"

    def bulk_energy_per_atom(self, element: str) -> tuple[float, str] | None:
        """Lowest energy per atom among stored elemental bulks of ``element``."""
        best: tuple[float, str] | None = None
        for source, rec in self._records("bulk"):
            store = dict(self.stores)[source]
            symbols = set(store.get(rec.id).interface.get_chemical_symbols())
            if symbols != {element}:
                continue
            per_atom = float(rec.energy) / rec.natoms
            if best is None or per_atom < best[0]:
                best = (per_atom, self._describe(source, rec))
        return best

    def molecule_energy(
        self, formula_counts: Mapping[str, int]
    ) -> tuple[float, str] | None:
        """Lowest energy among stored molecules with exactly this composition."""
        best: tuple[float, str] | None = None
        for source, rec in self._records("molecule"):
            store = dict(self.stores)[source]
            symbols = store.get(rec.id).interface.get_chemical_symbols()
            if composition_of(symbols) != dict(formula_counts):
                continue
            if best is None or float(rec.energy) < best[0]:
                best = (float(rec.energy), self._describe(source, rec))
        return best

    def clean_surface(self, interface: Interface) -> tuple[float, str] | None:
        """Lowest-energy stored clean slab matching the substrate of ``interface``.

        A match has the same substrate composition, the same number of atoms,
        and the same in-plane cell.
        """
        substrate = interface.substrate
        target = composition_of(substrate.get_chemical_symbols())
        cell = np.asarray(substrate.cell)[:2]
        best: tuple[float, str] | None = None
        for source, rec in self._records("surface"):
            if rec.natoms != len(substrate):
                continue
            candidate = dict(self.stores)[source].get(rec.id).interface
            if composition_of(candidate.get_chemical_symbols()) != target:
                continue
            if not np.allclose(np.asarray(candidate.cell)[:2], cell, atol=1e-3):
                continue
            if best is None or float(rec.energy) < best[0]:
                best = (float(rec.energy), self._describe(source, rec))
        return best

    def che_potentials(self) -> tuple[dict[str, float], dict[str, str], list[str]]:
        """CHE chemical potentials available at this level.

        Returns:
            ``(potentials, references used, missing molecules)``.
        """
        found: dict[str, tuple[float, str]] = {}
        missing = []
        for name, counts in (
            ("H2", {"H": 2}),
            ("H2O", {"H": 2, "O": 1}),
            ("CO", {"C": 1, "O": 1}),
        ):
            hit = self.molecule_energy(counts)
            if hit is None:
                missing.append(name)
            else:
                found[name] = hit
        potentials: dict[str, float] = {}
        if "H2" in found:
            potentials["H"] = found["H2"][0] / 2.0
        if "H2" in found and "H2O" in found:
            potentials["O"] = found["H2O"][0] - found["H2"][0]
        if "O" in potentials and "CO" in found:
            potentials["C"] = found["CO"][0] - potentials["O"]
        return (
            potentials,
            {name: description for name, (_, description) in found.items()},
            missing,
        )


def analyze(
    interface: Interface,
    role: str,
    level: str,
    stores: Iterable[tuple[str, StructureStore]],
    energy: float | None = None,
) -> Energetics:
    """Derive every energy the available references allow for one structure.

    Args:
        interface: The structure.
        role: Its role label.
        level: Level of theory of its energy.
        stores: Stores to take references from, in priority order.
        energy: Its total energy; taken from the attached calculator if omitted.

    Returns:
        The derived quantities with the references used, and a list of what
        is missing for the rest.
    """
    atoms = interface.interface
    if energy is None and atoms.calc is not None:
        try:
            energy = float(atoms.get_potential_energy())
        except Exception:  # noqa: BLE001 - no usable energy
            energy = None
    result = Energetics(role=role, level=level, energy=energy)
    if energy is None:
        result.missing.append(
            "the structure has no energy; run a single point or relaxation"
        )
        return result
    result.energy_per_atom = energy / len(atoms)
    if not level:
        result.missing.append(
            "the structure has no level-of-theory label, so references cannot be matched"
        )
        return result
    if not role:
        result.missing.append(
            "the structure has no role label (bulk, molecule, surface, adsorbed)"
        )
        return result

    references = ReferenceSet(stores, level)
    composition = composition_of(atoms.get_chemical_symbols())

    def bulk_references(elements: Iterable[str]) -> dict[str, float] | None:
        energies = {}
        for element in elements:
            hit = references.bulk_energy_per_atom(element)
            if hit is None:
                result.missing.append(
                    f"no bulk reference for {element} at level {level}"
                )
                return None
            energies[element] = hit[0]
            result.references[f"bulk {element}"] = hit[1]
        return energies

    if role == "bulk" and len(composition) > 1:
        bulks = bulk_references(composition)
        if bulks is not None:
            result.values["formation energy per atom (eV)"] = formation_energy(
                energy, composition, bulks
            ) / len(atoms)

    if role == "surface":
        bulks = bulk_references(composition)
        if bulks is not None:
            cell = np.asarray(atoms.cell)
            area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
            gamma = surface_energy(energy, composition, bulks, area)
            result.values["surface energy (eV/Å²)"] = gamma
            result.values["surface energy (J/m²)"] = gamma * EV_PER_A2_TO_J_PER_M2

    if role == "adsorbed":
        slab = references.clean_surface(interface)
        if slab is None:
            result.missing.append(
                f"no clean-surface reference with this substrate and cell at level {level}"
            )
        symbols = atoms.get_chemical_symbols()
        appended = [
            i for group in interface.adsList + interface.clusterList for i in group
        ]
        adsorbate_composition = composition_of(symbols[i] for i in appended)
        potentials, used, missing_molecules = references.che_potentials()
        if slab is not None:
            result.references["clean surface"] = slab[1]
            uncovered = [el for el in adsorbate_composition if el not in potentials]
            if uncovered:
                result.missing.append(
                    f"no gas reference for {sorted(uncovered)} at level {level} "
                    f"(needs molecules: {', '.join(missing_molecules) or 'H2, H2O, CO'})"
                )
            else:
                result.references.update(
                    {f"gas {name}": text for name, text in used.items()}
                )
                reference = sum(
                    count * potentials[el]
                    for el, count in adsorbate_composition.items()
                )
                result.values["adsorbate formation energy vs H2, H2O, CO (eV)"] = (
                    adsorption_energy(energy, slab[0], reference)
                )
            # Direct adsorption energy when every adsorbate has its own gas-phase molecule.
            direct = 0.0
            for group in interface.adsList:
                hit = references.molecule_energy(
                    composition_of(symbols[i] for i in group)
                )
                if hit is None:
                    direct = None
                    break
                direct += hit[0]
            if direct is not None and interface.adsList and not interface.clusterList:
                result.values["adsorption energy vs gas molecules (eV)"] = (
                    adsorption_energy(energy, slab[0], direct)
                )
    return result
