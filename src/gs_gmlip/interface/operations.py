"""Structure-building and editing operations for surface systems.

All geometry is done with pymatgen (``Structure``, ``Molecule``, ``SymmOp``,
``SlabGenerator``, ``AdsorbateSiteFinder``); ASE ``Atoms`` only appear at the
``Interface`` boundary. Every editing function is pure: it returns a new
``Interface`` and leaves its input untouched, which is what makes undo in the
interactive builder trivial.

Units: Å for lengths, degrees for angles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.build import molecule as ase_molecule
from ase.constraints import FixAtoms
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.io.ase import AseAtomsAdaptor

from gs_gmlip.interface.builders import primitive_slab_from_bulk, slab_from_bulk
from gs_gmlip.interface.core import Interface

# Lattice type -> (space group, fractional coordinates of the basis).
_BULK_PROTOTYPES: dict[str, tuple[str, list[list[float]]]] = {
    "fcc": ("Fm-3m", [[0.0, 0.0, 0.0]]),
    "bcc": ("Im-3m", [[0.0, 0.0, 0.0]]),
    "sc": ("Pm-3m", [[0.0, 0.0, 0.0]]),
    "diamond": ("Fd-3m", [[0.0, 0.0, 0.0]]),
    "hcp": ("P6_3/mmc", [[1 / 3, 2 / 3, 0.25]]),
}

BULK_LATTICES: tuple[str, ...] = tuple(_BULK_PROTOTYPES)

# Adsorbate name -> (ase.build.molecule name, index of the surface-binding atom).
_ADSORBATES: dict[str, tuple[str, int]] = {
    "H": ("H", 0),
    "C": ("C", 0),
    "N": ("N", 0),
    "O": ("O", 0),
    "S": ("S", 0),
    "OH": ("OH", 0),
    "CO": ("CO", 1),
    "NO": ("NO", 0),
    "H2": ("H2", 0),
    "N2": ("N2", 0),
    "O2": ("O2", 0),
    "H2O": ("H2O", 0),
    "NH3": ("NH3", 0),
    "CH3": ("CH3", 0),
    "CH4": ("CH4", 0),
    "CO2": ("CO2", 0),
    "HCO": ("HCO", 0),
    "CH3OH": ("CH3OH", 1),
}

ADSORBATES: tuple[str, ...] = tuple(_ADSORBATES)


@dataclass(frozen=True)
class AdsorptionSite:
    """One adsorption site above a slab.

    Attributes:
        kind: ``"ontop"``, ``"bridge"``, or ``"hollow"``.
        coords: Cartesian position (Å) of the binding atom, already lifted
            ``distance`` above the surface along the surface normal.
    """

    kind: str
    coords: tuple[float, float, float]


# ---------------------------------------------------------------------------
# Bulk and slab construction
# ---------------------------------------------------------------------------


def make_bulk(
    element: str, lattice: str, a: float, c: float | None = None
) -> Structure:
    """Build a conventional bulk cell of an elemental crystal.

    Args:
        element: Element symbol, e.g. ``"Cu"``.
        lattice: One of ``BULK_LATTICES``.
        a: Lattice constant in Å.
        c: Out-of-plane lattice constant in Å (hcp only; defaults to the ideal
            ratio ``a * sqrt(8/3)``).

    Returns:
        The conventional cell as a pymatgen ``Structure``.
    """
    if lattice not in _BULK_PROTOTYPES:
        raise ValueError(f"unknown lattice {lattice!r}; choose from {BULK_LATTICES}")
    spacegroup, basis = _BULK_PROTOTYPES[lattice]
    if lattice == "hcp":
        cell = Lattice.hexagonal(a, c if c else a * np.sqrt(8.0 / 3.0))
    else:
        cell = Lattice.cubic(a)
    return Structure.from_spacegroup(spacegroup, cell, [element], basis)


def load_structure(path: str | Path) -> Structure:
    """Read a periodic structure from any format pymatgen understands."""
    return Structure.from_file(str(path))


def orient_slab(structure: Structure) -> Structure:
    """Rotate a slab so that a is along x, b lies in the xy plane, and the normal is +z.

    Fractional coordinates are preserved, so this is a rigid rotation (plus a
    mirror if the input lattice was left-handed).
    """
    oriented = Lattice.from_parameters(*structure.lattice.parameters, vesta=True)
    return Structure(
        oriented,
        structure.species,
        structure.frac_coords,
        site_properties=structure.site_properties,
    )


def set_vacuum(structure: Structure, vacuum: float) -> Structure:
    """Resize the cell so the slab is separated by ``vacuum`` Å, centred, with c along the normal.

    The c vector is replaced by one perpendicular to the surface. For a slab
    this only changes which periodic images are stacked across the vacuum, so
    the Cartesian geometry of the slab itself is unchanged.
    """
    matrix = structure.lattice.matrix.copy()
    normal = np.cross(matrix[0], matrix[1])
    normal /= np.linalg.norm(normal)
    if matrix[2] @ normal < 0:
        normal = -normal
    heights = structure.cart_coords @ normal
    thickness = float(np.ptp(heights))
    new_height = thickness + vacuum
    matrix[2] = normal * new_height
    shift = normal * (new_height / 2.0 - (heights.min() + thickness / 2.0))
    return Structure(
        Lattice(matrix),
        structure.species,
        structure.cart_coords + shift,
        coords_are_cartesian=True,
        to_unit_cell=True,
        site_properties=structure.site_properties,
    )


def build_slab(
    bulk: Structure | Atoms,
    miller_index: Sequence[int],
    layers: int = 4,
    vacuum: float = 15.0,
    supercell: tuple[int, int] = (1, 1),
    fix_layers: int = 0,
    primitive: bool = False,
) -> Interface:
    """Build a bare slab ``Interface`` from a bulk structure.

    Args:
        bulk: Bulk cell.
        miller_index: Surface Miller index (h, k, l).
        layers: Slab thickness in (hkl) unit planes (pymatgen ``in_unit_planes``);
            this can exceed the number of atomic layers for non-primitive stackings.
        vacuum: Vacuum thickness in Å.
        supercell: In-plane repetitions (na, nb).
        fix_layers: Number of bottom atomic layers to freeze.
        primitive: Reduce the slab to its primitive surface cell first.

    Returns:
        A bare-slab ``Interface`` with a..x, normal..z, and constraints applied.
    """
    builder = primitive_slab_from_bulk if primitive else slab_from_bulk
    slab_atoms = builder(bulk, tuple(int(i) for i in miller_index), layer=layers)
    slab = AseAtomsAdaptor.get_structure(slab_atoms)
    slab = set_vacuum(orient_slab(slab), vacuum)
    slab.make_supercell([int(supercell[0]), int(supercell[1]), 1])
    slab = slab.get_sorted_structure(key=lambda site: site.coords[2])
    atoms = AseAtomsAdaptor.get_atoms(slab)
    atoms.set_pbc([True, True, True])
    interface = Interface(substrate=atoms)
    if fix_layers > 0:
        interface.fix_substrate(int(fix_layers))
    return interface


# ---------------------------------------------------------------------------
# Adsorbates
# ---------------------------------------------------------------------------


def _rotation_onto(source: np.ndarray, target: np.ndarray) -> SymmOp:
    """Proper rotation about the origin taking unit vector ``source`` onto ``target``."""
    axis = np.cross(source, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine < 1e-8:
        if cosine > 0:
            return SymmOp.from_rotation_and_translation(np.eye(3), [0, 0, 0])
        # Antiparallel: rotate by pi about any axis perpendicular to source.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(source @ helper) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, helper)
    return SymmOp.from_origin_axis_angle(
        [0, 0, 0], axis / np.linalg.norm(axis), np.degrees(np.arctan2(sine, cosine))
    )


def get_adsorbate(name: str) -> Molecule:
    """Return a library adsorbate with its binding atom at the origin, pointing down.

    The molecule is rotated so that the rest of it sits above the binding atom
    (+z). Molecules whose binding atom is at their centroid (e.g. CO2) are laid
    flat along x instead.

    Args:
        name: One of ``ADSORBATES``.
    """
    if name not in _ADSORBATES:
        raise ValueError(f"unknown adsorbate {name!r}; choose from {ADSORBATES}")
    ase_name, bind_index = _ADSORBATES[name]
    mol = AseAtomsAdaptor.get_molecule(ase_molecule(ase_name))
    mol.translate_sites(list(range(len(mol))), -mol.cart_coords[bind_index])
    if len(mol) == 1:
        return mol
    rest = np.delete(mol.cart_coords, bind_index, axis=0)
    upward = rest.mean(axis=0)
    if np.linalg.norm(upward) > 1e-3:
        op = _rotation_onto(upward / np.linalg.norm(upward), np.array([0.0, 0.0, 1.0]))
    else:
        _, _, vt = np.linalg.svd(rest - rest.mean(axis=0))
        op = _rotation_onto(vt[0], np.array([1.0, 0.0, 0.0]))
    mol.apply_operation(op)
    return mol


def adsorbate_atoms(name: str) -> Atoms:
    """Library adsorbate as ASE ``Atoms`` (for search-method building blocks)."""
    return AseAtomsAdaptor.get_atoms(get_adsorbate(name))


def _bare_structure(atoms: Atoms) -> Structure:
    """Convert to pymatgen without constraints leaking in as site properties."""
    plain = atoms.copy()
    plain.set_constraint()
    plain.calc = None
    return AseAtomsAdaptor.get_structure(plain)


def surface_normal(interface: Interface) -> np.ndarray:
    """Unit surface normal (a x b), flipped if needed to point towards +c."""
    cell = np.asarray(interface.substrate.cell)
    normal = np.cross(cell[0], cell[1])
    normal /= np.linalg.norm(normal)
    return normal if normal @ cell[2] >= 0 else -normal


def find_adsorption_sites(
    interface: Interface,
    distance: float = 2.0,
    symmetry_reduce: bool = False,
    surface_height: float = 0.9,
) -> list[AdsorptionSite]:
    """Find ontop/bridge/hollow sites on the top face of the substrate.

    Uses pymatgen's Delaunay-based ``AdsorbateSiteFinder`` on the bare
    substrate, so existing adsorbates do not distort the site set.

    Args:
        interface: The surface system.
        distance: Height (Å) of the returned positions above the surface.
        symmetry_reduce: Keep only symmetry-distinct sites.
        surface_height: Atoms within this distance (Å) of the topmost atom
            count as surface atoms.

    Returns:
        Sites ordered ontop, bridge, hollow.
    """
    slab = _bare_structure(interface.substrate)
    finder = AdsorbateSiteFinder(slab, height=surface_height)
    found = finder.find_adsorption_sites(
        distance=distance,
        symm_reduce=0.01 if symmetry_reduce else 0,
        near_reduce=0.01,
        no_obtuse_hollow=True,
    )
    sites = []
    for kind in ("ontop", "bridge", "hollow"):
        for coords in found.get(kind, []):
            frac = slab.lattice.get_fractional_coords(coords)
            frac[:2] %= 1.0
            wrapped = slab.lattice.get_cartesian_coords(frac)
            sites.append(AdsorptionSite(kind, tuple(float(x) for x in wrapped)))
    return sites


def place_adsorbate(
    interface: Interface,
    adsorbate: str | Molecule,
    position: Sequence[float],
    rotation: float = 0.0,
) -> Interface:
    """Add an adsorbate with its binding atom at ``position``.

    The molecule's +z axis is aligned with the surface normal, then it is
    rotated by ``rotation`` degrees about that normal.

    Args:
        interface: The surface system.
        adsorbate: Library name or a ``Molecule`` whose binding atom is at the
            origin with the rest of the molecule along +z.
        position: Cartesian position (Å) of the binding atom.
        rotation: Azimuthal angle about the surface normal, in degrees.

    Returns:
        A new ``Interface`` with the adsorbate appended.
    """
    mol = get_adsorbate(adsorbate) if isinstance(adsorbate, str) else adsorbate.copy()
    normal = surface_normal(interface)
    mol.apply_operation(_rotation_onto(np.array([0.0, 0.0, 1.0]), normal))
    if rotation:
        mol.apply_operation(SymmOp.from_origin_axis_angle([0, 0, 0], normal, rotation))
    mol.translate_sites(list(range(len(mol))), np.asarray(position, dtype=float))
    result = interface.copy()
    result.add_adsorbate(AseAtomsAdaptor.get_atoms(mol))
    _apply_constraints(result)
    return result


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------


def _apply_constraints(interface: Interface) -> None:
    """Make the ASE FixAtoms constraints agree with ``interface.fix``."""
    for atoms in (interface.substrate, interface.interface):
        atoms.set_constraint(FixAtoms(indices=interface.fix) if interface.fix else None)


def _from_structure(
    structure: Structure, n_substrate: int, fix: list[int]
) -> Interface:
    """Rebuild an ``Interface`` from an edited pymatgen structure."""
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.set_pbc([True, True, True])
    result = Interface(substrate=atoms[:n_substrate], interface=atoms, fixlist=fix)
    _apply_constraints(result)
    return result


def _check_indices(interface: Interface, indices: Sequence[int]) -> list[int]:
    unique = sorted({int(i) for i in indices})
    if unique and (unique[0] < 0 or unique[-1] >= len(interface.interface)):
        raise IndexError("atom index out of range")
    return unique


def delete_atoms(interface: Interface, indices: Sequence[int]) -> Interface:
    """Remove atoms (substrate vacancies or adsorbate atoms) by interface index."""
    removed = _check_indices(interface, indices)
    if not removed:
        return interface.copy()
    n_substrate = len(interface.substrate)
    structure = _bare_structure(interface.interface)
    structure.remove_sites(removed)
    kept = [i for i in range(len(interface.interface)) if i not in set(removed)]
    remap = {old: new for new, old in enumerate(kept)}
    fix = [remap[i] for i in interface.fix if i in remap]
    return _from_structure(structure, sum(1 for i in kept if i < n_substrate), fix)


def substitute_atoms(
    interface: Interface, indices: Sequence[int], element: str
) -> Interface:
    """Replace the species of the given atoms (dopants, single-atom sites)."""
    structure = _bare_structure(interface.interface)
    for index in _check_indices(interface, indices):
        structure.replace(index, element)
    return _from_structure(structure, len(interface.substrate), list(interface.fix))


def translate_atoms(
    interface: Interface, indices: Sequence[int], vector: Sequence[float]
) -> Interface:
    """Translate the given atoms by a Cartesian vector (Å), without wrapping."""
    structure = _bare_structure(interface.interface)
    structure.translate_sites(
        _check_indices(interface, indices),
        np.asarray(vector, dtype=float),
        frac_coords=False,
        to_unit_cell=False,
    )
    return _from_structure(structure, len(interface.substrate), list(interface.fix))


def set_fixed(interface: Interface, indices: Sequence[int], fixed: bool) -> Interface:
    """Freeze or release substrate atoms. Non-substrate indices are ignored."""
    n_substrate = len(interface.substrate)
    chosen = {i for i in _check_indices(interface, indices) if i < n_substrate}
    current = set(interface.fix)
    new_fix = sorted(current | chosen if fixed else current - chosen)
    result = interface.copy()
    result.fix = new_fix
    result.relax = sorted(set(range(n_substrate)) - set(new_fix))
    _apply_constraints(result)
    return result


def fix_bottom_layers(interface: Interface, layers: int) -> Interface:
    """Freeze the bottom ``layers`` atomic layers of the substrate (0 releases all)."""
    result = interface.copy()
    result.fix_substrate(int(layers))
    _apply_constraints(result)
    return result
