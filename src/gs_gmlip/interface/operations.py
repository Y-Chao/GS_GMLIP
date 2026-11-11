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
from ase.constraints import FixAtoms, FixBondLengths, FixSubsetCom, Hookean
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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


def conventional_cell(structure: Structure | Atoms, symprec: float = 0.01) -> Structure:
    """Reduce a bulk structure (e.g. a relaxed supercell) to its conventional cell.

    Lattice parameters are taken from the input, so a cell-relaxed bulk keeps
    its optimized lattice constant. If no symmetry is found within
    ``symprec`` (for example an MD snapshot), the structure is returned as is.

    Args:
        structure: Bulk structure.
        symprec: Symmetry tolerance in Å.
    """
    if isinstance(structure, Atoms):
        structure = _bare_structure(structure)
    try:
        return SpacegroupAnalyzer(
            structure, symprec=symprec
        ).get_conventional_standard_structure()
    except Exception:  # noqa: BLE001 - spglib can fail on disordered snapshots
        return structure


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


VACUUM_POSITIONS: tuple[str, ...] = ("center", "bottom")
# Gap (Å) left below the lowest atom when a slab is placed at the bottom of the
# cell, so that no atom sits exactly on the periodic boundary.
BOTTOM_MARGIN = 1.0


def _vacuum_transform(
    matrix: np.ndarray, coords: np.ndarray, vacuum: float, position: str, margin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """New cell, rigid shift, and unit normal that give ``vacuum`` Å of empty space.

    Returns:
        ``(new lattice matrix, Cartesian shift for every atom, unit surface normal)``.
    """
    if position not in VACUUM_POSITIONS:
        raise ValueError(
            f"unknown position {position!r}; choose from {VACUUM_POSITIONS}"
        )
    if vacuum <= 0:
        raise ValueError("vacuum must be positive")
    if position == "bottom" and margin >= vacuum:
        raise ValueError("the bottom margin must be smaller than the vacuum")
    new_matrix = np.array(matrix, dtype=float)
    normal = np.cross(new_matrix[0], new_matrix[1])
    normal /= np.linalg.norm(normal)
    if new_matrix[2] @ normal < 0:
        normal = -normal
    heights = coords @ normal
    thickness = float(np.ptp(heights))
    new_height = thickness + vacuum
    new_matrix[2] = normal * new_height
    lowest = margin if position == "bottom" else (new_height - thickness) / 2.0
    return new_matrix, normal * (lowest - float(heights.min())), normal


def set_vacuum(
    structure: Structure,
    vacuum: float,
    position: str = "center",
    margin: float = BOTTOM_MARGIN,
) -> Structure:
    """Resize the cell so the slab is separated by ``vacuum`` Å, with c along the normal.

    The c vector is replaced by one perpendicular to the surface. For a slab
    this only changes which periodic images are stacked across the vacuum, so
    the Cartesian geometry of the slab itself is unchanged.

    Args:
        structure: The slab.
        vacuum: Total empty space (Å) along the normal: cell height minus slab thickness.
        position: ``"center"`` puts the slab in the middle of the cell;
            ``"bottom"`` puts its lowest atom ``margin`` Å above the cell floor.
        margin: Gap below the slab for ``position="bottom"``.
    """
    matrix, shift, _ = _vacuum_transform(
        structure.lattice.matrix, structure.cart_coords, vacuum, position, margin
    )
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


def orient_adsorbate(molecule: Molecule | Atoms, bind_index: int) -> Molecule:
    """Put the binding atom at the origin with the rest of the molecule above it (+z).

    Molecules whose binding atom is at their centroid (e.g. CO2) are laid flat
    along x instead.

    Args:
        molecule: The molecule in any orientation.
        bind_index: Index of the surface-binding atom.
    """
    if isinstance(molecule, Atoms):
        # Built directly: the ASE adaptor rejects radicals (odd electron counts).
        molecule = Molecule(molecule.get_chemical_symbols(), molecule.get_positions())
    mol = molecule.copy()
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


def get_adsorbate(name: str) -> Molecule:
    """Return a built-in adsorbate with its binding atom at the origin, pointing down.

    Args:
        name: One of ``ADSORBATES``.
    """
    if name not in _ADSORBATES:
        raise ValueError(f"unknown adsorbate {name!r}; choose from {ADSORBATES}")
    ase_name, bind_index = _ADSORBATES[name]
    return orient_adsorbate(ase_molecule(ase_name), bind_index)


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


def extra_constraints(interface: Interface) -> list:
    """Constraints on the interface other than the frozen-atom ``FixAtoms``."""
    return [c for c in interface.interface.constraints if not isinstance(c, FixAtoms)]


def _apply_constraints(interface: Interface, extras: list | None = None) -> None:
    """Make the ASE constraints agree with ``interface.fix`` plus the extra constraints.

    Args:
        interface: Modified in place.
        extras: Non-``FixAtoms`` constraints to keep; defaults to those
            currently on ``interface.interface``.
    """
    extras = extra_constraints(interface) if extras is None else list(extras)
    frozen = [FixAtoms(indices=interface.fix)] if interface.fix else []
    interface.substrate.set_constraint(frozen or None)
    interface.interface.set_constraint((frozen + extras) or None)


def _from_structure(
    structure: Structure, n_substrate: int, fix: list[int], extras: list | None = None
) -> Interface:
    """Rebuild an ``Interface`` from an edited pymatgen structure."""
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.set_pbc([True, True, True])
    result = Interface(substrate=atoms[:n_substrate], interface=atoms, fixlist=fix)
    _apply_constraints(result, extras or [])
    return result


def _check_indices(interface: Interface, indices: Sequence[int]) -> list[int]:
    unique = sorted({int(i) for i in indices})
    if unique and (unique[0] < 0 or unique[-1] >= len(interface.interface)):
        raise IndexError("atom index out of range")
    return unique


def _remap_constraints(extras: list, remap: dict[int, int]) -> list:
    """Re-index extra constraints after atoms were removed.

    A bond constraint or wall on a removed atom is dropped; a centre-of-mass
    constraint keeps its surviving atoms and is dropped when none remain.
    """
    result = []
    for constraint in extras:
        if isinstance(constraint, FixBondLengths):
            pairs = [
                [remap[int(i)], remap[int(j)]]
                for i, j in constraint.pairs
                if int(i) in remap and int(j) in remap
            ]
            if pairs:
                result.append(FixBondLengths(pairs))
        elif isinstance(constraint, FixSubsetCom):
            kept = [remap[int(i)] for i in constraint.index if int(i) in remap]
            if kept:
                result.append(FixSubsetCom(kept))
        elif (
            isinstance(constraint, Hookean)
            and getattr(constraint, "_type", "") == "plane"
        ):
            if int(constraint.index) in remap:
                plane = tuple(float(x) for x in constraint.plane)
                result.append(
                    Hookean(
                        a1=remap[int(constraint.index)], a2=plane, k=constraint.spring
                    )
                )
    return result


def delete_atoms(interface: Interface, indices: Sequence[int]) -> Interface:
    """Remove atoms (substrate vacancies or adsorbate atoms) by interface index.

    Frozen atoms and bond, centre-of-mass, and wall constraints are re-indexed;
    a constraint that involves a removed atom is dropped (a centre-of-mass
    constraint keeps its remaining atoms).
    """
    removed = _check_indices(interface, indices)
    if not removed:
        return interface.copy()
    n_substrate = len(interface.substrate)
    structure = _bare_structure(interface.interface)
    structure.remove_sites(removed)
    kept = [i for i in range(len(interface.interface)) if i not in set(removed)]
    remap = {old: new for new, old in enumerate(kept)}
    fix = [remap[i] for i in interface.fix if i in remap]
    return _from_structure(
        structure,
        sum(1 for i in kept if i < n_substrate),
        fix,
        _remap_constraints(extra_constraints(interface), remap),
    )


def substitute_atoms(
    interface: Interface, indices: Sequence[int], element: str
) -> Interface:
    """Replace the species of the given atoms (dopants, single-atom sites)."""
    structure = _bare_structure(interface.interface)
    for index in _check_indices(interface, indices):
        structure.replace(index, element)
    return _from_structure(
        structure,
        len(interface.substrate),
        list(interface.fix),
        extra_constraints(interface),
    )


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
    return _from_structure(
        structure,
        len(interface.substrate),
        list(interface.fix),
        extra_constraints(interface),
    )


def rotate_atoms(
    interface: Interface,
    indices: Sequence[int],
    angle: float,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    anchor: Sequence[float] | None = None,
) -> Interface:
    """Rigidly rotate the given atoms.

    Args:
        interface: The surface system.
        indices: Interface indices of the atoms to rotate.
        angle: Rotation angle in degrees (right-handed about ``axis``).
        axis: Rotation axis in Cartesian coordinates.
        anchor: Point (Å) the axis passes through; defaults to the centroid of
            the rotated atoms.
    """
    chosen = _check_indices(interface, indices)
    if not chosen:
        return interface.copy()
    structure = _bare_structure(interface.interface)
    if anchor is None:
        anchor = structure.cart_coords[chosen].mean(axis=0)
    direction = np.asarray(axis, dtype=float)
    structure.rotate_sites(
        chosen,
        theta=float(np.radians(angle)),
        axis=direction / np.linalg.norm(direction),
        anchor=np.asarray(anchor, dtype=float),
        to_unit_cell=False,
    )
    return _from_structure(
        structure,
        len(interface.substrate),
        list(interface.fix),
        extra_constraints(interface),
    )


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


def adjust_vacuum(
    interface: Interface,
    vacuum: float,
    position: str = "center",
    margin: float = BOTTOM_MARGIN,
) -> Interface:
    """Change the vacuum of a slab model and place it at the centre or bottom of the cell.

    All atoms (substrate and adsorbates) move rigidly along the surface
    normal; in-plane positions, frozen atoms, and bond or centre-of-mass
    constraints are unchanged. Walls move with the atoms.

    Args:
        interface: The surface system.
        vacuum: Total empty space (Å) along the normal: cell height minus the
            thickness of everything in the cell, adsorbates included.
        position: ``"center"`` or ``"bottom"`` (lowest atom ``margin`` Å above
            the cell floor, leaving the rest of the vacuum on top).
        margin: Gap below the slab for ``position="bottom"``.

    Raises:
        ValueError: For an unknown position or a non-positive vacuum.
    """
    structure = _bare_structure(interface.interface)
    matrix, shift, normal = _vacuum_transform(
        structure.lattice.matrix,
        structure.cart_coords,
        float(vacuum),
        position,
        float(margin),
    )
    moved = Structure(
        Lattice(matrix),
        structure.species,
        structure.cart_coords + shift,
        coords_are_cartesian=True,
        to_unit_cell=False,
    )
    lift = float(shift @ normal)
    extras = []
    for constraint in extra_constraints(interface):
        if (
            isinstance(constraint, Hookean)
            and getattr(constraint, "_type", "") == "plane"
        ):
            a, b, c, d = (float(x) for x in constraint.plane)
            extras.append(
                Hookean(
                    a1=int(constraint.index),
                    a2=(a, b, c, d - lift),
                    k=constraint.spring,
                )
            )
        else:
            extras.append(constraint)
    return _from_structure(moved, len(interface.substrate), list(interface.fix), extras)


def fix_bottom_layers(interface: Interface, layers: int) -> Interface:
    """Freeze the bottom ``layers`` atomic layers of the substrate (0 releases all)."""
    extras = extra_constraints(interface)
    result = interface.copy()
    result.fix_substrate(int(layers))
    _apply_constraints(result, extras)
    return result


# ---------------------------------------------------------------------------
# Constraints beyond frozen atoms (ASE constraint objects on the interface)
# ---------------------------------------------------------------------------


def add_bond_constraint(interface: Interface, first: int, second: int) -> Interface:
    """Hold the distance between two atoms at its current value (``FixBondLengths``)."""
    first, second = (
        _check_indices(interface, [first, second])
        if first != second
        else (first, first)
    )
    if first == second:
        raise ValueError("a bond constraint needs two different atoms")
    extras = extra_constraints(interface)
    pairs = [[first, second]]
    kept = []
    for constraint in extras:
        if isinstance(constraint, FixBondLengths):
            pairs = [list(map(int, pair)) for pair in constraint.pairs] + pairs
        else:
            kept.append(constraint)
    unique = [
        list(pair) for pair in dict.fromkeys(tuple(sorted(pair)) for pair in pairs)
    ]
    result = interface.copy()
    _apply_constraints(result, [*kept, FixBondLengths(unique)])
    return result


def add_com_constraint(interface: Interface, indices: Sequence[int]) -> Interface:
    """Hold the centre of mass of the given atoms in place (``FixSubsetCom``)."""
    chosen = _check_indices(interface, indices)
    if not chosen:
        raise ValueError("a centre-of-mass constraint needs at least one atom")
    result = interface.copy()
    _apply_constraints(result, [*extra_constraints(interface), FixSubsetCom(chosen)])
    return result


def add_wall(
    interface: Interface,
    indices: Sequence[int],
    height: float,
    spring_constant: float = 5.0,
) -> Interface:
    """Add a soft wall keeping the given atoms below a plane parallel to the surface.

    Each atom gets an ASE ``Hookean`` plane restraint: no force below the
    plane, a restoring force ``spring_constant`` (eV/Å²) times the excursion
    above it. This keeps adsorbates from desorbing into the vacuum during MD.

    Args:
        interface: The surface system.
        indices: Interface indices of the restrained atoms.
        height: Wall position in Å along the surface normal, measured from the
            cell origin (the same axis as the atomic z coordinates).
        spring_constant: Restoring force constant in eV/Å².
    """
    chosen = _check_indices(interface, indices)
    if not chosen:
        raise ValueError("a wall needs at least one atom")
    normal = surface_normal(interface)
    plane = (float(normal[0]), float(normal[1]), float(normal[2]), -float(height))
    walls = [Hookean(a1=index, a2=plane, k=float(spring_constant)) for index in chosen]
    result = interface.copy()
    _apply_constraints(result, [*extra_constraints(interface), *walls])
    return result


def clear_constraints(interface: Interface) -> Interface:
    """Remove bond, centre-of-mass, and wall constraints (frozen atoms are kept)."""
    result = interface.copy()
    _apply_constraints(result, [])
    return result


def describe_constraints(interface: Interface) -> dict[str, list]:
    """Summarize the extra constraints for display.

    Returns:
        Dict with ``bonds`` (list of index pairs), ``com`` (list of index
        lists), and ``walls`` (list of ``{"index", "height"}``).
    """
    summary: dict[str, list] = {"bonds": [], "com": [], "walls": []}
    for constraint in extra_constraints(interface):
        if isinstance(constraint, FixBondLengths):
            summary["bonds"].extend([int(i), int(j)] for i, j in constraint.pairs)
        elif isinstance(constraint, FixSubsetCom):
            summary["com"].append([int(i) for i in constraint.index])
        elif (
            isinstance(constraint, Hookean)
            and getattr(constraint, "_type", "") == "plane"
        ):
            summary["walls"].append(
                {"index": int(constraint.index), "height": -float(constraint.plane[3])}
            )
    return summary


# ---------------------------------------------------------------------------
# Itemized view of a structure: what can be listed, switched off, and removed
# ---------------------------------------------------------------------------


def _wall_heights(interface: Interface) -> list[float]:
    return sorted(
        {round(wall["height"], 6) for wall in describe_constraints(interface)["walls"]}
    )


def list_items(interface: Interface) -> list[dict]:
    """Itemize a structure: substrate, frozen atoms, adsorbates, clusters, constraints.

    Returns:
        Dicts with ``key`` (stable while the structure is unchanged), ``kind``,
        ``label``, ``atoms`` (interface indices the item refers to), and
        ``removable``. Keys: ``substrate``, ``fixed``, ``ads:<n>``,
        ``cluster:<n>``, ``bond:<n>``, ``com:<n>``, ``wall:<n>``.
    """
    atoms = interface.interface
    symbols = atoms.get_chemical_symbols()
    items: list[dict] = []
    if len(interface.substrate):
        items.append(
            {
                "key": "substrate",
                "kind": "substrate",
                "label": f"Substrate {interface.substrate.get_chemical_formula()}",
                "atoms": list(range(len(interface.substrate))),
                "removable": False,
            }
        )
    if interface.fix:
        items.append(
            {
                "key": "fixed",
                "kind": "constraint",
                "label": f"Fixed atoms ({len(interface.fix)})",
                "atoms": list(interface.fix),
                "removable": True,
            }
        )
    for kind, groups in (
        ("ads", interface.adsList),
        ("cluster", interface.clusterList),
    ):
        for number, group in enumerate(groups):
            name = "Adsorbate" if kind == "ads" else "Cluster"
            formula = atoms[list(group)].get_chemical_formula()
            items.append(
                {
                    "key": f"{kind}:{number}",
                    "kind": "adsorbate" if kind == "ads" else "cluster",
                    "label": f"{name} {formula} (atoms {group[0]}-{group[-1]})"
                    if len(group) > 1
                    else f"{name} {formula} (atom {group[0]})",
                    "atoms": [int(i) for i in group],
                    "removable": True,
                }
            )
    summary = describe_constraints(interface)
    for number, (first, second) in enumerate(summary["bonds"]):
        distance = atoms.get_distance(first, second)
        items.append(
            {
                "key": f"bond:{number}",
                "kind": "constraint",
                "label": f"Fixed bond {symbols[first]}{first}-{symbols[second]}{second} ({distance:.3f} Å)",
                "atoms": [first, second],
                "removable": True,
            }
        )
    for number, group in enumerate(summary["com"]):
        items.append(
            {
                "key": f"com:{number}",
                "kind": "constraint",
                "label": f"Fixed centre of mass ({len(group)} atoms)",
                "atoms": list(group),
                "removable": True,
            }
        )
    for number, height in enumerate(_wall_heights(interface)):
        members = [
            w["index"] for w in summary["walls"] if round(w["height"], 6) == height
        ]
        items.append(
            {
                "key": f"wall:{number}",
                "kind": "constraint",
                "label": f"Wall at z = {height:.2f} Å ({len(members)} atoms)",
                "atoms": members,
                "removable": True,
            }
        )
    return items


def remove_items(interface: Interface, keys: Sequence[str]) -> Interface:
    """Remove itemized parts of a structure (see ``list_items`` for the keys).

    Constraints are removed first, then adsorbate and cluster atoms; remaining
    constraints are re-indexed. Unknown keys and ``substrate`` are ignored.

    Args:
        interface: The structure.
        keys: Item keys to remove.
    """
    chosen = set(keys)
    known = {item["key"]: item for item in list_items(interface)}
    heights = _wall_heights(interface)
    drop_bonds = {
        int(key.split(":")[1]) for key in chosen & set(known) if key.startswith("bond:")
    }
    drop_com = {
        int(key.split(":")[1]) for key in chosen & set(known) if key.startswith("com:")
    }
    drop_walls = {
        heights[int(key.split(":")[1])]
        for key in chosen & set(known)
        if key.startswith("wall:")
    }

    extras: list = []
    bond_number = com_number = 0
    for constraint in extra_constraints(interface):
        if isinstance(constraint, FixBondLengths):
            pairs = []
            for pair in constraint.pairs:
                if bond_number not in drop_bonds:
                    pairs.append([int(pair[0]), int(pair[1])])
                bond_number += 1
            if pairs:
                extras.append(FixBondLengths(pairs))
        elif isinstance(constraint, FixSubsetCom):
            if com_number not in drop_com:
                extras.append(constraint)
            com_number += 1
        elif (
            isinstance(constraint, Hookean)
            and getattr(constraint, "_type", "") == "plane"
        ):
            if round(-float(constraint.plane[3]), 6) not in drop_walls:
                extras.append(constraint)
        else:
            extras.append(constraint)

    result = interface.copy()
    if "fixed" in chosen:
        result.fix = []
        result.relax = list(range(len(result.substrate)))
    _apply_constraints(result, extras)
    doomed = sorted(
        {
            i
            for key in chosen & set(known)
            if key.startswith(("ads:", "cluster:"))
            for i in known[key]["atoms"]
        }
    )
    return delete_atoms(result, doomed) if doomed else result
