"""Server-side session: windows, the structures in them, and everything done to them.

The browser is a view. It sends intents ("cleave this slab", "rotate these
atoms", "run this task") and gets back plain JSON; all chemistry and geometry
stays here, in Python, on top of ``gs_gmlip.interface.operations``, the
``StructureStore``, and the job executor. No web framework is imported, so the
whole workflow is testable without a server.

A session holds several workspaces ("windows" in the UI), each with its own
structure, undo history, and store link: a bulk window to prepare the crystal,
surface windows cleaved from it, and molecule windows to build adsorbates that
go into the molecule library and onto surfaces.
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ase import Atoms
from ase.build import molecule as ase_molecule
from ase.collections import g2
from ase.io import read as ase_read
from ase.io import write as ase_write
from pymatgen.io.ase import AseAtomsAdaptor

from gs_gmlip.app.scene import SITE_COLORS, build_scene
from gs_gmlip.evaluator import vasp
from gs_gmlip.evaluator.calculators import calculator_entries, matgl_models
from gs_gmlip.interface import operations as ops
from gs_gmlip.config import recent_projects, remember_project
from gs_gmlip.interface.core import Interface
from gs_gmlip.interface.intermediates import REACTIONS
from gs_gmlip.interface.molecules import USER_SCOPES, MoleculeLibrary
from gs_gmlip.interface.solvation import fill_solvent, packmol_available
from gs_gmlip.jobs.executor import BaseExecutor, LocalExecutor
from gs_gmlip.jobs.spec import JobSpec
from gs_gmlip.jobs.tasks import OPTIMIZERS, TASKS
from gs_gmlip.plugins import load_plugins
from gs_gmlip.energetics import analyze
from gs_gmlip.store import ROLES, STAGES, Project, StructureRecord, global_store

_HISTORY_LIMIT = 50
EXPORT_FORMATS = {"extxyz": "xyz", "vasp": "vasp", "cif": "cif"}
WORKSPACE_KINDS = ("bulk", "surface", "molecule")
# Edge (Å) of the cubic box a molecule is centred in.
MOLECULE_BOX = 15.0


def _indices(params: dict[str, Any]) -> list[int]:
    chosen = [int(i) for i in params.get("indices", [])]
    if not chosen:
        raise ValueError("select atoms first")
    return chosen


def _boxed(atoms: Atoms) -> Atoms:
    """Centre a molecule in a cubic box so every task can treat it like any structure."""
    boxed = Atoms(atoms.get_chemical_symbols(), positions=atoms.get_positions())
    boxed.set_cell([MOLECULE_BOX] * 3)
    boxed.center()
    boxed.set_pbc([True, True, True])
    return boxed


@dataclass
class Workspace:
    """One window: a structure being edited, with its own history and store link.

    Attributes:
        id: Stable identifier, e.g. ``surface-1``.
        name: Display name.
        kind: ``bulk``, ``surface``, or ``molecule``.
        interface: The structure, ``None`` while the window is empty.
        history: Undo stack of ``(structure, store id)``.
        sites: Adsorption sites currently shown (surface windows).
        current_id: Row id of the structure if it is stored and unmodified.
        parent_id: Row id the (possibly edited) structure descends from.
        level: Level-of-theory label of the structure's energy, if it has one.
        origin: Where the structure was loaded from (e.g. ``global #12``).
        excluded: Item keys the user unchecked; they stay visible but are left
            out of what is stored, exported, and simulated.
    """

    id: str
    name: str
    kind: str
    interface: Interface | None = None
    history: list[tuple] = field(default_factory=list)
    sites: list[ops.AdsorptionSite] = field(default_factory=list)
    show_sites: bool = False
    distinct_sites: bool = False
    site_height: float = 2.0
    current_id: int | None = None
    parent_id: int | None = None
    # Level of theory of the structure's energy, and where it was loaded from.
    level: str = ""
    origin: str = ""
    # Keys of items (see ``operations.list_items``) switched off by the user.
    excluded: set[str] = field(default_factory=set)


class Session:
    """One user's working state inside a project.

    Args:
        project: Project (store + jobs directory) the session works in.
        executor: Job executor; a ``LocalExecutor`` by default.
    """

    def __init__(self, project: Project, executor: BaseExecutor | None = None):
        self.project = project
        self.executor = executor or LocalExecutor(project)
        self.plugins = load_plugins(project.root)
        self.library = MoleculeLibrary(project.root)
        remember_project(project.root)
        self.workspaces: dict[str, Workspace] = {}
        self._counters = dict.fromkeys(WORKSPACE_KINDS, 0)
        for kind in WORKSPACE_KINDS:
            self.add_window(kind)

    # ------------------------------------------------------------------
    # Windows
    # ------------------------------------------------------------------

    def add_window(self, kind: str, name: str | None = None) -> str:
        """Open a new window and return its id."""
        if kind not in WORKSPACE_KINDS:
            raise ValueError(
                f"unknown window kind {kind!r}; choose from {WORKSPACE_KINDS}"
            )
        self._counters[kind] += 1
        number = self._counters[kind]
        workspace_id = f"{kind}-{number}"
        self.workspaces[workspace_id] = Workspace(
            workspace_id, name or f"{kind.capitalize()} {number}", kind
        )
        return workspace_id

    def remove_window(self, workspace_id: str) -> None:
        """Close a window (the last one cannot be closed)."""
        self._ws(workspace_id)
        if len(self.workspaces) == 1:
            raise ValueError("the last window cannot be closed")
        del self.workspaces[workspace_id]

    def windows(self) -> list[dict[str, Any]]:
        """Summary of every window, in creation order."""
        return [
            {
                "id": ws.id,
                "name": ws.name,
                "kind": ws.kind,
                "formula": ws.interface.interface.get_chemical_formula()
                if ws.interface
                else None,
            }
            for ws in self.workspaces.values()
        ]

    def _ws(self, workspace_id: str | None) -> Workspace:
        if not workspace_id:
            return next(iter(self.workspaces.values()))
        if workspace_id not in self.workspaces:
            raise ValueError(f"no window {workspace_id!r}")
        return self.workspaces[workspace_id]

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _set(
        self,
        ws: Workspace,
        interface: Interface,
        current_id: int | None = None,
        level: str = "",
        origin: str = "",
    ) -> None:
        if ws.interface is not None:
            ws.history.append((ws.interface, ws.current_id, ws.level, ws.origin))
            del ws.history[:-_HISTORY_LIMIT]
        ws.interface = interface
        ws.current_id = current_id
        ws.parent_id = current_id if current_id is not None else ws.parent_id
        ws.level = level
        ws.origin = origin
        ws.excluded &= {item["key"] for item in ops.list_items(interface)}
        self._refresh_sites(ws)

    def _set_new(self, ws: Workspace, interface: Interface) -> None:
        """Replace the structure with one that has no stored ancestor."""
        self._set(ws, interface)
        ws.parent_id = None
        ws.excluded.clear()

    @staticmethod
    def _role(ws: Workspace) -> str:
        """Role label a window's structure gets when stored."""
        if ws.kind != "surface":
            return ws.kind
        interface = ws.interface
        has_adsorbates = interface is not None and (
            interface.adsList or interface.clusterList
        )
        return "adsorbed" if has_adsorbates else "surface"

    def _effective(self, ws: Workspace) -> Interface:
        """The structure that leaves the window: unchecked items removed."""
        interface = self._require(ws)
        return (
            ops.remove_items(interface, sorted(ws.excluded))
            if ws.excluded
            else interface
        )

    def _refresh_sites(self, ws: Workspace) -> None:
        ws.sites = []
        show = ws.show_sites and ws.kind == "surface" and ws.interface is not None
        if show and len(ws.interface.substrate):
            try:
                ws.sites = ops.find_adsorption_sites(
                    ws.interface,
                    distance=ws.site_height,
                    symmetry_reduce=ws.distinct_sites,
                )
            except Exception:  # noqa: BLE001 - site search is best effort on odd inputs
                ws.sites = []

    @staticmethod
    def _require(ws: Workspace) -> Interface:
        if ws.interface is None:
            raise ValueError(
                f"the {ws.name} window is empty; build or load a structure first"
            )
        return ws.interface

    def state(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Everything the browser needs to redraw one window, plus the window list."""
        ws = self._ws(workspace_id)
        result: dict[str, Any] = {
            "workspace": {"id": ws.id, "name": ws.name, "kind": ws.kind},
            "windows": self.windows(),
            "molecules": self.library.entries(),
            "scene": None,
            "info": None,
            "current_id": ws.current_id,
            "can_undo": bool(ws.history),
            "role": self._role(ws),
            "level": ws.level,
            "origin": ws.origin,
            "items": [],
            "lineage": [],
            "sites": {
                "show": ws.show_sites,
                "distinct": ws.distinct_sites,
                "height": ws.site_height,
            },
        }
        interface = ws.interface
        if interface is None:
            return result
        atoms = interface.interface
        constraints = ops.describe_constraints(interface)
        energy = None
        if atoms.calc is not None:
            try:
                energy = float(atoms.get_potential_energy())
            except Exception:  # noqa: BLE001 - a stale calculator just means "no energy"
                energy = None
        result["scene"] = build_scene(interface, ws.sites, ws.excluded)
        result["items"] = [
            {**item, "checked": item["key"] not in ws.excluded}
            for item in ops.list_items(interface)
        ]
        result["info"] = {
            "formula": atoms.get_chemical_formula(),
            "natoms": len(atoms),
            "n_substrate": len(interface.substrate),
            "n_fixed": len(interface.fix),
            "n_adsorbates": len(interface.adsList),
            "n_clusters": len(interface.clusterList),
            "n_bond_constraints": len(constraints["bonds"]),
            "n_com_constraints": len(constraints["com"]),
            "n_walls": len(constraints["walls"]),
            "energy": energy,
            "cell_lengths": [float(x) for x in atoms.cell.lengths()],
            "cell_angles": [float(x) for x in atoms.cell.angles()],
            "top_z": float(atoms.positions[:, 2].max()) if len(atoms) else 0.0,
        }
        if ws.current_id is not None:
            result["lineage"] = [
                rec.id for rec in self.project.store.lineage(ws.current_id)
            ]
        return result

    def meta(self) -> dict[str, Any]:
        """Static choices for the forms: lattices, molecules, calculators, tasks."""
        calculators = []
        for name, entry in calculator_entries().items():
            choices = dict(entry.choices)
            if name == "matgl" and entry.available:
                choices["model"] = matgl_models()
            calculators.append(
                {
                    "name": name,
                    "available": entry.available,
                    "requires": entry.requires,
                    "description": entry.description,
                    "defaults": entry.defaults,
                    "choices": choices,
                }
            )
        return {
            "project": str(self.project.root),
            "recent_projects": recent_projects(),
            "reactions": list(REACTIONS),
            "library_scopes": list(USER_SCOPES),
            "lattices": list(ops.BULK_LATTICES),
            "molecule_names": sorted(g2.names),
            "window_kinds": list(WORKSPACE_KINDS),
            "site_colors": SITE_COLORS,
            "stages": list(STAGES),
            "roles": list(ROLES),
            "optimizers": list(OPTIMIZERS),
            "export_formats": list(EXPORT_FORMATS),
            "packmol": packmol_available(),
            "plugins": list(self.plugins),
            "vasp": {
                "presets": [
                    {
                        "name": preset.name,
                        "task": preset.task,
                        "description": preset.description,
                    }
                    for preset in vasp.VASP_PRESETS.values()
                ],
                "precisions": list(vasp.PRECISION_TAGS),
                "dispersions": list(vasp.DISPERSION_TAGS),
                "potcars": vasp.potcars_available(),
            },
            "calculators": calculators,
            "tasks": [
                {
                    "kind": task.kind,
                    "description": task.description,
                    "defaults": task.defaults,
                }
                for task in TASKS.values()
            ],
        }

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def create_bulk(
        self,
        workspace_id: str | None,
        element: str,
        lattice: str,
        a: float,
        c: float | None = None,
    ) -> None:
        """Create an elemental bulk cell in a window."""
        bulk = ops.make_bulk(
            element.strip(), lattice, float(a), float(c) if c else None
        )
        atoms = AseAtomsAdaptor.get_atoms(bulk)
        atoms.set_pbc([True, True, True])
        self._set_new(self._ws(workspace_id), Interface(substrate=atoms))

    def create_molecule(self, workspace_id: str | None, name: str) -> None:
        """Load a molecule from the ASE G2 database into a window, centred in a box."""
        if name not in g2.names:
            raise ValueError(
                f"unknown molecule {name!r} (ASE G2 database names are accepted)"
            )
        self._set_new(
            self._ws(workspace_id), Interface(substrate=_boxed(ase_molecule(name)))
        )

    def load_file(self, workspace_id: str | None, name: str, content: bytes) -> None:
        """Read an uploaded structure file into a window.

        Molecule windows centre the atoms in a box; other windows keep the
        file's cell.

        Args:
            workspace_id: Target window.
            name: Original file name (its extension selects the reader).
            content: File bytes.
        """
        ws = self._ws(workspace_id)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / Path(name).name
            path.write_bytes(content)
            try:
                if ws.kind == "molecule":
                    atoms = _boxed(ase_read(str(path)))
                else:
                    try:
                        structure = ops.load_structure(path)
                    except Exception:  # noqa: BLE001 - fall back to ASE readers (extxyz, traj, ...)
                        structure = AseAtomsAdaptor.get_structure(ase_read(str(path)))
                    atoms = AseAtomsAdaptor.get_atoms(structure)
                    atoms.set_pbc([True, True, True])
            except Exception as error:
                raise ValueError(f"could not read {name}: {error}") from error
        self._set_new(ws, Interface(substrate=atoms))

    def cleave(
        self,
        source_id: str | None,
        miller: list[int],
        target_id: str | None = None,
        layers: int = 4,
        vacuum: float = 15.0,
        supercell: tuple[int, int] = (1, 1),
        fix_layers: int = 0,
        primitive: bool = False,
        conventional: bool = True,
    ) -> str:
        """Cleave a slab from the bulk in one window into a surface window.

        Args:
            source_id: Window holding the bulk structure.
            miller: Miller index (h, k, l).
            target_id: Surface window to fill; a new one is opened if omitted.
            layers: Slab thickness in (hkl) unit planes.
            vacuum: Vacuum thickness in Å.
            supercell: In-plane repetitions (na, nb).
            fix_layers: Number of bottom atomic layers to freeze.
            primitive: Reduce the slab to its primitive surface cell.
            conventional: Reduce the bulk to its conventional cell first (so a
                relaxed supercell cleaves like the crystal it represents).

        Returns:
            Id of the surface window that received the slab.
        """
        source = self._ws(source_id)
        bulk_atoms = self._require(source).interface
        bulk = (
            ops.conventional_cell(bulk_atoms)
            if conventional
            else AseAtomsAdaptor.get_structure(bulk_atoms)
        )
        interface = ops.build_slab(
            bulk,
            [int(i) for i in miller],
            layers=int(layers),
            vacuum=float(vacuum),
            supercell=(int(supercell[0]), int(supercell[1])),
            fix_layers=int(fix_layers),
            primitive=bool(primitive),
        )
        if target_id:
            target = self._ws(target_id)
        else:
            label = f"{bulk.composition.reduced_formula}({''.join(str(int(i)) for i in miller)})"
            target = self._ws(self.add_window("surface", label))
        self._set_new(target, interface)
        target.parent_id = source.current_id
        return target.id

    def set_sites(
        self,
        workspace_id: str | None,
        show: bool,
        distinct: bool = False,
        height: float = 2.0,
    ) -> None:
        """Show or hide adsorption-site markers in a surface window."""
        ws = self._ws(workspace_id)
        ws.show_sites = bool(show)
        ws.distinct_sites = bool(distinct)
        ws.site_height = float(height)
        self._refresh_sites(ws)

    # ------------------------------------------------------------------
    # Molecule library
    # ------------------------------------------------------------------

    def library_add(
        self,
        workspace_id: str | None,
        name: str,
        scope: str = "temporary",
        bind_index: int | None = None,
        reactions: list[str] | None = None,
        note: str = "",
    ) -> None:
        """Add the molecule in a window to the library as an adsorbate.

        Args:
            workspace_id: Window holding the molecule.
            name: Library name.
            scope: ``temporary`` (this session), ``project``, or ``global``
                (shared by all projects).
            bind_index: Surface-binding atom; the lowest atom along z if omitted.
            reactions: Reaction tags.
            note: Provenance (method, reference).
        """
        atoms = self._require(self._ws(workspace_id)).interface
        self.library.add(
            name,
            atoms,
            bind_index=bind_index,
            scope=scope,
            reactions=reactions,
            note=note,
        )

    def library_open(self, workspace_id: str | None, name: str) -> None:
        """Load a library molecule into a window for editing or optimization."""
        self._set_new(
            self._ws(workspace_id),
            Interface(substrate=_boxed(self.library.atoms(name))),
        )

    def library_remove(self, name: str) -> None:
        """Remove a user molecule from the library."""
        self.library.remove(name)

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def _place(
        self, ws: Workspace, interface: Interface, params: dict[str, Any]
    ) -> Interface:
        if "site" in params:
            index = int(params["site"])
            if not 0 <= index < len(ws.sites):
                raise ValueError("that adsorption site no longer exists")
            position = ws.sites[index].coords
        else:
            position = [float(x) for x in params["position"]]
        molecule = self.library.get(str(params["adsorbate"]))
        return ops.place_adsorbate(
            interface, molecule, position, float(params.get("rotation", 0.0))
        )

    @staticmethod
    def _bond(interface: Interface, params: dict[str, Any]) -> Interface:
        chosen = _indices(params)
        if len(chosen) != 2:
            raise ValueError("a bond constraint needs exactly two selected atoms")
        return ops.add_bond_constraint(interface, chosen[0], chosen[1])

    @staticmethod
    def _water(interface: Interface, params: dict[str, Any]) -> Interface:
        count = params.get("n_molecules")
        thickness = params.get("thickness")
        return fill_solvent(
            interface,
            n_molecules=int(count) if count else None,
            density=float(params.get("density", 1.0)),
            gap=float(params.get("gap", 2.0)),
            thickness=float(thickness) if thickness else None,
            tolerance=float(params.get("tolerance", 2.0)),
            seed=int(params.get("seed", 0)),
        )

    def edit(self, workspace_id: str | None, op: str, **params: Any) -> None:
        """Apply one editing operation to the structure in a window.

        Args:
            workspace_id: Target window.
            op: One of ``place``, ``delete``, ``substitute``, ``fix``,
                ``release``, ``move``, ``rotate``, ``fix_layers``, ``bond``,
                ``com``, ``wall``, ``clear_constraints``, ``vacuum``, ``water``.
            **params: Operation parameters (``indices`` for selection-based ones).

        Raises:
            ValueError: For an unknown operation or invalid parameters.
        """
        ws = self._ws(workspace_id)
        handlers: dict[str, Callable[[Interface], Interface]] = {
            "place": lambda i: self._place(ws, i, params),
            "delete": lambda i: ops.delete_atoms(i, _indices(params)),
            "substitute": lambda i: ops.substitute_atoms(
                i, _indices(params), str(params["element"]).strip()
            ),
            "fix": lambda i: ops.set_fixed(i, _indices(params), True),
            "release": lambda i: ops.set_fixed(i, _indices(params), False),
            "move": lambda i: ops.translate_atoms(
                i, _indices(params), [float(x) for x in params["vector"]]
            ),
            "rotate": lambda i: ops.rotate_atoms(
                i,
                _indices(params),
                float(params["angle"]),
                axis=[float(x) for x in params.get("axis", (0, 0, 1))],
                anchor=[float(x) for x in params["anchor"]]
                if params.get("anchor")
                else None,
            ),
            "fix_layers": lambda i: ops.fix_bottom_layers(i, int(params["layers"])),
            "bond": lambda i: self._bond(i, params),
            "com": lambda i: ops.add_com_constraint(i, _indices(params)),
            "wall": lambda i: ops.add_wall(
                i,
                _indices(params),
                float(params["height"]),
                float(params.get("spring_constant", 5.0)),
            ),
            "clear_constraints": ops.clear_constraints,
            "remove_item": lambda i: ops.remove_items(i, [str(params["key"])]),
            "vacuum": lambda i: ops.adjust_vacuum(
                i, float(params["vacuum"]), str(params.get("position", "center"))
            ),
            "water": lambda i: self._water(i, params),
        }
        if op not in handlers:
            raise ValueError(f"unknown edit operation {op!r}")
        self._set(ws, handlers[op](self._require(ws)))

    def undo(self, workspace_id: str | None = None) -> None:
        """Restore the structure before the last change in a window."""
        ws = self._ws(workspace_id)
        if not ws.history:
            raise ValueError("nothing to undo")
        ws.interface, ws.current_id, ws.level, ws.origin = ws.history.pop()
        ws.excluded &= {item["key"] for item in ops.list_items(ws.interface)}
        self._refresh_sites(ws)

    # ------------------------------------------------------------------
    # Store, export, jobs
    # ------------------------------------------------------------------

    def save(self, workspace_id: str | None = None, label: str = "") -> int:
        """Store a window's structure if it is not stored yet; return its row id.

        Unchecked items are left out of the stored structure. In that case the
        window keeps showing the full structure and stays "unsaved", and the
        stored row becomes its parent.
        """
        ws = self._ws(workspace_id)
        interface = self._effective(ws)
        if ws.current_id is not None and not ws.excluded:
            return ws.current_id
        row_id = self.project.store.add(
            interface,
            stage="built",
            parent_id=ws.parent_id,
            label=label,
            role=self._role(ws),
            level=ws.level if interface.interface.calc is not None else "",
        )
        ws.parent_id = row_id
        if not ws.excluded:
            ws.current_id = row_id
        return row_id

    def set_item(self, workspace_id: str | None, key: str, checked: bool) -> None:
        """Switch an item (adsorbate, cluster, or constraint) on or off for a window."""
        ws = self._ws(workspace_id)
        known = {item["key"]: item for item in ops.list_items(self._require(ws))}
        if key not in known or not known[key]["removable"]:
            raise ValueError(f"no switchable item {key!r} in this window")
        if checked:
            ws.excluded.discard(key)
        else:
            ws.excluded.add(key)

    def export(self, workspace_id: str | None, fmt: str) -> tuple[str, str]:
        """Serialize a window's structure.

        Returns:
            ``(text, file name)``.
        """
        if fmt not in EXPORT_FORMATS:
            raise ValueError(
                f"unknown format {fmt!r}; choose from {sorted(EXPORT_FORMATS)}"
            )
        atoms = self._effective(self._ws(workspace_id)).interface.copy()
        atoms.calc = None
        buffer = io.StringIO()
        ase_write(buffer, atoms, format=fmt)
        return (
            buffer.getvalue(),
            f"{atoms.get_chemical_formula()}.{EXPORT_FORMATS[fmt]}",
        )

    def submit(
        self,
        workspace_id: str | None,
        kind: str,
        calculator: str,
        calculator_params: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        label: str = "",
    ) -> str:
        """Store a window's structure if needed and start a job on it."""
        entries = calculator_entries()
        if calculator not in entries:
            raise ValueError(f"unknown calculator {calculator!r}")
        if not entries[calculator].available:
            raise ValueError(
                f"calculator {calculator!r} needs the '{entries[calculator].requires}' package"
            )
        if kind not in TASKS:
            raise ValueError(f"unknown task {kind!r}")
        spec = JobSpec(
            kind=kind,
            structure_id=self.save(workspace_id),
            calculator=calculator,
            calculator_params=dict(calculator_params or {}),
            params=dict(params or {}),
            label=label,
        )
        return self.executor.submit(spec)

    def jobs(self) -> dict[str, Any]:
        """All jobs with their status, newest first, plus the store size."""
        items = []
        for spec, status in self.executor.jobs():
            items.append(
                {
                    "job_id": spec.job_id,
                    "kind": spec.kind,
                    "calculator": spec.calculator,
                    "structure_id": spec.structure_id,
                    "label": spec.label,
                    "state": status.state,
                    "active": status.active,
                    "progress": status.progress,
                    "message": status.message,
                    "n_structures": status.n_structures,
                }
            )
        return {"jobs": items, "n_structures": len(self.project.store)}

    def _store(self, scope: str):
        if scope == "project":
            return self.project.store
        if scope == "global":
            return global_store()
        raise ValueError(f"unknown database {scope!r}; choose 'project' or 'global'")

    def records(
        self,
        stage: str | None = None,
        job_id: str | None = None,
        order: str = "time",
        scope: str = "project",
        role: str | None = None,
    ) -> dict[str, Any]:
        """List stored structures, optionally ranked.

        The ranking metric is the stored score (e.g. the GCMC grand potential)
        when any listed row has one, otherwise the potential energy. Each row
        gets its ``rank`` (1 = most stable) and ``delta``, the metric relative
        to the best listed row, in eV.

        Args:
            stage: Keep only this workflow stage.
            job_id: Keep only rows produced by this job.
            order: ``"time"`` (ascending id) or ``"rank"`` (most stable first).
            scope: ``"project"`` or ``"global"`` (the user-level reference store).
            role: Keep only this role (bulk, molecule, surface, adsorbed, other).
        """
        found = self._store(scope).records(
            stage=stage or None, job_id=job_id or None, role=role or None
        )
        metric = "score" if any(rec.score is not None for rec in found) else "energy"

        def value(rec: StructureRecord) -> float | None:
            raw = rec.score if metric == "score" else rec.energy
            return None if raw is None else float(raw)

        ranked = sorted((rec for rec in found if value(rec) is not None), key=value)
        rank = {rec.id: position + 1 for position, rec in enumerate(ranked)}
        best = value(ranked[0]) if ranked else None
        if order == "rank":
            found = ranked + [rec for rec in found if value(rec) is None]
        rows = [
            {
                "id": rec.id,
                "formula": rec.formula,
                "natoms": rec.natoms,
                "stage": rec.stage,
                "method": rec.method,
                "calculator": rec.calculator,
                "energy": None if rec.energy is None else float(rec.energy),
                "fmax": None if rec.fmax is None else float(rec.fmax),
                "score": None if rec.score is None else float(rec.score),
                "value": value(rec),
                "delta": None if value(rec) is None else value(rec) - best,
                "rank": rank.get(rec.id),
                "step": rec.step,
                "parent_id": rec.parent_id,
                "job_id": rec.job_id,
                "label": rec.label,
                "role": rec.role,
                "level": rec.level,
                "potential": rec.tags.get("potential"),
            }
            for rec in found
        ]
        return {"rows": rows, "metric": metric, "order": order, "scope": scope}

    def load_record(
        self, workspace_id: str | None, structure_id: int, scope: str = "project"
    ) -> None:
        """Load a stored structure into a window.

        A structure from the global store arrives as an unsaved copy: it keeps
        its level-of-theory label and origin, and is written to the project
        only when saved or used as a job input.
        """
        ws = self._ws(workspace_id)
        store = self._store(scope)
        record = store.record(int(structure_id))
        interface = store.get(int(structure_id))
        if scope == "project":
            self._set(ws, interface, current_id=int(structure_id), level=record.level)
        else:
            self._set(ws, interface, level=record.level, origin=f"global #{record.id}")
            ws.parent_id = None

    def references(self, role: str = "bulk") -> list[dict[str, Any]]:
        """Stored structures of one role from the project and global stores.

        Used to pick a ready-made reference directly, e.g. a relaxed bulk at a
        chosen functional and cutoff as the starting point of a slab.
        """
        items = []
        for scope in ("project", "global"):
            for rec in self._store(scope).records(role=role):
                cell = self._store(scope).get(rec.id).interface.cell
                items.append(
                    {
                        "scope": scope,
                        "id": rec.id,
                        "formula": rec.formula,
                        "level": rec.level,
                        "label": rec.label,
                        "energy_per_atom": None
                        if rec.energy is None
                        else float(rec.energy) / rec.natoms,
                        "cell_lengths": [float(x) for x in cell.lengths()],
                    }
                )
        items.sort(
            key=lambda item: (item["formula"], item["level"], item["scope"], item["id"])
        )
        return items

    def energetics(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Derived energies of a window's structure from matching references.

        References are matched by role and level of theory in the project
        store first, then the global store (see ``gs_gmlip.energetics``).
        """
        ws = self._ws(workspace_id)
        stores = [("project", self.project.store), ("global", global_store())]
        return analyze(self._require(ws), self._role(ws), ws.level, stores).as_dict()

    def delete_record(self, structure_id: int) -> None:
        """Delete a stored structure."""
        self.project.store.delete([int(structure_id)])
        for ws in self.workspaces.values():
            if ws.current_id == int(structure_id):
                ws.current_id = None
                ws.parent_id = None

    def vasp_preview(
        self, workspace_id: str | None = None, **options: Any
    ) -> dict[str, Any]:
        """Resolved VASP tags for a window's structure (see ``vasp.resolve_incar``)."""
        return vasp.resolve_incar(self._effective(self._ws(workspace_id)), **options)

    # ------------------------------------------------------------------
    # Provenance graph
    # ------------------------------------------------------------------

    # Job kinds whose result is the last structure written; any other kind is a
    # search, whose result is its best-ranked structure.
    _TRAJECTORY_KINDS = frozenset({"singlepoint", "relax", "cell_relax", "nvt", "npt"})

    @staticmethod
    def _metric(rec: StructureRecord) -> float:
        """Ranking value of a record: its score if it has one, otherwise its energy."""
        return rec.score if rec.score is not None else rec.energy

    def graph(self) -> dict[str, Any]:
        """Provenance graph of the project: structures and jobs as nodes.

        Intermediate structures of a job (optimizer steps, MD frames, search
        candidates) are folded into the job node; a structure is drawn when it
        is hand-built, the input of a job or of an edit, or a job's result.

        Returns:
            ``{"nodes": [...], "edges": [...]}`` with node ``type`` either
            ``structure`` or ``job``.
        """
        records = {rec.id: rec for rec in self.project.store.records()}
        current_ids = {ws.current_id for ws in self.workspaces.values()}
        jobs = {spec.job_id: (spec, status) for spec, status in self.executor.jobs()}
        by_job: dict[str, list[StructureRecord]] = {}
        for rec in records.values():
            if rec.job_id:
                by_job.setdefault(rec.job_id, []).append(rec)

        shown: set[int] = {rec.id for rec in records.values() if not rec.job_id}
        shown |= {
            spec.structure_id
            for spec, _ in jobs.values()
            if spec.structure_id in records
        }
        shown |= {
            rec.parent_id
            for rec in records.values()
            if not rec.job_id and rec.parent_id in records
        }
        for job_id, produced in by_job.items():
            kind = jobs[job_id][0].kind if job_id in jobs else produced[0].stage
            if kind in self._TRAJECTORY_KINDS:
                shown.add(max(produced, key=lambda rec: rec.id).id)
            else:
                scored = [
                    rec for rec in produced if (rec.score, rec.energy) != (None, None)
                ]
                if scored:
                    best = min(scored, key=self._metric)
                    shown.add(best.id)

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for structure_id in sorted(shown):
            rec = records[structure_id]
            nodes.append(
                {
                    "id": f"s{rec.id}",
                    "type": "structure",
                    "structure_id": rec.id,
                    "label": rec.formula,
                    "stage": rec.stage,
                    "energy": None if rec.energy is None else float(rec.energy),
                    "score": None if rec.score is None else float(rec.score),
                    "current": rec.id in current_ids,
                }
            )
            if rec.job_id:
                edges.append(
                    {
                        "source": f"j{rec.job_id}",
                        "target": f"s{rec.id}",
                        "kind": "output",
                    }
                )
            elif rec.parent_id in records:
                edges.append(
                    {
                        "source": f"s{rec.parent_id}",
                        "target": f"s{rec.id}",
                        "kind": "edit",
                    }
                )
        for job_id in sorted(set(jobs) | set(by_job)):
            spec, status = jobs.get(job_id, (None, None))
            produced = by_job.get(job_id, [])
            nodes.append(
                {
                    "id": f"j{job_id}",
                    "type": "job",
                    "job_id": job_id,
                    "label": spec.kind if spec else produced[0].stage,
                    "calculator": spec.calculator if spec else produced[0].calculator,
                    "state": status.state if status else "done",
                    "n_structures": len(produced),
                }
            )
            input_id = spec.structure_id if spec else produced[0].parent_id
            if input_id in records:
                edges.append(
                    {"source": f"s{input_id}", "target": f"j{job_id}", "kind": "input"}
                )
        return {"nodes": nodes, "edges": edges}
