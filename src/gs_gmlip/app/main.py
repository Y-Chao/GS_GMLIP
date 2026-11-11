"""The interactive app: build a surface model, run or submit jobs, browse the store.

Single-user local tool. One ``AppState`` holds the structure being edited;
every edit goes through ``gs_gmlip.interface.operations`` (pymatgen), every
simulation goes through a ``JobSpec`` handed to an executor, and everything
that is produced is read back from the project's ``StructureStore``.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase.io import read as ase_read
from ase.io import write as ase_write
from nicegui import ui
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from gs_gmlip.app.scene import SITE_COLORS, build_scene
from gs_gmlip.app.viewer import SurfaceViewer, install_viewer_assets
from gs_gmlip.evaluator.calculators import calculator_entries
from gs_gmlip.interface import operations as ops
from gs_gmlip.interface.core import Interface
from gs_gmlip.jobs.executor import LocalExecutor
from gs_gmlip.jobs.spec import JobSpec
from gs_gmlip.jobs.tasks import OPTIMIZERS, TASKS
from gs_gmlip.store import STAGES, Project

_HISTORY_LIMIT = 50
_POLL_SECONDS = 1.5
_STATE_COLORS = {
    "queued": "grey",
    "running": "primary",
    "done": "positive",
    "failed": "negative",
    "cancelled": "warning",
}


class AppState:
    """The structure being edited plus the project it belongs to.

    Args:
        project: Project (store + jobs directory) the app works in.
    """

    def __init__(self, project: Project):
        self.project = project
        self.executor = LocalExecutor(project)
        self.bulk: Structure | None = None
        self.interface: Interface | None = None
        self.history: list[tuple[Interface, int | None]] = []
        self.selection: set[int] = set()
        self.sites: list[ops.AdsorptionSite] = []
        self.show_sites = False
        self.distinct_sites = False
        self.site_height = 2.0
        self.repeat = 0
        # Row id of the current structure if it is stored and unmodified.
        self.current_id: int | None = None
        # Row id the current (possibly edited) structure descends from.
        self.parent_id: int | None = None

    def set_interface(
        self, interface: Interface, current_id: int | None = None
    ) -> None:
        """Replace the edited structure, pushing the previous one onto the undo stack."""
        if self.interface is not None:
            self.history.append((self.interface, self.current_id))
            del self.history[:-_HISTORY_LIMIT]
        self.interface = interface
        self.current_id = current_id
        if current_id is not None:
            self.parent_id = current_id
        self.selection = {i for i in self.selection if i < len(interface.interface)}

    def undo(self) -> bool:
        """Restore the previous structure. Returns False if there is none."""
        if not self.history:
            return False
        self.interface, self.current_id = self.history.pop()
        self.selection.clear()
        return True

    def ensure_saved(self, label: str = "") -> int:
        """Store the current structure if needed and return its row id."""
        if self.interface is None:
            raise ValueError("no structure to save")
        if self.current_id is None:
            self.current_id = self.project.store.add(
                self.interface, stage="built", parent_id=self.parent_id, label=label
            )
            self.parent_id = self.current_id
        return self.current_id


def _param_form(defaults: dict[str, Any]) -> Callable[[], dict[str, Any]]:
    """Render one input per parameter and return a getter for the edited values."""
    getters: dict[str, Callable[[], Any]] = {}
    with ui.grid(columns=2).classes("w-full gap-x-3 gap-y-0"):
        for key, default in defaults.items():
            if key == "optimizer":
                widget = ui.select(list(OPTIMIZERS), value=default, label=key).props(
                    "dense"
                )
                getters[key] = lambda w=widget: w.value
            elif isinstance(default, bool):
                widget = ui.switch(key, value=default)
                getters[key] = lambda w=widget: bool(w.value)
            elif isinstance(default, int):
                widget = ui.number(key, value=default, precision=0).props("dense")
                getters[key] = lambda w=widget: int(w.value)
            elif isinstance(default, float):
                widget = ui.number(key, value=default, format="%g").props("dense")
                getters[key] = lambda w=widget: float(w.value)
            elif isinstance(default, (dict, list)):
                widget = ui.input(f"{key} (JSON)", value=json.dumps(default)).props(
                    "dense"
                )
                getters[key] = lambda w=widget: json.loads(w.value)
            else:
                widget = ui.input(key, value=str(default)).props("dense")
                getters[key] = lambda w=widget: w.value
    return lambda: {key: getter() for key, getter in getters.items()}


def build_page(state: AppState) -> None:  # noqa: C901, PLR0915 - one page, many small closures
    """Build the single app page for ``state``."""
    install_viewer_assets()
    ui.query(".nicegui-content").classes("p-0")
    store = state.project.store
    last_cell: dict[str, Any] = {"cell": None}

    # ------------------------------------------------------------------
    # Shared refresh helpers
    # ------------------------------------------------------------------

    def refresh_sites() -> None:
        state.sites = []
        if (
            state.show_sites
            and state.interface is not None
            and len(state.interface.substrate)
        ):
            try:
                state.sites = ops.find_adsorption_sites(
                    state.interface,
                    distance=state.site_height,
                    symmetry_reduce=state.distinct_sites,
                )
            except Exception as error:  # noqa: BLE001 - surface analysis can fail on odd inputs
                ui.notify(f"Site search failed: {error}", type="warning")

    def refresh_view(sites: bool = False) -> None:
        if sites:
            refresh_sites()
        iface = state.interface
        if iface is None:
            info.set_text(
                "No structure. Build a slab, upload one, or pick a database row."
            )
            return
        cell = np.asarray(iface.interface.cell)
        keep = last_cell["cell"] is not None and np.allclose(
            cell, last_cell["cell"], atol=1e-6
        )
        last_cell["cell"] = cell
        viewer.render(
            build_scene(iface, state.selection, state.sites, state.repeat),
            keep_view=keep,
        )
        atoms = iface.interface
        parts = [
            atoms.get_chemical_formula(),
            f"{len(atoms)} atoms",
            f"{len(iface.fix)} fixed",
            f"{len(iface.adsList)} adsorbates",
            f"{len(iface.clusterList)} clusters",
        ]
        if atoms.calc is not None:
            parts.append(f"E = {atoms.get_potential_energy():.4f} eV")
        parts.append(
            f"id {state.current_id}" if state.current_id is not None else "unsaved"
        )
        info.set_text("  |  ".join(parts))
        selected = sorted(state.selection)
        shown = ", ".join(f"{atoms[i].symbol}{i}" for i in selected[:12])
        more = f" (+{len(selected) - 12})" if len(selected) > 12 else ""
        selection_label.set_text(
            f"Selected: {shown}{more}" if selected else "Selected: none"
        )
        input_label.set_text(
            f"Input: structure id {state.current_id}"
            if state.current_id is not None
            else "Input: current structure (saved to the database on run)"
        )

    def apply_edit(edit: Callable[[Interface], Interface], sites: bool = True) -> None:
        if state.interface is None:
            ui.notify("Build or load a structure first", type="warning")
            return
        try:
            state.set_interface(edit(state.interface))
        except Exception as error:  # noqa: BLE001 - show any builder error to the user
            ui.notify(f"{type(error).__name__}: {error}", type="negative")
            return
        refresh_view(sites=sites)

    def need_selection() -> list[int] | None:
        if not state.selection:
            ui.notify("Select atoms first (click; shift-click to add)", type="warning")
            return None
        return sorted(state.selection)

    # ------------------------------------------------------------------
    # Viewer callbacks
    # ------------------------------------------------------------------

    def on_atom_click(index: int, extend: bool) -> None:
        if extend:
            state.selection ^= {index}
        else:
            state.selection = set() if state.selection == {index} else {index}
        refresh_view()

    def on_site_click(index: int) -> None:
        if index >= len(state.sites):
            return
        site = state.sites[index]
        apply_edit(
            lambda iface: ops.place_adsorbate(
                iface,
                adsorbate_select.value,
                site.coords,
                float(rotation_input.value or 0),
            ),
            sites=False,
        )

    # ------------------------------------------------------------------
    # Build tab actions
    # ------------------------------------------------------------------

    def describe_bulk() -> None:
        bulk = state.bulk
        if bulk is None:
            bulk_label.set_text("No bulk structure yet")
            return
        spacegroup = SpacegroupAnalyzer(bulk).get_space_group_symbol()
        a, b, c = bulk.lattice.abc
        bulk_label.set_text(
            f"Bulk: {bulk.composition.reduced_formula}, {spacegroup}, "
            f"a = {a:.3f}, b = {b:.3f}, c = {c:.3f} Å, {len(bulk)} atoms"
        )

    def create_bulk() -> None:
        try:
            state.bulk = ops.make_bulk(
                element_input.value.strip(),
                lattice_select.value,
                float(a_input.value),
                float(c_input.value) if c_input.value else None,
            )
        except Exception as error:  # noqa: BLE001
            ui.notify(f"{type(error).__name__}: {error}", type="negative")
            return
        describe_bulk()

    async def on_upload(event) -> None:
        name = Path(event.file.name).name
        content = await event.file.read()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(content)
            try:
                try:
                    structure = ops.load_structure(path)
                except Exception:  # noqa: BLE001 - fall back to ASE readers (extxyz, traj, ...)
                    structure = AseAtomsAdaptor.get_structure(ase_read(str(path)))
            except Exception as error:  # noqa: BLE001
                ui.notify(f"Could not read {name}: {error}", type="negative")
                return
        if upload_mode.value == "bulk":
            state.bulk = structure
            describe_bulk()
            ui.notify(f"Loaded {name} as bulk")
        else:
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.set_pbc([True, True, True])
            state.set_interface(Interface(substrate=atoms))
            state.parent_id = None
            refresh_view(sites=True)
            ui.notify(f"Loaded {name} as slab (all atoms are substrate)")

    def build_slab() -> None:
        if state.bulk is None:
            create_bulk()
        if state.bulk is None:
            return
        try:
            interface = ops.build_slab(
                state.bulk,
                (int(h_input.value), int(k_input.value), int(l_input.value)),
                layers=int(layers_input.value),
                vacuum=float(vacuum_input.value),
                supercell=(int(na_input.value), int(nb_input.value)),
                fix_layers=int(fix_input.value),
                primitive=bool(primitive_switch.value),
            )
        except Exception as error:  # noqa: BLE001
            ui.notify(f"{type(error).__name__}: {error}", type="negative")
            return
        state.selection.clear()
        state.set_interface(interface)
        state.parent_id = None
        refresh_view(sites=True)

    def toggle_sites() -> None:
        state.show_sites = bool(sites_switch.value)
        state.distinct_sites = bool(distinct_switch.value)
        state.site_height = float(height_input.value or 2.0)
        refresh_view(sites=True)

    def delete_selected() -> None:
        chosen = need_selection()
        if chosen:
            state.selection.clear()
            apply_edit(lambda iface: ops.delete_atoms(iface, chosen))

    def substitute_selected() -> None:
        chosen = need_selection()
        if chosen:
            apply_edit(
                lambda iface: ops.substitute_atoms(
                    iface, chosen, substitute_input.value.strip()
                )
            )

    def fix_selected(fixed: bool) -> None:
        chosen = need_selection()
        if chosen:
            apply_edit(lambda iface: ops.set_fixed(iface, chosen, fixed), sites=False)

    def move_selected() -> None:
        chosen = need_selection()
        if chosen:
            vector = [float(w.value or 0) for w in (dx_input, dy_input, dz_input)]
            apply_edit(lambda iface: ops.translate_atoms(iface, chosen, vector))

    def clear_selection() -> None:
        state.selection.clear()
        refresh_view()

    def undo() -> None:
        if state.undo():
            refresh_view(sites=True)
        else:
            ui.notify("Nothing to undo")

    def save_structure() -> None:
        if state.interface is None:
            ui.notify("Nothing to save", type="warning")
            return
        if state.current_id is not None:
            ui.notify(f"Already stored as id {state.current_id}")
            return
        row_id = state.ensure_saved(label=save_label.value or "")
        ui.notify(f"Saved as id {row_id}", type="positive")
        refresh_view()
        refresh_database(force=True)

    def download(fmt: str, extension: str) -> None:
        if state.interface is None:
            ui.notify("Nothing to download", type="warning")
            return
        atoms = state.interface.interface.copy()
        buffer = io.StringIO()
        ase_write(buffer, atoms, format=fmt)
        ui.download.content(
            buffer.getvalue(), f"{atoms.get_chemical_formula()}.{extension}"
        )

    # ------------------------------------------------------------------
    # Run tab actions
    # ------------------------------------------------------------------

    task_getter: dict[str, Callable[[], dict[str, Any]]] = {}

    def render_task_form() -> None:
        task_form.clear()
        task = TASKS[task_select.value]
        with task_form:
            ui.label(task.description).classes("text-caption text-grey-8")
            task_getter["get"] = _param_form(task.defaults)

    def render_calculator_form() -> None:
        entry = calculator_entries()[calculator_select.value]
        calculator_params.set_value(json.dumps(entry.defaults))
        note = entry.description + (
            "" if entry.available else f"  (needs '{entry.requires}')"
        )
        calculator_note.set_text(note)

    def submit_job() -> None:
        if state.interface is None:
            ui.notify("Build or load a structure first", type="warning")
            return
        entry = calculator_entries()[calculator_select.value]
        if not entry.available:
            ui.notify(
                f"Calculator '{entry.name}' needs the '{entry.requires}' package",
                type="negative",
            )
            return
        try:
            params = task_getter["get"]()
            calc_params = json.loads(calculator_params.value or "{}")
        except (ValueError, TypeError) as error:
            ui.notify(f"Invalid parameters: {error}", type="negative")
            return
        spec = JobSpec(
            kind=task_select.value,
            structure_id=state.ensure_saved(),
            calculator=entry.name,
            calculator_params=calc_params,
            params=params,
            label=job_label.value or "",
        )
        job_id = state.executor.submit(spec)
        ui.notify(f"Started {job_id}", type="positive")
        refresh_view()
        poll(force=True)

    def show_log(job_id: str) -> None:
        with (
            ui.dialog() as dialog,
            ui.card().classes("w-full").style("max-width: 900px"),
        ):
            ui.label(job_id).classes("text-subtitle2")
            ui.code(state.executor.log(job_id) or "(empty)", language="text").classes(
                "w-full"
            )
            ui.button("Close", on_click=dialog.close).props("flat")
        dialog.open()

    def show_results(job_id: str) -> None:
        job_filter.set_options(job_options(), value=job_id)
        stage_filter.set_value("all")
        tabs.set_value("Database")
        panels.set_value("Database")
        refresh_database(force=True)

    @ui.refreshable
    def jobs_view() -> None:
        jobs = state.executor.jobs()
        if not jobs:
            ui.label("No jobs yet").classes("text-grey-7")
        for spec, status in jobs:
            with ui.card().classes("w-full q-pa-sm").props("flat bordered"):
                with ui.row().classes("w-full items-center no-wrap"):
                    ui.badge(
                        status.state, color=_STATE_COLORS.get(status.state, "grey")
                    )
                    ui.label(spec.job_id).classes("text-caption text-weight-medium")
                    ui.space()
                    ui.label(
                        f"{spec.calculator}, input id {spec.structure_id}"
                    ).classes("text-caption")
                ui.linear_progress(value=status.progress, show_value=False).props(
                    "instant-feedback"
                )
                ui.label(status.message or "waiting").classes(
                    "text-caption text-grey-8"
                )
                with ui.row().classes("items-center"):
                    ui.label(f"{status.n_structures} structures").classes(
                        "text-caption"
                    )
                    ui.button(
                        "Results", on_click=lambda j=spec.job_id: show_results(j)
                    ).props("flat dense")
                    ui.button("Log", on_click=lambda j=spec.job_id: show_log(j)).props(
                        "flat dense"
                    )
                    if status.active:
                        ui.button(
                            "Cancel",
                            on_click=lambda j=spec.job_id: (
                                state.executor.cancel(j),
                                poll(force=True),
                            ),
                        ).props("flat dense color=negative")

    # ------------------------------------------------------------------
    # Database tab actions
    # ------------------------------------------------------------------

    visible: dict[str, list] = {"records": []}

    def job_options() -> list[str]:
        return ["all"] + [spec.job_id for spec, _ in state.executor.jobs()]

    def load_record(row_id: int) -> None:
        try:
            interface = store.get(row_id)
        except KeyError:
            ui.notify(f"Structure {row_id} no longer exists", type="warning")
            return
        state.set_interface(interface, current_id=row_id)
        refresh_view(sites=True)
        chain = " → ".join(str(rec.id) for rec in store.lineage(row_id))
        lineage_label.set_text(f"Lineage: {chain}")

    def refresh_database(force: bool = False) -> None:
        records = store.records(
            stage=None if stage_filter.value == "all" else stage_filter.value,
            job_id=None if job_filter.value == "all" else job_filter.value,
        )
        if not force and [r.id for r in records] == [r.id for r in visible["records"]]:
            return
        visible["records"] = records
        table.rows = [
            {
                "id": rec.id,
                "formula": rec.formula,
                "stage": rec.stage,
                "method": rec.method,
                "calc": rec.calculator,
                "energy": None if rec.energy is None else round(float(rec.energy), 4),
                "fmax": None if rec.fmax is None else round(float(rec.fmax), 4),
                "step": rec.step,
                "parent": rec.parent_id,
                "label": rec.label,
            }
            for rec in reversed(records)
        ]
        table.update()
        with_energy = [rec for rec in records if rec.energy is not None]
        plot.update_figure(
            {
                "data": [
                    {
                        "x": list(range(len(with_energy))),
                        "y": [float(rec.energy) for rec in with_energy],
                        "customdata": [rec.id for rec in with_energy],
                        "text": [f"id {rec.id}, {rec.stage}" for rec in with_energy],
                        "mode": "lines+markers",
                        "type": "scatter",
                        "marker": {"size": 5, "color": "#1f77b4"},
                        "line": {"width": 1, "color": "#1f77b4"},
                        "hovertemplate": "%{text}<br>E = %{y:.4f} eV<extra></extra>",
                    }
                ],
                "layout": {
                    "margin": {"l": 60, "r": 10, "t": 10, "b": 40},
                    "height": 230,
                    "xaxis": {"title": {"text": "Structure (in listed order)"}},
                    "yaxis": {"title": {"text": "Energy (eV)"}},
                    "font": {"size": 11},
                },
            }
        )
        frame_slider.props(f"max={max(len(records) - 1, 0)}")
        frame_slider.set_enabled(len(records) > 1)

    def on_frame(event) -> None:
        records = visible["records"]
        index = int(event.value or 0)
        if 0 <= index < len(records):
            load_record(records[index].id)

    def on_plot_click(event) -> None:
        points = event.args.get("points") or []
        if points and points[0].get("customdata") is not None:
            load_record(int(points[0]["customdata"]))

    def delete_current_row() -> None:
        if state.current_id is None:
            ui.notify("The current structure is not a stored row", type="warning")
            return
        store.delete([state.current_id])
        ui.notify(f"Deleted id {state.current_id}")
        state.current_id = None
        state.parent_id = None
        refresh_view()
        refresh_database(force=True)

    # ------------------------------------------------------------------
    # Polling (jobs write to the store from other processes)
    # ------------------------------------------------------------------

    seen: dict[str, Any] = {"jobs": None, "count": -1}

    def poll(force: bool = False) -> None:
        jobs = state.executor.jobs()
        signature = [
            (
                spec.job_id,
                status.state,
                round(status.progress, 2),
                status.n_structures,
                status.message,
            )
            for spec, status in jobs
        ]
        if force or signature != seen["jobs"]:
            previous = job_filter.value
            options = ["all"] + [spec.job_id for spec, _ in jobs]
            job_filter.set_options(
                options, value=previous if previous in options else "all"
            )
            seen["jobs"] = signature
            jobs_view.refresh()
        count = len(store)
        if force or count != seen["count"]:
            seen["count"] = count
            refresh_database(force=force)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    with ui.header().classes("items-center q-py-xs bg-blue-grey-10"):
        ui.label("GS-GMLIP").classes("text-h6")
        ui.label("surface and reaction-mechanism exploration").classes("text-caption")
        ui.space()
        ui.label(f"Project: {state.project.root}").classes("text-caption")

    with (
        ui.splitter(value=36)
        .classes("w-full")
        .style("height: calc(100vh - 52px)") as splitter
    ):
        with splitter.before:
            with ui.tabs().classes("w-full").props("dense") as tabs:
                ui.tab("Build")
                ui.tab("Run")
                ui.tab("Database")
            with (
                ui.tab_panels(tabs, value="Build")
                .classes("w-full")
                .style("height: calc(100vh - 92px); overflow-y: auto") as panels
            ):
                # ---------------- Build ----------------
                with ui.tab_panel("Build").classes("q-pa-sm"):
                    with ui.expansion("1. Bulk", value=True).classes("w-full"):
                        with ui.grid(columns=4).classes("w-full gap-x-3 gap-y-0"):
                            element_input = ui.input("Element", value="Cu").props(
                                "dense"
                            )
                            lattice_select = ui.select(
                                list(ops.BULK_LATTICES), value="fcc", label="Lattice"
                            ).props("dense")
                            a_input = ui.number("a (Å)", value=3.61, format="%g").props(
                                "dense"
                            )
                            c_input = ui.number("c (Å, hcp)", format="%g").props(
                                "dense"
                            )
                        with ui.row().classes("items-center"):
                            ui.button("Create bulk", on_click=create_bulk).props(
                                "dense outline"
                            )
                            upload_mode = ui.toggle(
                                {"bulk": "as bulk", "slab": "as slab"}, value="bulk"
                            ).props("dense")
                        ui.upload(
                            on_upload=on_upload,
                            auto_upload=True,
                            label="Upload CIF / POSCAR / extxyz",
                        ).props("dense flat bordered").classes("w-full")
                        bulk_label = ui.label("No bulk structure yet").classes(
                            "text-caption text-grey-8"
                        )

                    with ui.expansion("2. Slab", value=True).classes("w-full"):
                        with ui.grid(columns=5).classes("w-full gap-x-3 gap-y-0"):
                            h_input = ui.number("h", value=1, precision=0).props(
                                "dense"
                            )
                            k_input = ui.number("k", value=1, precision=0).props(
                                "dense"
                            )
                            l_input = ui.number("l", value=1, precision=0).props(
                                "dense"
                            )
                            na_input = ui.number(
                                "Repeat a", value=3, min=1, precision=0
                            ).props("dense")
                            nb_input = ui.number(
                                "Repeat b", value=3, min=1, precision=0
                            ).props("dense")
                            layers_input = ui.number(
                                "Unit planes", value=4, min=1, precision=0
                            ).props("dense")
                            vacuum_input = ui.number(
                                "Vacuum (Å)", value=15.0, format="%g"
                            ).props("dense")
                            fix_input = ui.number(
                                "Fix layers", value=2, min=0, precision=0
                            ).props("dense")
                        with ui.row().classes("items-center"):
                            primitive_switch = ui.switch("Primitive surface cell")
                            ui.button("Build slab", on_click=build_slab).props(
                                "dense unelevated"
                            )

                    with ui.expansion("3. Adsorbates", value=True).classes("w-full"):
                        with ui.grid(columns=3).classes("w-full gap-x-3 gap-y-0"):
                            adsorbate_select = ui.select(
                                list(ops.ADSORBATES), value="CO", label="Adsorbate"
                            ).props("dense")
                            height_input = ui.number(
                                "Height (Å)",
                                value=2.0,
                                format="%g",
                                on_change=lambda: toggle_sites(),
                            ).props("dense")
                            rotation_input = ui.number(
                                "Rotation (°)", value=0, format="%g"
                            ).props("dense")
                        with ui.row().classes("items-center"):
                            sites_switch = ui.switch(
                                "Show sites", on_change=lambda: toggle_sites()
                            )
                            distinct_switch = ui.switch(
                                "Symmetry-distinct only",
                                on_change=lambda: toggle_sites(),
                            )
                        with ui.row().classes("items-center gap-3"):
                            for kind, color in SITE_COLORS.items():
                                ui.html(
                                    f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
                                    f'background:{color};margin-right:4px"></span>{kind}'
                                ).classes("text-caption")
                        ui.label(
                            "Click a site marker in the viewer to place the adsorbate there."
                        ).classes("text-caption text-grey-8")

                    with ui.expansion("4. Edit selection", value=True).classes(
                        "w-full"
                    ):
                        selection_label = ui.label("Selected: none").classes(
                            "text-caption"
                        )
                        with ui.row().classes("items-center"):
                            ui.button("Delete", on_click=delete_selected).props(
                                "dense outline color=negative"
                            )
                            ui.button("Fix", on_click=lambda: fix_selected(True)).props(
                                "dense outline"
                            )
                            ui.button(
                                "Release", on_click=lambda: fix_selected(False)
                            ).props("dense outline")
                            ui.button("Clear", on_click=clear_selection).props(
                                "dense flat"
                            )
                        with ui.row().classes("items-center no-wrap"):
                            substitute_input = (
                                ui.input("Element", value="Pt")
                                .props("dense")
                                .classes("w-24")
                            )
                            ui.button("Substitute", on_click=substitute_selected).props(
                                "dense outline"
                            )
                        with ui.row().classes("items-center no-wrap"):
                            dx_input = (
                                ui.number("dx (Å)", value=0, format="%g")
                                .props("dense")
                                .classes("w-20")
                            )
                            dy_input = (
                                ui.number("dy (Å)", value=0, format="%g")
                                .props("dense")
                                .classes("w-20")
                            )
                            dz_input = (
                                ui.number("dz (Å)", value=0, format="%g")
                                .props("dense")
                                .classes("w-20")
                            )
                            ui.button("Move", on_click=move_selected).props(
                                "dense outline"
                            )

                    with ui.row().classes("items-center no-wrap w-full q-pt-sm"):
                        save_label = ui.input("Label").props("dense").classes("grow")
                        ui.button("Save to database", on_click=save_structure).props(
                            "dense unelevated"
                        )

                # ---------------- Run ----------------
                with ui.tab_panel("Run").classes("q-pa-sm"):
                    input_label = ui.label("Input: none").classes("text-caption")
                    entries = calculator_entries()
                    calculator_select = (
                        ui.select(
                            {
                                name: name
                                if entry.available
                                else f"{name} (not installed)"
                                for name, entry in entries.items()
                            },
                            value="emt",
                            label="Calculator",
                            on_change=lambda: render_calculator_form(),
                        )
                        .props("dense")
                        .classes("w-full")
                    )
                    calculator_note = ui.label("").classes("text-caption text-grey-8")
                    calculator_params = (
                        ui.input("Calculator parameters (JSON)", value="{}")
                        .props("dense")
                        .classes("w-full")
                    )
                    task_select = (
                        ui.select(
                            list(TASKS),
                            value="relax",
                            label="Task",
                            on_change=lambda: render_task_form(),
                        )
                        .props("dense")
                        .classes("w-full")
                    )
                    task_form = ui.column().classes("w-full gap-0")
                    with ui.row().classes("items-center no-wrap w-full"):
                        job_label = ui.input("Label").props("dense").classes("grow")
                        ui.button("Run locally", on_click=submit_job).props(
                            "dense unelevated"
                        )
                    ui.separator().classes("q-my-sm")
                    ui.label("Jobs").classes("text-subtitle2")
                    jobs_view()

                # ---------------- Database ----------------
                with ui.tab_panel("Database").classes("q-pa-sm"):
                    with ui.row().classes("items-center no-wrap w-full"):
                        stage_filter = (
                            ui.select(
                                ["all", *STAGES],
                                value="all",
                                label="Stage",
                                on_change=lambda: refresh_database(force=True),
                            )
                            .props("dense")
                            .classes("w-32")
                        )
                        job_filter = (
                            ui.select(
                                job_options(),
                                value="all",
                                label="Job",
                                on_change=lambda: refresh_database(force=True),
                            )
                            .props("dense")
                            .classes("grow")
                        )
                    plot = ui.plotly({"data": [], "layout": {"height": 230}}).classes(
                        "w-full"
                    )
                    plot.on("plotly_click", on_plot_click)
                    with ui.row().classes("items-center no-wrap w-full"):
                        ui.label("Frame").classes("text-caption")
                        frame_slider = ui.slider(
                            min=0, max=0, value=0, on_change=on_frame
                        ).classes("grow")
                    lineage_label = ui.label("Lineage: none").classes("text-caption")
                    table = (
                        ui.table(
                            columns=[
                                {
                                    "name": name,
                                    "label": label,
                                    "field": name,
                                    "sortable": True,
                                    "align": "left",
                                }
                                for name, label in [
                                    ("id", "id"),
                                    ("formula", "Formula"),
                                    ("stage", "Stage"),
                                    ("method", "Method"),
                                    ("calc", "Calculator"),
                                    ("energy", "E (eV)"),
                                    ("fmax", "fmax (eV/Å)"),
                                    ("step", "Step"),
                                    ("parent", "Parent"),
                                    ("label", "Label"),
                                ]
                            ],
                            rows=[],
                            row_key="id",
                            pagination=15,
                        )
                        .props("dense flat")
                        .classes("w-full")
                    )
                    table.on(
                        "rowClick", lambda event: load_record(int(event.args[1]["id"]))
                    )
                    ui.button("Delete current row", on_click=delete_current_row).props(
                        "dense flat color=negative"
                    )

        with splitter.after:
            with ui.column().classes("w-full h-full no-wrap gap-0"):
                with ui.row().classes("items-center q-px-sm q-py-xs gap-2"):
                    ui.button("Top", on_click=lambda: viewer.preset("top")).props(
                        "dense flat"
                    )
                    ui.button("Side", on_click=lambda: viewer.preset("side")).props(
                        "dense flat"
                    )
                    ui.button("Front", on_click=lambda: viewer.preset("front")).props(
                        "dense flat"
                    )

                    def set_repeat(event) -> None:
                        state.repeat = 1 if event.value else 0
                        refresh_view()

                    ui.switch("Periodic images", on_change=set_repeat)
                    ui.button("Undo", on_click=undo).props("dense flat")
                    with ui.dropdown_button("Download", auto_close=True).props(
                        "dense flat"
                    ):
                        ui.item("extxyz", on_click=lambda: download("extxyz", "xyz"))
                        ui.item("POSCAR", on_click=lambda: download("vasp", "vasp"))
                        ui.item("CIF", on_click=lambda: download("cif", "cif"))
                info = ui.label("").classes("text-caption q-px-sm")
                with ui.element("div").classes("w-full grow").style("min-height: 0"):
                    viewer = SurfaceViewer(on_atom_click, on_site_click)

    render_calculator_form()
    render_task_form()
    refresh_view()
    poll(force=True)
    ui.timer(_POLL_SECONDS, poll)


def run_app(
    project: str | Path = "gs_project",
    host: str = "127.0.0.1",
    port: int = 8080,
    show: bool = True,
) -> None:
    """Start the app server (blocks until stopped).

    Args:
        project: Project directory holding ``structures.db`` and ``jobs/``.
        host: Interface to bind.
        port: Port to serve on.
        show: Open a browser window on start.
    """
    state = AppState(Project(project))

    @ui.page("/")
    def index() -> None:
        build_page(state)

    ui.run(
        host=host,
        port=port,
        title="GS-GMLIP",
        reload=False,
        show=show,
        show_welcome_message=False,
    )
