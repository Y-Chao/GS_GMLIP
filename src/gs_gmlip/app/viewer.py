"""3Dmol.js viewer adapted to surface systems, as a NiceGUI component.

Python owns the structure; the browser only draws the scene produced by
``gs_gmlip.app.scene.build_scene`` and reports clicks back. Compared with a
stock molecule viewer this one draws the cell, faded in-plane periodic
images (never across the vacuum), frozen atoms in a darker tint, adsorbate
bonds, a selection halo, and clickable adsorption-site markers, with an
orthographic camera and top/side presets.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable

from nicegui import ui

THREEDMOL_URL = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"

VIEWER_JS = r"""
window.gsViewer = {
  states: {},
  ready(id) {
    return typeof $3Dmol !== 'undefined' && document.getElementById(id) !== null;
  },
  state(id) {
    if (!this.states[id]) {
      const viewer = $3Dmol.createViewer(document.getElementById(id), {backgroundColor: 'white'});
      viewer.setProjection('orthographic');
      this.states[id] = {viewer: viewer, hasView: false};
      window.addEventListener('resize', () => { viewer.resize(); viewer.render(); });
    }
    return this.states[id];
  },
  render(id, scene, keepView) {
    if (!this.ready(id)) { setTimeout(() => this.render(id, scene, keepView), 150); return; }
    const state = this.state(id), viewer = state.viewer;
    const view = viewer.getView();
    viewer.removeAllModels(); viewer.removeAllShapes(); viewer.removeAllLabels();

    const point = (p) => ({x: p[0], y: p[1], z: p[2]});
    const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

    // Periodic images: faded, not clickable, in-plane only.
    if (scene.ghosts.length) {
      const ghosts = viewer.addModel();
      ghosts.addAtoms(scene.ghosts.map((g) => ({elem: g.elem, x: g.x, y: g.y, z: g.z, bonds: [], bondOrder: [],
        properties: {color: g.color, radius: g.radius}})));
      ghosts.setStyle({}, {sphere: {colorfunc: (atom) => atom.properties.color, scale: 0.42, opacity: 0.28}});
    }

    // Real atoms: substrate space-filling, adsorbates smaller with bonds.
    const model = viewer.addModel();
    model.addAtoms(scene.atoms.map((a) => ({elem: a.elem, x: a.x, y: a.y, z: a.z, bonds: [], bondOrder: [],
      properties: {index: a.index, color: a.color, substrate: a.role === 'substrate'}})));
    const radii = {};
    scene.atoms.forEach((a) => { radii[a.elem] = a.radius; });
    Object.keys(radii).forEach((elem) => {
      model.setStyle({elem: elem, predicate: (atom) => atom.properties.substrate},
        {sphere: {colorfunc: (atom) => atom.properties.color, radius: radii[elem] * 0.95}});
      model.setStyle({elem: elem, predicate: (atom) => !atom.properties.substrate},
        {sphere: {colorfunc: (atom) => atom.properties.color, radius: Math.max(radii[elem] * 0.62, 0.3)}});
    });
    model.setClickable({}, true, (atom, v, event) => {
      emitEvent('gsv_atom', {id: id, index: atom.properties.index,
        shift: !!(event && (event.shiftKey || event.metaKey || event.ctrlKey))});
    });

    scene.bonds.forEach((pair) => {
      const a = scene.atoms[pair[0]], b = scene.atoms[pair[1]];
      viewer.addCylinder({start: {x: a.x, y: a.y, z: a.z}, end: {x: b.x, y: b.y, z: b.z},
        radius: 0.11, color: '#5a5a5a', fromCap: 1, toCap: 1});
    });

    scene.atoms.filter((a) => a.selected).forEach((a) => {
      const radius = (a.role === 'substrate' ? a.radius * 0.95 : Math.max(a.radius * 0.62, 0.3)) + 0.22;
      viewer.addSphere({center: {x: a.x, y: a.y, z: a.z}, radius: radius, color: '#ffd400', opacity: 0.6});
    });

    // Cell edges.
    const [va, vb, vc] = scene.cell, origin = [0, 0, 0];
    const corners = [origin, va, vb, add(va, vb)];
    const edge = (p, q) => viewer.addCylinder({start: point(p), end: point(q), radius: 0.035,
      color: '#333333', fromCap: 1, toCap: 1});
    [[0, 1], [0, 2], [1, 3], [2, 3]].forEach(([i, j]) => {
      edge(corners[i], corners[j]);
      edge(add(corners[i], vc), add(corners[j], vc));
    });
    corners.forEach((c) => edge(c, add(c, vc)));

    // Adsorption sites: clickable markers.
    scene.sites.forEach((site) => {
      viewer.addSphere({center: {x: site.x, y: site.y, z: site.z}, radius: 0.3, color: site.color,
        opacity: 0.9, clickable: true,
        callback: () => emitEvent('gsv_site', {id: id, index: site.index})});
    });

    if (keepView && state.hasView) {
      viewer.setView(view);
    } else {
      viewer.zoomTo();
      state.hasView = true;
      this.orient(viewer, 'top');
    }
    viewer.render();
  },
  orient(viewer, preset) {
    const view = viewer.getView();
    const h = Math.SQRT1_2;
    const quaternion = {top: [0, 0, 0, 1], side: [-h, 0, 0, h], front: [-0.5, -0.5, -0.5, 0.5]}[preset];
    view.splice(4, 4, ...quaternion);
    viewer.setView(view);
    viewer.zoomTo();
    viewer.render();
  },
  preset(id, preset) {
    if (this.ready(id)) { this.orient(this.state(id).viewer, preset); }
  },
};
"""


def install_viewer_assets() -> None:
    """Add 3Dmol.js and the surface-viewer script to the current page."""
    ui.add_head_html(f'<script src="{THREEDMOL_URL}"></script>')
    ui.add_head_html(f"<script>{VIEWER_JS}</script>")


class SurfaceViewer:
    """A 3Dmol canvas that draws scenes and reports atom and site clicks.

    Args:
        on_atom_click: Called with ``(atom_index, extend_selection)``.
        on_site_click: Called with the index of the clicked site marker.
    """

    def __init__(
        self,
        on_atom_click: Callable[[int, bool], Any],
        on_site_click: Callable[[int], Any],
    ):
        self.id = f"gsv-{uuid.uuid4().hex[:8]}"
        self.element = (
            ui.element("div")
            .props(f"id={self.id}")
            .classes("w-full")
            .style("position: relative; height: 100%; min-height: 420px;")
        )

        def atom_event(event) -> None:
            if event.args.get("id") == self.id:
                on_atom_click(int(event.args["index"]), bool(event.args.get("shift")))

        def site_event(event) -> None:
            if event.args.get("id") == self.id:
                on_site_click(int(event.args["index"]))

        ui.on("gsv_atom", atom_event)
        ui.on("gsv_site", site_event)

    def render(self, scene: dict[str, Any], keep_view: bool = True) -> None:
        """Draw a scene; ``keep_view=False`` resets the camera to the top view."""
        ui.run_javascript(
            f"gsViewer.render({json.dumps(self.id)}, {json.dumps(scene)}, {json.dumps(keep_view)})"
        )

    def preset(self, name: str) -> None:
        """Switch the camera to the ``top``, ``side``, or ``front`` view."""
        ui.run_javascript(f"gsViewer.preset({json.dumps(self.id)}, {json.dumps(name)})")
