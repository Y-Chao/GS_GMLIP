// 3Dmol.js viewer adapted to surface systems.
//
// Python owns the structure. This module draws the scene it sends (atoms,
// bonds, cell, sites, constraints), adds what is purely visual (periodic
// images, selection halos, style), and turns mouse gestures into intents:
// picks, site clicks, and drag transforms of the selection.

const STYLES = {
  // PyMOL-like ball and stick: small spheres, sticks coloured by element.
  'ball-stick': {substrate: 0.42, other: 0.42, stick: 0.13, substrateBonds: true},
  // Space-filling substrate, ball-and-stick adsorbates.
  'space-fill': {substrate: 0.95, other: 0.5, stick: 0.13, substrateBonds: false},
};

const PRESETS = {
  top: [0, 0, 0, 1],
  side: [-Math.SQRT1_2, 0, 0, Math.SQRT1_2],
  front: [-0.5, -0.5, -0.5, 0.5],
};

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
const xyz = (p) => ({x: p[0], y: p[1], z: p[2]});

// Rodrigues rotation of point p about the unit axis through anchor by angle (rad).
function rotate(p, anchor, axis, angle) {
  const v = sub(p, anchor);
  const c = Math.cos(angle), s = Math.sin(angle);
  const term = add(add(scale(v, c), scale(cross(axis, v), s)), scale(axis, dot(axis, v) * (1 - c)));
  return add(anchor, term);
}

export class SurfaceViewer {
  constructor(element, dragLayer, handlers) {
    this.element = element;
    this.dragLayer = dragLayer;
    this.handlers = handlers;
    this.viewer = $3Dmol.createViewer(element, {backgroundColor: 'black'});
    this.viewer.setProjection('orthographic');
    this.scene = null;
    this.selection = new Set();
    this.options = {style: 'ball-stick', images: false, white: false};
    this.tool = 'select';
    this.preview = null;
    this.hasView = false;
    this.showCell = true;
    this.frame = 0;
    window.addEventListener('resize', () => { this.viewer.resize(); this.viewer.render(); });
    new ResizeObserver(() => { this.viewer.resize(); this.viewer.render(); }).observe(element);
    this._bindDrag();
  }

  setScene(scene, resetView) {
    this.scene = scene;
    this.preview = null;
    if (resetView) this.hasView = false;
    this.render();
  }

  setSelection(selection) {
    this.selection = selection;
    this._updateDragLayer();
    this.render();
  }

  setOptions(options) {
    Object.assign(this.options, options);
    this.viewer.setBackgroundColor(this.options.white ? 'white' : 'black');
    this.render();
  }

  setTool(tool) {
    this.tool = tool;
    this._updateDragLayer();
  }

  preset(name) {
    const view = this.viewer.getView();
    view.splice(4, 4, ...PRESETS[name]);
    this.viewer.setView(view);
    this.fit();
    this.viewer.render();
  }

  // Zoom so the atoms fill most of the canvas, whatever its aspect ratio.
  fit() {
    const viewer = this.viewer;
    viewer.zoomTo();
    if (!this.scene || !this.scene.atoms.length) return;
    viewer.render();
    const points = this.scene.atoms.map((atom) => viewer.modelToScreen({x: atom.x, y: atom.y, z: atom.z}));
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const width = Math.max(...xs) - Math.min(...xs) + 110, height = Math.max(...ys) - Math.min(...ys) + 110;
    const factor = 0.85 * Math.min(this.element.clientWidth / width, this.element.clientHeight / height);
    if (Number.isFinite(factor) && factor > 0) viewer.zoom(Math.min(Math.max(factor, 0.3), 8));
  }

  screenshot() {
    return this.viewer.pngURI();
  }

  _radius(atom) {
    const style = STYLES[this.options.style];
    const factor = atom.role === 'substrate' ? style.substrate : style.other;
    return Math.max(atom.radius * factor, 0.2);
  }

  _position(atom) {
    return this.preview && this.preview.has(atom.index) ? this.preview.get(atom.index) : [atom.x, atom.y, atom.z];
  }

  render() {
    const viewer = this.viewer, scene = this.scene;
    const view = viewer.getView();
    viewer.removeAllModels();
    viewer.removeAllShapes();
    viewer.removeAllLabels();
    if (!scene) { viewer.render(); return; }

    const style = STYLES[this.options.style];
    const atoms = scene.atoms;
    const positions = atoms.map((atom) => this._position(atom));
    const radii = atoms.map((atom) => this._radius(atom));

    const neighbours = atoms.map(() => []);
    scene.bonds.forEach(([i, j]) => {
      const anySubstrate = atoms[i].role === 'substrate' || atoms[j].role === 'substrate';
      if (style.substrateBonds || !anySubstrate) {
        neighbours[i].push(j);
        neighbours[j].push(i);
      }
    });

    const addModel = (offset, opacity, clickable) => {
      const model = viewer.addModel();
      model.addAtoms(atoms.map((atom, i) => ({
        elem: atom.elem,
        x: positions[i][0] + offset[0],
        y: positions[i][1] + offset[1],
        z: positions[i][2] + offset[2],
        bonds: neighbours[i].slice(),
        bondOrder: neighbours[i].map(() => 1),
        properties: {index: atom.index, color: atom.color, radius: radii[i], faded: !!atom.excluded},
      })));
      const colorfunc = (atom) => atom.properties.color;
      // Items the user switched off stay visible but faded.
      [false, true].forEach((faded) => {
        const alpha = faded ? opacity * 0.22 : opacity;
        new Set(radii).forEach((radius) => {
          model.setStyle({predicate: (atom) => atom.properties.radius === radius && atom.properties.faded === faded}, {
            sphere: {colorfunc, radius, opacity: alpha},
            stick: {colorfunc, radius: style.stick, opacity: alpha},
          });
        });
      });
      if (clickable) {
        model.setClickable({}, true, (atom, _viewer, event) => {
          const extend = !!(event && (event.shiftKey || event.metaKey || event.ctrlKey));
          this.handlers.onPick(atom.properties.index, extend);
        });
      }
    };

    const [va, vb, vc] = scene.cell;
    if (this.options.images) {
      for (let i = -1; i <= 1; i++) {
        for (let j = -1; j <= 1; j++) {
          if (i || j) addModel(add(scale(va, i), scale(vb, j)), 0.4, false);
        }
      }
    }
    addModel([0, 0, 0], 1, true);

    // Selection halos.
    atoms.forEach((atom, i) => {
      if (this.selection.has(atom.index)) {
        viewer.addSphere({center: xyz(positions[i]), radius: radii[i] + 0.2, color: '#ffd24a', opacity: 0.55});
      }
    });

    // Constraints: fixed bonds (magenta), centre-of-mass anchors (cyan), walls (orange).
    scene.constraints.bonds.forEach(([i, j]) => {
      viewer.addCylinder({start: xyz(positions[i]), end: xyz(positions[j]), radius: 0.2,
        color: '#ff4fd8', opacity: 0.7, fromCap: 1, toCap: 1});
    });
    scene.constraints.com.forEach((group) => {
      viewer.addSphere({center: xyz(group.center), radius: 0.2, color: '#35e0e0'});
      group.indices.forEach((i) => viewer.addCylinder({start: xyz(group.center), end: xyz(positions[i]),
        radius: 0.035, color: '#35e0e0', fromCap: 1, toCap: 1}));
    });
    const edge = (p, q, color, radius) => viewer.addCylinder({start: xyz(p), end: xyz(q), radius, color, fromCap: 1, toCap: 1});
    const corners = [[0, 0, 0], va, add(va, vb), vb];
    new Set(scene.constraints.walls.map((wall) => wall.height)).forEach((height) => {
      const lift = scale(vc, height / (vc[2] || 1));
      const quad = corners.map((corner) => add(corner, lift));
      quad.forEach((corner, k) => edge(corner, quad[(k + 1) % 4], '#ffa94d', 0.05));
      // Filled, translucent plane (both windings so it shows from above and below).
      const normal = scale(vc, 1 / norm(vc));
      viewer.addCustom({
        vertexArr: quad.map(xyz),
        normalArr: quad.map(() => xyz(normal)),
        faceArr: [0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2],
        color: '#ffc46b',
        opacity: 0.6,
      });
    });

    // Cell.
    const cellColor = this.options.white ? '#555555' : '#6b7686';
    if (this.showCell) {
      corners.forEach((corner, k) => {
        const next = corners[(k + 1) % 4];
        edge(corner, next, cellColor, 0.03);
        edge(add(corner, vc), add(next, vc), cellColor, 0.03);
        edge(corner, add(corner, vc), cellColor, 0.03);
      });
    }

    // Adsorption sites.
    scene.sites.forEach((site) => {
      viewer.addSphere({center: {x: site.x, y: site.y, z: site.z}, radius: 0.26, color: site.color,
        opacity: 0.95, clickable: true, callback: () => this.handlers.onSite(site.index)});
    });

    viewer.setViewStyle({style: 'outline', color: 'black', width: 0.04});
    if (this.hasView) {
      viewer.setView(view);
    } else {
      viewer.zoomTo();
      const fresh = viewer.getView();
      fresh.splice(4, 4, ...PRESETS.top);
      viewer.setView(fresh);
      this.fit();
      this.hasView = true;
    }
    viewer.render();
  }

  // ----- drag transforms of the selection (move / rotate tools) -----

  _updateDragLayer() {
    const transform = (this.tool === 'move' || this.tool === 'rotate') && this.selection.size > 0;
    this.dragLayer.hidden = !(transform || this.tool === 'box');
    this.dragLayer.classList.toggle('box', this.tool === 'box');
  }

  // Gradients of screen x and y with respect to model coordinates at point c (px per Å).
  _screenGradients(c) {
    const at = (p) => {
      const s = this.viewer.modelToScreen(xyz(p));
      return [s.x, s.y];
    };
    const origin = at(c);
    const gx = [0, 0, 0], gy = [0, 0, 0];
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((unit, k) => {
      const s = at(add(c, unit));
      gx[k] = s[0] - origin[0];
      gy[k] = s[1] - origin[1];
    });
    return {gx, gy};
  }

  _bindDrag() {
    let drag = null;
    const layer = this.dragLayer;

    // Box selection: drag a rectangle, select every atom whose centre projects inside it.
    let box = null;
    const band = document.createElement('div');
    band.className = 'band';
    band.hidden = true;
    layer.append(band);
    const drawBand = () => {
      const rect = layer.getBoundingClientRect();
      band.style.left = `${Math.min(box.x0, box.x1) - rect.left}px`;
      band.style.top = `${Math.min(box.y0, box.y1) - rect.top}px`;
      band.style.width = `${Math.abs(box.x1 - box.x0)}px`;
      band.style.height = `${Math.abs(box.y1 - box.y0)}px`;
    };
    window.addEventListener('mousemove', (event) => {
      if (!box) return;
      box.x1 = event.clientX;
      box.y1 = event.clientY;
      drawBand();
    });
    window.addEventListener('mouseup', (event) => {
      if (!box) return;
      const [left, right] = [Math.min(box.x0, box.x1), Math.max(box.x0, box.x1)];
      const [top, bottom] = [Math.min(box.y0, box.y1), Math.max(box.y0, box.y1)];
      const extend = box.extend;
      box = null;
      band.hidden = true;
      if (right - left < 3 && bottom - top < 3) {
        // A click, not a drag: clear the selection unless extending.
        if (!extend) this.handlers.onBox([], false);
        return;
      }
      const inside = this.scene.atoms.filter((atom) => {
        const p = this.viewer.modelToScreen({x: atom.x, y: atom.y, z: atom.z});
        return p.x >= left && p.x <= right && p.y >= top && p.y <= bottom;
      }).map((atom) => atom.index);
      this.handlers.onBox(inside, extend || event.shiftKey);
    });

    layer.addEventListener('mousedown', (event) => {
      if (event.button === 0 && this.scene && this.tool === 'box') {
        box = {x0: event.clientX, y0: event.clientY, x1: event.clientX, y1: event.clientY,
          extend: event.shiftKey || event.metaKey || event.ctrlKey};
        band.hidden = false;
        drawBand();
        event.preventDefault();
        return;
      }
      if (event.button !== 0 || !this.scene || !this.selection.size) return;
      const chosen = this.scene.atoms.filter((atom) => this.selection.has(atom.index));
      const centroid = scale(chosen.reduce((sum, atom) => add(sum, [atom.x, atom.y, atom.z]), [0, 0, 0]), 1 / chosen.length);
      const {gx, gy} = this._screenGradients(centroid);
      const axis = cross(gx, gy);
      if (norm(axis) < 1e-9) return;
      drag = {x: event.clientX, y: event.clientY, chosen, centroid, gx, gy, axis: scale(axis, 1 / norm(axis)), result: null};
      layer.classList.add('dragging');
      event.preventDefault();
    });

    window.addEventListener('mousemove', (event) => {
      if (!drag) return;
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      const preview = new Map();
      if (this.tool === 'move') {
        // Minimum-norm model displacement whose screen projection is (dx, dy).
        const a = dot(drag.gx, drag.gx), b = dot(drag.gx, drag.gy), d = dot(drag.gy, drag.gy);
        const det = a * d - b * b;
        if (Math.abs(det) < 1e-12) return;
        const wx = (d * dx - b * dy) / det, wy = (a * dy - b * dx) / det;
        const vector = add(scale(drag.gx, wx), scale(drag.gy, wy));
        drag.chosen.forEach((atom) => preview.set(atom.index, add([atom.x, atom.y, atom.z], vector)));
        drag.result = {op: 'move', vector};
      } else {
        const degrees = dx * 0.6;
        const angle = degrees * Math.PI / 180;
        drag.chosen.forEach((atom) => preview.set(atom.index, rotate([atom.x, atom.y, atom.z], drag.centroid, drag.axis, angle)));
        drag.result = {op: 'rotate', angle: degrees, axis: drag.axis, anchor: drag.centroid};
      }
      this.preview = preview;
      cancelAnimationFrame(this.frame);
      this.frame = requestAnimationFrame(() => this.render());
    });

    window.addEventListener('mouseup', () => {
      if (!drag) return;
      const result = drag.result;
      drag = null;
      layer.classList.remove('dragging');
      if (result) this.handlers.onTransform(result);
    });
  }
}
