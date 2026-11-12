// GM2-Surf frontend: all interaction and rendering. The Python server does the
// chemistry; this file only sends intents and redraws from the JSON it gets back.
//
// The stage holds several windows (bulk, surface, molecule), each with its own
// 3D viewer, selection, and server-side structure. One window is active: the
// tool column, the side panel, and the Run/Data tabs act on it.

import {SurfaceViewer} from './viewer.js';
import {drawChart, drawGraph} from './graph.js';

const $ = (id) => document.getElementById(id);
const num = (id) => parseFloat($(id).value);
const int = (id) => parseInt($(id).value, 10);

const app = {
  meta: null,
  windows: new Map(), // id -> {id, name, kind, element, viewer, state, selection, cellKey}
  active: null,
  molecules: [],
  tool: 'select',
  toolWindow: null,
  options: {style: 'ball-stick', images: false, white: false},
  records: {rows: [], metric: 'energy'},
  jobsKey: '',
  nStructures: -1,
  forms: {calc: () => ({}), task: () => ({})},
  jobStates: new Map(), // job id -> last seen state
  following: new Map(), // job id -> {ws, last} for jobs shown live in a window
  waiting: false,
};

const EMPTY_TEXT = {
  bulk: 'Empty bulk window.\nCreate a crystal in the Build tab, load a file, or pick a row in the Data tab.',
  surface: 'Empty surface window.\nCleave a slab from a bulk window, load a file, or pick a row in the Data tab.',
  molecule: 'Empty molecule window.\nCreate a molecule in the Build tab or load an xyz file.',
};

const USER_SOURCES = ['temporary', 'project', 'global'];

// ---------------------------------------------------------------- helpers

function toast(message, kind = 'info') {
  const node = document.createElement('div');
  node.className = `toast ${kind}`;
  node.textContent = message;
  $('toasts').append(node);
  setTimeout(() => node.remove(), kind === 'error' ? 7000 : 3200);
}

// Append the window id to an API path.
function withWs(path, ws = app.active) {
  return `${path}${path.includes('?') ? '&' : '?'}ws=${encodeURIComponent(ws || '')}`;
}

async function api(path, {method = 'GET', body, raw} = {}) {
  const options = {method};
  if (raw !== undefined) {
    options.body = raw;
  } else if (body !== undefined) {
    options.headers = {'Content-Type': 'application/json'};
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch { /* keep status text */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return response.json();
}

// Run a request that returns a window state, redraw, and report errors.
async function act(request, done) {
  try {
    const result = await request();
    const state = result && result.state ? result.state : result;
    if (state && 'scene' in state) applyState(state);
    if (done) done(state && 'scene' in state ? state : result, result);
    return result;
  } catch (error) {
    toast(error.message, 'error');
    return null;
  }
}

const post = (path, body = {}, ws = app.active) => api(withWs(path, ws), {method: 'POST', body});
const current = () => app.windows.get(app.active);
const selected = (win = current()) => [...win.selection].sort((a, b) => a - b);
const edit = (op, params = {}, done, ws = app.active) => act(() => post('/api/edit', {op, ...params}, ws), done);

function fillSelect(select, items, value) {
  const previous = value !== undefined ? value : select.value;
  select.replaceChildren(...items.map((item) => {
    const [key, label] = Array.isArray(item) ? item : [item, item];
    const option = document.createElement('option');
    option.value = key;
    option.textContent = label;
    return option;
  }));
  if ([...select.options].some((option) => option.value === previous)) select.value = previous;
}

// Build one input per parameter; returns a getter for the edited values.
function buildForm(container, defaults, choices = {}) {
  container.replaceChildren();
  const getters = {};
  Object.entries(defaults).forEach(([key, value]) => {
    const label = document.createElement('label');
    label.textContent = key;
    let input;
    if (choices[key]) {
      input = document.createElement('select');
      fillSelect(input, choices[key], value);
      getters[key] = () => input.value;
    } else if (typeof value === 'boolean') {
      input = document.createElement('select');
      fillSelect(input, [['true', 'true'], ['false', 'false']], String(value));
      getters[key] = () => input.value === 'true';
    } else if (typeof value === 'number') {
      input = document.createElement('input');
      input.type = 'number';
      input.step = 'any';
      input.value = value;
      getters[key] = () => parseFloat(input.value);
    } else if (value !== null && typeof value === 'object') {
      input = document.createElement('input');
      input.value = JSON.stringify(value);
      label.textContent = `${key} (JSON)`;
      label.classList.add('wide');
      getters[key] = () => JSON.parse(input.value || 'null');
    } else {
      input = document.createElement('input');
      input.value = value ?? '';
      getters[key] = () => input.value;
    }
    label.append(input);
    container.append(label);
  });
  return () => Object.fromEntries(Object.entries(getters).map(([key, get]) => [key, get()]));
}

// ---------------------------------------------------------------- windows

function createWindow(info) {
  const element = document.createElement('div');
  element.className = 'win';
  element.dataset.id = info.id;
  element.innerHTML = `
    <div class="win-head"><span class="kind ${info.kind}">${info.kind}</span><span class="name"></span>
      <span class="formula muted"></span><span class="spacer"></span>
      <button class="close" title="Close this window">×</button></div>
    <div class="win-body"><div class="viewer"></div><div class="drag-layer" hidden></div><div class="empty"></div></div>`;
  element.querySelector('.empty').innerText = EMPTY_TEXT[info.kind];
  $('panes').append(element);
  const win = {id: info.id, name: info.name, kind: info.kind, element, state: null, selection: new Set(), cellKey: ''};
  win.viewer = new SurfaceViewer(element.querySelector('.viewer'), element.querySelector('.drag-layer'), {
    onPick(index, extend) {
      activate(win.id);
      if (extend) {
        const next = new Set(win.selection);
        if (next.has(index)) next.delete(index); else next.add(index);
        setSelection(next, win);
      } else {
        setSelection(win.selection.size === 1 && win.selection.has(index) ? [] : [index], win);
      }
    },
    onSite(index) {
      activate(win.id);
      edit('place', {site: index, adsorbate: $('ads-name').value, rotation: num('ads-rotation') || 0}, null, win.id);
    },
    onBox(indices, extend) {
      activate(win.id);
      setSelection(extend ? new Set([...win.selection, ...indices]) : indices, win);
    },
    onTransform(result) {
      const {op, ...params} = result;
      edit(op, {...params, indices: selected(win)}, null, win.id);
    },
  });
  win.viewer.showCell = info.kind !== 'molecule';
  win.viewer.setOptions(app.options);
  element.addEventListener('mousedown', () => activate(win.id), true);
  element.querySelector('.close').addEventListener('click', async (event) => {
    event.stopPropagation();
    try {
      syncWindows((await api(`/api/windows/${win.id}`, {method: 'DELETE'})).windows);
      renderPanel();
    } catch (error) {
      toast(error.message, 'error');
    }
  });
  app.windows.set(win.id, win);
  return win;
}

// Make the browser's windows match the server's list.
function syncWindows(list) {
  const ids = new Set(list.map((info) => info.id));
  [...app.windows.values()].forEach((win) => {
    if (!ids.has(win.id)) {
      win.element.remove();
      app.windows.delete(win.id);
    }
  });
  list.forEach((info) => {
    const win = app.windows.get(info.id) || createWindow(info);
    win.name = info.name;
    win.element.querySelector('.name').textContent = info.name;
    win.element.querySelector('.formula').textContent = info.formula || '';
  });
  if (!app.windows.has(app.active)) app.active = list.length ? list[0].id : null;
  renderWindowTabs();
  renderWindowTargets();
  app.windows.forEach((win) => win.element.classList.toggle('active', win.id === app.active));
}

function renderWindowTabs() {
  $('wintabs').replaceChildren(...[...app.windows.values()].map((win) => {
    const tab = document.createElement('button');
    tab.className = win.id === app.active ? 'active' : '';
    tab.innerHTML = `<span class="kind ${win.kind}">${win.kind}</span><span></span>`;
    tab.lastChild.textContent = win.name;
    tab.addEventListener('click', () => activate(win.id));
    return tab;
  }));
}

function renderWindowTargets() {
  const surfaces = [...app.windows.values()].filter((win) => win.kind === 'surface').map((win) => [win.id, win.name]);
  fillSelect($('slab-target'), [...surfaces, ['new', 'New surface window']]);
  fillSelect($('lib-target'), surfaces.length ? surfaces : [['', 'no surface window']]);
}

function activate(id) {
  if (!app.windows.has(id)) return;
  const changed = app.active !== id;
  app.active = id;
  app.windows.forEach((win) => {
    win.element.classList.toggle('active', win.id === id);
    win.viewer.setTool(win.id === id ? toolMode() : 'select');
  });
  if (changed) {
    renderWindowTabs();
    renderPanel();
  }
}

function setSelection(indices, win = current()) {
  win.selection = new Set(indices);
  win.viewer.setSelection(win.selection);
  if (win.id === app.active) updateStatus();
}

// ---------------------------------------------------------------- state -> screen

function applyState(state) {
  syncWindows(state.windows);
  app.molecules = state.molecules;
  const win = app.windows.get(state.workspace.id);
  if (!win) return;
  win.state = state;
  const scene = state.scene;
  win.element.querySelector('.empty').hidden = !!scene;
  win.selection = scene ? new Set(selected(win).filter((index) => index < scene.atoms.length)) : new Set();
  const cellKey = scene ? JSON.stringify(scene.cell.map((row) => row.map((x) => x.toFixed(4)))) : '';
  const reset = cellKey !== win.cellKey;
  win.cellKey = cellKey;
  win.viewer.selection = win.selection;
  win.viewer.setScene(scene, reset);
  win.viewer.setSelection(win.selection);
  renderMolecules();
  if (win.id === app.active) renderPanel();
}

// Everything in the side panel and status bar that depends on the active window.
function renderPanel() {
  const win = current();
  if (!win) return;
  const state = win.state;
  document.querySelectorAll('#pane-build [data-kind]').forEach((section) => { section.hidden = section.dataset.kind !== win.kind; });
  const label = $('active-window');
  label.innerHTML = `<span class="kind ${win.kind}">${win.kind}</span><strong></strong><span class="muted"></span>`;
  label.children[1].textContent = win.name;
  label.children[2].textContent = state && state.info ? state.info.formula : 'empty';
  if (state) {
    $('ads-sites').checked = state.sites.show;
    $('ads-distinct').checked = state.sites.distinct;
    $('run-input').textContent = state.current_id !== null
      ? `Input: ${win.name}, structure #${state.current_id}`
      : (state.scene ? `Input: ${win.name} (stored when the job starts)` : `Input: ${win.name} is empty`);
    $('lineage').textContent = state.lineage.length ? `Lineage: ${state.lineage.map((id) => `#${id}`).join(' → ')}` : 'Lineage: none';
  }
  renderItems();
  if (app.taskKind !== win.kind) { app.taskKind = win.kind; renderTaskChoices(); }
  updateStatus();
  updateTools();
  markCurrentRow();
  if (activeTab() === 'data') refreshEnergetics();
}

function updateStatus() {
  const win = current();
  const info = win && win.state && win.state.info;
  if (!info) {
    $('status-main').textContent = win ? `${win.name}: empty` : 'No window';
    $('status-selection').textContent = '';
    return;
  }
  const parts = [win.name, info.formula, `${info.natoms} atoms`];
  if (win.kind === 'surface') parts.push(`${info.n_fixed} fixed`, `${info.n_adsorbates} ads`);
  parts.push(`cell ${info.cell_lengths.map((x) => x.toFixed(2)).join(' × ')} Å`);
  const constraints = [];
  if (info.n_bond_constraints) constraints.push(`${info.n_bond_constraints} bond`);
  if (info.n_com_constraints) constraints.push(`${info.n_com_constraints} COM`);
  if (info.n_walls) constraints.push(`${info.n_walls} wall`);
  if (constraints.length) parts.push(`constraints: ${constraints.join(', ')}`);
  if (info.energy !== null) parts.push(`E = ${info.energy.toFixed(4)} eV`);
  if (win.state.level) parts.push(win.state.level);
  parts.push(win.state.current_id !== null ? `#${win.state.current_id}` : (win.state.origin ? `copy of ${win.state.origin}` : 'unsaved'));
  $('status-main').textContent = parts.join('  |  ');
  const atoms = win.state.scene.atoms;
  const chosen = selected(win);
  $('status-selection').textContent = chosen.length
    ? `selected ${chosen.length}: ${chosen.slice(0, 10).map((i) => `${atoms[i].elem}${i}`).join(' ')}${chosen.length > 10 ? ' …' : ''}`
    : '';
}

// ---------------------------------------------------------------- objects, references, energetics

function renderItems() {
  const win = current();
  const box = $('items');
  box.replaceChildren();
  const items = win && win.state ? win.state.items : [];
  if (!items.length) {
    box.innerHTML = '<div class="muted">Nothing in this window yet.</div>';
    return;
  }
  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = `item${item.checked ? '' : ' off'}`;
    const check = document.createElement('input');
    check.type = 'checkbox';
    check.checked = item.checked;
    check.disabled = !item.removable;
    check.title = item.removable ? 'Include this item in what is saved, exported, and run' : 'The substrate is always included';
    check.addEventListener('change', () => act(() => post('/api/items', {key: item.key, checked: check.checked})));
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = item.kind;
    const text = document.createElement('span');
    text.className = 'text';
    text.textContent = item.label;
    text.title = 'Click to select these atoms';
    text.addEventListener('click', () => setSelection(item.atoms));
    row.append(check, tag, text);
    if (item.removable) {
      const remove = document.createElement('button');
      remove.textContent = '×';
      remove.title = 'Remove this item';
      remove.addEventListener('click', () => edit('remove_item', {key: item.key}, () => setSelection([])));
      row.append(remove);
    }
    box.append(row);
  });
}

async function refreshReferences() {
  try {
    const {references} = await api('/api/references?role=bulk');
    fillSelect($('ref-bulk'), references.length
      ? references.map((ref) => [`${ref.scope}:${ref.id}`,
        `${ref.formula} · ${ref.level || 'no level'} · a = ${ref.cell_lengths[0].toFixed(3)} Å (${ref.scope} #${ref.id})`])
      : [['', 'no stored bulk structures yet']]);
  } catch (error) {
    toast(error.message, 'error');
  }
}

async function refreshEnergetics() {
  const win = current();
  const box = $('energetics');
  if (!win || !win.state || !win.state.scene) { box.textContent = 'The active window is empty.'; return; }
  let result;
  try {
    result = await api(withWs('/api/energetics'));
  } catch (error) {
    box.textContent = error.message;
    return;
  }
  box.replaceChildren();
  const head = document.createElement('div');
  head.className = 'head';
  head.textContent = `role: ${result.role || 'none'}   level: ${result.level || 'none'}`;
  box.append(head);
  const line = (name, value, cls = 'value') => {
    const row = document.createElement('div');
    row.className = cls;
    row.innerHTML = '<span></span><span></span>';
    row.children[0].textContent = name;
    row.children[1].textContent = value;
    box.append(row);
  };
  if (result.energy !== null) {
    line('total energy (eV)', result.energy.toFixed(4));
    line('energy per atom (eV)', result.energy_per_atom.toFixed(4));
  }
  Object.entries(result.values).forEach(([name, value]) => line(name, value.toFixed(4)));
  Object.entries(result.references).forEach(([name, text]) => line(`reference: ${name}`, text, 'value muted'));
  result.missing.forEach((text) => {
    const row = document.createElement('div');
    row.className = 'warn';
    row.textContent = text;
    box.append(row);
  });
}

// ---------------------------------------------------------------- molecule library

function renderMolecules() {
  const reaction = $('ads-reaction').value;
  const matches = app.molecules.filter((molecule) => {
    if (reaction === '') return true;
    if (reaction === 'mine') return USER_SOURCES.includes(molecule.source);
    return molecule.reactions.includes(reaction);
  });
  const label = (molecule) => `${molecule.name}  (${molecule.formula}, ${molecule.source})`;
  fillSelect($('ads-name'), matches.map((molecule) => [molecule.name, label(molecule)]));
  fillSelect($('lib-open-name'), app.molecules.map((molecule) => [molecule.name, label(molecule)]));
  showMoleculeNote();

  const mine = app.molecules.filter((molecule) => USER_SOURCES.includes(molecule.source));
  const list = $('lib-list');
  list.replaceChildren();
  if (!mine.length) {
    list.innerHTML = '<div class="muted">No user molecules yet.</div>';
    return;
  }
  mine.forEach((molecule) => {
    const item = document.createElement('div');
    item.className = 'lib-item';
    item.innerHTML = `<span class="badge ${molecule.source === 'temporary' ? 'temporary' : 'permanent'}">${molecule.source}</span>
      <strong></strong><span class="muted mono"></span><span class="spacer"></span><button class="danger" title="Remove from the library">Remove</button>`;
    item.children[1].textContent = molecule.name;
    item.children[2].textContent = `${molecule.formula} ${molecule.reactions.join(' ')}`;
    item.title = molecule.note;
    item.querySelector('button').addEventListener('click', () => act(
      () => api(withWs(`/api/library/${encodeURIComponent(molecule.name)}`), {method: 'DELETE'}),
      () => toast(`Removed ${molecule.name}`)));
    list.append(item);
  });
}

function showMoleculeNote() {
  const molecule = app.molecules.find((item) => item.name === $('ads-name').value);
  $('ads-note').textContent = molecule ? molecule.note : '';
}

async function addToLibrary() {
  const win = current();
  const name = $('lib-name').value.trim() || (win.state && win.state.info ? win.state.info.formula : '');
  const chosen = selected(win);
  const body = {
    name,
    scope: $('lib-scope').value,
    bind_index: chosen.length === 1 ? chosen[0] : null,
    reactions: $('lib-reactions').value.split(',').map((tag) => tag.trim()).filter(Boolean),
    note: $('lib-note').value,
  };
  const result = await act(() => post('/api/library', body), () => toast(`Added ${name} to the ${body.scope} library`));
  return result ? name : null;
}

// ---------------------------------------------------------------- tool column

const ICONS = {
  select: '<path d="M5 3l14 8-6 2-2 6z"/>',
  box: '<rect x="4" y="5" width="16" height="14" stroke-dasharray="3 3"/><circle cx="9" cy="10" r="1.6"/><circle cx="14" cy="14" r="1.6"/>',
  all: '<rect x="4" y="4" width="16" height="16" stroke-dasharray="3 3"/><circle cx="12" cy="12" r="2.5"/>',
  none: '<rect x="4" y="4" width="16" height="16" stroke-dasharray="3 3"/><path d="M9 9l6 6M15 9l-6 6"/>',
  move: '<path d="M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3"/>',
  rotate: '<path d="M20 12a8 8 0 1 1-3-6.2M20 4v5h-5"/>',
  fix: '<path d="M7 11V8a5 5 0 0 1 10 0v3M6 11h12v9H6z"/>',
  release: '<path d="M7 11V8a5 5 0 0 1 9.5-2M6 11h12v9H6z"/>',
  layers: '<path d="M12 4l9 5-9 5-9-5zM3 14l9 5 9-5"/>',
  vacuum: '<rect x="5" y="3" width="14" height="18" stroke-dasharray="3 3"/><path d="M8 15h8M8 18h8M12 5v6M12 5l-2 2M12 5l2 2M12 11l-2-2M12 11l2-2"/>',
  substitute: '<path d="M4 8h13l-3-3M20 16H7l3 3"/>',
  remove: '<path d="M5 7h14M9 7V4h6v3M7 7l1 13h8l1-13"/>',
  bond: '<circle cx="6" cy="12" r="3"/><circle cx="18" cy="12" r="3"/><path d="M9 12h6"/>',
  com: '<circle cx="12" cy="12" r="8"/><path d="M12 4v16M4 12h16"/>',
  wall: '<path d="M3 6h18M6 6l-2 3M11 6l-2 3M16 6l-2 3M21 6l-2 3"/><circle cx="12" cy="16" r="3"/>',
  clear: '<circle cx="6" cy="12" r="3"/><circle cx="18" cy="12" r="3"/><path d="M10 9l4 6M14 9l-4 6"/>',
  water: '<path d="M12 3c4 5 6 8 6 11a6 6 0 0 1-12 0c0-3 2-6 6-11z"/>',
  undo: '<path d="M9 6L4 11l5 5M4 11h9a6 6 0 0 1 0 12h-2"/>',
};

const needSelection = (count) => {
  const size = current().selection.size;
  if (count ? size !== count : size === 0) {
    toast(count ? `Select exactly ${count} atoms (shift-click adds)` : 'Select atoms first (click; shift-click adds)', 'error');
    return false;
  }
  return true;
};

const TOOLS = [
  {id: 'select', name: 'Select', title: 'Select and orbit: click atoms, shift-click to add (Esc)', mode: 'select'},
  {
    id: 'box', name: 'Box', title: 'Box select: drag a rectangle to select the atoms inside (shift adds)', mode: 'box',
    hint: 'Drag a rectangle to select atoms; hold shift to add to the selection',
  },
  {id: 'all', name: 'Adsorb.', title: 'Select all adsorbate atoms', run: () => setSelection(current().state.scene.atoms.filter((a) => a.role !== 'substrate').map((a) => a.index))},
  {id: 'none', name: 'Deselect', title: 'Clear the selection', run: () => setSelection([])},
  'sep',
  {
    id: 'move', name: 'Move', title: 'Move selected atoms: drag in the window or enter a vector', mode: 'move',
    hint: 'Drag to move the selection in the screen plane',
    fields: {'dx (Å)': 0, 'dy (Å)': 0, 'dz (Å)': 0},
    apply: (v) => needSelection() && edit('move', {indices: selected(), vector: [v['dx (Å)'], v['dy (Å)'], v['dz (Å)']]}),
  },
  {
    id: 'rotate', name: 'Rotate', title: 'Rotate selected atoms: drag in the window or enter an angle', mode: 'rotate',
    hint: 'Drag left or right to rotate the selection about the view axis',
    fields: {'angle (°)': 90, axis: 'z'},
    choices: {axis: ['x', 'y', 'z']},
    apply: (v) => needSelection() && edit('rotate', {indices: selected(), angle: v['angle (°)'],
      axis: {x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1]}[v.axis]}),
  },
  {
    id: 'substitute', name: 'Element', title: 'Substitute the element of selected atoms',
    fields: {element: 'Pt'},
    apply: (v) => needSelection() && edit('substitute', {indices: selected(), element: v.element}),
  },
  {id: 'remove', name: 'Delete', title: 'Delete selected atoms', run: () => needSelection() && edit('delete', {indices: selected()}, () => setSelection([]))},
  'sep',
  {id: 'fix', name: 'Fix', title: 'Fix selected substrate atoms', run: () => needSelection() && edit('fix', {indices: selected()})},
  {id: 'release', name: 'Release', title: 'Release selected substrate atoms', run: () => needSelection() && edit('release', {indices: selected()})},
  {
    id: 'layers', name: 'Layers', title: 'Fix the bottom atomic layers',
    fields: {layers: 2},
    apply: (v) => edit('fix_layers', {layers: v.layers}),
  },
  {id: 'bond', name: 'Bond', title: 'Constrain the distance between two selected atoms (ASE FixBondLengths)', run: () => needSelection(2) && edit('bond', {indices: selected()})},
  {id: 'com', name: 'COM', title: 'Constrain the centre of mass of the selection (ASE FixSubsetCom)', run: () => needSelection() && edit('com', {indices: selected()})},
  {
    id: 'wall', name: 'Wall', title: 'Add a soft wall above the selection (ASE Hookean plane)',
    fields: () => ({'height z (Å)': Math.round((current().state.info.top_z + 3) * 100) / 100, 'k (eV/Å²)': 5}),
    apply: (v) => needSelection() && edit('wall', {indices: selected(), height: v['height z (Å)'], spring_constant: v['k (eV/Å²)']}),
  },
  {id: 'clear', name: 'Unconstr.', title: 'Remove bond, centre-of-mass, and wall constraints', run: () => edit('clear_constraints')},
  'sep',
  {
    id: 'vacuum', name: 'Vacuum', title: 'Vacuum: set its thickness and place the slab at the centre or bottom of the cell',
    fields: () => {
      const info = current().state.info;
      const atoms = current().state.scene.atoms;
      const thickness = Math.max(...atoms.map((a) => a.z)) - Math.min(...atoms.map((a) => a.z));
      return {'vacuum (Å)': Math.round((info.cell_lengths[2] - thickness) * 100) / 100, position: 'center'};
    },
    choices: {position: ['center', 'bottom']},
    note: 'Vacuum is the empty space along the normal (cell height minus slab and adsorbate thickness). "bottom" leaves 1 Å below the slab.',
    apply: (v) => edit('vacuum', {vacuum: v['vacuum (Å)'], position: v.position}),
  },
  {
    id: 'water', name: 'Water', title: 'Fill the space above the surface with water (Packmol)',
    fields: {'density (g/cm³)': 1.0, molecules: '', 'gap (Å)': 2.0, 'thickness (Å)': '', 'tolerance (Å)': 2.0, seed: 0},
    note: 'Leave "molecules" empty to use the density, and "thickness" empty to fill all the vacuum.',
    apply: (v) => edit('water', {density: v['density (g/cm³)'], n_molecules: v.molecules || null, gap: v['gap (Å)'],
      thickness: v['thickness (Å)'] || null, tolerance: v['tolerance (Å)'], seed: v.seed}),
  },
  'sep',
  {id: 'undo', name: 'Undo', title: 'Undo the last change in the active window (Ctrl+Z)', run: () => act(() => post('/api/undo'))},
];

function toolMode() {
  const tool = TOOLS.find((item) => item.id === app.tool);
  return (tool && tool.mode) || 'select';
}

function openToolWindow(tool) {
  const box = $('tool-window');
  if (!tool || !tool.fields) { box.hidden = true; app.toolWindow = null; return; }
  app.toolWindow = tool.id;
  box.hidden = false;
  box.replaceChildren();
  const title = document.createElement('h4');
  title.textContent = tool.title.split(':')[0];
  const form = document.createElement('div');
  form.className = 'grid g2';
  const fields = typeof tool.fields === 'function' ? tool.fields() : tool.fields;
  const values = buildForm(form, fields, tool.choices || {});
  const row = document.createElement('div');
  row.className = 'row';
  const apply = document.createElement('button');
  apply.className = 'primary';
  apply.textContent = 'Apply';
  apply.addEventListener('click', () => {
    const raw = values();
    Object.keys(raw).forEach((key) => {
      if (fields[key] === '' && raw[key] !== '') raw[key] = parseFloat(raw[key]);
    });
    tool.apply(raw);
  });
  const close = document.createElement('button');
  close.textContent = 'Close';
  close.addEventListener('click', () => { openToolWindow(null); if (tool.mode) setTool('select'); else updateTools(); });
  row.append(apply, close);
  box.append(title, form);
  if (tool.hint || tool.note) {
    const note = document.createElement('p');
    note.textContent = tool.note || `${tool.hint}, or enter values.`;
    box.append(note);
  }
  box.append(row);
}

function setTool(id) {
  const tool = TOOLS.find((item) => item.id === id);
  app.tool = id;
  app.windows.forEach((win) => win.viewer.setTool(win.id === app.active ? (tool.mode || 'select') : 'select'));
  $('hint').hidden = !tool.hint;
  $('hint').textContent = tool.hint ? `${tool.hint}. Esc returns to select.` : '';
  openToolWindow(tool.fields ? tool : null);
  updateTools();
}

function updateTools() {
  const win = current();
  const state = win && win.state;
  const hasScene = !!(state && state.scene);
  document.querySelectorAll('#tools button').forEach((button) => {
    const tool = TOOLS.find((item) => item.id === button.dataset.tool);
    button.classList.toggle('active', tool.mode ? app.tool === tool.id : app.toolWindow === tool.id);
    button.disabled = !hasScene || (tool.id === 'undo' && !state.can_undo) || (tool.id === 'water' && !app.meta.packmol);
  });
}

function buildTools() {
  TOOLS.forEach((tool) => {
    if (tool === 'sep') {
      const sep = document.createElement('div');
      sep.className = 'sep';
      $('tools').append(sep);
      return;
    }
    const button = document.createElement('button');
    button.dataset.tool = tool.id;
    button.title = tool.id === 'water' && !app.meta.packmol ? `${tool.title} — packmol is not installed` : tool.title;
    button.innerHTML = `<svg viewBox="0 0 24 24">${ICONS[tool.id]}</svg><span>${tool.name}</span>`;
    button.addEventListener('click', () => {
      if (tool.run) { tool.run(); return; }
      if (tool.mode) { setTool(tool.id); return; }
      openToolWindow(app.toolWindow === tool.id ? null : tool);
      updateTools();
    });
    $('tools').append(button);
  });
}

// ---------------------------------------------------------------- build tab

function bindBuild() {
  $('bulk-create').addEventListener('click', () => act(() => post('/api/bulk/create', {
    element: $('bulk-element').value, lattice: $('bulk-lattice').value, a: num('bulk-a'), c: num('bulk-c') || null})));

  $('ref-bulk-load').addEventListener('click', () => {
    const [scope, id] = $('ref-bulk').value.split(':');
    if (!id) { toast('No stored bulk structures yet; import or calculate one first', 'error'); return; }
    act(() => api(withWs(`/api/records/${id}/load?scope=${scope}`), {method: 'POST', body: {}}), (state) => toast(`Loaded ${state.info.formula} (${state.level || 'no level'})`));
  });

  $('slab-build').addEventListener('click', () => act(() => post('/api/cleave', {
    miller: [int('slab-h'), int('slab-k'), int('slab-l')], layers: int('slab-layers'), vacuum: num('slab-vacuum'),
    supercell: [int('slab-na'), int('slab-nb')], fix_layers: int('slab-fix'), primitive: $('slab-primitive').checked,
    conventional: $('slab-conventional').checked,
    target: $('slab-target').value === 'new' ? null : $('slab-target').value,
  }), (state) => {
    setSelection([], app.windows.get(state.workspace.id));
    activate(state.workspace.id);
    toast(`Cleaved into ${state.workspace.name}`);
  }));

  const sites = () => act(() => post('/api/sites', {
    show: $('ads-sites').checked, distinct: $('ads-distinct').checked, height: num('ads-height') || 2.0}));
  ['ads-sites', 'ads-distinct', 'ads-height'].forEach((id) => $(id).addEventListener('change', sites));
  $('ads-reaction').addEventListener('change', renderMolecules);
  $('ads-name').addEventListener('change', showMoleculeNote);

  $('mol-create').addEventListener('click', () => act(() => post('/api/molecule/create', {name: $('mol-name').value.trim()})));
  $('lib-open').addEventListener('click', () => act(
    () => post(`/api/library/${encodeURIComponent($('lib-open-name').value)}/open`), () => setSelection([])));
  $('lib-add').addEventListener('click', addToLibrary);
  $('lib-send').addEventListener('click', async () => {
    const target = $('lib-target').value;
    if (!target) { toast('Open a surface window first', 'error'); return; }
    const name = await addToLibrary();
    if (!name) return;
    activate(target);
    $('ads-reaction').value = '';
    renderMolecules();
    $('ads-name').value = name;
    showMoleculeNote();
    await act(() => post('/api/sites', {show: true, distinct: $('ads-distinct').checked, height: num('ads-height') || 2.0}, target));
    toast(`${name} is selected: click a site marker in ${app.windows.get(target).name} to place it`);
  });

  $('upload-file').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    await act(async () => api(withWs(`/api/upload?name=${encodeURIComponent(file.name)}`),
      {method: 'POST', raw: await file.arrayBuffer()}), () => toast(`Loaded ${file.name} into ${current().name}`));
    event.target.value = '';
  });
  $('save-btn').addEventListener('click', () => act(() => post('/api/save', {label: $('save-label').value}), (state) => {
    toast(`Stored as #${state.current_id}`);
    refreshData();
  }));

  document.querySelectorAll('[data-add]').forEach((button) => button.addEventListener('click', () => act(
    () => api('/api/windows', {method: 'POST', body: {kind: button.dataset.add}}), (state) => activate(state.workspace.id))));
  $('layout').addEventListener('change', (event) => { $('panes').className = `${event.target.value}${app.options.white ? ' white' : ''}`; });
}

// ---------------------------------------------------------------- run tab

function renderCalculator() {
  const entry = app.meta.calculators.find((item) => item.name === $('calc-name').value);
  $('calc-note').textContent = entry.description + (entry.available ? '' : `  (needs the '${entry.requires}' package)`);
  app.forms.calc = buildForm($('calc-params'), entry.defaults, entry.choices);
}

// Tasks offered depend on the active window's model type and the chosen category.
function renderTaskChoices() {
  const win = current();
  const kind = win ? win.kind : 'surface';
  const usable = app.meta.tasks.filter((task) => task.applies_to.includes(kind));
  const categories = app.meta.task_categories.filter((name) => usable.some((task) => task.category === name));
  fillSelect($('task-category'), categories, $('task-category').value || 'optimization');
  const inCategory = usable.filter((task) => task.category === $('task-category').value);
  fillSelect($('task-kind'), inCategory.map((task) => task.kind));
  renderTask();
}

function renderTask() {
  const task = app.meta.tasks.find((item) => item.kind === $('task-kind').value);
  if (!task) { $('task-note').textContent = 'No task of this category applies to this window.'; $('task-params').replaceChildren(); return; }
  $('task-note').textContent = task.description;
  const choices = {optimizer: app.meta.optimizers};
  const isVasp = task.kind === 'vasp_inputs';
  if (isVasp) {
    choices.preset = app.meta.vasp.presets.map((preset) => [preset.name, `${preset.name}: ${preset.description}`]);
    choices.precision = app.meta.vasp.precisions;
    choices.dispersion = app.meta.vasp.dispersions;
  }
  app.forms.task = buildForm($('task-params'), task.defaults, choices);
  $('vasp-preview-row').hidden = !isVasp;
  $('vasp-incar').hidden = true;
}

function renderJobs(jobs) {
  const box = $('jobs');
  box.replaceChildren();
  if (!jobs.length) {
    box.innerHTML = '<div class="muted">No jobs yet</div>';
    return;
  }
  jobs.forEach((job) => {
    const card = document.createElement('div');
    card.className = 'job';
    card.innerHTML = `
      <div class="head"><span class="badge ${job.state}">${job.state}</span><span class="id"></span><span class="spacer"></span>
        <span class="muted where"></span></div>
      <div class="bar"><div style="width:${Math.round(job.progress * 100)}%"></div></div>
      <div class="msg"></div>
      <div class="row"><span class="muted">${job.n_structures} structures</span></div>`;
    card.querySelector('.id').textContent = job.job_id;
    card.querySelector('.where').textContent = `${job.calculator}, input #${job.structure_id}, ${job.host || 'local'}`;
    card.querySelector('.msg').textContent = job.message || 'waiting';
    const row = card.querySelector('.row');
    const button = (text, handler, cls = '') => {
      const node = document.createElement('button');
      node.textContent = text;
      node.className = cls;
      node.addEventListener('click', handler);
      row.append(node);
    };
    button('Results', () => showJob(job.job_id));
    button('Log', async () => {
      const result = await api(`/api/jobs/${job.job_id}/log`);
      $('log-title').textContent = job.job_id;
      $('log-text').textContent = result.log || '(empty)';
      $('log-dialog').showModal();
    });
    if (job.active) button('Cancel', () => api(`/api/jobs/${job.job_id}/cancel`, {method: 'POST', body: {}}).then(poll), 'danger');
    box.append(card);
  });
}

function bindRun() {
  $('calc-name').addEventListener('change', renderCalculator);
  $('task-kind').addEventListener('change', renderTask);
  $('vasp-preview').addEventListener('click', async () => {
    try {
      const result = await post('/api/vasp/preview', app.forms.task());
      const lines = Object.entries(result.incar).map(([key, value]) => `${key.padEnd(10)} = ${Array.isArray(value) ? value.join(' ') : value}`);
      lines.push('', `KPOINTS   : ${result.kpoints ? result.kpoints.map((row) => row.join(' ')).join('; ') : 'from KSPACING'}`,
        `POTCAR    : ${result.potcar.join(' ')}`, `slab tags : ${result.slab ? 'applied' : 'not applied (bulk)'}`,
        `selective dynamics: ${result.selective_dynamics ? 'yes (fixed atoms)' : 'no'}`);
      $('vasp-incar').textContent = lines.join('\n');
      $('vasp-incar').hidden = false;
    } catch (error) {
      toast(error.message, 'error');
    }
  });
  $('job-run').addEventListener('click', async () => {
    let params, calculatorParams;
    try {
      params = app.forms.task();
      calculatorParams = app.forms.calc();
    } catch (error) {
      toast(`Invalid parameters: ${error.message}`, 'error');
      return;
    }
    const ws = app.active;
    const host = $('job-host').value;
    await act(() => post('/api/jobs', {kind: $('task-kind').value, calculator: $('calc-name').value,
      calculator_params: calculatorParams, params, label: $('job-label').value, host}), (_state, result) => {
      toast(host ? `Submitted ${result.job_id} to ${host}` : `Started ${result.job_id}`);
      if ($('job-follow').checked) app.following.set(result.job_id, {ws, last: null});
    });
    poll();
  });
  $('log-close').addEventListener('click', () => $('log-dialog').close());
  bindHosts();
  bindWorkflow();
  $('task-category').addEventListener('change', renderTaskChoices);
  $('jobs-wait').addEventListener('click', waitForAll);
}

// ---------------------------------------------------------------- workflows

// "temperature = 300, 600" per line -> {temperature: [300, 600]}; values are JSON where possible.
function parseSweep(text) {
  const sweep = {};
  text.split('\n').map((line) => line.trim()).filter(Boolean).forEach((line) => {
    const cut = line.indexOf('=');
    if (cut < 1) throw new Error(`sweep line needs "parameter = values": ${line}`);
    const values = line.slice(cut + 1).split(',').map((item) => item.trim()).filter(Boolean).map((item) => {
      try { return JSON.parse(item); } catch { return item; }
    });
    if (!values.length) throw new Error(`sweep over ${line.slice(0, cut).trim()} has no values`);
    sweep[line.slice(0, cut).trim()] = values;
  });
  return sweep;
}

function renderWorkflowSteps() {
  const box = $('wf-steps');
  box.replaceChildren();
  app.workflowSteps.forEach((step, index) => {
    const row = document.createElement('div');
    row.className = 'wf-step';
    const text = document.createElement('span');
    text.className = 'grow';
    const sweep = Object.entries(step.sweep).map(([name, values]) => `${name} × ${values.length}`).join(', ');
    text.textContent = `${index + 1}. ${step.kind} · ${step.calculator} · ${step.host || 'local'}`
      + `${sweep ? ` · sweep ${sweep}` : ''} → pass on ${step.take === 'best' ? `best ${step.n}` : step.take}`;
    const remove = document.createElement('button');
    remove.textContent = '×';
    remove.title = 'Remove this step';
    remove.addEventListener('click', () => { app.workflowSteps.splice(index, 1); renderWorkflowSteps(); });
    row.append(text, remove);
    box.append(row);
  });
  $('wf-run').disabled = !app.workflowSteps.length;
}

function bindWorkflow() {
  app.workflowSteps = [];
  renderWorkflowSteps();
  $('wf-add').addEventListener('click', () => {
    try {
      app.workflowSteps.push({kind: $('task-kind').value, calculator: $('calc-name').value, calculator_params: app.forms.calc(),
        params: app.forms.task(), host: $('job-host').value, take: $('wf-take').value, n: Number($('wf-n').value) || 1,
        sweep: parseSweep($('wf-sweep').value)});
    } catch (error) {
      toast(`Invalid step: ${error.message}`, 'error');
      return;
    }
    $('wf-sweep').value = '';
    renderWorkflowSteps();
  });
  $('wf-clear').addEventListener('click', () => { app.workflowSteps = []; renderWorkflowSteps(); });
  $('wf-run').addEventListener('click', async () => {
    const body = {name: $('wf-name').value.trim() || 'workflow', steps: app.workflowSteps, max_parallel: Number($('wf-parallel').value) || 4};
    if ($('wf-input').value === 'rows') {
      if (app.records.scope !== 'project') { toast('Batch runs start from the project database; switch the Data tab to Project', 'error'); return; }
      body.inputs = app.records.rows.map((row) => row.id);
      if (!body.inputs.length) { toast('The Data tab lists no rows; open it and set its filters first', 'error'); return; }
      if (!window.confirm(`Start "${body.name}" on ${body.inputs.length} structures?`)) return;
    }
    await act(() => post('/api/workflows', body), (_state, result) => {
      toast(`Started workflow ${result.workflow_id}`);
      app.workflowsActive = true;
      activateTab('flow');
    });
    poll();
  });
}

async function refreshWorkflows() {
  let workflows;
  try {
    workflows = (await api('/api/workflows')).workflows;
  } catch {
    return;
  }
  app.workflowsActive = workflows.some((workflow) => workflow.active);
  const box = $('workflows');
  if (!workflows.length) return;
  box.classList.remove('muted');
  box.replaceChildren();
  workflows.forEach((workflow) => {
    const card = document.createElement('div');
    card.className = 'wf-card';
    const head = document.createElement('div');
    head.className = 'row';
    const badge = document.createElement('span');
    badge.className = `badge ${workflow.state}`;
    badge.textContent = workflow.state;
    const name = document.createElement('strong');
    name.className = 'grow';
    name.textContent = `${workflow.name}  (${workflow.workflow_id})`;
    head.append(badge, name);
    if (workflow.active) {
      const cancel = document.createElement('button');
      cancel.textContent = 'Cancel';
      cancel.className = 'danger';
      cancel.addEventListener('click', () => api(`/api/workflows/${workflow.workflow_id}/cancel`, {method: 'POST', body: {}}).then(refreshWorkflows).then(poll));
      head.append(cancel);
    }
    const list = document.createElement('ol');
    workflow.steps.forEach((step, index) => {
      const item = document.createElement('li');
      const counts = Object.entries(step.counts).map(([state, count]) => `${count} ${state}`).join(', ');
      const sweep = Object.entries(step.sweep).map(([key, count]) => `${key} × ${count}`).join(', ');
      item.textContent = `${step.kind} · ${step.calculator} · ${step.host || 'local'}${sweep ? ` · sweep ${sweep}` : ''}`
        + ` — ${step.inputs === null ? 'not started' : `${step.inputs} input${step.inputs === 1 ? '' : 's'}; ${counts || 'no jobs yet'}`}`
        + ` → ${step.take}`;
      if (workflow.active && index === workflow.step) item.style.fontWeight = '600';
      list.append(item);
    });
    card.append(head, list);
    const note = [workflow.message, ...workflow.errors].filter(Boolean).join(' · ');
    if (note) {
      const message = document.createElement('div');
      message.className = 'muted';
      message.textContent = note;
      card.append(message);
    }
    box.append(card);
  });
}

// ---------------------------------------------------------------- remote hosts

const HOST_FIELDS = {name: 'host-name', host: 'host-host', scheduler: 'host-scheduler', remote_root: 'host-root',
  queue: 'host-queue', account: 'host-account', resources: 'host-resources', walltime: 'host-walltime',
  sync_interval: 'host-sync', command: 'host-command'};
const HOST_LISTS = {setup: 'host-setup', directives: 'host-directives'};

function renderHosts(hosts) {
  app.hosts = hosts;
  const select = $('job-host');
  const chosen = select.value;
  select.replaceChildren(new Option('This machine', ''), ...hosts.map((host) => new Option(`${host.name} (${host.scheduler}${host.queue ? `, ${host.queue}` : ''})`, host.name)));
  if (hosts.some((host) => host.name === chosen)) select.value = chosen;
  const list = $('hosts-list');
  list.replaceChildren();
  hosts.forEach((host) => {
    const row = document.createElement('div');
    row.className = 'row';
    const name = document.createElement('span');
    name.className = 'grow';
    name.textContent = `${host.name}: ${host.host}, ${host.scheduler}, ${host.remote_root}`;
    const edit = document.createElement('button');
    edit.textContent = 'Edit';
    edit.addEventListener('click', () => fillHostForm(host));
    const remove = document.createElement('button');
    remove.textContent = 'Remove';
    remove.className = 'danger';
    remove.addEventListener('click', async () => {
      try { renderHosts((await api(`/api/hosts/${encodeURIComponent(host.name)}`, {method: 'DELETE'})).hosts); } catch (error) { toast(error.message, 'error'); }
    });
    row.append(name, edit, remove);
    list.append(row);
  });
}

function fillHostForm(host) {
  Object.entries(HOST_FIELDS).forEach(([key, id]) => { $(id).value = host[key] ?? ''; });
  Object.entries(HOST_LISTS).forEach(([key, id]) => { $(id).value = (host[key] || []).join('\n'); });
  $('host-check').textContent = '';
}

// Cluster presets: predefined values for every entry of the host form.
function renderHostPresets(presets) {
  app.hostPresets = presets;
  fillSelect($('host-preset'), [['', 'Custom (no preset)'],
    ...presets.map((preset) => [preset.name, `${preset.name}: ${preset.description}${preset.configured ? '' : '  (no SSH alias found)'}`])], '');
}

function applyHostPreset(name) {
  const preset = (app.hostPresets || []).find((item) => item.name === name);
  const notes = $('host-preset-notes');
  if (!preset) { notes.textContent = ''; return; }
  fillHostForm({...preset, sync_interval: 15});
  const missing = preset.configured ? [] : [`No SSH alias for ${preset.name} in ~/.ssh/config: set SSH destination.`];
  notes.replaceChildren(...[...missing, ...preset.notes].map((line) => Object.assign(document.createElement('div'), {textContent: line})));
}

function readHostForm() {
  const host = {};
  Object.entries(HOST_FIELDS).forEach(([key, id]) => { host[key] = $(id).value.trim(); });
  Object.entries(HOST_LISTS).forEach(([key, id]) => { host[key] = $(id).value.split('\n').map((line) => line.trim()).filter(Boolean); });
  host.sync_interval = Number(host.sync_interval) || 15;
  return host;
}

async function saveHost() {
  const host = readHostForm();
  renderHosts((await api('/api/hosts', {method: 'POST', body: host})).hosts);
  $('job-host').value = host.name;
  return host;
}

function bindHosts() {
  $('hosts-btn').addEventListener('click', () => $('hosts-dialog').showModal());
  $('hosts-close').addEventListener('click', () => $('hosts-dialog').close());
  $('host-preset').addEventListener('change', (event) => applyHostPreset(event.target.value));
  $('host-save').addEventListener('click', async () => {
    try { toast(`Saved host ${(await saveHost()).name}`); } catch (error) { toast(error.message, 'error'); }
  });
  $('host-test').addEventListener('click', async () => {
    const note = $('host-check');
    try {
      const host = await saveHost();
      note.textContent = `Checking ${host.name}…`;
      const result = await api(`/api/hosts/${encodeURIComponent(host.name)}/check`, {method: 'POST', body: {}});
      note.textContent = result.checks.map((check) => `${check.ok ? 'ok' : 'FAILED'}: ${check.detail}`).join(' · ');
    } catch (error) {
      note.textContent = error.message;
    }
  });
}

// Job callbacks: report every job that finishes, and show new structures of
// followed jobs in the window they were started from.
async function onJobUpdates(jobs) {
  for (const job of jobs) {
    const before = app.jobStates.get(job.job_id);
    app.jobStates.set(job.job_id, job.state);
    if (before && before !== job.state && !job.active) {
      toast(`${job.job_id} ${job.state}: ${job.message || ''}`, job.state === 'done' ? 'info' : 'error');
    }
    const follow = app.following.get(job.job_id);
    if (follow && job.last_structure_id !== null && job.last_structure_id !== follow.last) {
      follow.last = job.last_structure_id;
      if (app.windows.has(follow.ws)) {
        try {
          applyState(await api(withWs(`/api/records/${job.last_structure_id}/load`, follow.ws), {method: 'POST', body: {}}));
        } catch { /* the row may have been removed; the next update retries */ }
      }
    }
    if (follow && !job.active) app.following.delete(job.job_id);
  }
}

// Optional blocking wait: long-poll until no job is active, then report once.
async function waitForAll() {
  if (app.waiting) { app.waiting = false; return; }
  app.waiting = true;
  $('jobs-wait').textContent = 'Stop waiting';
  const note = $('jobs-waiting');
  note.hidden = false;
  let jobs = [];
  while (app.waiting) {
    try {
      jobs = (await api('/api/jobs/wait?timeout=20')).jobs;
    } catch (error) {
      toast(error.message, 'error');
      break;
    }
    const active = jobs.filter((job) => job.active);
    note.textContent = `Waiting for ${active.length} running job${active.length === 1 ? '' : 's'}…`;
    await poll();
    if (!active.length) break;
  }
  if (app.waiting) {
    const failed = jobs.filter((job) => job.state === 'failed').length;
    toast(`All jobs finished: ${jobs.filter((job) => job.state === 'done').length} done, ${failed} failed`, failed ? 'error' : 'info');
    activateTab('data');
  }
  app.waiting = false;
  note.hidden = true;
  $('jobs-wait').textContent = 'Wait for all';
}

// ---------------------------------------------------------------- data tab

const COLUMNS = [
  ['rank', 'rank'], ['id', 'id'], ['formula', 'formula', 'left'], ['role', 'role', 'left'], ['stage', 'stage', 'left'],
  ['energy', 'E (eV)'], ['score', 'score (eV)'], ['delta', 'Δ (eV)'], ['fmax', 'fmax (eV/Å)'],
  ['potential', 'Φ (V)'], ['step', 'step'], ['parent_id', 'parent'], ['level', 'level of theory', 'left'], ['label', 'label', 'left'],
];

const format = (key, value) => {
  if (value === null || value === undefined || value === '') return '';
  if (['energy', 'score', 'delta', 'fmax'].includes(key)) return value.toFixed(4);
  if (key === 'potential') return Number(value).toFixed(2);
  return String(value);
};

async function loadRecord(id) {
  const scope = app.records.scope || 'project';
  await act(() => api(withWs(`/api/records/${id}/load?scope=${scope}`), {method: 'POST', body: {}}));
}

function markCurrentRow() {
  const win = current();
  const currentId = win && win.state && app.records.scope === 'project' ? win.state.current_id : null;
  document.querySelectorAll('#records tbody tr').forEach((row) => row.classList.toggle('current', Number(row.dataset.id) === currentId));
  const rows = app.records.rows;
  const index = rows.findIndex((row) => row.id === currentId);
  if (index >= 0) $('data-frame').value = index;
  $('data-frame-label').textContent = rows.length ? `${index >= 0 ? index + 1 : '-'}/${rows.length}` : '';
  drawChart($('chart'), rows, currentId, (row) => loadRecord(row.id));
}

async function refreshData() {
  let result;
  try {
    const scope = $('data-scope').value;
    $('data-job').disabled = scope !== 'project';
    const query = new URLSearchParams({stage: $('data-stage').value, job_id: scope === 'project' ? $('data-job').value : '',
      order: $('data-order').value, scope, role: $('data-role').value});
    result = await api(`/api/records?${query}`);
  } catch (error) {
    toast(error.message, 'error');
    return;
  }
  app.records = result;
  const head = document.querySelector('#records thead');
  head.innerHTML = `<tr>${COLUMNS.map(([, label, cls]) => `<th class="${cls || ''}">${label}</th>`).join('')}</tr>`;
  const body = document.querySelector('#records tbody');
  body.replaceChildren(...result.rows.map((row) => {
    const tr = document.createElement('tr');
    tr.dataset.id = row.id;
    COLUMNS.forEach(([key, , cls]) => {
      const td = document.createElement('td');
      td.className = cls || '';
      td.textContent = format(key, row[key]);
      tr.append(td);
    });
    tr.addEventListener('click', () => loadRecord(row.id));
    return tr;
  }));
  $('chart-title').textContent = result.metric === 'score'
    ? 'Score: grand potential relative to the starting surface (eV); Δ and rank use the score'
    : 'Potential energy (eV); Δ and rank use the energy';
  $('data-frame').max = Math.max(result.rows.length - 1, 0);
  $('data-frame').disabled = result.rows.length < 2;
  markCurrentRow();
  refreshEnergetics();
}

function showJob(jobId) {
  $('data-job').value = jobId;
  $('data-stage').value = '';
  activateTab('data');
}

function bindData() {
  ['data-stage', 'data-job', 'data-order', 'data-scope', 'data-role'].forEach((id) => $(id).addEventListener('change', refreshData));
  $('data-frame').addEventListener('input', (event) => {
    const row = app.records.rows[Number(event.target.value)];
    if (row) loadRecord(row.id);
  });
  $('data-to-global').addEventListener('click', async () => {
    const id = current().state.current_id;
    if (id === null || $('data-scope').value !== 'project') { toast('Load a row of the project database into the active window first', 'error'); return; }
    try {
      const result = await api('/api/records/to-global', {method: 'POST', body: {ids: [id]}});
      toast(`Project #${id} is in the global database as #${result.copied[id]}`);
    } catch (error) {
      toast(error.message, 'error');
    }
  });
  $('data-delete').addEventListener('click', async () => {
    const id = current().state.current_id;
    if (id === null || $('data-scope').value !== 'project') { toast('Only rows of the project database loaded in the active window can be deleted here', 'error'); return; }
    await act(() => api(withWs(`/api/records/${id}`), {method: 'DELETE'}), () => toast(`Deleted #${id}`));
    refreshData();
  });
}

// ---------------------------------------------------------------- workflow tab

async function refreshGraph() {
  try {
    drawGraph($('graph'), await api('/api/graph'), {
      onStructure: (id) => loadRecord(id).then(refreshGraph),
      onJob: showJob,
    });
  } catch (error) {
    toast(error.message, 'error');
  }
}

// ---------------------------------------------------------------- project

const NEW_PROJECT = '\u0000new';

function showProjectDialog() {
  // Without a project (first start) the dialog cannot be dismissed: nothing is stored before a folder is chosen.
  const first = !app.meta.project;
  $('project-first').hidden = !first;
  $('project-close').hidden = first;
  $('project-path').value = app.meta.project;
  const recent = app.meta.recent_projects.filter((path) => path !== app.meta.project);
  const options = first
    ? [new Option(recent.length ? 'Choose a recent project…' : 'No recent projects', ''), ...recent.map((path) => new Option(path, path))]
    : [new Option(`${app.meta.project}  (current)`, app.meta.project), ...recent.map((path) => new Option(path, path))];
  if (app.meta.folder_picker) options.push(new Option('New project: choose a folder on this Mac…', NEW_PROJECT));
  $('project-select').replaceChildren(...options);
  $('project-dialog').showModal();
}

// The folder dialog is native and runs on the machine that serves the app.
async function chooseProjectFolder() {
  try {
    const result = await api('/api/choose-folder', {method: 'POST', body: {start: app.meta.project.replace(/\/[^/]*$/, '')}});
    if (result.path) {
      $('project-path').value = result.path;
      await openProject(result.path);
      return;
    }
  } catch (error) {
    toast(error.message, 'error');
  }
  $('project-select').value = app.meta.project;
}

async function openProject(path) {
  if (!path) { toast('Choose a recent project, pick a folder, or type a path', 'error'); return; }
  const first = !app.meta.project;
  try {
    app.meta = await api('/api/project', {method: 'POST', body: {path}});
  } catch (error) {
    toast(error.message, 'error');
    return;
  }
  // The page was waiting for a project: start it properly now that one exists.
  if (first) { window.location.reload(); return; }
  $('project-dialog').close();
  // A new project is a new server-side session: rebuild every window from it.
  [...app.windows.values()].forEach((win) => win.element.remove());
  app.windows.clear();
  app.active = null;
  app.jobsKey = '';
  app.nStructures = -1;
  $('project-btn').textContent = app.meta.project;
  await loadAllWindows();
  await refreshReferences();
  await poll();
  toast(`Project: ${app.meta.project}`);
}

async function loadAllWindows() {
  const {windows} = await api('/api/windows');
  for (const info of windows) applyState(await api(withWs('/api/state', info.id)));
  const first = windows.find((info) => info.kind === 'bulk') || windows[0];
  app.active = null;
  activate(first.id);
}

// ---------------------------------------------------------------- tabs, polling, start

function activeTab() {
  return document.querySelector('.tab.active').dataset.tab;
}

function activateTab(name) {
  document.querySelectorAll('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === name));
  document.querySelectorAll('.pane').forEach((pane) => pane.classList.toggle('active', pane.id === `pane-${name}`));
  if (name === 'data') refreshData();
  if (name === 'flow') { refreshGraph(); refreshWorkflows(); }
}

async function poll() {
  let result;
  try {
    result = await api('/api/jobs');
  } catch {
    return;
  }
  const key = JSON.stringify(result.jobs);
  const jobsChanged = key !== app.jobsKey;
  const storeChanged = result.n_structures !== app.nStructures;
  if (jobsChanged) {
    app.jobsKey = key;
    await onJobUpdates(result.jobs);
    renderJobs(result.jobs);
    fillSelect($('data-job'), [['', 'all'], ...result.jobs.map((job) => [job.job_id, job.job_id])]);
  }
  app.nStructures = result.n_structures;
  if (storeChanged) refreshReferences();
  if (jobsChanged || storeChanged) {
    if (activeTab() === 'data') refreshData();
    if (activeTab() === 'flow') refreshGraph();
  }
  if (activeTab() === 'flow' || app.workflowsActive) refreshWorkflows();
}

function setViewerOptions(options) {
  Object.assign(app.options, options);
  app.windows.forEach((win) => win.viewer.setOptions(app.options));
}

function bindProject() {
  $('project-btn').addEventListener('click', showProjectDialog);
  $('project-open').addEventListener('click', () => openProject($('project-path').value.trim()));
  $('project-select').addEventListener('change', (event) => {
    if (event.target.value === NEW_PROJECT) chooseProjectFolder();
    else $('project-path').value = event.target.value;
  });
  $('project-close').addEventListener('click', () => $('project-dialog').close());
  $('project-dialog').addEventListener('cancel', (event) => { if (!app.meta.project) event.preventDefault(); });
  // Browsers may skip the cancel event (Escape pressed twice): reopen while there is no project.
  $('project-dialog').addEventListener('close', () => { if (!app.meta.project) $('project-dialog').showModal(); });
}

function bindQuit() {
  $('quit-btn').hidden = !app.meta.can_quit;
  $('quit-btn').addEventListener('click', async () => {
    if (!window.confirm('Stop GM2-Surf? Unsaved windows are lost; running jobs continue.')) return;
    try {
      await api('/api/quit', {method: 'POST', body: {}});
      document.body.innerHTML = '<p style="padding:24px;font:16px sans-serif">GM2-Surf has stopped. You can close this tab; open the app again to restart.</p>';
    } catch (error) {
      toast(error.message, 'error');
    }
  });
}

function bindTopbar() {
  document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => current().viewer.preset(button.dataset.view)));
  $('opt-style').addEventListener('change', (event) => setViewerOptions({style: event.target.value}));
  $('opt-images').addEventListener('change', (event) => setViewerOptions({images: event.target.checked}));
  $('opt-white').addEventListener('change', (event) => {
    $('panes').classList.toggle('white', event.target.checked);
    setViewerOptions({white: event.target.checked});
  });
  $('btn-png').addEventListener('click', () => {
    const win = current();
    const link = document.createElement('a');
    link.href = win.viewer.screenshot();
    link.download = `${win.state && win.state.info ? win.state.info.formula : 'view'}.png`;
    link.click();
  });
  $('btn-export').addEventListener('click', () => {
    const win = current();
    if (!win.state || !win.state.scene) { toast('The active window is empty', 'error'); return; }
    window.location.href = withWs(`/api/export?format=${$('export-format').value}`);
  });
  bindProject();
  document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => activateTab(tab.dataset.tab)));
  window.addEventListener('keydown', (event) => {
    if (event.target.matches('input, select, textarea')) return;
    if (event.key === 'Escape') setTool('select');
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') act(() => post('/api/undo'));
  });
}

async function start() {
  app.meta = await api('/api/meta');
  const meta = app.meta;
  if (!meta.project) {
    // No project yet: only the chooser works. Opening one reloads the page.
    $('project-btn').textContent = 'no project';
    bindProject();
    bindQuit();
    showProjectDialog();
    return;
  }
  $('project-btn').textContent = meta.project;
  fillSelect($('bulk-lattice'), meta.lattices, 'fcc');
  fillSelect($('export-format'), meta.export_formats);
  fillSelect($('data-stage'), [['', 'all'], ...meta.stages]);
  fillSelect($('data-job'), [['', 'all']]);
  fillSelect($('data-role'), [['', 'all'], ...meta.roles]);
  fillSelect($('ads-reaction'), [['', 'All molecules'], ['mine', 'My molecules'], ...meta.reactions.map((name) => [name, `${name} intermediates`])]);
  fillSelect($('calc-name'), meta.calculators.map((entry) => [entry.name, entry.available ? entry.name : `${entry.name} (not installed)`]), 'emt');
  $('mol-names').replaceChildren(...meta.molecule_names.map((name) => Object.assign(document.createElement('option'), {value: name})));
  $('site-legend').innerHTML = Object.entries(meta.site_colors)
    .map(([kind, color]) => `<span><span class="dot" style="background:${color}"></span>${kind}</span>`).join('');
  fillSelect($('host-scheduler'), meta.schedulers, 'pbs');
  fillSelect($('wf-take'), [['last', 'last structure'], ['best', 'best N'], ['all', 'all structures']], 'last');
  buildTools();
  bindTopbar();
  bindBuild();
  bindRun();
  bindData();
  renderCalculator();
  await loadAllWindows();
  await refreshReferences();
  renderHosts((await api('/api/hosts')).hosts);
  renderHostPresets((await api('/api/host-presets')).presets);
  await poll();
  setInterval(poll, 1500);
  bindQuit();
  if (meta.ask_project) showProjectDialog();
}

// Handle for debugging from the browser console.
window.gm2surf = {app};

start().catch((error) => toast(`Could not start: ${error.message}`, 'error'));
