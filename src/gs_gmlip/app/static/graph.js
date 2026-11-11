// Small SVG renderers: the energy/score chart and the provenance graph.

const SVG = 'http://www.w3.org/2000/svg';

function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  if (text !== undefined) node.textContent = text;
  return node;
}

// Line chart of rows[].value in listed order. Clicking a point calls onPick(row).
export function drawChart(element, rows, currentId, onPick) {
  element.replaceChildren();
  const width = 420, height = 190, left = 58, right = 10, top = 10, bottom = 22;
  const points = rows.map((row, index) => ({row, index})).filter((p) => p.row.value !== null);
  element.append(svg('line', {class: 'axis', x1: left, y1: top, x2: left, y2: height - bottom}));
  element.append(svg('line', {class: 'axis', x1: left, y1: height - bottom, x2: width - right, y2: height - bottom}));
  if (!points.length) {
    element.append(svg('text', {x: width / 2, y: height / 2, 'text-anchor': 'middle'}, 'no evaluated structures'));
    return;
  }
  const values = points.map((p) => p.row.value);
  let low = Math.min(...values), high = Math.max(...values);
  if (high - low < 1e-9) { low -= 0.5; high += 0.5; }
  const pad = (high - low) * 0.06;
  low -= pad; high += pad;
  const last = Math.max(rows.length - 1, 1);
  const x = (index) => left + (index / last) * (width - left - right);
  const y = (value) => top + (1 - (value - low) / (high - low)) * (height - top - bottom);

  [high - pad, (low + high) / 2, low + pad].forEach((value) => {
    element.append(svg('text', {x: left - 5, y: y(value) + 3, 'text-anchor': 'end'}, value.toFixed(3)));
  });
  element.append(svg('text', {x: left, y: height - 7}, '0'));
  element.append(svg('text', {x: width - right, y: height - 7, 'text-anchor': 'end'}, String(rows.length - 1)));

  element.append(svg('polyline', {class: 'line', points: points.map((p) => `${x(p.index)},${y(p.row.value)}`).join(' ')}));
  points.forEach((p) => {
    const dot = svg('circle', {cx: x(p.index), cy: y(p.row.value), r: p.row.id === currentId ? 4 : 2.6,
      class: p.row.id === currentId ? 'current' : ''});
    dot.append(svg('title', {}, `id ${p.row.id}  ${p.row.formula}  ${p.row.value.toFixed(4)} eV`));
    dot.addEventListener('click', () => onPick(p.row));
    element.append(dot);
  });
}

// Layered left-to-right provenance graph: ellipses are structures, rectangles are jobs.
export function drawGraph(container, graph, handlers) {
  container.replaceChildren();
  if (!graph.nodes.length) {
    const empty = document.createElement('div');
    empty.className = 'muted';
    empty.style.padding = '12px';
    empty.textContent = 'Nothing stored yet. Save a structure or run a job.';
    container.append(empty);
    return;
  }
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  const incoming = new Map(graph.nodes.map((node) => [node.id, []]));
  const edges = graph.edges.filter((edge) => byId.has(edge.source) && byId.has(edge.target));
  edges.forEach((edge) => incoming.get(edge.target).push(edge.source));

  // Depth = longest path from a source (memoised; cycles cannot occur in provenance).
  const depth = new Map();
  const depthOf = (id, seen = new Set()) => {
    if (depth.has(id)) return depth.get(id);
    if (seen.has(id)) return 0;
    seen.add(id);
    const value = Math.max(-1, ...incoming.get(id).map((source) => depthOf(source, seen))) + 1;
    depth.set(id, value);
    return value;
  };
  graph.nodes.forEach((node) => depthOf(node.id));

  const layers = [];
  graph.nodes.forEach((node) => {
    const level = depth.get(node.id);
    (layers[level] = layers[level] || []).push(node);
  });

  const dx = 168, dy = 58, marginX = 78, marginY = 36;
  const position = new Map();
  layers.forEach((layer, level) => {
    // Order by the mean row of predecessors to keep edges short.
    const key = (node) => {
      const rows = incoming.get(node.id).map((source) => position.get(source)?.row ?? 0);
      return rows.length ? rows.reduce((a, b) => a + b, 0) / rows.length : Number.MAX_SAFE_INTEGER;
    };
    if (level > 0) layer.sort((a, b) => key(a) - key(b));
    layer.forEach((node, row) => position.set(node.id, {row, x: marginX + level * dx, y: marginY + row * dy}));
  });

  const width = marginX * 2 + (layers.length - 1) * dx;
  const height = marginY * 2 + (Math.max(...layers.map((layer) => layer.length)) - 1) * dy;
  const root = svg('svg', {width, height, viewBox: `0 0 ${width} ${height}`});
  const defs = svg('defs');
  const marker = svg('marker', {id: 'arrow', viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto'});
  marker.append(svg('path', {d: 'M0 0L8 4L0 8z', fill: '#55637a'}));
  defs.append(marker);
  root.append(defs);

  const halfWidth = 62;
  edges.forEach((edge) => {
    const from = position.get(edge.source), to = position.get(edge.target);
    const x1 = from.x + halfWidth, x2 = to.x - halfWidth - 2, bend = (x2 - x1) / 2;
    root.append(svg('path', {class: `edge ${edge.kind}`, 'marker-end': 'url(#arrow)',
      d: `M${x1} ${from.y} C${x1 + bend} ${from.y} ${x2 - bend} ${to.y} ${x2} ${to.y}`}));
  });

  graph.nodes.forEach((node) => {
    const {x, y} = position.get(node.id);
    const classes = ['node', node.type];
    if (node.current) classes.push('current');
    if (node.state) classes.push(node.state);
    const group = svg('g', {class: classes.join(' '), transform: `translate(${x} ${y})`});
    if (node.type === 'structure') {
      group.append(svg('ellipse', {rx: halfWidth, ry: 19}));
      group.append(svg('text', {y: -2, 'text-anchor': 'middle'}, node.label));
      const energy = node.score !== null ? `${node.score.toFixed(3)} eV*` : node.energy !== null ? `${node.energy.toFixed(3)} eV` : node.stage;
      group.append(svg('text', {class: 'sub', y: 11, 'text-anchor': 'middle'}, `#${node.structure_id}  ${energy}`));
      group.addEventListener('click', () => handlers.onStructure(node.structure_id));
    } else {
      group.append(svg('rect', {x: -halfWidth, y: -19, width: halfWidth * 2, height: 38, rx: 3}));
      group.append(svg('text', {y: -2, 'text-anchor': 'middle'}, `${node.label} · ${node.calculator || ''}`));
      group.append(svg('text', {class: 'sub', y: 11, 'text-anchor': 'middle'}, `${node.state}, ${node.n_structures} structures`));
      group.addEventListener('click', () => handlers.onJob(node.job_id));
    }
    group.append(svg('title', {}, node.type === 'structure' ? `structure ${node.structure_id} (${node.stage})` : node.job_id));
    root.append(group);
  });
  container.append(root);
}
