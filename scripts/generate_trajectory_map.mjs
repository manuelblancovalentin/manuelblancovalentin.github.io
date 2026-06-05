import { readFileSync, writeFileSync } from 'node:fs';
import { geoNaturalEarth1, geoPath, geoInterpolate } from 'd3-geo';
import { feature, mesh } from 'topojson-client';

const width = 1600;
const height = 900;
const mapPadding = 24;
const cropPadding = { x: 130, y: 125 };

const trajectory = [
  {
    id: 'barcelona',
    label: 'Barcelona',
    coordinates: [2.1734, 41.3851],
  },
  {
    id: 'rio',
    label: 'Rio de Janeiro',
    coordinates: [-43.1729, -22.9068],
  },
  {
    id: 'chicago',
    label: 'Chicago',
    coordinates: [-87.6298, 41.8781],
  },
  {
    id: 'sanjose',
    label: 'San Jose',
    coordinates: [-121.8863, 37.3382],
  },
  {
    id: 'chicago-return',
    label: 'Chicago',
    coordinates: [-87.6298, 41.8781],
  },
  {
    id: 'austin',
    label: 'Austin',
    coordinates: [-97.7431, 30.2672],
  },
];

const pins = [
  trajectory[0],
  trajectory[1],
  trajectory[2],
  trajectory[3],
  trajectory[5],
];

const routeLegs = [
  {
    id: 'barcelona-rio',
    from: trajectory[0],
    to: trajectory[1],
    label: 'Barcelona to Rio de Janeiro',
    date: '2014',
  },
  {
    id: 'rio-chicago',
    from: trajectory[1],
    to: trajectory[2],
    label: 'Rio de Janeiro to Chicago',
    date: '2020',
  },
  {
    id: 'chicago-sanjose',
    from: trajectory[2],
    to: trajectory[3],
    label: 'Chicago to San Jose, CA',
    date: '2023',
  },
  {
    id: 'chicago-austin',
    from: trajectory[4],
    to: trajectory[5],
    label: 'Chicago to Austin, TX',
    date: '2026',
  },
];

const topo = JSON.parse(
  readFileSync('node_modules/world-atlas/countries-110m.json', 'utf8'),
);
const usTopo = JSON.parse(
  readFileSync('node_modules/us-atlas/states-10m.json', 'utf8'),
);

const countries = feature(topo, topo.objects.countries);
const borders = mesh(topo, topo.objects.countries, (a, b) => a !== b);
const states = feature(usTopo, usTopo.objects.states);
const stateBorders = mesh(usTopo, usTopo.objects.states, (a, b) => a !== b);
const projection = geoNaturalEarth1().fitExtent(
  [
    [mapPadding, mapPadding],
    [width - mapPadding, height - mapPadding],
  ],
  { type: 'Sphere' },
);
const path = geoPath(projection);

const projectedPins = pins.map((pin) => {
  const [x, y] = projection(pin.coordinates);
  return { ...pin, x, y };
});

const minX = Math.max(0, Math.min(...projectedPins.map((pin) => pin.x)) - cropPadding.x);
const maxX = Math.min(width, Math.max(...projectedPins.map((pin) => pin.x)) + cropPadding.x);
const minY = Math.max(0, Math.min(...projectedPins.map((pin) => pin.y)) - cropPadding.y);
const maxY = Math.min(height, Math.max(...projectedPins.map((pin) => pin.y)) + cropPadding.y);

const viewBox = {
  x: Math.round(minX * 10) / 10,
  y: Math.round(minY * 10) / 10,
  width: Math.round((maxX - minX) * 10) / 10,
  height: Math.round((maxY - minY) * 10) / 10,
};

function escapeXml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function routeSegment(from, to, samples = 42) {
  const interpolate = geoInterpolate(from.coordinates, to.coordinates);
  return Array.from({ length: samples }, (_, index) => {
    const t = index / (samples - 1);
    return projection(interpolate(t));
  }).filter(Boolean);
}

function catmullRomPath(points) {
  if (points.length < 2) return '';

  const d = [`M ${points[0][0].toFixed(2)} ${points[0][1].toFixed(2)}`];

  for (let i = 0; i < points.length - 1; i += 1) {
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(points.length - 1, i + 2)];
    const c1x = p1[0] + (p2[0] - p0[0]) / 6;
    const c1y = p1[1] + (p2[1] - p0[1]) / 6;
    const c2x = p2[0] - (p3[0] - p1[0]) / 6;
    const c2y = p2[1] - (p3[1] - p1[1]) / 6;
    d.push(
      `C ${c1x.toFixed(2)} ${c1y.toFixed(2)} ${c2x.toFixed(2)} ${c2y.toFixed(2)} ${p2[0].toFixed(2)} ${p2[1].toFixed(2)}`,
    );
  }

  return d.join(' ');
}

const routePaths = routeLegs.map((leg) => {
  const points = routeSegment(leg.from, leg.to, 48);
  const midpoint = points[Math.floor(points.length / 2)];
  return {
    ...leg,
    d: catmullRomPath(points),
    labelX: midpoint[0],
    labelY: midpoint[1],
  };
});

const highlightedCountries = new Set(['Brazil', 'Spain', 'United States of America']);
const countryPaths = countries.features.map((country) => {
  const name = country.properties?.name || '';
  const className = highlightedCountries.has(name) && name !== 'United States of America' ? 'country country--highlight' : 'country';
  return `<path class="${className}" data-country="${escapeXml(name)}" d="${path(country)}"/>`;
}).join('\n    ');
const highlightedStates = new Set(['06', '17', '48']);
const statePaths = states.features
  .filter((state) => highlightedStates.has(String(state.id)))
  .map((state) => {
    const name = state.properties?.name || '';
    return `<path class="state state--highlight" data-state="${escapeXml(name)}" d="${path(state)}"/>`;
  })
  .join('\n    ');

const pinGuide = projectedPins.map((pin) => {
  const left = ((pin.x - viewBox.x) / viewBox.width) * 100;
  const top = ((pin.y - viewBox.y) / viewBox.height) * 100;
  return `.${pin.id === 'chicago-return' ? 'chicago' : pin.id}: left ${left.toFixed(3)}%; top ${top.toFixed(3)}%;`;
}).join('\n');

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}" role="img" aria-labelledby="trajectory-map-title trajectory-map-desc">
  <title id="trajectory-map-title">Trajectory map</title>
  <desc id="trajectory-map-desc">A cropped world map showing a trajectory from Barcelona to Rio de Janeiro, Chicago, San Jose, Chicago, and Austin.</desc>
  <style>
    .sphere { fill: #d9eef6; }
    .country { fill: #d8d8d2; stroke: #ffffff; stroke-width: 0.75; vector-effect: non-scaling-stroke; }
    .country--highlight { fill: #e3c59c; }
    .state--highlight { fill: #e3c59c; stroke: #ffffff; stroke-width: 0.85; vector-effect: non-scaling-stroke; }
    .borders { fill: none; stroke: #ffffff; stroke-width: 0.55; opacity: 0.72; vector-effect: non-scaling-stroke; }
    .state-borders { fill: none; stroke: #ffffff; stroke-width: 0.45; opacity: 0.64; vector-effect: non-scaling-stroke; }
    @media (prefers-color-scheme: dark) {
      .sphere { fill: #203039; }
      .country { fill: #5a5d59; stroke: #32342f; }
      .country--highlight { fill: #82684b; }
      .state--highlight { fill: #82684b; stroke: #32342f; }
      .borders { stroke: #343832; opacity: 0.72; }
      .state-borders { stroke: #343832; opacity: 0.64; }
    }
  </style>
  <path class="sphere" d="${path({ type: 'Sphere' })}"/>
  <g class="countries">
    ${countryPaths}
  </g>
  <g class="states">
    ${statePaths}
  </g>
  <path class="borders" d="${path(borders)}"/>
  <path class="state-borders" d="${path(stateBorders)}"/>
</svg>
`;

writeFileSync('images/maps/trajectory-map.svg', svg);
const routeOverlay = `<svg class="landing-trajectory__routes" viewBox="${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}" preserveAspectRatio="none" aria-hidden="false" focusable="false">
  <defs>
    <filter id="landing-route-glow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="1.2" stdDeviation="1.7" flood-color="#1b2a31" flood-opacity="0.18"/>
    </filter>
  </defs>
  ${routePaths.map((leg) => `<g class="landing-trajectory__route-leg landing-trajectory__route-leg--${leg.id}">
    <path class="landing-trajectory__route-visible" d="${leg.d}"></path>
    <path class="landing-trajectory__route-hit" tabindex="0" role="img" aria-label="${escapeXml(leg.label)}: ${escapeXml(leg.date)}" onmouseenter="this.parentNode.classList.add('is-active')" onmouseleave="this.parentNode.classList.remove('is-active')" onfocus="this.parentNode.classList.add('is-active')" onblur="this.parentNode.classList.remove('is-active')" onclick="this.parentNode.classList.add('is-active')" d="${leg.d}">
      <title>${escapeXml(leg.label)}: ${escapeXml(leg.date)}</title>
    </path>
    <g class="landing-trajectory__route-label" transform="translate(${leg.labelX.toFixed(2)} ${leg.labelY.toFixed(2)})" aria-hidden="true">
      <rect x="-43" y="-16" width="86" height="28" rx="9"></rect>
      <text x="0" y="2">Moved ${escapeXml(leg.date)}</text>
    </g>
  </g>`).join('\n  ')}
</svg>`;
writeFileSync('_includes/landing-trajectory-routes.svg', routeOverlay);
console.log(`wrote images/maps/trajectory-map.svg`);
console.log(`wrote _includes/landing-trajectory-routes.svg`);
console.log(`viewBox ${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`);
console.log(`aspect ${viewBox.width} / ${viewBox.height}`);
console.log(pinGuide);
