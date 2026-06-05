import { readFileSync, writeFileSync } from 'node:fs';
import { geoNaturalEarth1, geoPath, geoInterpolate } from 'd3-geo';
import { feature, mesh } from 'topojson-client';

const width = 1600;
const height = 900;
const mapPadding = 24;
const cropPadding = { x: 230, y: 150 };

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
    label: 'Chicago/Evanston',
    coordinates: [-87.6298, 41.8781],
  },
  {
    id: 'sanjose',
    label: 'San Jose',
    coordinates: [-121.8863, 37.3382],
  },
  {
    id: 'chicago-return',
    label: 'Chicago/Evanston',
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

const topo = JSON.parse(
  readFileSync('node_modules/world-atlas/countries-110m.json', 'utf8'),
);

const countries = feature(topo, topo.objects.countries);
const borders = mesh(topo, topo.objects.countries, (a, b) => a !== b);
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

const routePoints = [];
for (let i = 0; i < trajectory.length - 1; i += 1) {
  const segment = routeSegment(trajectory[i], trajectory[i + 1]);
  routePoints.push(...(i === 0 ? segment : segment.slice(1)));
}
const routePath = catmullRomPath(routePoints);

const highlightedCountries = new Set(['Brazil', 'Spain', 'United States of America']);
const countryPaths = countries.features.map((country) => {
  const name = country.properties?.name || '';
  const className = highlightedCountries.has(name) ? 'country country--highlight' : 'country';
  return `<path class="${className}" data-country="${escapeXml(name)}" d="${path(country)}"/>`;
}).join('\n    ');

const pinGuide = projectedPins.map((pin) => {
  const left = ((pin.x - viewBox.x) / viewBox.width) * 100;
  const top = ((pin.y - viewBox.y) / viewBox.height) * 100;
  return `.${pin.id === 'chicago-return' ? 'chicago' : pin.id}: left ${left.toFixed(3)}%; top ${top.toFixed(3)}%;`;
}).join('\n');

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}" role="img" aria-labelledby="trajectory-map-title trajectory-map-desc">
  <title id="trajectory-map-title">Trajectory map</title>
  <desc id="trajectory-map-desc">A cropped world map showing a trajectory from Barcelona to Rio de Janeiro, Chicago, San Jose, Chicago, and Austin.</desc>
  <defs>
    <filter id="route-shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="1.5" stdDeviation="2" flood-color="#1b2a31" flood-opacity="0.18"/>
    </filter>
  </defs>
  <style>
    .sphere { fill: #d9eef6; }
    .country { fill: #d8d8d2; stroke: #ffffff; stroke-width: 0.75; vector-effect: non-scaling-stroke; }
    .country--highlight { fill: #e3c59c; }
    .borders { fill: none; stroke: #ffffff; stroke-width: 0.55; opacity: 0.72; vector-effect: non-scaling-stroke; }
    .route { fill: none; stroke: #4e7284; stroke-width: 4.3; stroke-linecap: round; stroke-linejoin: round; filter: url(#route-shadow); vector-effect: non-scaling-stroke; }
    @media (prefers-color-scheme: dark) {
      .sphere { fill: #203039; }
      .country { fill: #5a5d59; stroke: #32342f; }
      .country--highlight { fill: #82684b; }
      .borders { stroke: #343832; opacity: 0.72; }
      .route { stroke: #91b7c7; }
    }
  </style>
  <path class="sphere" d="${path({ type: 'Sphere' })}"/>
  <g class="countries">
    ${countryPaths}
  </g>
  <path class="borders" d="${path(borders)}"/>
  <path class="route" d="${routePath}"/>
</svg>
`;

writeFileSync('images/maps/trajectory-map.svg', svg);
console.log(`wrote images/maps/trajectory-map.svg`);
console.log(`viewBox ${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`);
console.log(`aspect ${viewBox.width} / ${viewBox.height}`);
console.log(pinGuide);
