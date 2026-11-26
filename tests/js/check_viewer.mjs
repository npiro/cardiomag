// Headless check of the viewer's data + field computation: decodes the embedded payload with the
// same code path as the page and prints B at a small grid for the first simulated source.
import fs from "node:fs";
import zlib from "node:zlib";
const html = fs.readFileSync(process.argv[2], "utf8");
const m = html.match(/const DATA = (\{.*?\});\nconst \$ =/s);
const DATA = JSON.parse(m[1]);
const dec = (b64, T) => { const b = zlib.gunzipSync(Buffer.from(b64, "base64")); return new T(b.buffer, b.byteOffset, b.byteLength / T.BYTES_PER_ELEMENT); };
const s = DATA.sources[0];
const vox = dec(s.vox, Float32Array), mm = dec(s.m, Int16Array), T = s.t_ms.length, V = s.n_vox;
const sensors = JSON.parse(process.argv[3]);   // [[x,y,z] mm]
const sc = s.m_scale / 32767 * 1e-7 * 1e12;
const out = sensors.map(([sx, sy, sz]) => {
  const B = new Float64Array(T);
  for (let v = 0; v < V; v++) {
    const rx = (sx - vox[3*v]) * 1e-3, ry = (sy - vox[3*v+1]) * 1e-3, rz = (sz - vox[3*v+2]) * 1e-3;
    const d3 = Math.hypot(rx, ry, rz) ** 3, L = [ry / d3, -rx / d3, 0];
    for (let t = 0; t < T; t++) { const o = t*V*3 + 3*v; B[t] += (mm[o]*L[0] + mm[o+1]*L[1]) * sc; }
  }
  return Array.from(B);
});
console.log(JSON.stringify({title: s.title, sources: DATA.sources.length, measured: DATA.measured.length, B: out}));
