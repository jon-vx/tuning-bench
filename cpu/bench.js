// bench.js

export const time = (fn) => {
  const t0 = performance.now();
  fn();
  return performance.now() - t0;
};

export const timeBatched = (fn, iters) => {
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn();
  return (performance.now() - t0) / iters;
};

export const calibrateIters = (fn, targetMs = 50, maxIters = 1e7) => {
  let iters = 1;
  while (true) {
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) fn();
    const dt = performance.now() - t0;
    if (dt >= targetMs || iters >= maxIters) {
      return { iters, perCall: dt / iters };
    }
    const factor = dt > 0 ? (targetMs / dt) * 1.2 : 2;
    iters = Math.min(maxIters, Math.max(iters * 2, Math.ceil(iters * factor)));
  }
};

export const stats = (s) => {
  const sorted = [...s].sort((a, b) => a - b);
  return {
    median: sorted[Math.floor(sorted.length / 2)],
    p95: sorted[Math.floor(sorted.length * 0.95)],
    mean: s.reduce((a, b) => a + b) / s.length,
    min: sorted[0],
  };
};

const download = (text, name, type) => {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
};

export const downloadJSON = (obj, name = "bench.json") =>
  download(JSON.stringify(obj, null, 2), name, "application/json");

export const downloadCSV = (text, name = "bench.csv") =>
  download(text, name, "text/csv");

const csvCell = (v) => {
  const s = v == null ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export const toCsv = (header, rows) =>
  [header, ...rows].map((row) => row.map(csvCell).join(",")).join("\n");
