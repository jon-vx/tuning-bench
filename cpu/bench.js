import { quantile } from "../shared/benchmark-utils.js";

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
  const median = quantile(s, 0.5);
  const q1 = quantile(s, 0.25);
  const q3 = quantile(s, 0.75);
  return {
    median,
    q1,
    q3,
    iqr: q3 - q1,
    iqrPct: median > 0 ? (q3 - q1) / median : null,
    p95: quantile(s, 0.95),
    mean: s.reduce((a, b) => a + b) / s.length,
    min: sorted[0],
  };
};
