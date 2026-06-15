// cpu.js

import * as np from "https://esm.sh/numpy-ts@1.4.0";
import { matmulNumpy, matmulIJK, matmulIKJ } from "./matmul.js";
import { timeBatched, calibrateIters, stats, downloadJSON, downloadCSV, toCsv } from "./bench.js";

const SIZES = [128, 256, 512, 1024];
const MEASURED_RUNS = 10;
const TARGET_BATCH_MS = 50;

const njFrom = (flat, N) =>
  np.array(flat).reshape([N, N]).astype("float32");

export function verifyCorrectness(logFn = console.log) {
  const N = 128;
  const flatA = new Float32Array(N * N).map(() => Math.random());
  const flatB = new Float32Array(N * N).map(() => Math.random());
  const ref = matmulNumpy(njFrom(flatA, N), njFrom(flatB, N)).data;

  const maxDiff = (out) => {
    let m = 0;
    for (let i = 0; i < out.length; i++) m = Math.max(m, Math.abs(out[i] - ref[i]));
    return m;
  };
  const dIjk = maxDiff(matmulIJK(flatA, flatB, N));
  const dIkj = maxDiff(matmulIKJ(flatA, flatB, N));
  const worst = Math.max(dIjk, dIkj);

  logFn(
    `verify N=128: max|ijk-dot|=${dIjk.toExponential(3)}, ` +
    `max|ikj-dot|=${dIkj.toExponential(3)} (threshold 1e-3)`
  );
  if (worst > 1e-3) {
    throw new Error(
      `verification failed: max abs diff ${worst.toExponential(3)} > 1e-3`
    );
  }
  logFn(`verify pass (worst ${worst.toExponential(3)})`);
  return worst;
}

export async function runCpuBenchmark(logFn = console.log, statusFn = () => { }, label = "") {
  verifyCorrectness(logFn);

  let sink = 0;
  const results = [];
  for (const N of SIZES) {
    const flatA = new Float32Array(N * N).map(() => Math.random());
    const flatB = new Float32Array(N * N).map(() => Math.random());
    const njA = njFrom(flatA, N);
    const njB = njFrom(flatB, N);

    const runs = [
      { name: "numpy-ts.dot", fn: () => { sink += matmulNumpy(njA, njB).data[0]; } },
      { name: "handrolled-ijk", fn: () => { sink += matmulIJK(flatA, flatB, N)[0]; } },
      { name: "handrolled-ikj", fn: () => { sink += matmulIKJ(flatA, flatB, N)[0]; } },
    ];

    for (const { name, fn } of runs) {
      statusFn(`calibrating N=${N} ${name}...`);
      const { iters } = calibrateIters(fn, TARGET_BATCH_MS);

      statusFn(`running N=${N} ${name} (x${iters}/sample)...`);
      const samples = [];
      for (let i = 0; i < MEASURED_RUNS; i++) samples.push(timeBatched(fn, iters));
      const s = stats(samples);
      results.push({ N, variant: name, iters, ...s });
      logFn(
        `N=${N} ${name}: median ${s.median.toFixed(4)} ms/call ` +
        `(min ${s.min.toFixed(4)}, p95 ${s.p95.toFixed(4)}, ` +
        `mean ${s.mean.toFixed(4)}, x${iters}/sample)`
      );
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  const payload = {
    device: navigator.userAgent,
    label,
    backend: "cpu",
    timestamp: new Date().toISOString(),
    results,
  };
  logFn(`checksum: ${sink}`);
  return payload;
}

const $ = (id) => document.getElementById(id);
const logEl = $("log");
const statusEl = $("status");
const runBtn = $("run-btn");
const labelEl = $("label");
const exportBtn = $("export-btn");
const exportCsvBtn = $("export-csv-btn");

let lastPayload = null;

const log = (msg) => {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
};
const setStatus = (msg) => {
  statusEl.textContent = msg;
};

function fileStem(p) {
  const slug = p.label ? p.label.replace(/[^a-z0-9]+/gi, "-") + "-" : "";
  const stamp = p.timestamp.replace(/[:.]/g, "-");
  return `bench-cpu-${slug}${stamp}`;
}

function exportJson() {
  downloadJSON(lastPayload, `${fileStem(lastPayload)}.json`);
}

function exportCsv() {
  const header = [
    "timestamp", "label", "device", "backend",
    "N", "variant", "iters", "median_ms", "p95_ms", "mean_ms", "min_ms",
  ];
  const rows = lastPayload.results.map((r) => [
    lastPayload.timestamp, lastPayload.label, lastPayload.device, lastPayload.backend,
    r.N, r.variant, r.iters,
    r.median.toFixed(4), r.p95.toFixed(4), r.mean.toFixed(4), r.min.toFixed(4),
  ]);
  downloadCSV(toCsv(header, rows), `${fileStem(lastPayload)}.csv`);
}

runBtn.addEventListener("click", async () => {
  runBtn.disabled = true;
  exportBtn.disabled = true;
  exportCsvBtn.disabled = true;
  logEl.textContent = "";
  setStatus("starting...");
  try {
    lastPayload = await runCpuBenchmark(log, setStatus, labelEl.value.trim());
    exportBtn.disabled = false;
    exportCsvBtn.disabled = false;
    setStatus("done");
    log("\n=== done ===");
  } catch (err) {
    setStatus(`error: ${err.message}`);
    log(`ERROR: ${err.stack ?? err}`);
  } finally {
    runBtn.disabled = false;
  }
});
exportBtn.addEventListener("click", exportJson);
exportCsvBtn.addEventListener("click", exportCsv);
