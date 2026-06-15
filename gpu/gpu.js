// gpu.js

import {
  CreateMLCEngine,
  prebuiltAppConfig,
} from "https://esm.run/@mlc-ai/web-llm@0.2.79";

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const logEl = $("log");
const modelSelect = $("model-select");
const loadBtn = $("load-btn");
const runBtn = $("run-btn");
const exportBtn = $("export-btn");
const exportCsvBtn = $("export-csv-btn");
const summaryTable = $("summary");
const promptEl = $("prompt");
const labelEl = $("label");

const log = (msg) => {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
};
const setStatus = (msg) => {
  statusEl.textContent = msg;
};

let engine = null;
let gpuInfo = null;
let results = [];
let sessionMeta = {};

const BENCH_PROMPT =
  "Explain why transformers were such a pivotal moment in the development of artificial intelligence.";

const PREFERRED = [
  "Llama-3.2-1B-Instruct-q4f16_1-MLC",
  "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
  "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
  "SmolLM2-360M-Instruct-q4f16_1-MLC",
  "Phi-3.5-mini-instruct-q4f16_1-MLC",
];

function populateModelList() {
  const available = new Set(prebuiltAppConfig.model_list.map((m) => m.model_id));
  const ids = PREFERRED.filter((id) => available.has(id));
  const list = ids.length ? ids : [...available];
  for (const id of list) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = id;
    modelSelect.appendChild(opt);
  }
}

async function getGpuInfo() {
  if (!navigator.gpu) return { error: "WebGPU not available" };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { error: "No WebGPU adapter" };
  const info =
    adapter.info ??
    (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  return {
    vendor: info.vendor ?? "",
    architecture: info.architecture ?? "",
    device: info.device ?? "",
    description: info.description ?? "",
    limits: {
      maxBufferSize: adapter.limits?.maxBufferSize,
      maxComputeWorkgroupStorageSize:
        adapter.limits?.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup:
        adapter.limits?.maxComputeInvocationsPerWorkgroup,
    },
  };
}

function parseStats(text) {
  const prefill = /prefill:\s*([\d.]+)/.exec(text);
  const decode = /decod\w*:\s*([\d.]+)/.exec(text);
  return {
    prefillTokS: prefill ? parseFloat(prefill[1]) : null,
    decodeTokS: decode ? parseFloat(decode[1]) : null,
  };
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
const min = (xs) => xs.reduce((a, b) => Math.min(a, b));
const max = (xs) => xs.reduce((a, b) => Math.max(a, b));

async function oneRun(maxTokens, prompt) {
  await engine.resetChat();
  const t0 = performance.now();
  const reply = await engine.chat.completions.create({
    messages: [{ role: "user", content: prompt }],
    max_tokens: maxTokens,
    temperature: 0,
  });
  const wallMs = performance.now() - t0;
  const statsText = await engine.runtimeStatsText();
  const stats = parseStats(statsText);
  return {
    wallMs,
    ...stats,
    completionTokens: reply.usage?.completion_tokens ?? null,
    promptTokens: reply.usage?.prompt_tokens ?? null,
    finishText: reply.choices[0]?.message?.content?.slice(0, 60) ?? "",
  };
}

async function runBenchmark() {
  const warmup = parseInt($("warmup").value, 10);
  const runs = parseInt($("runs").value, 10);
  const maxTokens = parseInt($("max-tokens").value, 10);
  const prompt = promptEl.value.trim() || BENCH_PROMPT;

  runBtn.disabled = true;
  results = [];
  sessionMeta = {
    model: modelSelect.value,
    label: labelEl.value.trim(),
    warmup,
    runs,
    maxTokens,
    prompt,
    gpu: gpuInfo,
    userAgent: navigator.userAgent,
    timestamp: new Date().toISOString(),
  };

  log(`\n=== session ${sessionMeta.timestamp} ===`);
  log(`model: ${sessionMeta.model}`);
  log(`gpu: ${JSON.stringify(gpuInfo)}`);

  try {
    for (let i = 0; i < warmup; i++) {
      setStatus(`warmup ${i + 1}/${warmup}...`);
      const r = await oneRun(maxTokens, prompt);
      log(
        `warmup ${i + 1}: prefill ${r.prefillTokS} tok/s, ` +
        `decode ${r.decodeTokS} tok/s, wall ${r.wallMs.toFixed(0)}ms (discarded)`
      );
    }

    for (let i = 0; i < runs; i++) {
      setStatus(`run ${i + 1}/${runs}...`);
      const r = await oneRun(maxTokens, prompt);
      results.push(r);
      log(
        `run ${i + 1}: prefill ${r.prefillTokS} tok/s, ` +
        `decode ${r.decodeTokS} tok/s, wall ${r.wallMs.toFixed(0)}ms, ` +
        `tokens ${r.completionTokens}`
      );
    }

    const last = results.at(-1);
    if (last) log(`sanity (first 60 chars): "${last.finishText}"`);

    renderSummary();
    exportBtn.disabled = false;
    exportCsvBtn.disabled = false;
    setStatus("done");
  } catch (err) {
    setStatus(`error: ${err.message}`);
    log(`ERROR: ${err.stack ?? err}`);
  } finally {
    runBtn.disabled = false;
  }
}

function renderSummary() {
  const tbody = summaryTable.querySelector("tbody");
  tbody.replaceChildren();
  const cols = {
    prefill: results.map((r) => r.prefillTokS).filter((x) => x != null),
    decode: results.map((r) => r.decodeTokS).filter((x) => x != null),
    wall: results.map((r) => r.wallMs),
  };
  const rows = [
    ["median", median(cols.prefill), median(cols.decode), median(cols.wall)],
    ["min", min(cols.prefill), min(cols.decode), min(cols.wall)],
    ["max", max(cols.prefill), max(cols.decode), max(cols.wall)],
  ];
  for (const [label, p, d, w] of rows) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = label;
    tr.appendChild(th);
    for (const val of [p.toFixed(1), d.toFixed(1), w.toFixed(0)]) {
      const td = document.createElement("td");
      td.textContent = val;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  summaryTable.hidden = false;
}

function fileStem() {
  const slug = sessionMeta.label ? sessionMeta.label.replace(/[^a-z0-9]+/gi, "-") + "-" : "";
  const stamp = sessionMeta.timestamp.replace(/[:.]/g, "-");
  return `bench-${slug}${sessionMeta.model}-${stamp}`;
}

function exportJson() {
  const blob = new Blob(
    [JSON.stringify({ meta: sessionMeta, results }, null, 2)],
    { type: "application/json" }
  );
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${fileStem()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

const csvCell = (v) => {
  const s = v == null ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

function exportCsv() {
  const gpu = sessionMeta.gpu ?? {};
  const header = [
    "timestamp", "label", "model", "run_idx", "prompt_tokens", "completion_tokens",
    "max_tokens", "prefill_tok_s", "decode_tok_s", "wall_ms",
    "gpu_vendor", "gpu_architecture", "gpu_device", "user_agent", "prompt",
  ];
  const rows = results.map((r, i) => [
    sessionMeta.timestamp,
    sessionMeta.label,
    sessionMeta.model,
    i + 1,
    r.promptTokens,
    r.completionTokens,
    sessionMeta.maxTokens,
    r.prefillTokS,
    r.decodeTokS,
    r.wallMs.toFixed(3),
    gpu.vendor,
    gpu.architecture,
    gpu.device,
    sessionMeta.userAgent,
    sessionMeta.prompt,
  ]);
  const csv = [header, ...rows]
    .map((row) => row.map(csvCell).join(","))
    .join("\n");

  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${fileStem()}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

async function loadModel() {
  loadBtn.disabled = true;
  runBtn.disabled = true;
  exportBtn.disabled = true;
  exportCsvBtn.disabled = true;
  summaryTable.hidden = true;

  const modelId = modelSelect.value;
  try {
    if (engine) {
      await engine.unload();
      engine = null;
    }
    engine = await CreateMLCEngine(modelId, {
      initProgressCallback: (p) => setStatus(p.text),
    });
    setStatus(`loaded ${modelId}`);
    log(`loaded ${modelId}`);
    runBtn.disabled = false;
  } catch (err) {
    setStatus(`load error: ${err.message}`);
    log(`LOAD ERROR: ${err.stack ?? err}`);
  } finally {
    loadBtn.disabled = false;
  }
}

async function init() {
  if (!navigator.gpu) {
    setStatus("WebGPU not available in this browser. Use Chrome or Edge.");
    loadBtn.disabled = true;
    return;
  }
  gpuInfo = await getGpuInfo();
  log(`adapter: ${JSON.stringify(gpuInfo, null, 2)}`);
  promptEl.value = BENCH_PROMPT;
  populateModelList();
}

loadBtn.addEventListener("click", loadModel);
runBtn.addEventListener("click", runBenchmark);
exportBtn.addEventListener("click", exportJson);
exportCsvBtn.addEventListener("click", exportCsv);

init();
