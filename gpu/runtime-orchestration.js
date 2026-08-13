import { getWebGpuProfile, makeDefaultLabel } from "./device-profile.js";
import { FIXED_MODELS } from "./models.js";
import { downloadJSON } from "../shared/benchmark-utils.js";
import {
  ORCHESTRATION_POLICIES,
  ORCHESTRATION_WORKLOADS,
  rotatedPolicyOrder,
  summarizeOrchestrationRows,
} from "./runtime-orchestration-core.js";

const BUNDLE_URL = "../vendor/generated/webllm-0.2.79-runtime-orchestration.js";
const ids = [
  "campaign-id", "model", "repetitions", "warmups", "policy-controls", "workload-controls",
  "run", "export", "status", "progress", "summary", "log",
];
const elements = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
let lastPayload = null;
let running = false;

function log(message) {
  elements.log.textContent += `${message}\n`;
  elements.log.scrollTop = elements.log.scrollHeight;
}

function setStatus(message) {
  elements.status.textContent = message;
  log(message);
}

function addChecks(container, name, definitions) {
  for (const [index, definition] of definitions.entries()) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = name;
    input.value = definition.id;
    input.checked = definition.defaultSelected ?? name === "workloads";
    if (name === "policies" && index === 0) input.disabled = true;
    label.append(input, definition.label);
    container.append(label);
  }
}

function selected(name, definitions) {
  const ids = new Set(
    [...document.querySelectorAll(`input[name="${name}"]:checked`)].map((input) => input.value)
  );
  return definitions.filter((definition) => ids.has(definition.id));
}

function pct(value) {
  return Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(1)}%` : "—";
}

function num(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function equalityCell(rate) {
  const cell = document.createElement("td");
  cell.textContent = Number.isFinite(rate) ? `${Math.round(rate * 100)}%` : "—";
  cell.className = rate === 1 ? "pass" : "fail";
  return cell;
}

function renderSummary(summaries) {
  const body = elements.summary.querySelector("tbody");
  body.textContent = "";
  for (const summary of summaries) {
    const row = document.createElement("tr");
    const values = [
      summary.workloadLabel,
      summary.policyLabel,
      `${summary.successfulRuns}/${summary.measuredRuns}`,
      num(summary.medianTtftMs),
      pct(summary.medianTtftDeltaPct),
      num(summary.medianWallMs),
      pct(summary.medianWallDeltaPct),
      num(summary.medianChunkGapMs, 2),
      num(summary.p95ChunkGapMs, 2),
      num(summary.medianSubmitCalls, 0),
      num(summary.medianDispatches, 0),
    ];
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    row.append(equalityCell(summary.outputEqualityRate));
    row.append(equalityCell(summary.traceEqualityRate));
    const status = document.createElement("td");
    status.textContent = summary.screenStatus;
    status.className = summary.screenStatus === "promising signal" ? "pass" : "";
    row.append(status);
    body.append(row);
  }
  elements.summary.hidden = false;
}

async function consumeCompletion(engine, workload) {
  const request = {
    messages: [
      { role: "system", content: "Follow the requested format exactly. Be concise and deterministic." },
      { role: "user", content: workload.prompt },
    ],
    stream: true,
    stream_options: { include_usage: true },
    max_tokens: workload.maxTokens,
    temperature: 0,
    top_p: 1,
    repetition_penalty: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
    seed: 42,
  };
  if (workload.stop) request.stop = workload.stop;
  let output = "";
  let usage = {};
  let finishReason = "";
  let firstTokenAt = null;
  let lastChunkAt = null;
  const chunkGapsMs = [];
  const startedAt = performance.now();
  const response = await engine.chat.completions.create(request);
  for await (const chunk of response) {
    const now = performance.now();
    const choice = chunk.choices?.[0];
    const delta = choice?.delta?.content || "";
    if (delta) {
      if (firstTokenAt === null) firstTokenAt = now;
      if (lastChunkAt !== null) chunkGapsMs.push(now - lastChunkAt);
      lastChunkAt = now;
      output += delta;
    }
    if (choice?.finish_reason) finishReason = choice.finish_reason;
    if (chunk.usage) usage = chunk.usage;
  }
  return {
    output,
    usage,
    finishReason,
    ttftMs: firstTokenAt === null ? null : firstTokenAt - startedAt,
    wallMs: performance.now() - startedAt,
    chunkGapsMs,
  };
}

async function runOne(engine, control, policy, workload, repetition, runKind, executionIndex) {
  await control.setPolicy({
    dispatchesPerEncoder: policy.dispatchesPerEncoder,
    reuseUniformBuffers: policy.reuseUniformBuffers,
  });
  await engine.resetChat();
  control.resetStats();
  const startedAt = new Date().toISOString();
  try {
    const completion = await consumeCompletion(engine, workload);
    return {
      policyId: policy.id,
      policyLabel: policy.label,
      dispatchesPerEncoder: policy.dispatchesPerEncoder,
      reuseUniformBuffers: policy.reuseUniformBuffers,
      workloadId: workload.id,
      workloadLabel: workload.label,
      maxTokens: workload.maxTokens,
      repetition,
      runKind,
      executionIndex,
      startedAt,
      success: completion.finishReason === "stop" || completion.finishReason === "length",
      ...completion,
      orchestration: control.getStats(),
    };
  } catch (error) {
    return {
      policyId: policy.id,
      policyLabel: policy.label,
      dispatchesPerEncoder: policy.dispatchesPerEncoder,
      reuseUniformBuffers: policy.reuseUniformBuffers,
      workloadId: workload.id,
      workloadLabel: workload.label,
      maxTokens: workload.maxTokens,
      repetition,
      runKind,
      executionIndex,
      startedAt,
      success: false,
      error: error?.message || String(error),
      orchestration: control.getStats(),
      chunkGapsMs: [],
    };
  }
}

async function runExperiment() {
  if (running) return;
  running = true;
  elements.run.disabled = true;
  elements.export.disabled = true;
  elements.log.textContent = "";
  elements.summary.hidden = true;
  const policies = selected("policies", ORCHESTRATION_POLICIES);
  const workloads = selected("workloads", ORCHESTRATION_WORKLOADS);
  const repetitions = Number(elements.repetitions.value);
  const warmups = Number(elements.warmups.value);
  const campaignId = elements["campaign-id"].value.trim();
  let engine = null;
  try {
    if (policies.length < 2) throw new Error("Select the baseline and at least one treatment");
    if (!workloads.length) throw new Error("Select at least one workload");
    if (!/^[a-z0-9][a-z0-9-]*$/.test(campaignId)) throw new Error("Campaign ID must use lowercase letters, numbers, and hyphens");
    setStatus("Loading local WebLLM experiment bundle…");
    const runtimeBuild = await fetch("../vendor/generated/runtime-orchestration-build.json").then((response) => {
      if (!response.ok) throw new Error("Local experiment bundle metadata is missing; run node scripts/build-runtime-orchestration.mjs");
      return response.json();
    });
    const webllm = await import(BUNDLE_URL);
    const model = elements.model.value;
    const deviceProfile = await getWebGpuProfile();
    setStatus(`Loading ${model}; the first run may download model files…`);
    engine = await webllm.CreateMLCEngine(model, {
      initProgressCallback: (progress) => setStatus(progress.text),
    }, { context_window_size: 4096 });
    const control = globalThis.__WEBLLM_ORCHESTRATION_EXPERIMENT__;
    if (!control || control.version !== "0.2.79") {
      throw new Error("The experimental runtime control was not installed by the local bundle");
    }

    let executionIndex = 0;
    const rows = [];
    const total = policies.length * workloads.length * (warmups + repetitions);
    elements.progress.max = total;
    elements.progress.value = 0;
    for (const workload of workloads) {
      for (const policy of policies) {
        for (let index = 0; index < warmups; index += 1) {
          setStatus(`Warmup: ${workload.label}, ${policy.label}`);
          const row = await runOne(engine, control, policy, workload, -1 - index, "warmup", executionIndex++);
          if (!row.success) throw new Error(`Warmup failed for ${policy.label}: ${row.error || row.finishReason}`);
          elements.progress.value += 1;
        }
      }
      for (let repetition = 0; repetition < repetitions; repetition += 1) {
        for (const policy of rotatedPolicyOrder(policies, repetition)) {
          setStatus(`Measured ${repetition + 1}/${repetitions}: ${workload.label}, ${policy.label}`);
          const row = await runOne(engine, control, policy, workload, repetition, "measured", executionIndex++);
          rows.push(row);
          log(`${row.success ? "ok" : "FAIL"} ${policy.id}: TTFT ${num(row.ttftMs)} ms, wall ${num(row.wallMs)} ms, submits ${row.orchestration.submitCalls}`);
          elements.progress.value += 1;
        }
      }
    }

    const summaries = summarizeOrchestrationRows(rows, policies, workloads);
    lastPayload = {
      schemaVersion: "webllm-runtime-orchestration-v1",
      campaignId,
      createdAt: new Date().toISOString(),
      label: makeDefaultLabel(deviceProfile),
      webLlmVersion: "0.2.79",
      runtimeBundle: BUNDLE_URL,
      runtimeBuild,
      model,
      repetitions,
      warmups,
      policies,
      workloads: workloads.map(({ id, label, maxTokens, prompt, stop }) => ({
        id,
        label,
        maxTokens,
        prompt,
        stop: stop ?? [],
      })),
      deviceProfile,
      fixedGeneration: {
        temperature: 0,
        topP: 1,
        repetitionPenalty: 1,
        frequencyPenalty: 0,
        presencePenalty: 0,
        seed: 42,
      },
      summaries,
      rows,
    };
    renderSummary(summaries);
    const saveResponse = await fetch("/api/runtime-orchestration-results", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(lastPayload),
    });
    const saved = await saveResponse.json();
    if (!saveResponse.ok) throw new Error(`Result save failed: ${saved.error || saveResponse.status}`);
    elements.export.disabled = false;
    setStatus(`Complete and saved to ${saved.path}. Check output and trace equality before interpreting timing.`);
  } catch (error) {
    setStatus(`Stopped: ${error?.message || error}`);
    console.error(error);
  } finally {
    try {
      if (engine) await engine.unload();
    } catch (error) {
      log(`Unload warning: ${error?.message || error}`);
    }
    running = false;
    elements.run.disabled = false;
  }
}

for (const model of FIXED_MODELS) {
  const option = document.createElement("option");
  option.value = model.model;
  option.textContent = `${model.id} — ${model.model}`;
  elements.model.append(option);
}
addChecks(elements["policy-controls"], "policies", ORCHESTRATION_POLICIES);
addChecks(elements["workload-controls"], "workloads", ORCHESTRATION_WORKLOADS);
elements.run.addEventListener("click", runExperiment);
elements.export.addEventListener("click", () => {
  if (!lastPayload) return;
  downloadJSON(lastPayload, `webllm-runtime-orchestration-${lastPayload.label || "device"}.json`);
});
