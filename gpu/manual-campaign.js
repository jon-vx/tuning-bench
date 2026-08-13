import { experimentSupport } from "./controlled-experiment.js";
import { getWebGpuProfile } from "./device-profile.js";
import { runExperiment } from "./manual-tune.js";
import { loadWebLlmModel, unloadWebLlm } from "./webllm-runtime.js";

const $ = (id) => document.getElementById(id);

function changeValue(id, value) {
  const control = $(id);
  control.value = value;
  control.dispatchEvent(new Event("change"));
}

function targetSupported(target) {
  return experimentSupport(
    target.parameterId,
    target.modelConfigId,
    target.taskProfileId
  ).supported;
}

async function request(path, options = {}) {
  const response = await fetch(`/__campaign/${path}`, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function runTarget(campaignId, target, execution, orderOffset) {
  changeValue("model", target.modelConfigId);
  changeValue("task", target.taskProfileId);
  changeValue("parameter", target.parameterId);
  changeValue("warmups", String(execution.warmups ?? 1));
  changeValue("latency-target", "10000");
  $("label").value =
    `${campaignId}:${target.modelConfigId}:${target.taskProfileId}:${target.parameterId}`;
  delete window.__manualExperimentPayload;
  await request("claim", {
    method: "POST",
    body: JSON.stringify({ id: target.id }),
  });
  let runOptions = {};
  if (execution.mode === "rapid") {
    runOptions = {
      blocks: execution.blocks,
      warmups: execution.warmups,
      measuredRunsPerScenario: execution.measuredRunsPerScenario,
      completeCounterbalance: execution.completeCounterbalance,
      orderOffset,
      mode: execution.mode,
    };
  }
  if (execution.mode === "validation") {
    runOptions = {
      blocks: execution.blocks,
      warmups: execution.warmups,
      measuredRunsPerCase: execution.measuredRunsPerCase,
      candidatePolicyId: target.candidatePolicyId,
      warmupEachPolicy: execution.warmupEachPolicy,
      completeCounterbalance: execution.completeCounterbalance,
      orderOffset,
      mode: execution.mode,
    };
  }
  if (execution.mode === "latency") {
    runOptions = {
      blocks: execution.blocks,
      warmups: execution.warmups,
      measuredRunsPerCase: execution.measuredRunsPerCase,
      candidatePolicyId: target.candidatePolicyId,
      warmupEachPolicy: execution.warmupEachPolicy,
      completeCounterbalance: execution.completeCounterbalance,
      orderOffset,
      mode: execution.mode,
      conversationScenarios: ["single-turn"],
    };
  }
  await runExperiment(runOptions);
  const payload = window.__manualExperimentPayload;
  if (!payload) {
    throw new Error($("status").textContent || "experiment did not produce a payload");
  }
  await request("result", {
    method: "POST",
    body: JSON.stringify({ id: target.id, payload }),
  });
}

async function preflightArtifacts(manifest) {
  const pending = (manifest.artifactPreflight ?? []).filter((artifact) =>
    artifact.status === "pending"
  );
  for (const artifact of pending) {
    let status = "available";
    let error = "";
    try {
      $("status").textContent =
        `checking ${artifact.modelConfigId} ${artifact.quantization}...`;
      await loadWebLlmModel(
        artifact.artifactId,
        {
          contextWindowSize: 4096,
          penaltyProcessingMode: "explicit-zero",
          kvCacheMode: "context",
          engineThreadMode: "main",
        },
        (message) => {
          $("status").textContent = message;
        }
      );
    } catch (caught) {
      status = "failed";
      error = caught.message;
    } finally {
      await unloadWebLlm(() => {});
    }
    await request("preflight", {
      method: "POST",
      body: JSON.stringify({
        modelConfigId: artifact.modelConfigId,
        quantization: artifact.quantization,
        status,
        error,
      }),
    });
  }
}

export async function runManualCampaign() {
  let manifest = await request("manifest");
  const deviceProfile = await getWebGpuProfile();
  await request("start", {
    method: "POST",
    body: JSON.stringify({ deviceProfile }),
  });
  await preflightArtifacts(manifest);
  manifest = await request("manifest");
  const pending = manifest.entries.filter((entry) =>
    targetSupported(entry) && !["complete", "skipped"].includes(entry.status)
  );
  window.__manualCampaignState = {
    campaignId: manifest.campaignId,
    completed: 0,
    total: pending.length,
    current: "",
    stop: false,
  };

  for (const [targetIndex, target] of pending.entries()) {
    if (window.__manualCampaignState.stop) break;
    window.__manualCampaignState.current = target.id;
    let completed = false;
    let lastError;
    for (let attempt = 1; attempt <= 2 && !completed; attempt++) {
      try {
        await runTarget(
          manifest.campaignId,
          target,
          manifest.execution ?? { mode: "confirmatory" },
          targetIndex
        );
        completed = true;
      } catch (error) {
        lastError = error;
        await request("error", {
          method: "POST",
          body: JSON.stringify({
            id: target.id,
            attempt,
            final: attempt === 2,
            message: error.message,
            stack: error.stack,
          }),
        });
      }
    }
    window.__manualCampaignState.completed += 1;
    window.__manualCampaignState.lastError = completed ? "" : lastError?.message ?? "";
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  window.__manualCampaignState.current = "";
  window.__manualCampaignState.finished = !window.__manualCampaignState.stop;
  return request("manifest");
}

export function pauseManualCampaign() {
  if (window.__manualCampaignState) window.__manualCampaignState.stop = true;
}

const queuePanel = $("queue-panel");
const queueModels = $("queue-models");
const queueTasks = $("queue-tasks");
const queueParameters = $("queue-parameters");
const queueConfigure = $("queue-configure");
const queueStart = $("queue-start");
const queuePause = $("queue-pause");
const queueResume = $("queue-resume");
const queueProgress = $("queue-progress");

function selectedValues(select) {
  return [...select.selectedOptions].map((option) => option.value);
}

function targetCounts(manifest) {
  const complete = manifest.entries.filter((entry) => entry.status === "complete").length;
  const failed = manifest.entries.filter((entry) => entry.status === "failed").length;
  const runnable = manifest.entries.filter((entry) => entry.status !== "skipped").length;
  return { complete, failed, runnable };
}

function renderQueueProgress(manifest) {
  const counts = targetCounts(manifest);
  const inference = manifest.inferenceProgress ?? {};
  const inferenceText = Number.isFinite(inference.expected)
    ? `; measured inferences ${inference.completed}/${inference.expected}`
    : "";
  const state = window.__manualCampaignState;
  const current = state?.current ? `; current ${state.current}` : "";
  queueProgress.textContent =
    `Targets ${counts.complete}/${counts.runnable}; failed ${counts.failed}${inferenceText}${current}`;
  queueStart.disabled = Boolean(manifest.startedAt) || counts.complete === counts.runnable;
  queueResume.disabled = !manifest.startedAt || counts.complete === counts.runnable || Boolean(state && !state.stop);
  queuePause.disabled = !state || state.stop || state.finished;
  queueConfigure.disabled = Boolean(manifest.startedAt);
}

async function refreshQueue() {
  const manifest = await request("manifest");
  renderQueueProgress(manifest);
  return manifest;
}

async function runQueue() {
  queueStart.disabled = true;
  queueResume.disabled = true;
  queuePause.disabled = false;
  try {
    await runManualCampaign();
  } finally {
    await refreshQueue();
  }
}

async function initializeQueueUi() {
  let manifest;
  try {
    manifest = await request("manifest");
  } catch {
    return;
  }
  queuePanel.hidden = false;
  const selectedParameters = new Set(manifest.parameters);
  for (const model of manifest.models) {
    queueModels.add(new Option(model.model, model.id, true, true));
  }
  for (const task of manifest.tasks) {
    queueTasks.add(new Option(task.label, task.id, true, true));
  }
  const definitions = (await import("./controlled-experiment.js")).EXPERIMENT_DEFINITIONS;
  for (const definition of definitions.filter((item) => selectedParameters.has(item.id))) {
    const suffix = definition.scope === "application" ? " [APP-LEVEL]" : "";
    queueParameters.add(new Option(`${definition.label}${suffix}`, definition.id, true, true));
  }
  renderQueueProgress(manifest);
  queueConfigure.addEventListener("click", async () => {
    try {
      manifest = await request("configure", {
        method: "POST",
        body: JSON.stringify({
          modelIds: selectedValues(queueModels),
          taskIds: selectedValues(queueTasks),
          parameterIds: selectedValues(queueParameters),
        }),
      });
      renderQueueProgress(manifest);
    } catch (error) {
      queueProgress.textContent = `Queue error: ${error.message}`;
    }
  });
  queueStart.addEventListener("click", runQueue);
  queueResume.addEventListener("click", runQueue);
  queuePause.addEventListener("click", () => {
    pauseManualCampaign();
    queuePause.disabled = true;
    queueProgress.textContent += "; pausing after current target";
  });
  setInterval(() => refreshQueue().catch(() => {}), 2000);
}

initializeQueueUi();
