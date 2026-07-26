import {
  MATRIX_MEASURED_RUNS,
  MATRIX_WARMUP_RUNS,
  buildMatrixCases,
  matrixInferenceCount,
  matrixProgressSteps,
} from "./collection-matrix.js";
import { getAvailableFixedModels } from "./fixed-model-bench.js";
import { getWebGpuProfile, makeDefaultLabel } from "./device-profile.js";
import { LATENCY_BUDGETS, TASK_PROFILES } from "./task-profiles.js";
import { AUTOMATIC_POLICY_GROUPS } from "./tuning-policies.js";
import {
  RETENTION_MEASURED_RUNS,
  runCollection,
} from "./collection-runner.js";
import { collectionCsvText, collectionFileStem } from "./export-format.js";
import { downloadCSV, downloadJSON } from "../shared/benchmark-utils.js";

const $ = (id) => document.getElementById(id);
const startButton = $("start-btn");
const exportButton = $("export-btn");
const labelElement = $("label");
const modelOptions = $("model-options");
const matrixSize = $("matrix-size");
const latencyBudgetSelect = $("latency-budget");
const workflowSelect = $("workflow");
const statusElement = $("status");
const logElement = $("log");
const progressPhase = $("progress-phase");
const progressPercent = $("progress-percent");
const progressTrack = $("progress-track");
const progressFill = $("progress-fill");

const stepElements = {
  profile: $("step-profile"),
  probe: $("step-probe"),
  baseline: $("step-baseline"),
  chat: $("step-chat"),
  export: $("step-export"),
};

let payload = null;
let progress = null;

function log(message) {
  logElement.textContent += `${message}\n`;
  logElement.scrollTop = logElement.scrollHeight;
}

function selectedModels() {
  const selectedIds = new Set(
    [...modelOptions.querySelectorAll("input:checked")].map((input) => input.value)
  );
  return getAvailableFixedModels().filter((model) => selectedIds.has(model.id));
}

function selectedMatrixCases() {
  return buildMatrixCases(selectedModels(), TASK_PROFILES);
}

function selectedPolicyGroups() {
  const selected = AUTOMATIC_POLICY_GROUPS.find((group) => group.id === workflowSelect.value);
  return selected ? [selected] : [];
}

function setBusy(busy) {
  startButton.disabled = busy;
  labelElement.disabled = busy;
  latencyBudgetSelect.disabled = busy;
  workflowSelect.disabled = busy;
  for (const input of modelOptions.querySelectorAll("input")) input.disabled = busy;
}

function resetSteps() {
  for (const element of Object.values(stepElements)) element.textContent = "waiting";
}

function resetProgress(cases, policyCount) {
  progress = {
    phase: "waiting to start",
    profileDone: false,
    probeCompleted: 0,
    probeTotal: 12,
    tuningCompleted: 0,
    tuningTotal: matrixProgressSteps(cases, policyCount),
    expectedPerPolicy: 1 + MATRIX_WARMUP_RUNS + MATRIX_MEASURED_RUNS,
    perPolicyCompleted: new Map(),
    exportDone: false,
  };
  renderProgress();
}

function renderProgress() {
  const total = 1 + progress.probeTotal + progress.tuningTotal + 1;
  const completed =
    (progress.profileDone ? 1 : 0) +
    progress.probeCompleted +
    progress.tuningCompleted +
    (progress.exportDone ? 1 : 0);
  const percent = total ? Math.min(100, Math.round(completed / total * 100)) : 0;
  progressPhase.textContent = progress.phase;
  progressPercent.textContent = `${percent}%`;
  progressTrack.setAttribute("aria-valuenow", String(percent));
  progressFill.style.width = `${percent}%`;
}

function updateTuningProgress(tuning, matrixCase) {
  const policyKey = `${matrixCase.id}:${tuning.policyId}`;
  if (["policy_start", "warmup", "measured"].includes(tuning.event)) {
    progress.tuningCompleted += 1;
    progress.perPolicyCompleted.set(
      policyKey,
      (progress.perPolicyCompleted.get(policyKey) ?? 0) + 1
    );
  }
  if (tuning.event === "policy_failed") {
    const completed = progress.perPolicyCompleted.get(policyKey) ?? 0;
    progress.tuningCompleted += Math.max(0, progress.expectedPerPolicy - completed);
  }
  progress.phase =
    `${matrixCase.model.family} / ${matrixCase.task.label}: ` +
    `${progress.tuningCompleted}/${progress.tuningTotal}`;
  renderProgress();
}

function updateProbeProgress(probe) {
  progress.probeTotal = probe.total || progress.probeTotal;
  progress.probeCompleted = probe.completed || 0;
  progress.phase = `WebGPU probe ${progress.probeCompleted}/${progress.probeTotal}`;
  renderProgress();
}

function updateTuningPlan(plan) {
  if (plan.total != null) progress.tuningTotal = plan.total;
  if (plan.expectedPerPolicy != null) {
    progress.expectedPerPolicy = plan.expectedPerPolicy;
  }
  progress.perPolicyCompleted.clear();
  renderProgress();
}

const collectionUi = {
  log,
  status(message) {
    statusElement.textContent = message;
  },
  phase(message) {
    progress.phase = message;
    renderProgress();
  },
  step(step, message) {
    stepElements[step].textContent = message;
  },
  profileComplete() {
    progress.profileDone = true;
    renderProgress();
  },
  probeProgress: updateProbeProgress,
  tuningProgress: updateTuningProgress,
  tuningPlan: updateTuningPlan,
};

function updateMatrixSize() {
  const cases = selectedMatrixCases();
  const policyGroups = selectedPolicyGroups();
  const baselineInferences = matrixInferenceCount(cases, 1);
  const policyCount = policyGroups.reduce(
    (total, group) => total + group.policies.length,
    0
  );
  const tuningInferences = matrixInferenceCount(cases, policyCount);
  const needsRetentionGate = policyGroups.some((group) => group.requiresHistoryRequirements);
  const retentionInferences = needsRetentionGate
    ? cases.length * (MATRIX_WARMUP_RUNS + RETENTION_MEASURED_RUNS)
    : 0;

  if (!cases.length) {
    matrixSize.textContent = "select at least one model";
  } else if (!policyGroups.length) {
    matrixSize.textContent = `${cases.length} baseline cases, ${baselineInferences} inferences`;
  } else {
    matrixSize.textContent =
      `${cases.length} baselines (${baselineInferences} inferences), then up to ` +
      `${retentionInferences ? `${retentionInferences} retention-gate and ` : ""}` +
      `${tuningInferences} gated tuning inferences`;
  }
  startButton.disabled = cases.length === 0;
}

function populateControls() {
  for (const model of getAvailableFixedModels()) {
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = model.id;
    checkbox.checked = true;
    checkbox.addEventListener("change", updateMatrixSize);
    const label = document.createElement("label");
    label.append(checkbox, ` ${model.model}`);
    modelOptions.appendChild(label);
  }
  for (const budget of LATENCY_BUDGETS) {
    latencyBudgetSelect.add(new Option(budget.label, budget.id));
  }
  workflowSelect.addEventListener("change", updateMatrixSize);
  updateMatrixSize();
}

async function startCollection() {
  setBusy(true);
  exportButton.disabled = true;
  payload = null;
  logElement.textContent = "";
  resetSteps();

  const matrixCases = selectedMatrixCases();
  if (!matrixCases.length) {
    statusElement.textContent = "select at least one model";
    setBusy(false);
    updateMatrixSize();
    return;
  }

  const policyGroups = selectedPolicyGroups();
  const policyCount = policyGroups.reduce(
    (total, group) => total + group.policies.length,
    1
  );
  resetProgress(matrixCases, policyCount);
  const latencyBudget = LATENCY_BUDGETS.find(
    (budget) => budget.id === latencyBudgetSelect.value
  ) ?? LATENCY_BUDGETS[0];

  try {
    const result = await runCollection({
      matrixCases,
      policyGroups,
      latencyTargetMs: latencyBudget.targetMs,
      matrixSeed: Date.now() >>> 0,
      label: labelElement.value.trim(),
      workflow: workflowSelect.value,
      ui: collectionUi,
    });
    payload = result.payload;
    progress.exportDone = true;
    progress.phase = "ready to export";
    stepElements.export.textContent = "ready";
    exportButton.disabled = false;
    statusElement.textContent = policyGroups.length
      ? `done; ${result.qualifiedCaseCount}/${matrixCases.length} baselines qualified; ` +
        `${result.recommendationCount}/${result.tuningRunCount} tuned cases beat untouched WebLLM; export data`
      : `baseline validation done; ${result.qualifiedCaseCount}/${matrixCases.length} cases qualified; export data`;
    renderProgress();
  } catch (error) {
    progress.phase = "collection stopped";
    statusElement.textContent = `error: ${error.message}`;
    log(`ERROR: ${error.stack || error}`);
    renderProgress();
  } finally {
    setBusy(false);
    updateMatrixSize();
  }
}

async function initializePage() {
  if (!navigator.gpu) {
    statusElement.textContent = "WebGPU not available in this browser. Use Chrome or Edge.";
    startButton.disabled = true;
    return;
  }
  populateControls();
  resetProgress(selectedMatrixCases(), 1);
  const profile = await getWebGpuProfile();
  if (!labelElement.value.trim()) labelElement.value = makeDefaultLabel(profile);
  stepElements.profile.textContent = "ready";
}

startButton.addEventListener("click", startCollection);
exportButton.addEventListener("click", () => {
  downloadJSON(payload, `${collectionFileStem(payload)}.json`);
  downloadCSV(collectionCsvText(payload), `${collectionFileStem(payload)}.csv`);
});
initializePage();
