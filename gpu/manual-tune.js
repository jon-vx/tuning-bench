import {
  analyzeControlledExperiment,
  buildExperimentPolicies,
  controlledExperimentSettings,
  counterbalancedOrder,
  EXPERIMENT_DEFINITIONS,
  getExperimentDefinition,
} from "./controlled-experiment.js";
import {
  getAvailableFixedModels,
  runFixedModelPoliciesCore,
} from "./fixed-model-bench.js";
import { getWebGpuProfile, makeDefaultLabel } from "./device-profile.js";
import { TASK_PROFILES } from "./task-profiles.js";
import { renderDoseResponsePlots } from "./experiment-plots.js";
import { downloadCSV, downloadJSON } from "../shared/benchmark-utils.js";
import { manualCsvText, manualFileStem } from "./export-format.js";

const $ = (id) => document.getElementById(id);
const runButton = $("run-experiment");
const exportJsonButton = $("export-json");
const exportCsvButton = $("export-csv");
const resultTable = $("results");
const statusElement = $("status");
const logElement = $("log");
const progressElement = $("experiment-progress");
const progressFill = $("progress-fill");
const modelSelect = $("model");
const taskSelect = $("task");
const parameterSelect = $("parameter");
const labelInput = $("label");
const blocksInput = $("blocks");
const warmupsInput = $("warmups");
const latencyTargetInput = $("latency-target");
const libraryDefaultInput = $("include-library");
const selectorBudgetInput = $("selector-budget");
const selectorBudgetControl = $("selector-budget-control");
const valuesElement = $("values");
const decisionRuleElement = $("decision-rule");
const repeatabilityHeading = $("repeatability-heading");
const matchedDeltaHeading = $("matched-delta-heading");
const lockedSettingsElement = $("locked-settings");
const workloadElement = $("workload");
const plotsElement = $("plots");
const plotGrid = $("plot-grid");

let payload = null;

function selectedTaskProfile() {
  return TASK_PROFILES.find((task) => task.id === taskSelect.value) ?? TASK_PROFILES[0];
}

function experimentBaseOverrides(parameterId) {
  const task = selectedTaskProfile();
  const overrides = { maxTokens: task.outputTokenLimits.standard };
  if (parameterId === "historySelectionPolicy") {
    overrides.historyTokenBudget = Number(selectorBudgetInput.value);
  }
  return overrides;
}

function experimentDefinitionOverrides(parameterId) {
  if (parameterId !== "maxTokens") return {};
  const limits = selectedTaskProfile().outputTokenLimits;
  return {
    values: [limits.short, limits.standard, limits.expanded],
    controlledValue: limits.standard,
  };
}

function selectedPolicies(parameterId) {
  const policies = buildExperimentPolicies(
    parameterId,
    experimentBaseOverrides(parameterId),
    experimentDefinitionOverrides(parameterId)
  );
  return libraryDefaultInput.checked
    ? policies
    : policies.filter((policy) => policy.baselineType !== "library_default");
}

function experimentWorkload(parameterId) {
  const workload = getExperimentDefinition(parameterId).workload;
  return {
    measuredRuns: workload.measuredRuns,
    conversationScenarios: workload.scenarios,
    warmupPerConfiguration: Boolean(workload.warmupEach),
  };
}

function log(message) {
  logElement.textContent += `${message}\n`;
  logElement.scrollTop = logElement.scrollHeight;
}

function setBusy(busy) {
  runButton.disabled = busy;
  for (const control of document.querySelectorAll("fieldset input, fieldset select")) {
    control.disabled = busy;
  }
  exportJsonButton.disabled = busy || !payload;
  exportCsvButton.disabled = busy || !payload;
}

function displayNumber(value, digits = 0) {
  return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

function displayPercent(value, digits = 0) {
  return Number.isFinite(value) ? `${value.toFixed(digits)}%` : "n/a";
}

function updateDefinition() {
  const parameterId = parameterSelect.value;
  selectorBudgetControl.hidden = parameterId !== "historySelectionPolicy";
  const settings = controlledExperimentSettings(
    parameterId,
    experimentBaseOverrides(parameterId),
    experimentDefinitionOverrides(parameterId)
  );
  const policies = selectedPolicies(parameterId);
  const experiment = experimentWorkload(parameterId);
  let blocks = Number(blocksInput.value);
  const warmups = Number(warmupsInput.value);
  if (blocks % policies.length !== 0) {
    blocks = policies.length;
    blocksInput.value = String(blocks);
  }
  valuesElement.textContent = settings.values.join(", ");
  const timingMetric = settings.objectiveMetric === "wall" ? "WALL" : "TTFT";
  if (parameterId === "distractionHistoryBudget") {
    decisionRuleElement.textContent = "task quality + no truncation + repeatable TTFT gain";
  } else if (settings.objectiveMetric === "quality") {
    decisionRuleElement.textContent =
      "quality recovery + no truncation + no meaningful TTFT regression";
  } else {
    decisionRuleElement.textContent =
      `quality + retention + no truncation + repeatable ${timingMetric} gain`;
  }
  repeatabilityHeading.textContent = `${timingMetric} repeat IQR`;
  matchedDeltaHeading.textContent = settings.objectiveMetric === "quality"
    ? "quality pass delta"
    : `matched ${timingMetric} delta`;
  lockedSettingsElement.textContent = Object.entries(settings.lockedSettings)
    .filter(([key]) => !["frequencyPenalty", "presencePenalty", "repetitionPenalty"].includes(key))
    .map(([key, value]) => `${key}=${value}`)
    .join("; ");
  const warmupInferences = experiment.warmupPerConfiguration
    ? policies.length * blocks * warmups
    : blocks * warmups;
  const inferences = policies.length * blocks * experiment.measuredRuns + warmupInferences;
  const firstScenario = experiment.conversationScenarios[0];
  let scenarioText;
  if (experiment.conversationScenarios.length === 2 &&
      firstScenario === "long-history-distraction") {
    scenarioText = "3 distraction + 3 retention prompts/block";
  } else if (firstScenario === "long-history-distraction") {
    scenarioText = `${experiment.measuredRuns} distraction prompts/block`;
  } else if (experiment.conversationScenarios.length === 1 &&
      firstScenario === "single-turn") {
    scenarioText = "3 single-turn prompts/block";
  } else if (experiment.conversationScenarios.length === 1) {
    scenarioText = "3 retention prompts/block";
  } else {
    scenarioText = "3 clean + 3 retention prompts/block";
  }
  workloadElement.textContent =
    `${policies.length} configurations; ${blocks} balanced blocks; ${scenarioText}; ` +
    `${warmupInferences} warmups; ${inferences} inferences`;
}

function updateProgress(completed, total, phase) {
  const percent = total ? Math.round(completed / total * 100) : 0;
  statusElement.textContent = `${phase} (${completed}/${total}, ${percent}%)`;
  progressElement.setAttribute("aria-valuenow", String(percent));
  progressFill.style.width = `${percent}%`;
}

function renderResults(analysis) {
  const body = resultTable.querySelector("tbody");
  body.replaceChildren();
  for (const summary of analysis.summaries) {
    const row = document.createElement("tr");
    row.className = summary.status;
    const values = [
      summary.policyLabel,
      summary.experimentValue,
      summary.runCount,
      displayNumber(summary.medianPromptTokens),
      displayNumber(summary.medianTtftMs),
      displayPercent(summary.matchedObjectiveDeltaPct, 1),
      displayPercent(summary.matchedRepeatabilityIqrPct * 100, 1),
      displayNumber(summary.medianWallMs),
      displayPercent(summary.wallDeltaPct, 1),
      displayPercent(summary.qualityPassRate * 100),
      summary.cleanRunCount ? displayPercent(summary.cleanQualityPassRate * 100) : "n/a",
      summary.requiredHistoryRunCount
        ? displayPercent(summary.requiredHistoryQualityPassRate * 100)
        : "n/a",
      summary.anyTruncated ? "yes" : "no",
      summary.status,
    ];
    for (const [index, value] of values.entries()) {
      const cell = document.createElement(index === 0 ? "th" : "td");
      cell.textContent = value;
      row.appendChild(cell);
    }
    body.appendChild(row);
  }
  resultTable.hidden = false;
}

async function runExperiment() {
  setBusy(true);
  payload = null;
  resultTable.hidden = true;
  plotsElement.hidden = true;
  logElement.textContent = "";
  progressFill.style.width = "0";
  const parameterId = parameterSelect.value;
  const settings = controlledExperimentSettings(
    parameterId,
    experimentBaseOverrides(parameterId),
    experimentDefinitionOverrides(parameterId)
  );
  const policies = selectedPolicies(parameterId);
  const experiment = experimentWorkload(parameterId);
  const blocks = Number(blocksInput.value);
  const warmups = Number(warmupsInput.value);
  const latencyTargetMs = Number(latencyTargetInput.value);
  const measuredRuns = experiment.measuredRuns;
  const warmupInferences = experiment.warmupPerConfiguration
    ? policies.length * blocks * warmups
    : blocks * warmups;
  const totalInferences = policies.length * blocks * measuredRuns + warmupInferences;
  let completedInferences = 0;
  const results = [];
  const executions = [];

  try {
    if (blocks % policies.length !== 0) {
      throw new Error(`block count must be a multiple of ${policies.length}`);
    }
    statusElement.textContent = "collecting device profile...";
    const deviceProfile = await getWebGpuProfile();
    const label = labelInput.value.trim() || makeDefaultLabel(deviceProfile);
    labelInput.value = label;

    for (let blockIndex = 0; blockIndex < blocks; blockIndex++) {
      const order = counterbalancedOrder(policies, blockIndex);
      log(`\n=== block ${blockIndex + 1}/${blocks}: ${order.map((policy) => policy.id).join(" -> ")} ===`);
      for (const [orderIndex, policy] of order.entries()) {
        const phase = `block ${blockIndex + 1}/${blocks}, ${policy.label}`;
        const policyWarmups = experiment.warmupPerConfiguration || orderIndex === 0 ? warmups : 0;
        let completedForPolicy = 0;
        const result = await runFixedModelPoliciesCore({
          modelConfigId: modelSelect.value,
          taskProfileId: taskSelect.value,
          latencyTargetMs,
          policies: [policy],
          warmup: policyWarmups,
          runs: measuredRuns,
          randomizeOrder: false,
          longHistoryScenario: "long-history-required",
          measuredConversationScenarios: experiment.conversationScenarios,
          label,
          deviceProfile,
          logFn: (message) => log(`[block ${blockIndex + 1}/${policy.id}] ${message}`),
          statusFn: (message) => { statusElement.textContent = `${phase}: ${message}`; },
          progressFn: (event) => {
            if (["warmup", "measured"].includes(event.event)) {
              completedForPolicy += 1;
              completedInferences += 1;
              updateProgress(completedInferences, totalInferences, phase);
            }
            if (event.event === "policy_failed") {
              const missing = policyWarmups + measuredRuns - completedForPolicy;
              completedInferences += Math.max(0, missing);
              updateProgress(completedInferences, totalInferences, `${phase}: failed`);
            }
          },
        });
        const annotatedRows = result.results.map((row) => ({
          ...row,
          experimentBlock: blockIndex + 1,
          experimentOrderIndex: orderIndex,
          experimentParameter: parameterId,
          experimentValue: policy.experimentValue,
        }));
        results.push(...annotatedRows);
        const executionMeta = { ...result.meta };
        delete executionMeta.deviceProfile;
        executions.push({
          block: blockIndex + 1,
          orderIndex,
          policyId: policy.id,
          meta: executionMeta,
        });
      }
    }

    const analysis = analyzeControlledExperiment(policies, results, parameterId);
    payload = {
      schema: "webllm-controlled-experiment-v2",
      timestamp: new Date().toISOString(),
      label,
      deviceProfile,
      experiment: {
        ...settings,
        modelConfigId: modelSelect.value,
        taskProfileId: taskSelect.value,
        blocks,
        warmupsPerBlock: experiment.warmupPerConfiguration ? null : warmups,
        warmupsPerConfiguration: experiment.warmupPerConfiguration ? warmups : null,
        measuredRunsPerBlock: measuredRuns,
        conversationScenarios: experiment.conversationScenarios,
        executionOrders: Array.from({ length: blocks }, (_, index) =>
          counterbalancedOrder(policies, index).map((policy) => policy.id)
        ),
      },
      policies,
      executions,
      results,
      ...analysis,
    };
    renderResults(analysis);
    renderDoseResponsePlots(
      plotsElement,
      plotGrid,
      analysis,
      parameterId
    );
    window.__manualExperimentPayload = payload;
    exportJsonButton.disabled = false;
    exportCsvButton.disabled = false;
    statusElement.textContent = analysis.recommendedPolicyId
      ? `done; candidate: ${analysis.recommendedPolicyId}`
      : "done; no changed value met every quality and repeatability requirement";
    progressFill.style.width = "100%";
    progressElement.setAttribute("aria-valuenow", "100");
  } catch (error) {
    statusElement.textContent = `error: ${error.message}`;
    log(error.stack || String(error));
  } finally {
    setBusy(false);
  }
}

function populate() {
  const models = getAvailableFixedModels();
  for (const model of models) modelSelect.add(new Option(model.model, model.id));
  const qwen = models.find((model) => model.family === "qwen-0.5b");
  if (qwen) modelSelect.value = qwen.id;
  for (const task of TASK_PROFILES) taskSelect.add(new Option(task.label, task.id));
  taskSelect.value = "chat-summarize";
  for (const definition of EXPERIMENT_DEFINITIONS) {
    parameterSelect.add(new Option(definition.label, definition.id));
  }
  for (const control of [
    parameterSelect,
    taskSelect,
    blocksInput,
    warmupsInput,
    libraryDefaultInput,
    selectorBudgetInput,
  ]) {
    control.addEventListener("change", updateDefinition);
  }
  updateDefinition();
}

runButton.addEventListener("click", runExperiment);
exportJsonButton.addEventListener("click", () => {
  downloadJSON(payload, `${manualFileStem(payload)}.json`);
});
exportCsvButton.addEventListener("click", () => {
  downloadCSV(manualCsvText(payload), `${manualFileStem(payload)}.csv`);
});

if (!navigator.gpu) {
  statusElement.textContent = "WebGPU is unavailable. Use Chrome or Edge on a WebGPU-capable device.";
  runButton.disabled = true;
} else {
  populate();
  getWebGpuProfile().then((profile) => {
    if (!labelInput.value) labelInput.value = makeDefaultLabel(profile);
  });
}
