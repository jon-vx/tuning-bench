import {
  analyzeControlledExperiment,
  buildExperimentPolicies,
  controlledExperimentSettings,
  counterbalancedOrder,
  EXPERIMENT_DEFINITIONS,
  experimentSupport,
  getExperimentDefinition,
} from "./controlled-experiment.js";
import {
  getAvailableFixedModels,
  runFixedModelPoliciesCore,
} from "./fixed-model-bench.js";
import { getWebGpuProfile, makeDefaultLabel } from "./device-profile.js";
import { MANUAL_TASK_PROFILES } from "./task-profiles.js";
import { renderDoseResponsePlots } from "./experiment-plots.js";
import { downloadCSV, downloadJSON } from "../shared/benchmark-utils.js";
import { manualCsvText, manualFileStem } from "./export-format.js";
import { modelArtifactId } from "./models.js";

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
let running = false;

function selectedTaskProfile() {
  return MANUAL_TASK_PROFILES.find((task) => task.id === taskSelect.value)
    ?? MANUAL_TASK_PROFILES[0];
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

function selectedPolicies(parameterId, candidatePolicyId = "") {
  const policies = buildExperimentPolicies(
    parameterId,
    experimentBaseOverrides(parameterId),
    experimentDefinitionOverrides(parameterId)
  );
  const selected = candidatePolicyId
    ? policies.filter((policy) =>
      ["library_default", "controlled"].includes(policy.baselineType) ||
      policy.id === candidatePolicyId
    )
    : policies;
  if (candidatePolicyId && !selected.some((policy) => policy.id === candidatePolicyId)) {
    throw new Error(`unknown candidate policy: ${candidatePolicyId}`);
  }
  return selected;
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
  running = busy;
  const support = experimentSupport(
    parameterSelect.value,
    modelSelect.value,
    taskSelect.value
  );
  runButton.disabled = busy || !support.supported;
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
  const support = experimentSupport(parameterId, modelSelect.value, taskSelect.value);
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
  const timingMetric = settings.objectiveMetric === "wall" ||
      settings.qualityGuardMetric === "wall"
    ? "WALL"
    : "TTFT";
  if (parameterId === "distractionHistoryBudget") {
    decisionRuleElement.textContent = "task quality + no truncation + repeatable TTFT gain";
  } else if (settings.objectiveMetric === "quality") {
    decisionRuleElement.textContent =
      `quality recovery + no truncation + no meaningful ${timingMetric} regression`;
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
  runButton.disabled = running || !support.supported;
  if (!support.supported) {
    statusElement.textContent = support.reason;
    statusElement.dataset.selectionError = "true";
  } else if (statusElement.dataset.selectionError) {
    statusElement.textContent = "idle";
    delete statusElement.dataset.selectionError;
  }
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

export async function runExperiment({
  blocks: blockOverride,
  warmups: warmupOverride,
  measuredRunsPerScenario,
  measuredRunsPerCase,
  candidatePolicyId = "",
  warmupEachPolicy = false,
  completeCounterbalance = true,
  orderOffset = 0,
  mode = "confirmatory",
  conversationScenarios: scenarioOverride = null,
} = {}) {
  const support = experimentSupport(
    parameterSelect.value,
    modelSelect.value,
    taskSelect.value
  );
  if (!support.supported) {
    statusElement.textContent = support.reason;
    return;
  }
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
  const policies = selectedPolicies(parameterId, candidatePolicyId);
  const experiment = experimentWorkload(parameterId);
  const conversationScenarios = scenarioOverride ?? experiment.conversationScenarios;
  const blocks = blockOverride ?? Number(blocksInput.value);
  const warmups = warmupOverride ?? Number(warmupsInput.value);
  const latencyTargetMs = Number(latencyTargetInput.value);
  const taskCaseCount = selectedTaskProfile().cases.length;
  const measuredRuns = measuredRunsPerCase == null
    ? measuredRunsPerScenario == null
      ? experiment.measuredRuns
      : measuredRunsPerScenario * conversationScenarios.length
    : measuredRunsPerCase * taskCaseCount * conversationScenarios.length;
  const warmupPerConfiguration = warmupEachPolicy || experiment.warmupPerConfiguration;
  const warmupInferences = warmupPerConfiguration
    ? policies.length * blocks * warmups
    : blocks * warmups;
  const totalInferences = policies.length * blocks * measuredRuns + warmupInferences;
  let completedInferences = 0;
  const results = [];
  const executions = [];

  try {
    if (completeCounterbalance && blocks % policies.length !== 0) {
      throw new Error(`block count must be a multiple of ${policies.length}`);
    }
    statusElement.textContent = "collecting device profile...";
    const deviceProfile = await getWebGpuProfile();
    const label = labelInput.value.trim() || makeDefaultLabel(deviceProfile);
    labelInput.value = label;

    for (let blockIndex = 0; blockIndex < blocks; blockIndex++) {
      const order = counterbalancedOrder(policies, blockIndex + orderOffset);
      log(`\n=== block ${blockIndex + 1}/${blocks}: ${order.map((policy) => policy.id).join(" -> ")} ===`);
      for (const [orderIndex, policy] of order.entries()) {
        const phase = `block ${blockIndex + 1}/${blocks}, ${policy.label}`;
        const policyWarmups = warmupPerConfiguration || orderIndex === 0 ? warmups : 0;
        const artifactId = policy.modelQuantization
          ? modelArtifactId(modelSelect.value, policy.modelQuantization)
          : "";
        if (policy.modelQuantization && !artifactId) {
          throw new Error(
            `${policy.modelQuantization} is unavailable for ${modelSelect.value}`
          );
        }
        let completedForPolicy = 0;
        const result = await runFixedModelPoliciesCore({
          modelConfigId: modelSelect.value,
          modelArtifactId: artifactId,
          taskProfileId: taskSelect.value,
          latencyTargetMs,
          policies: [policy],
          warmup: policyWarmups,
          runs: measuredRuns,
          randomizeOrder: false,
          longHistoryScenario: "long-history-required",
          measuredConversationScenarios: conversationScenarios,
          measuredScenarioOrder: mode === "rapid" ? "scenarios-first" : "cases-first",
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
        values: candidatePolicyId
          ? policies
            .filter((policy) => policy.baselineType !== "library_default")
            .map((policy) => policy.experimentValue)
          : settings.values,
        modelConfigId: modelSelect.value,
        taskProfileId: taskSelect.value,
        blocks,
        warmupsPerBlock: warmupPerConfiguration ? null : warmups,
        warmupsPerConfiguration: warmupPerConfiguration ? warmups : null,
        measuredRunsPerBlock: measuredRuns,
        measuredRunsPerCase: measuredRunsPerCase ?? null,
        taskCaseIds: selectedTaskProfile().cases.map((taskCase) => taskCase.id),
        candidatePolicyId: candidatePolicyId || null,
        mode,
        completeCounterbalance,
        measuredScenarioOrder: mode === "rapid" ? "scenarios-first" : "cases-first",
        conversationScenarios,
        executionOrders: Array.from({ length: blocks }, (_, index) =>
          counterbalancedOrder(policies, index + orderOffset).map((policy) => policy.id)
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
  for (const task of MANUAL_TASK_PROFILES) {
    taskSelect.add(new Option(task.label, task.id));
  }
  taskSelect.value = "chat-summarize";
  const scopeGroups = [
    ["webllm-setting", "Direct WebLLM settings"],
    ["webllm-operation", "WebLLM operations / artifacts"],
    ["application", "App-level (not direct WebLLM settings)"],
  ];
  for (const [scope, label] of scopeGroups) {
    const group = document.createElement("optgroup");
    group.label = label;
    for (const definition of EXPERIMENT_DEFINITIONS.filter((item) => item.scope === scope)) {
      group.appendChild(new Option(definition.label, definition.id));
    }
    parameterSelect.appendChild(group);
  }
  for (const control of [
    parameterSelect,
    modelSelect,
    taskSelect,
    blocksInput,
    warmupsInput,
    selectorBudgetInput,
  ]) {
    control.addEventListener("change", updateDefinition);
  }
  updateDefinition();
}

runButton.addEventListener("click", () => runExperiment());
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
