import { CONTROLLED_GENERATION } from "./tuning-policies.js";
import {
  matchedEqualityRate,
  matchedRows,
  median,
  quantile,
} from "../shared/benchmark-utils.js";

export const EXPERIMENT_DEFINITIONS = [
  {
    id: "historyTokenBudget",
    label: "History budget",
    values: [128, 256, 384, 512],
    controlledValue: 512,
    objectiveMetric: "ttft",
    workload: { measuredRuns: 3, scenarios: ["long-history-required"] },
  },
  {
    id: "relevanceHistoryBudget",
    label: "Relevance history budget",
    values: [64, 96, 128, 192],
    controlledValue: 128,
    objectiveMetric: "ttft",
    policyParameter: "historyTokenBudget",
    baseOverrides: { historySelectionPolicy: "relevance" },
    workload: { measuredRuns: 3, scenarios: ["long-history-required"] },
  },
  {
    id: "distractionHistoryBudget",
    label: "Distraction history",
    values: [128, 256, 384, 512],
    controlledValue: 512,
    objectiveMetric: "ttft",
    policyParameter: "historyTokenBudget",
    workload: { measuredRuns: 3, scenarios: ["long-history-distraction"] },
  },
  {
    id: "maxTokens",
    label: "Output limit",
    values: [128, 192, 256],
    controlledValue: 256,
    objectiveMetric: "wall",
    workload: { measuredRuns: 6, scenarios: ["single-turn", "long-history-required"] },
  },
  {
    id: "contextWindowSize",
    label: "Context window",
    values: [1024, 2048, 4096],
    controlledValue: 4096,
    objectiveMetric: "ttft",
    workload: { measuredRuns: 3, scenarios: ["long-history-required"], warmupEach: true },
  },
  {
    id: "promptTemplateId",
    label: "Prompt format",
    values: ["full", "compact"],
    controlledValue: "full",
    objectiveMetric: "ttft",
    workload: { measuredRuns: 6, scenarios: ["single-turn", "long-history-required"] },
  },
  {
    id: "historySelectionPolicy",
    label: "History selector",
    values: ["tail", "relevance"],
    controlledValue: "tail",
    objectiveMetric: "quality",
    baseOverrides: { historyTokenBudget: 320 },
    workload: { measuredRuns: 3, scenarios: ["long-history-required"] },
  },
  {
    id: "historyConfiguration",
    label: "History configuration",
    values: ["512-tail", "128-relevance"],
    controlledValue: "512-tail",
    objectiveMetric: "ttft",
    configurations: {
      "512-tail": { historyTokenBudget: 512, historySelectionPolicy: "tail" },
      "128-relevance": { historyTokenBudget: 128, historySelectionPolicy: "relevance" },
    },
    workload: { measuredRuns: 3, scenarios: ["long-history-required"] },
  },
  {
    id: "adaptiveHistoryPolicy",
    label: "Adaptive history policy",
    values: ["512-tail", "96-relevance"],
    controlledValue: "512-tail",
    objectiveMetric: "ttft",
    configurations: {
      "512-tail": { historyTokenBudget: 512, historySelectionPolicy: "tail" },
      "96-relevance": { historyTokenBudget: 96, historySelectionPolicy: "relevance" },
    },
    workload: {
      measuredRuns: 6,
      scenarios: ["long-history-distraction", "long-history-required"],
    },
  },
  {
    id: "conversationCacheMode",
    label: "Conversation prefix cache",
    values: ["full-prefill", "prefix-reuse"],
    controlledValue: "full-prefill",
    objectiveMetric: "ttft",
    workload: { measuredRuns: 3, scenarios: ["long-history-distraction"] },
    requirePrefixEquality: true,
  },
  {
    id: "penaltyProcessingMode",
    label: "Zero-penalty processing",
    values: ["explicit-zero", "bypass-zero"],
    controlledValue: "explicit-zero",
    objectiveMetric: "wall",
    workload: { measuredRuns: 3, scenarios: ["single-turn"], warmupEach: true },
    requireOutputEquality: true,
  },
  {
    id: "responseDeliveryMode",
    label: "Response delivery",
    values: ["streaming", "non-streaming"],
    controlledValue: "streaming",
    objectiveMetric: "wall",
    workload: { measuredRuns: 3, scenarios: ["single-turn"] },
    requireOutputEquality: true,
  },
  {
    id: "kvCacheMode",
    label: "KV cache mode at 4096 tokens",
    values: ["context", "sliding"],
    controlledValue: "context",
    objectiveMetric: "wall",
    baseOverrides: {
      historyTokenBudget: 3000,
      conversationCacheMode: "prefix-reuse",
    },
    workload: { measuredRuns: 1, scenarios: ["long-history-distraction"], warmupEach: true },
    requireOutputEquality: true,
    requirePrefixEquality: true,
  },
  {
    id: "slidingWindowSize",
    label: "Sliding-window size",
    values: [4096, 3072],
    controlledValue: 4096,
    objectiveMetric: "wall",
    baseOverrides: {
      historyTokenBudget: 3000,
      conversationCacheMode: "prefix-reuse",
      kvCacheMode: "sliding",
    },
    workload: { measuredRuns: 1, scenarios: ["long-history-distraction"], warmupEach: true },
    requireOutputEquality: true,
    requirePrefixEquality: true,
  },
  {
    id: "engineThreadMode",
    label: "Engine thread",
    values: ["main", "worker"],
    controlledValue: "main",
    objectiveMetric: "wall",
    workload: { measuredRuns: 3, scenarios: ["single-turn"], warmupEach: true },
    requireOutputEquality: true,
  },
];

const BASE_SETTINGS = {
  contextWindowSize: 4096,
  historyTokenBudget: 512,
  maxTokens: 192,
  promptTemplateId: "full",
  historySelectionPolicy: "tail",
  useLibraryGenerationDefaults: false,
  outputTokenTier: "custom",
  ...CONTROLLED_GENERATION,
};

function valueId(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

export function getExperimentDefinition(id, overrides = {}) {
  const definition = EXPERIMENT_DEFINITIONS.find((item) => item.id === id)
    ?? EXPERIMENT_DEFINITIONS[0];
  return { ...definition, ...overrides };
}

export function buildExperimentPolicies(
  parameterId,
  baseOverrides = {},
  definitionOverrides = {}
) {
  const definition = getExperimentDefinition(parameterId, definitionOverrides);
  const baseSettings = { ...BASE_SETTINGS, ...definition.baseOverrides, ...baseOverrides };
  const libraryDefault = {
    id: "library-default",
    label: "Untouched WebLLM reference",
    baselineType: "library_default",
    experimentValue: "library-default",
    contextWindowSize: null,
    historyTokenBudget: baseSettings.historyTokenBudget,
    historySelectionPolicy: baseSettings.historySelectionPolicy,
    maxTokens: null,
    promptTemplateId: BASE_SETTINGS.promptTemplateId,
    useLibraryGenerationDefaults: true,
    outputTokenTier: "library-default",
  };
  const controlled = definition.values.map((value) => {
    const isBaseline = value === definition.controlledValue;
    return {
      ...baseSettings,
      id: isBaseline ? "controlled-baseline" : `${parameterId}-${valueId(value)}`,
      label: isBaseline ? `Controlled baseline (${value})` : `${definition.label}: ${value}`,
      baselineType: isBaseline ? "controlled" : "",
      experimentValue: value,
      ...(definition.configurations?.[value] ?? {
        [definition.policyParameter ?? parameterId]: value,
      }),
    };
  });
  return [libraryDefault, ...controlled];
}

export function counterbalancedOrder(policies, blockIndex) {
  if (!policies.length) return [];
  const rotation = blockIndex % policies.length;
  return policies.slice(rotation).concat(policies.slice(0, rotation));
}

function scenarioSummary(rows, scenario) {
  const selected = rows.filter((row) => row.conversationScenario === scenario);
  return {
    runCount: selected.length,
    qualityPassRate: selected.length
      ? selected.filter((row) => row.qualityPassed).length / selected.length
      : 0,
    medianTtftMs: median(selected.map((row) => row.timeToFirstTokenMs)),
    medianWallMs: median(selected.map((row) => row.wallMs)),
  };
}

function matchedRepeatabilityIqrPct(rows, metric) {
  const groups = new Map();
  for (const row of rows.filter((item) => item.success)) {
    const key = `${row.taskCaseId}|${row.conversationScenario}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row[metric]);
  }
  const ratios = [...groups.values()].map((values) => {
    const center = median(values);
    const q1 = quantile(values, 0.25);
    const q3 = quantile(values, 0.75);
    return Number.isFinite(center) && center > 0 ? (q3 - q1) / center : null;
  });
  return median(ratios);
}

function matchedDelta(rows, controlledRows, metric) {
  const deltas = matchedRows(rows, controlledRows)
    .map(({ row, reference }) => {
      if (!Number.isFinite(reference[metric]) || reference[metric] <= 0) return null;
      return (row[metric] / reference[metric] - 1) * 100;
    })
    .filter(Number.isFinite);
  return {
    pairCount: deltas.length,
    medianDeltaPct: median(deltas),
    fasterPairRate: deltas.length
      ? deltas.filter((delta) => delta < 0).length / deltas.length
      : 0,
  };
}

function summarizeConfiguration(policy, rows) {
  const successful = rows.filter((row) => row.success);
  const ttftValues = successful.map((row) => row.timeToFirstTokenMs);
  const medianTtftMs = median(ttftValues);
  const q1 = quantile(ttftValues, 0.25);
  const q3 = quantile(ttftValues, 0.75);
  const blockMedians = [...new Set(rows.map((row) => row.experimentBlock))]
    .map((block) => median(
      successful
        .filter((row) => row.experimentBlock === block)
        .map((row) => row.timeToFirstTokenMs)
    ));
  return {
    policyId: policy.id,
    policyLabel: policy.label,
    baselineType: policy.baselineType || "",
    experimentValue: policy.experimentValue,
    runCount: rows.length,
    successRate: rows.length ? successful.length / rows.length : 0,
    qualityPassRate: successful.length
      ? successful.filter((row) => row.qualityPassed).length / successful.length
      : 0,
    contentQualityPassRate: successful.length
      ? successful.filter((row) => row.contentQualityPassed ?? row.qualityPassed).length /
      successful.length
      : 0,
    outputCompletionRate: successful.length
      ? successful.filter((row) => row.outputComplete ?? !row.outputTruncated).length /
      successful.length
      : 0,
    cleanRunCount: scenarioSummary(rows, "single-turn").runCount,
    cleanQualityPassRate: scenarioSummary(rows, "single-turn").qualityPassRate,
    requiredHistoryRunCount: scenarioSummary(rows, "long-history-required").runCount,
    requiredHistoryQualityPassRate:
      scenarioSummary(rows, "long-history-required").qualityPassRate,
    medianQualityScore: median(successful.map((row) => row.qualityScore)),
    medianPromptTokens: median(successful.map((row) => row.promptTokens)),
    medianCompletionTokens: median(successful.map((row) => row.completionTokens)),
    medianTtftMs,
    medianWallMs: median(successful.map((row) => row.wallMs)),
    medianPrefillTokS: median(successful.map((row) => row.prefillTokS)),
    medianDecodeTokS: median(successful.map((row) => row.decodeTokS)),
    ttftIqrMs: Number.isFinite(q1) && Number.isFinite(q3) ? q3 - q1 : null,
    ttftIqrPct: Number.isFinite(medianTtftMs) && medianTtftMs > 0 ? (q3 - q1) / medianTtftMs : null,
    matchedTtftRepeatabilityIqrPct:
      matchedRepeatabilityIqrPct(successful, "timeToFirstTokenMs"),
    matchedWallRepeatabilityIqrPct: matchedRepeatabilityIqrPct(successful, "wallMs"),
    blockMedianTtftMs: blockMedians,
    anyTruncated: successful.some((row) => row.outputTruncated),
    failureClasses: [...new Set(rows.filter((row) => !row.success).map((row) => row.failureClass))],
  };
}

export function analyzeControlledExperiment(policies, rows, parameterId) {
  const definition = getExperimentDefinition(
    parameterId ?? rows.find((row) => row.experimentParameter)?.experimentParameter
  );
  const objectiveMetric = definition.objectiveMetric;
  const summaries = policies.map((policy) =>
    summarizeConfiguration(policy, rows.filter((row) => row.policyId === policy.id))
  );
  const controlled = summaries.find((summary) => summary.baselineType === "controlled");
  if (!controlled) throw new Error("controlled baseline is missing");
  const controlledRows = rows.filter((row) => row.policyId === controlled.policyId);
  const repeatabilityField = objectiveMetric === "wall"
    ? "matchedWallRepeatabilityIqrPct"
    : "matchedTtftRepeatabilityIqrPct";
  const objectiveValueField = objectiveMetric === "wall" ? "medianWallMs" : "medianTtftMs";
  const minimumImprovement = Math.max(0.1, controlled[repeatabilityField] ?? 0);

  for (const summary of summaries) {
    const policyRows = rows.filter((row) => row.policyId === summary.policyId);
    const matchedTtft = matchedDelta(policyRows, controlledRows, "timeToFirstTokenMs");
    const matchedWall = matchedDelta(policyRows, controlledRows, "wallMs");
    summary.ttftDeltaPct = Number.isFinite(summary.medianTtftMs)
      ? (summary.medianTtftMs / controlled.medianTtftMs - 1) * 100
      : null;
    summary.wallDeltaPct = Number.isFinite(summary.medianWallMs)
      ? (summary.medianWallMs / controlled.medianWallMs - 1) * 100
      : null;
    summary.qualityPassDeltaPoints =
      (summary.qualityPassRate - controlled.qualityPassRate) * 100;
    summary.matchedPairCount = matchedTtft.pairCount;
    summary.matchedTtftDeltaPct = matchedTtft.medianDeltaPct;
    summary.matchedWallDeltaPct = matchedWall.medianDeltaPct;
    summary.matchedPrefixEqualityRate = matchedEqualityRate(
      policyRows,
      controlledRows,
      "conversationPrefixOutput"
    );
    summary.matchedOutputEqualityRate = matchedEqualityRate(
      policyRows,
      controlledRows,
      "output"
    );
    summary.matchedFasterPairRate = objectiveMetric === "wall"
      ? matchedWall.fasterPairRate
      : matchedTtft.fasterPairRate;
    if (objectiveMetric === "quality") {
      summary.matchedObjectiveDeltaPct = summary.qualityPassDeltaPoints;
    } else if (objectiveMetric === "wall") {
      summary.matchedObjectiveDeltaPct = matchedWall.medianDeltaPct;
    } else {
      summary.matchedObjectiveDeltaPct = matchedTtft.medianDeltaPct;
    }
    summary.matchedRepeatabilityIqrPct = summary[repeatabilityField];
    summary.minimumRequiredImprovementPct = Math.max(
      minimumImprovement,
      summary.matchedRepeatabilityIqrPct ?? 0
    ) * 100;
    const qualityEligible =
      summary.qualityPassRate >= Math.max(0.9, controlled.qualityPassRate) &&
      (!summary.cleanRunCount || summary.cleanQualityPassRate >= 0.9) &&
      (!summary.requiredHistoryRunCount || summary.requiredHistoryQualityPassRate >= 0.9) &&
      summary.medianQualityScore >= controlled.medianQualityScore - 0.1;
    let speedEligible = summary.baselineType === "controlled";
    if (!speedEligible && objectiveMetric === "quality") {
      speedEligible =
        Number.isFinite(summary.matchedTtftDeltaPct) &&
        summary.matchedTtftDeltaPct <= summary.minimumRequiredImprovementPct;
    } else if (!speedEligible) {
      speedEligible =
        Number.isFinite(summary.matchedObjectiveDeltaPct) &&
        summary.matchedObjectiveDeltaPct <= -summary.minimumRequiredImprovementPct &&
        summary.matchedFasterPairRate >= 2 / 3;
    }
    summary.eligible =
      summary.baselineType !== "library_default" &&
      summary.successRate === 1 &&
      !summary.anyTruncated &&
      (!definition.requireOutputEquality || summary.matchedOutputEqualityRate === 1) &&
      (!definition.requirePrefixEquality || summary.matchedPrefixEqualityRate === 1) &&
      qualityEligible &&
      speedEligible;
    if (summary.baselineType === "library_default") {
      summary.status = "reference";
    } else if (summary.baselineType === "controlled") {
      summary.status = "control";
    } else {
      summary.status = summary.eligible ? "candidate" : "rejected";
    }
  }

  const winner = summaries
    .filter((summary) => summary.status === "candidate")
    .sort((a, b) => objectiveMetric === "quality"
      ? b.qualityPassRate - a.qualityPassRate || a.medianTtftMs - b.medianTtftMs
      : a.matchedObjectiveDeltaPct - b.matchedObjectiveDeltaPct ||
      a[objectiveValueField] - b[objectiveValueField]
    )[0];
  return {
    summaries,
    controlledPolicyId: controlled.policyId,
    recommendedPolicyId: winner?.policyId ?? "",
    criteria: {
      objectiveMetric,
      minimumSuccessRate: 1,
      minimumQualityPassRate: 0.9,
      minimumQualityPassRatePerScenario: 0.9,
      noTruncation: true,
      maximumQualityScoreLoss: 0.1,
      minimumObjectiveImprovementPct: minimumImprovement * 100,
      candidateGainMustExceedOwnRepeatability: true,
      minimumMatchedFasterPairRate: 2 / 3,
      requireIdenticalOutput: Boolean(definition.requireOutputEquality),
      requireIdenticalConversationPrefix: Boolean(definition.requirePrefixEquality),
    },
  };
}

export function controlledExperimentSettings(
  parameterId,
  baseOverrides = {},
  definitionOverrides = {}
) {
  const definition = getExperimentDefinition(parameterId, definitionOverrides);
  const baseSettings = { ...BASE_SETTINGS, ...definition.baseOverrides, ...baseOverrides };
  const configurationValues = Object.values(definition.configurations ?? {});
  const variedKeys = new Set(
    Object.keys(configurationValues[0] ?? {}).filter((key) =>
      configurationValues.some((configuration) =>
        configuration[key] !== configurationValues[0][key]
      )
    )
  );
  variedKeys.add(definition.policyParameter ?? parameterId);
  return {
    variedParameter: definition.id,
    values: definition.values,
    controlledValue: definition.controlledValue,
    objectiveMetric: definition.objectiveMetric,
    lockedSettings: Object.fromEntries(
      Object.entries(baseSettings)
        .filter(([key]) => !variedKeys.has(key) && key !== "outputTokenTier")
    ),
  };
}
