import { getWebGpuProfile } from "./device-profile.js";
import {
  FIXED_MODELS,
  parseQuantization,
  resolveModelArtifact,
} from "./models.js";
import { evaluateTaskOutput } from "./quality-evaluator.js";
import { chooseRecommendedPolicy } from "./recommendation.js";
import { getTaskCase, getTaskProfile } from "./task-profiles.js";
import { TUNING_POLICIES } from "./tuning-policies.js";
import { selectRelevantHistory } from "./history-selection.js";
import { matchedEqualityRate, median } from "../shared/benchmark-utils.js";
import { prebuiltAppConfig } from "https://esm.run/@mlc-ai/web-llm@0.2.79";
import {
  classifyWebLlmError,
  completeWithWebLlm,
  loadWebLlmModel,
  readWebLlmStats,
  resetWebLlmChat,
  unloadWebLlm,
} from "./webllm-runtime.js";

const WEBLLM_VERSION = "0.2.79";
const DEFAULT_TIMEOUT_MS = 45_000;
const MIN_PASS_RATE = 0.9;
const MIN_SCREENING_TTFT_IMPROVEMENT = 0.1;
const QUALITY_TOLERANCE = 0.1;
const HISTORY_FIXTURE = [
  "The user is comparing browser inference behavior across several computers.",
  "Startup delay and time to first token are measured separately from decode speed.",
  "Each result records the browser, operating system, GPU identity, and WebGPU limits.",
  "The model and quantization stay fixed while exposed runtime settings change.",
  "Answer quality must remain comparable to the controlled baseline.",
  "Warmup runs are excluded from the reported medians.",
  "Failures and timeouts remain in the exported dataset.",
  "The final selector should learn which policy fits a device profile.",
].join(" ");

export function getAvailableFixedModels() {
  const available = new Set(
    prebuiltAppConfig.model_list.map((record) => record.model_id)
  );
  return FIXED_MODELS.filter((model) => available.has(model.model));
}

function estimateTokens(text) {
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return words ? Math.ceil(words * 1.35) : 0;
}

function isLongHistoryScenario(conversationScenario) {
  return conversationScenario.startsWith("long-history");
}

function repeatFixtureWords(wordCount) {
  const sourceWords = HISTORY_FIXTURE.split(/\s+/);
  return Array.from({ length: wordCount }, (_, index) => sourceWords[index % sourceWords.length]);
}

function buildHistoryText(tokenBudget, taskCase, conversationScenario, selectionPolicy) {
  if (!tokenBudget) return "";
  const wordBudget = Math.max(1, Math.floor(tokenBudget / 1.35));
  if (conversationScenario === "long-history-required" && taskCase.historyRequirement) {
    const factWords = taskCase.historyRequirement.factText.split(/\s+/);
    const trailingWords = repeatFixtureWords(
      Math.max(1, Math.floor(taskCase.historyRequirement.distanceTokens / 1.35))
    );
    const sourceWords = [...factWords, ...trailingWords];
    if (selectionPolicy === "relevance") {
      return selectRelevantHistory(sourceWords, taskCase.historyRequirement.prompt, wordBudget);
    }
    return sourceWords.slice(-wordBudget).join(" ");
  }
  return repeatFixtureWords(wordBudget).join(" ");
}

function materializeTaskCase(taskCase, conversationScenario) {
  if (conversationScenario !== "long-history-required") return taskCase;
  const requirement = taskCase.historyRequirement;
  if (!requirement) throw new Error(`task case ${taskCase.id} has no history requirement`);
  return {
    id: `${taskCase.id}-history-required`,
    prompt: requirement.prompt,
    quality: requirement.quality,
    historyRequirement: requirement,
  };
}

function buildPolicyMessages(taskProfile, taskCase, policy, conversationScenario) {
  const activeHistoryBudget = isLongHistoryScenario(conversationScenario)
    ? policy.historyTokenBudget
    : 0;
  const history = buildHistoryText(
    activeHistoryBudget,
    taskCase,
    conversationScenario,
    policy.historySelectionPolicy
  );
  const compact = policy.promptTemplateId === "compact";
  const systemPrompt = compact
    ? taskProfile.compactSystemPrompt || taskProfile.systemPrompt
    : taskProfile.systemPrompt;
  const messages = [];
  if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
  if (history) {
    const prefix = compact
      ? "Prior context, use only if relevant: "
      : "Use this earlier conversation context only when it is relevant to the next request: ";
    messages.push({ role: "user", content: prefix + history });
    if (!compact) {
      messages.push({
        role: "assistant",
        content: "Understood. I will use relevant context while prioritizing the next request.",
      });
    }
  }
  messages.push({ role: "user", content: taskCase.prompt });
  return {
    messages,
    estimatedSystemTokens: estimateTokens(systemPrompt || ""),
    estimatedHistoryTokens: estimateTokens(history),
    estimatedUserTokens: estimateTokens(taskCase.prompt),
    estimatedInputTokens: messages.reduce((sum, message) => sum + estimateTokens(message.content), 0),
  };
}

function validatePolicy(policy, estimatedInputTokens) {
  if (policy.contextWindowSize != null && policy.contextWindowSize < 256) {
    throw new Error("context window must be at least 256");
  }
  if (policy.historyTokenBudget < 0 || (policy.maxTokens != null && policy.maxTokens < 1)) {
    throw new Error("token budgets must be positive");
  }
  if (
    policy.contextWindowSize != null &&
    policy.maxTokens != null &&
    estimatedInputTokens + policy.maxTokens > policy.contextWindowSize
  ) {
    throw new Error(
      `estimated input (${estimatedInputTokens}) plus output (${policy.maxTokens}) exceeds context window (${policy.contextWindowSize})`
    );
  }
}

function modelRecord(modelId) {
  return prebuiltAppConfig.model_list.find((record) => record.model_id === modelId);
}

function effectiveContextWindow(modelId, policy) {
  return policy.contextWindowSize ?? modelRecord(modelId)?.overrides?.context_window_size ?? null;
}

function basePolicyFields({
  model,
  taskProfile,
  taskCase,
  policy,
  executionIndex,
  conversationScenario,
}) {
  return {
    policyId: policy.id,
    policyLabel: policy.label,
    baselineType: policy.baselineType || "",
    executionIndex,
    modelFamily: model.family,
    model: model.model,
    quantization: parseQuantization(model.model),
    taskProfileId: taskProfile.id,
    taskType: taskProfile.taskType,
    taskCaseId: taskCase?.id ?? "",
    conversationScenario: conversationScenario ?? "",
    requestedContextWindowSize: policy.contextWindowSize,
    effectiveContextWindowSize: effectiveContextWindow(model.model, policy),
    configuredHistoryTokenBudget: policy.historyTokenBudget,
    historySelectionPolicy: policy.historySelectionPolicy || "tail",
    historyTokenBudget: isLongHistoryScenario(conversationScenario)
      ? policy.historyTokenBudget
      : 0,
    outputTokenTier: policy.outputTokenTier || "custom",
    maxTokens: policy.maxTokens,
    promptTemplateId: policy.promptTemplateId,
    useLibraryGenerationDefaults: policy.useLibraryGenerationDefaults,
    requestedTemperature: policy.useLibraryGenerationDefaults ? null : policy.temperature,
    requestedTopP: policy.useLibraryGenerationDefaults ? null : policy.topP,
    requestedRepetitionPenalty: policy.useLibraryGenerationDefaults ? null : policy.repetitionPenalty,
    requestedFrequencyPenalty: policy.useLibraryGenerationDefaults ? null : policy.frequencyPenalty,
    requestedPresencePenalty: policy.useLibraryGenerationDefaults ? null : policy.presencePenalty,
    requestedSeed: policy.useLibraryGenerationDefaults ? null : policy.seed,
    conversationCacheMode: policy.conversationCacheMode ?? "",
    penaltyProcessingMode: policy.penaltyProcessingMode ?? "",
    responseDeliveryMode: policy.useLibraryGenerationDefaults
      ? "library-default"
      : policy.responseDeliveryMode ?? "streaming",
    kvCacheMode: policy.kvCacheMode ?? "context",
    slidingWindowSize: policy.slidingWindowSize ?? null,
    engineThreadMode: policy.engineThreadMode ?? "main",
    topLogprobs: policy.topLogprobs ?? null,
  };
}

async function runInference({
  model,
  taskProfile,
  taskCase,
  policy,
  built,
  conversationScenario,
  runKind,
  runIndex,
  executionIndex,
  loadInfo,
  latencyTargetMs,
  timeoutMs,
}) {
  await resetWebLlmChat();
  let measuredMessages = built.messages;
  let conversationPrefixOutput = "";
  if (policy.conversationCacheMode) {
    const prefix = await completeWithWebLlm(
      built.messages,
      policy,
      timeoutMs
    );
    conversationPrefixOutput = prefix.output;
    measuredMessages = [
      ...built.messages,
      { role: "assistant", content: conversationPrefixOutput },
      {
        role: "user",
        content: policy.kvCacheMode
          ? "Repeat your immediately previous answer verbatim. Preserve every bullet on its own line; do not add, remove, or merge anything."
          : "Give the requested answer again. Preserve every required fact and the requested format.",
      },
    ];
    if (policy.conversationCacheMode === "full-prefill") {
      await resetWebLlmChat();
    }
  }
  const start = performance.now();
  let firstTokenAt = null;
  const completion = await completeWithWebLlm(
    measuredMessages,
    policy,
    timeoutMs,
    () => {
      if (firstTokenAt == null) firstTokenAt = performance.now();
    }
  );
  const { output, usage, finishReason } = completion;
  if (firstTokenAt == null && Number.isFinite(usage.extra?.time_to_first_token_s)) {
    firstTokenAt = start + usage.extra.time_to_first_token_s * 1000;
  }
  const wallMs = performance.now() - start;
  const stats = await readWebLlmStats();
  const quality = await evaluateTaskOutput(taskCase, output);
  const outputLimitReached = finishReason === "length";
  const latencyOnly = taskProfile.taskType === "latency-only";
  const outputTruncated = outputLimitReached && !latencyOnly;
  const outputComplete = finishReason === "stop" || (latencyOnly && outputLimitReached);
  const promptTokens = usage.prompt_tokens ?? null;
  const completionTokens = usage.completion_tokens ?? null;
  const tokenMetrics = { ...built };
  delete tokenMetrics.messages;
  return {
    ...basePolicyFields({
      model,
      taskProfile,
      taskCase,
      policy,
      executionIndex,
      conversationScenario,
    }),
    ...tokenMetrics,
    runKind,
    runIndex,
    repetitionBlock: runKind === "measured"
      ? Math.floor((runIndex - 1) / (taskProfile.cases.length * 2)) + 1
      : 0,
    success: true,
    failureClass: "",
    failureKeep: "",
    errorStage: "",
    errorType: "",
    timeout: false,
    policyLoadMs: loadInfo.loadMs,
    engineCacheState: loadInfo.engineCacheState,
    promptTokens,
    completionTokens,
    timeToFirstTokenMs: firstTokenAt == null ? null : firstTokenAt - start,
    reportedTimeToFirstTokenMs: Number.isFinite(usage.extra?.time_to_first_token_s)
      ? usage.extra.time_to_first_token_s * 1000
      : null,
    wallMs,
    ...stats,
    estimatedPrefillMs: promptTokens && stats.prefillTokS ? promptTokens / stats.prefillTokS * 1000 : null,
    decodeMsPerToken: stats.decodeTokS ? 1000 / stats.decodeTokS : null,
    latencyTargetMs,
    latencyTargetMet: wallMs <= latencyTargetMs,
    finishReason,
    outputLimitReached,
    outputTruncated,
    outputComplete,
    outputChars: output.length,
    output,
    conversationPrefixChars: conversationPrefixOutput.length,
    conversationPrefixOutput,
    qualityScore: quality.score,
    contentQualityScore: quality.score,
    contentQualityPassed: quality.passed,
    qualityPassed: quality.passed && outputComplete,
    qualityDetail: outputComplete
      ? quality.detail
      : `${quality.detail}|incomplete:${finishReason || "unknown"}`,
  };
}

function failureRow({
  model,
  taskProfile,
  taskCase,
  policy,
  executionIndex,
  conversationScenario = "",
  runKind = "measured",
  runIndex = 0,
  loadInfo = null,
  error,
  stage,
  latencyTargetMs,
}) {
  const classification = classifyWebLlmError(error, stage);
  return {
    ...basePolicyFields({
      model,
      taskProfile,
      taskCase,
      policy,
      executionIndex,
      conversationScenario,
    }),
    runKind,
    runIndex,
    success: false,
    ...classification,
    errorType: error?.message ?? String(error),
    timeout: classification.failureClass === "timeout",
    policyLoadMs: loadInfo?.loadMs ?? null,
    engineCacheState: loadInfo?.engineCacheState ?? "",
    promptTokens: null,
    completionTokens: null,
    timeToFirstTokenMs: null,
    wallMs: null,
    prefillTokS: null,
    decodeTokS: null,
    latencyTargetMs,
    latencyTargetMet: false,
    outputLimitReached: false,
    outputTruncated: false,
    outputComplete: false,
    qualityScore: 0,
    contentQualityScore: 0,
    contentQualityPassed: false,
    qualityPassed: false,
    qualityDetail: "not_evaluated",
    output: "",
  };
}

function scenarioMetrics(rows, scenario) {
  const successful = rows.filter(
    (row) => row.success && row.conversationScenario === scenario
  );
  return {
    runCount: successful.length,
    qualityPassRate: successful.length
      ? successful.filter((row) => row.qualityPassed).length / successful.length
      : 0,
    contentQualityPassRate: successful.length
      ? successful.filter((row) => row.contentQualityPassed).length / successful.length
      : 0,
    outputCompletionRate: successful.length
      ? successful.filter((row) => row.outputComplete).length / successful.length
      : 0,
    medianTtftMs: median(successful.map((row) => row.timeToFirstTokenMs)),
    medianWallMs: median(successful.map((row) => row.wallMs)),
  };
}

function scenarioMetricKey(scenario) {
  return scenario.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}


function summarizePolicies(results, policies) {
  const summaries = policies.map((policy) => {
    const rows = results.filter((row) => row.policyId === policy.id && row.runKind === "measured");
    const successful = rows.filter((row) => row.success);
    return {
      policyId: policy.id,
      policyLabel: policy.label,
      baselineType: policy.baselineType || "",
      requestedContextWindowSize: policy.contextWindowSize,
      effectiveContextWindowSize: successful[0]?.effectiveContextWindowSize ?? null,
      historyTokenBudget: policy.historyTokenBudget,
      outputTokenTier: policy.outputTokenTier || "custom",
      maxTokens: policy.maxTokens,
      promptTemplateId: policy.promptTemplateId,
      conversationCacheMode: policy.conversationCacheMode ?? "",
      runCount: rows.length,
      successRate: rows.length ? successful.length / rows.length : 0,
      qualityPassRate: successful.length
        ? successful.filter((row) => row.qualityPassed).length / successful.length
        : 0,
      contentQualityPassRate: successful.length
        ? successful.filter((row) => row.contentQualityPassed).length / successful.length
        : 0,
      outputCompletionRate: successful.length
        ? successful.filter((row) => row.outputComplete).length / successful.length
        : 0,
      medianQualityScore: median(successful.map((row) => row.qualityScore)),
      medianPromptTokens: median(successful.map((row) => row.promptTokens)),
      medianCompletionTokens: median(successful.map((row) => row.completionTokens)),
      medianTtftMs: median(successful.map((row) => row.timeToFirstTokenMs)),
      medianWallMs: median(successful.map((row) => row.wallMs)),
      medianPrefillTokS: median(successful.map((row) => row.prefillTokS)),
      medianDecodeTokS: median(successful.map((row) => row.decodeTokS)),
      latencyTargetHitRate: successful.length
        ? successful.filter((row) => row.latencyTargetMet).length / successful.length
        : 0,
      anyTruncated: successful.some((row) => row.outputTruncated),
      scenarioMetrics: Object.fromEntries(
        [...new Set(successful.map((row) => row.conversationScenario))]
          .map((scenario) => [scenarioMetricKey(scenario), scenarioMetrics(rows, scenario)])
      ),
      failureClasses: [...new Set(rows.filter((row) => !row.success).map((row) => row.failureClass))],
    };
  });
  const controlledQuality = summaries.find(
    (summary) => summary.baselineType === "controlled"
  )?.medianQualityScore;
  const controlledPolicy = policies.find((policy) => policy.baselineType === "controlled");
  const controlledRows = results.filter((row) =>
    row.policyId === controlledPolicy?.id && row.runKind === "measured"
  );
  for (const summary of summaries) {
    const policyRows = results.filter((row) =>
      row.policyId === summary.policyId && row.runKind === "measured"
    );
    summary.qualityDeltaVsControlled =
      Number.isFinite(controlledQuality) && Number.isFinite(summary.medianQualityScore)
        ? summary.medianQualityScore - controlledQuality
        : null;
    summary.qualityRetainedVsControlled =
      Number.isFinite(summary.qualityDeltaVsControlled) &&
      summary.qualityDeltaVsControlled >= -QUALITY_TOLERANCE;
    summary.matchedOutputEqualityRate = matchedEqualityRate(
      policyRows,
      controlledRows,
      "output"
    );
    summary.matchedPrefixEqualityRate = matchedEqualityRate(
      policyRows,
      controlledRows,
      "conversationPrefixOutput"
    );
  }
  return summaries;
}

function seededShuffle(items, seed) {
  let state = seed >>> 0;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const shuffled = [...items];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

function orderPolicies(policies, randomizeOrder, seed) {
  if (!randomizeOrder) return [...policies];
  const groups = new Map();
  for (const policy of policies) {
    const key = policy.contextWindowSize == null ? "library-default" : String(policy.contextWindowSize);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(policy);
  }
  const groupList = [...groups.values()];
  const offset = (seed >>> 0) % groupList.length;
  const rotated = groupList.slice(offset).concat(groupList.slice(0, offset));
  return rotated.flatMap((group, index) =>
    seededShuffle(group, (seed ^ Math.imul(index + 1, 0x9e3779b1)) >>> 0)
  );
}

function resolvePolicies(taskProfile, policies) {
  return policies.map((policy) => {
    const tier = policy.outputTokenTier;
    if (!tier || !taskProfile.outputTokenLimits?.[tier]) return { ...policy };
    return { ...policy, maxTokens: taskProfile.outputTokenLimits[tier] };
  });
}

function runVariant({
  taskProfile,
  runIndex,
  runKind,
  longHistoryScenario,
  measuredConversationScenarios,
  measuredScenarioOrder,
}) {
  if (runKind === "warmup") {
    return {
      taskCase: getTaskCase(taskProfile, runIndex - 1),
      conversationScenario: "single-turn",
    };
  }
  const caseCount = taskProfile.cases.length;
  const scenarios = measuredConversationScenarios?.length
    ? measuredConversationScenarios
    : ["single-turn", longHistoryScenario];
  const scenarioIndex = measuredScenarioOrder === "scenarios-first"
    ? (runIndex - 1) % scenarios.length
    : Math.floor((runIndex - 1) / caseCount) % scenarios.length;
  const taskCaseIndex = measuredScenarioOrder === "scenarios-first"
    ? Math.floor((runIndex - 1) / scenarios.length)
    : runIndex - 1;
  const conversationScenario = scenarios[scenarioIndex];
  const taskCase = materializeTaskCase(
    getTaskCase(taskProfile, taskCaseIndex),
    conversationScenario
  );
  return {
    taskCase,
    conversationScenario,
  };
}

async function runPolicyPhase({
  model,
  taskProfile,
  policy,
  executionIndex,
  loadInfo,
  runKind,
  runCount,
  longHistoryScenario,
  measuredConversationScenarios,
  measuredScenarioOrder,
  latencyTargetMs,
  timeoutMs,
  results,
  logFn,
  statusFn,
  progressFn,
}) {
  for (let index = 1; index <= runCount; index++) {
    const variant = runVariant({
      taskProfile,
      runIndex: index,
      runKind,
      longHistoryScenario,
      measuredConversationScenarios,
      measuredScenarioOrder,
    });
    const built = buildPolicyMessages(
      taskProfile,
      variant.taskCase,
      policy,
      variant.conversationScenario
    );
    try {
      validatePolicy(policy, built.estimatedInputTokens);
      statusFn(`${runKind === "warmup" ? "warming up" : "running"} ${policy.label} ${index}/${runCount}...`);
      const row = await runInference({
        model,
        taskProfile,
        taskCase: variant.taskCase,
        policy,
        built,
        conversationScenario: variant.conversationScenario,
        runKind,
        runIndex: index,
        executionIndex,
        loadInfo,
        latencyTargetMs,
        timeoutMs,
      });
      if (runKind === "measured") {
        results.push(row);
        const ttft = Number.isFinite(row.timeToFirstTokenMs)
          ? `${row.timeToFirstTokenMs.toFixed(0)}ms`
          : "n/a";
        const wall = Number.isFinite(row.wallMs)
          ? `${row.wallMs.toFixed(0)}ms`
          : "n/a";
        const quality = Number.isFinite(row.qualityScore)
          ? row.qualityScore.toFixed(2)
          : "n/a";
        logFn(
          `${policy.id} ${index}/${runCount}: TTFT ${ttft}, ` +
          `wall ${wall}, quality ${quality} ` +
          `(${row.qualityDetail})`
        );
      }
    } catch (error) {
      if (runKind === "measured") {
        results.push(failureRow({
          model,
          taskProfile,
          taskCase: variant.taskCase,
          policy,
          executionIndex,
          conversationScenario: variant.conversationScenario,
          runKind,
          runIndex: index,
          loadInfo,
          error,
          stage: "inference",
          latencyTargetMs,
        }));
      }
      logFn(`ERROR ${policy.id} ${runKind} ${index}/${runCount}: ${error.stack || error}`);
    }
    progressFn({ event: runKind, policyId: policy.id, run: index });
  }
}

export async function runFixedModelPoliciesCore({
  modelConfigId,
  modelArtifactId = "",
  taskProfileId,
  latencyTargetMs = 5000,
  policies = TUNING_POLICIES,
  warmup = 1,
  runs = 6,
  timeoutMs = DEFAULT_TIMEOUT_MS,
  randomizeOrder = true,
  shuffleSeed = Date.now() >>> 0,
  label = "",
  deviceProfile = null,
  longHistoryScenario = "long-history-distraction",
  measuredConversationScenarios = null,
  measuredScenarioOrder = "cases-first",
  logFn = () => {},
  statusFn = () => {},
  progressFn = () => {},
} = {}) {
  const availableModels = getAvailableFixedModels();
  const configuredModel = availableModels.find((candidate) => candidate.id === modelConfigId)
    ?? availableModels[0];
  if (!configuredModel) throw new Error("no configured fixed model is available");
  const model = resolveModelArtifact(configuredModel.id, modelArtifactId);
  const taskProfile = getTaskProfile(taskProfileId);
  const resolvedPolicies = resolvePolicies(taskProfile, policies);
  const executionPolicies = orderPolicies(resolvedPolicies, randomizeOrder, shuffleSeed);
  const results = [];
  progressFn({ event: "init", policyTotal: executionPolicies.length, warmup, runs });

  for (const [executionIndex, policy] of executionPolicies.entries()) {
    try {
      statusFn(`loading ${policy.label}...`);
      progressFn({ event: "policy_start", policyId: policy.id, executionIndex });
      const loadInfo = await loadWebLlmModel(model.model, policy, statusFn);
      logFn(`${policy.id}: model ready in ${loadInfo.loadMs.toFixed(0)}ms`);
      const common = {
        model, taskProfile, policy, executionIndex, loadInfo, longHistoryScenario,
        measuredConversationScenarios, measuredScenarioOrder, latencyTargetMs,
        timeoutMs, results,
        logFn, statusFn, progressFn,
      };
      await runPolicyPhase({ ...common, runKind: "warmup", runCount: warmup });
      await runPolicyPhase({ ...common, runKind: "measured", runCount: runs });
    } catch (error) {
      results.push(failureRow({
        model,
        taskProfile,
        policy,
        executionIndex,
        taskCase: null,
        error,
        stage: "load",
        latencyTargetMs
      }));
      progressFn({ event: "policy_failed", policyId: policy.id });
      logFn(`ERROR ${policy.id}: ${error.stack || error}`);
      await unloadWebLlm(logFn);
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  const summaries = summarizePolicies(results, resolvedPolicies);
  const recommendationCriteria = {
    minimumPassRate: MIN_PASS_RATE,
    qualityTolerance: QUALITY_TOLERANCE,
    minimumTtftImprovement: MIN_SCREENING_TTFT_IMPROVEMENT,
  };
  const recommended = chooseRecommendedPolicy(summaries, recommendationCriteria);
  const resolvedDeviceProfile = deviceProfile ?? await getWebGpuProfile();
  return {
    meta: {
      schema: "webllm-fixed-model-bench-v4",
      timestamp: new Date().toISOString(),
      label,
      webllmVersion: WEBLLM_VERSION,
      fixedModel: model.model,
      modelFamily: model.family,
      quantization: parseQuantization(model.model),
      modelLibraryOverrides: modelRecord(model.model)?.overrides ?? null,
      taskProfileId: taskProfile.id,
      taskType: taskProfile.taskType,
      latencyTargetMs,
      warmup,
      runs,
      timeoutMs,
      taskCaseIds: taskProfile.cases.map((taskCase) => taskCase.id),
      conversationScenarios: measuredConversationScenarios?.length
        ? measuredConversationScenarios
        : ["single-turn", longHistoryScenario],
      measuredScenarioOrder,
      policyOrderStrategy: randomizeOrder ? "context_grouped_balanced_rotation" : "provided_order",
      shuffleSeed,
      policyExecutionOrder: executionPolicies.map((policy) => policy.id),
      deviceProfile: resolvedDeviceProfile,
    },
    policies: resolvedPolicies,
    results,
    summaries,
    recommendedPolicyId: recommended?.policyId ?? "",
    recommendationCriteria: {
      requiredSuccessRate: 1,
      minimumQualityPassRate: MIN_PASS_RATE,
      minimumQualityPassRatePerScenario: MIN_PASS_RATE,
      minimumOutputCompletionRate: 1,
      noTruncation: true,
      maxQualityLossVsControlled: QUALITY_TOLERANCE,
      minimumScreeningTtftImprovement: MIN_SCREENING_TTFT_IMPROVEMENT,
      mustBeatUntouchedWebllm: true,
      latencyTargetHitRateMustMatchOrBeatUntouchedWebllm: true,
      prefixReuseRequiresExactPrefixAndOutput: true,
      ranking: ["median_ttft_ms", "median_wall_ms", "median_quality", "capacity"],
    },
  };
}
