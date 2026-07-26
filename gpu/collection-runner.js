import { runDeviceProbeCore } from "./device-probe.js";
import {
  BASELINE_QUALIFICATION_CRITERIA,
  MATRIX_MEASURED_RUNS,
  MATRIX_WARMUP_RUNS,
  matrixProgressSteps,
  qualifyBaselineRun,
  runMatrixCases,
} from "./collection-matrix.js";
import { runFixedModelPoliciesCore } from "./fixed-model-bench.js";
import { getWebGpuProfile } from "./device-profile.js";
import { TASK_PROFILES } from "./task-profiles.js";
import { TUNING_POLICIES } from "./tuning-policies.js";

export const RETENTION_MEASURED_RUNS = 3;

const controlledBaselinePolicy = TUNING_POLICIES.find(
  (policy) => policy.id === "controlled-baseline"
);

function eligibleMatrixCases(group, cases, retentionQualifiedIds = null) {
  if (!group.requiresHistoryRequirements) return cases;
  return cases.filter((matrixCase) => {
    const hasFixtures = matrixCase.task.cases.every((taskCase) => taskCase.historyRequirement);
    return hasFixtures && (!retentionQualifiedIds || retentionQualifiedIds.has(matrixCase.id));
  });
}

function retentionProgressSteps(cases) {
  return cases.length * (1 + MATRIX_WARMUP_RUNS + RETENTION_MEASURED_RUNS);
}

function qualificationsFor(runs, cases) {
  return runs.map((run, index) => ({
    matrixCaseId: cases[index].id,
    modelId: cases[index].model.id,
    taskProfileId: cases[index].task.id,
    ...qualifyBaselineRun(run),
  }));
}

async function runPolicyMatrix({
  cases,
  policies,
  runs,
  latencyTargetMs,
  seed,
  label,
  profile,
  logPrefix,
  heading,
  statusPrefix = "",
  scenarios = null,
  randomizeOrder = false,
  ui,
  onCase = () => {},
}) {
  return runMatrixCases(cases, async (matrixCase, caseIndex) => {
    const caseLabel = `${matrixCase.model.family} / ${matrixCase.task.label}`;
    ui.log(`\n=== ${heading}: ${caseLabel} ===`);
    return runFixedModelPoliciesCore({
      modelConfigId: matrixCase.model.id,
      taskProfileId: matrixCase.task.id,
      latencyTargetMs,
      policies,
      warmup: MATRIX_WARMUP_RUNS,
      runs,
      randomizeOrder,
      measuredConversationScenarios: scenarios,
      shuffleSeed: seed + caseIndex,
      label,
      deviceProfile: profile,
      logFn: (message) => ui.log(`[${logPrefix}/${matrixCase.id}] ${message}`),
      statusFn: (message) => ui.status(`${statusPrefix}${caseLabel}: ${message}`),
      progressFn: (progress) => ui.tuningProgress(progress, matrixCase),
    });
  }, onCase);
}

async function collectDevice(label, ui) {
  ui.phase("collecting device profile");
  ui.step("profile", "running");
  ui.status("collecting device profile...");
  const profile = await getWebGpuProfile();
  ui.profileComplete();
  ui.step("profile", "done");

  ui.phase("starting WebGPU probe");
  ui.step("probe", "running");
  ui.status("running WebGPU device probe...");
  const probe = await runDeviceProbeCore({
    label,
    deviceProfile: profile,
    logFn: ui.log,
    statusFn: ui.status,
    progressFn: ui.probeProgress,
  });
  ui.step("probe", probe.probeSummary.success ? "done" : "failed");
  return { profile, probe };
}

async function qualifyBaselines(options) {
  const {
    matrixCases,
    policyGroups,
    latencyTargetMs,
    matrixSeed,
    label,
    profile,
    ui,
  } = options;
  ui.phase("starting baseline qualification");
  ui.step("baseline", "running");
  ui.status("validating controlled baselines...");
  const runs = await runPolicyMatrix({
    cases: matrixCases,
    policies: [controlledBaselinePolicy],
    runs: MATRIX_MEASURED_RUNS,
    latencyTargetMs,
    seed: matrixSeed,
    label,
    profile,
    logPrefix: "baseline",
    heading: "baseline",
    ui,
    onCase: (matrixCase, caseIndex, caseTotal) => ui.step(
      "baseline",
      `${caseIndex + 1}/${caseTotal}: ${matrixCase.model.family} / ${matrixCase.task.label}`
    ),
  });
  const qualifications = qualificationsFor(runs, matrixCases);
  const qualifiedIds = new Set(
    qualifications.filter((item) => item.qualified).map((item) => item.matrixCaseId)
  );
  const cases = matrixCases.filter((matrixCase) => qualifiedIds.has(matrixCase.id));
  for (const qualification of qualifications.filter((item) => !item.qualified)) {
    ui.log(`EXCLUDED ${qualification.matrixCaseId}: ${qualification.reasons.join("|")}`);
  }
  ui.step("baseline", `${cases.length}/${matrixCases.length} qualified`);

  const needsRetentionGate = policyGroups.some((group) => group.requiresHistoryRequirements);
  const tuningTotal = matrixProgressSteps(matrixCases, 1) + (
    policyGroups.length
      ? (needsRetentionGate ? retentionProgressSteps(cases) : 0) +
        policyGroups.reduce(
          (total, group) =>
            total + matrixProgressSteps(eligibleMatrixCases(group, cases), group.policies.length),
          0
        )
      : 0
  );
  ui.tuningPlan({ total: tuningTotal });
  return { runs, qualifications, cases };
}

async function qualifyRetention(options) {
  const {
    enabled,
    qualifiedCases,
    policyGroups,
    matrixCases,
    latencyTargetMs,
    matrixSeed,
    label,
    profile,
    ui,
  } = options;
  if (!enabled || !qualifiedCases.length) {
    return { runs: [], qualifications: [], qualifiedIds: new Set() };
  }

  ui.phase("validating required-history baselines");
  ui.tuningPlan({
    expectedPerPolicy: 1 + MATRIX_WARMUP_RUNS + RETENTION_MEASURED_RUNS,
  });
  ui.step("baseline", "running retention gate");
  ui.status("validating required-history baselines...");
  const runs = await runPolicyMatrix({
    cases: qualifiedCases,
    policies: [controlledBaselinePolicy],
    runs: RETENTION_MEASURED_RUNS,
    latencyTargetMs,
    seed: matrixSeed + matrixCases.length,
    label,
    profile,
    logPrefix: "retention",
    heading: "retention gate",
    statusPrefix: "Retention gate, ",
    scenarios: ["long-history-required"],
    ui,
  });
  const qualifications = qualificationsFor(runs, qualifiedCases);
  const qualifiedIds = new Set(
    qualifications.filter((item) => item.qualified).map((item) => item.matrixCaseId)
  );
  for (const qualification of qualifications.filter((item) => !item.qualified)) {
    ui.log(
      `HISTORY EXCLUDED ${qualification.matrixCaseId}: ${qualification.reasons.join("|")}`
    );
  }
  ui.step(
    "baseline",
    `${qualifiedCases.length}/${matrixCases.length} clean; ` +
      `${qualifiedIds.size}/${qualifiedCases.length} retention`
  );
  const tuningTotal = matrixProgressSteps(matrixCases, 1) +
    retentionProgressSteps(qualifiedCases) +
    policyGroups.reduce(
      (total, group) => total + matrixProgressSteps(
        eligibleMatrixCases(group, qualifiedCases, qualifiedIds),
        group.policies.length
      ),
      0
    );
  ui.tuningPlan({
    total: tuningTotal,
    expectedPerPolicy: 1 + MATRIX_WARMUP_RUNS + MATRIX_MEASURED_RUNS,
  });
  return { runs, qualifications, qualifiedIds };
}

async function runTuningGroups(options) {
  const {
    policyGroups,
    qualifiedCases,
    retentionQualifiedIds,
    matrixCases,
    latencyTargetMs,
    matrixSeed,
    label,
    profile,
    ui,
  } = options;
  if (!policyGroups.length || !qualifiedCases.length) {
    ui.step(
      "chat",
      policyGroups.length ? "no qualified cases" : "skipped (validation only)"
    );
    return [];
  }

  ui.phase("starting qualified model/task matrix");
  ui.step("chat", "running");
  ui.status(`running policies for ${qualifiedCases.length} qualified cases...`);
  const tuningRuns = [];
  for (const [groupIndex, group] of policyGroups.entries()) {
    const cases = eligibleMatrixCases(group, qualifiedCases, retentionQualifiedIds);
    const groupRuns = await runPolicyMatrix({
      cases,
      policies: group.policies,
      runs: MATRIX_MEASURED_RUNS,
      latencyTargetMs,
      seed: matrixSeed + matrixCases.length * (groupIndex + 1),
      label,
      profile,
      logPrefix: group.id,
      heading: group.label,
      statusPrefix: `${group.label}, `,
      scenarios: group.measuredConversationScenarios,
      randomizeOrder: true,
      ui,
      onCase: (matrixCase, caseIndex, caseTotal) => ui.step(
        "chat",
        `${group.label} ${caseIndex + 1}/${caseTotal}: ` +
          `${matrixCase.model.family} / ${matrixCase.task.label}`
      ),
    });
    for (const run of groupRuns) {
      run.meta.policyGroupId = group.id;
      run.meta.policyGroupLabel = group.label;
    }
    tuningRuns.push(...groupRuns);
  }
  const anySuccess = tuningRuns.some((tuning) => tuning.results.some((row) => row.success));
  ui.step("chat", anySuccess ? "done" : "all failed");
  return tuningRuns;
}

export async function runCollection({
  matrixCases,
  policyGroups,
  latencyTargetMs,
  matrixSeed,
  label,
  workflow,
  ui,
}) {
  const { profile, probe } = await collectDevice(label, ui);
  const baseline = await qualifyBaselines({
    matrixCases,
    policyGroups,
    latencyTargetMs,
    matrixSeed,
    label,
    profile,
    ui,
  });
  const retention = await qualifyRetention({
    enabled: policyGroups.some((group) => group.requiresHistoryRequirements),
    qualifiedCases: baseline.cases,
    policyGroups,
    matrixCases,
    latencyTargetMs,
    matrixSeed,
    label,
    profile,
    ui,
  });
  const tuningRuns = await runTuningGroups({
    policyGroups,
    qualifiedCases: baseline.cases,
    retentionQualifiedIds: retention.qualifiedIds,
    matrixCases,
    latencyTargetMs,
    matrixSeed,
    label,
    profile,
    ui,
  });

  return {
    payload: {
      schema: "webllm-model-task-matrix-v7",
      timestamp: new Date().toISOString(),
      label,
      workflow,
      deviceProfile: profile,
      probe,
      baselineValidation: {
        criteria: BASELINE_QUALIFICATION_CRITERIA,
        runs: baseline.runs,
        qualifications: baseline.qualifications,
      },
      retentionValidation: {
        criteria: BASELINE_QUALIFICATION_CRITERIA,
        measuredRuns: RETENTION_MEASURED_RUNS,
        runs: retention.runs,
        qualifications: retention.qualifications,
      },
      matrix: {
        modelIds: [...new Set(matrixCases.map((matrixCase) => matrixCase.model.id))],
        taskProfileIds: TASK_PROFILES.map((task) => task.id),
        taskCaseIds: Object.fromEntries(
          TASK_PROFILES.map((task) => [task.id, task.cases.map((taskCase) => taskCase.id)])
        ),
        conversationScenarios: ["single-turn", "long-history-distraction"],
        policyGroups: Object.fromEntries(
          policyGroups.map((group) => [
            group.id,
            {
              policyIds: group.policies.map((policy) => policy.id),
              conversationScenarios: group.measuredConversationScenarios,
            },
          ])
        ),
        qualifiedCaseIds: baseline.cases.map((matrixCase) => matrixCase.id),
        retentionQualifiedCaseIds: [...retention.qualifiedIds],
        excludedCases: baseline.qualifications
          .filter((item) => !item.qualified)
          .map(({ matrixCaseId, reasons }) => ({ matrixCaseId, reasons })),
        warmupRuns: MATRIX_WARMUP_RUNS,
        measuredRuns: MATRIX_MEASURED_RUNS,
        latencyTargetMs,
        matrixSeed,
      },
      tuningRuns,
    },
    qualifiedCaseCount: baseline.cases.length,
    tuningRunCount: tuningRuns.length,
    recommendationCount: tuningRuns.filter((tuning) => tuning.recommendedPolicyId).length,
  };
}
