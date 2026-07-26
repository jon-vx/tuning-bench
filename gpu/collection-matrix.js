export const MATRIX_WARMUP_RUNS = 1;
export const MATRIX_MEASURED_RUNS = 6;
export const BASELINE_QUALIFICATION_CRITERIA = {
  minimumSuccessRate: 1,
  minimumContentQualityPassRate: 0.8,
  minimumOutputCompletionRate: 1,
  minimumStrictQualityPassRate: 0.8,
  minimumStrictQualityPassRatePerScenario: 2 / 3,
};

export function buildMatrixCases(models, tasks) {
  return models.flatMap((model) =>
    tasks.map((task) => ({
      id: `${model.id}:${task.id}`,
      model,
      task,
    }))
  );
}

export function matrixInferenceCount(cases, policyCount) {
  return cases.length * policyCount * (MATRIX_WARMUP_RUNS + MATRIX_MEASURED_RUNS);
}

export function matrixProgressSteps(cases, policyCount) {
  return cases.length * policyCount * (1 + MATRIX_WARMUP_RUNS + MATRIX_MEASURED_RUNS);
}

export function qualifyBaselineRun(run, criteria = BASELINE_QUALIFICATION_CRITERIA) {
  const summary = run?.summaries?.find((item) => item.policyId === "controlled-baseline");
  const reasons = [];
  if (!summary) {
    reasons.push("controlled_baseline_missing");
    return { qualified: false, reasons, summary: null };
  }
  if (summary.successRate < criteria.minimumSuccessRate) reasons.push("runtime_failures");
  if (summary.contentQualityPassRate < criteria.minimumContentQualityPassRate) {
    reasons.push("content_quality_below_floor");
  }
  if (summary.outputCompletionRate < criteria.minimumOutputCompletionRate) {
    reasons.push("incomplete_outputs");
  }
  if (summary.qualityPassRate < criteria.minimumStrictQualityPassRate) {
    reasons.push("strict_quality_below_floor");
  }
  const weakScenarios = Object.entries(summary.scenarioMetrics ?? {})
    .filter(([, metrics]) =>
      metrics.qualityPassRate < criteria.minimumStrictQualityPassRatePerScenario
    )
    .map(([scenario]) => scenario);
  if (weakScenarios.length) reasons.push(`scenario_quality_below_floor:${weakScenarios.join(",")}`);
  return { qualified: reasons.length === 0, reasons, summary };
}

export async function runMatrixCases(cases, runCase, onCaseStart = () => {}) {
  const results = [];
  for (const [index, matrixCase] of cases.entries()) {
    onCaseStart(matrixCase, index, cases.length);
    results.push(await runCase(matrixCase, index));
  }
  return results;
}
