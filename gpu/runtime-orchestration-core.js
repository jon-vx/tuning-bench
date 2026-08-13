import { median, quantile } from "../shared/benchmark-utils.js";

export const ORCHESTRATION_POLICIES = [
  { id: "baseline", label: "stock baseline", dispatchesPerEncoder: 1, reuseUniformBuffers: false, defaultSelected: true },
  { id: "reuse-only", label: "reuse only / batch 1", dispatchesPerEncoder: 1, reuseUniformBuffers: true, defaultSelected: true },
  { id: "batch-2", label: "batch 2", dispatchesPerEncoder: 2, reuseUniformBuffers: true },
  { id: "batch-4", label: "batch 4", dispatchesPerEncoder: 4, reuseUniformBuffers: true, defaultSelected: true },
  { id: "batch-8", label: "batch 8", dispatchesPerEncoder: 8, reuseUniformBuffers: true },
  { id: "batch-16", label: "batch 16", dispatchesPerEncoder: 16, reuseUniformBuffers: true, defaultSelected: true },
  { id: "batch-32", label: "batch 32", dispatchesPerEncoder: 32, reuseUniformBuffers: true },
  { id: "batch-64", label: "batch 64", dispatchesPerEncoder: 64, reuseUniformBuffers: true },
  { id: "batch-boundary", label: "batch to boundary", dispatchesPerEncoder: null, reuseUniformBuffers: true },
];

const FILLER =
  "Browser inference keeps model data on the local device while deterministic measurements compare unchanged GPU computation. ";

export const ORCHESTRATION_WORKLOADS = [
  {
    id: "short-prefill",
    label: "short prefill / up to 32 output",
    maxTokens: 32,
    stop: ["."],
    prompt: "In one short sentence, state why fixed inputs matter. End with a period.",
  },
  {
    id: "long-prefill",
    label: "long prefill / up to 32 output",
    maxTokens: 32,
    stop: ["."],
    prompt:
      FILLER.repeat(150) +
      "\nIn one short sentence, state where model data stays. End with a period.",
  },
  {
    id: "long-decode",
    label: "short prefill / up to 128 output",
    maxTokens: 128,
    prompt:
      "Reply exactly with these eight lines:\n" +
      "1. Fix all benchmark inputs.\n" +
      "2. Record the tested hardware.\n" +
      "3. Pin every software version.\n" +
      "4. Warm up before measurement.\n" +
      "5. Rotate the execution order.\n" +
      "6. Repeat each timed configuration.\n" +
      "7. Check generated output equality.\n" +
      "8. Save every raw result.",
  },
];

export function rotatedPolicyOrder(policies, repetition) {
  if (!policies.length) return [];
  const offset = repetition % policies.length;
  const rotated = [...policies.slice(offset), ...policies.slice(0, offset)];
  return Math.floor(repetition / policies.length) % 2 ? rotated.reverse() : rotated;
}

export function summarizeOrchestrationRows(rows, policies, workloads) {
  const summaries = [];
  for (const workload of workloads) {
    const workloadRows = rows.filter((row) => row.workloadId === workload.id);
    const baseline = new Map(
      workloadRows
        .filter((row) => row.policyId === "baseline")
        .map((row) => [row.repetition, row])
    );
    for (const policy of policies) {
      const allPolicyRows = workloadRows.filter((row) => row.policyId === policy.id);
      const policyRows = allPolicyRows.filter((row) => row.success);
      const allPairs = allPolicyRows
        .map((row) => ({ row, baseline: baseline.get(row.repetition) }))
        .filter((pair) => pair.baseline);
      const pairs = allPairs.filter(({ row, baseline: base }) => row.success && base.success);
      const delta = (field) => pairs.map(({ row, baseline: base }) =>
        base[field] > 0 ? ((row[field] - base[field]) / base[field]) * 100 : NaN
      );
      const outputEqualityRate = policy.id === "baseline" || !allPairs.length
        ? 1
        : allPairs.filter(({ row, baseline: base }) =>
          row.success && base.success && row.output === base.output
        ).length / allPairs.length;
      const traceEqualityRate = policy.id === "baseline" || !allPairs.length
        ? 1
        : allPairs.filter(({ row, baseline: base }) =>
          row.success && base.success &&
          row.orchestration.dispatches === base.orchestration.dispatches &&
          row.orchestration.traceHash === base.orchestration.traceHash
        ).length / allPairs.length;
      const medianTtftDeltaPct = policy.id === "baseline" ? 0 : median(delta("ttftMs"));
      const medianWallDeltaPct = policy.id === "baseline" ? 0 : median(delta("wallMs"));
      const fasterTtftFraction = policy.id === "baseline" || !pairs.length
        ? null
        : pairs.filter(({ row, baseline: base }) => row.ttftMs < base.ttftMs).length / pairs.length;
      const fasterWallFraction = policy.id === "baseline" || !pairs.length
        ? null
        : pairs.filter(({ row, baseline: base }) => row.wallMs < base.wallMs).length / pairs.length;
      const complete = policyRows.length === allPolicyRows.length && allPolicyRows.length > 0;
      const objectiveDelta = workload.id === "long-decode" ? medianWallDeltaPct : medianTtftDeltaPct;
      const fasterFraction = workload.id === "long-decode" ? fasterWallFraction : fasterTtftFraction;
      let screenStatus = "reference";
      if (policy.id !== "baseline") {
        if (!complete) screenStatus = "reliability failure";
        else if (outputEqualityRate !== 1) screenStatus = "output mismatch";
        else if (traceEqualityRate !== 1) screenStatus = "trace mismatch";
        else if (objectiveDelta <= -10 && fasterFraction >= 2 / 3) screenStatus = "promising signal";
        else screenStatus = "no material win";
      }
      summaries.push({
        workloadId: workload.id,
        workloadLabel: workload.label,
        policyId: policy.id,
        policyLabel: policy.label,
        measuredRuns: allPolicyRows.length,
        successfulRuns: policyRows.length,
        medianTtftMs: median(policyRows.map((row) => row.ttftMs)),
        medianWallMs: median(policyRows.map((row) => row.wallMs)),
        medianChunkGapMs: median(policyRows.flatMap((row) => row.chunkGapsMs)),
        p95ChunkGapMs: quantile(policyRows.flatMap((row) => row.chunkGapsMs), 0.95),
        medianTtftDeltaPct,
        medianWallDeltaPct,
        fasterTtftFraction,
        fasterWallFraction,
        outputEqualityRate,
        traceEqualityRate,
        medianDispatches: median(policyRows.map((row) => row.orchestration.dispatches)),
        medianSubmitCalls: median(policyRows.map((row) => row.orchestration.submitCalls)),
        objective: workload.id === "long-decode" ? "wallMs" : "ttftMs",
        screenStatus,
      });
    }
  }
  return summaries;
}
