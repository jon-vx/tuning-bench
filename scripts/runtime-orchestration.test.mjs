import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { patchWebLlmBundle } from "./build-runtime-orchestration.mjs";
import {
  ORCHESTRATION_POLICIES,
  rotatedPolicyOrder,
  summarizeOrchestrationRows,
} from "../gpu/runtime-orchestration-core.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const generated = path.join(root, "vendor/generated/webllm-0.2.79-runtime-orchestration.js");

test("generated bundle contains the switchable experiment and mandatory flushes", () => {
  const source = readFileSync(generated, "utf8");
  assert.match(source, /__WEBLLM_ORCHESTRATION_EXPERIMENT__/);
  assert.match(source, /flushOrchestrationCommands\("gpu-to-cpu"\)/);
  assert.match(source, /flushOrchestrationCommands\("cpu-to-gpu"\)/);
  assert.match(source, /flushOrchestrationCommands\("buffer-free"\)/);
  assert.match(source, /this\.orchestrationPolicy\.dispatchesPerEncoder !== 1/);
  assert.match(source, /const reuseUniformBuffers = this\.orchestrationPolicy\.reuseUniformBuffers/);
  assert.match(source, /if \(reuseUniformBuffers\) this\.orchestrationBufferIndex = 0/);
  assert.doesNotMatch(source, /setPolicy:[\s\S]{0,300}destroyOrchestrationBufferPool/);
});

test("patcher refuses an unknown upstream bundle", () => {
  assert.throws(() => patchWebLlmBundle("not WebLLM"), /patch point not found/);
});

test("policy order rotates while retaining every policy", () => {
  const orders = [0, 1, 2, 3].map((index) => rotatedPolicyOrder(ORCHESTRATION_POLICIES, index));
  for (const order of orders) {
    assert.deepEqual(new Set(order.map((policy) => policy.id)), new Set(ORCHESTRATION_POLICIES.map((policy) => policy.id)));
  }
  assert.notDeepEqual(orders[0].map((policy) => policy.id), orders[1].map((policy) => policy.id));
});

test("summary uses matched baseline pairs and penalizes failed equality", () => {
  const policies = ORCHESTRATION_POLICIES.filter((policy) => ["baseline", "batch-4"].includes(policy.id));
  const workloads = [{ id: "w", label: "workload" }];
  const orchestration = (traceHash, dispatches, submitCalls) => ({ traceHash, dispatches, submitCalls });
  const rows = [
    { policyId: "baseline", workloadId: "w", repetition: 0, success: true, ttftMs: 100, wallMs: 200, chunkGapsMs: [10], output: "same", orchestration: orchestration(1, 8, 8) },
    { policyId: "batch-4", workloadId: "w", repetition: 0, success: true, ttftMs: 80, wallMs: 180, chunkGapsMs: [9], output: "same", orchestration: orchestration(1, 8, 2) },
    { policyId: "baseline", workloadId: "w", repetition: 1, success: true, ttftMs: 100, wallMs: 200, chunkGapsMs: [10], output: "same", orchestration: orchestration(1, 8, 8) },
    { policyId: "batch-4", workloadId: "w", repetition: 1, success: false, chunkGapsMs: [], orchestration: orchestration(1, 4, 1) },
  ];
  const candidate = summarizeOrchestrationRows(rows, policies, workloads).find((row) => row.policyId === "batch-4");
  assert.equal(candidate.medianTtftDeltaPct, -20);
  assert.equal(candidate.successfulRuns, 1);
  assert.equal(candidate.measuredRuns, 2);
  assert.equal(candidate.outputEqualityRate, 0.5);
  assert.equal(candidate.traceEqualityRate, 0.5);
});
