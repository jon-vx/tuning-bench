#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { rotatedPolicyOrder, summarizeOrchestrationRows } from "../gpu/runtime-orchestration-core.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const campaignDirs = process.argv.slice(2);
if (!campaignDirs.length) {
  throw new Error("usage: node scripts/audit-runtime-orchestration.mjs <campaign-dir> [...]");
}

function closeEnough(left, right, tolerance = 1e-9) {
  if (left == null || right == null) return left === right;
  return Math.abs(left - right) <= tolerance * Math.max(1, Math.abs(left), Math.abs(right));
}

function auditCampaign(inputDir) {
  const directory = path.resolve(root, inputDir);
  const rawPath = path.join(directory, "raw-results.json");
  const checksumPath = path.join(directory, "raw-results.sha256");
  const text = readFileSync(rawPath, "utf8");
  const payload = JSON.parse(text);
  const failures = [];
  const checks = [];
  const check = (condition, message) => {
    checks.push(message);
    if (!condition) failures.push(message);
  };

  const expectedChecksum = readFileSync(checksumPath, "utf8").trim().split(/\s+/)[0];
  const actualChecksum = createHash("sha256").update(text).digest("hex");
  check(actualChecksum === expectedChecksum, "checksum matches raw result");
  check(
    /^webllm-runtime-orchestration(?:-[a-z]+)?-v1$/.test(payload.schemaVersion),
    "schema version is recognized"
  );
  check(path.basename(directory) === payload.campaignId, "directory matches campaign ID");
  check(payload.runtimeBuild?.webLlmVersion === "0.2.79", "WebLLM version is pinned to 0.2.79");
  check(Boolean(payload.runtimeBuild?.upstreamSha256), "upstream bundle hash is present");
  check(Boolean(payload.runtimeBuild?.patchedSha256), "patched bundle hash is present");
  check(payload.fixedGeneration?.temperature === 0 && payload.fixedGeneration?.seed === 42, "generation is deterministic");

  const expectedRows = payload.policies.length * payload.workloads.length * payload.repetitions;
  check(payload.rows.length === expectedRows, `measured row count is ${expectedRows}`);
  check(payload.rows.every((row) => row.runKind === "measured"), "export contains measured rows only");
  check(payload.rows.every((row) => row.success), "all measured rows succeeded");
  check(payload.rows.every((row) => Number.isFinite(row.ttftMs) && row.ttftMs > 0), "all TTFT values are finite and positive");
  check(payload.rows.every((row) => Number.isFinite(row.wallMs) && row.wallMs >= row.ttftMs), "all wall times are valid");

  const keys = payload.rows.map((row) => `${row.workloadId}|${row.repetition}|${row.policyId}`);
  check(new Set(keys).size === keys.length, "row matching keys are unique");

  for (const workload of payload.workloads) {
    for (let repetition = 0; repetition < payload.repetitions; repetition += 1) {
      const block = payload.rows
        .filter((row) => row.workloadId === workload.id && row.repetition === repetition)
        .sort((left, right) => left.executionIndex - right.executionIndex);
      const expectedOrder = rotatedPolicyOrder(payload.policies, repetition).map((policy) => policy.id);
      check(
        block.map((row) => row.policyId).join("|") === expectedOrder.join("|"),
        `${workload.id} repetition ${repetition} follows counterbalanced order`
      );
      const baseline = block.find((row) => row.policyId === "baseline");
      for (const row of block) {
        check(row.output === baseline.output, `${workload.id} repetition ${repetition} ${row.policyId} output equals baseline`);
        check(
          row.orchestration.dispatches === baseline.orchestration.dispatches &&
          row.orchestration.traceHash === baseline.orchestration.traceHash,
          `${workload.id} repetition ${repetition} ${row.policyId} dispatch trace equals baseline`
        );
      }
    }
  }

  const recomputed = summarizeOrchestrationRows(payload.rows, payload.policies, payload.workloads);
  check(recomputed.length === payload.summaries.length, "summary row count recomputes");
  for (const expected of payload.summaries) {
    const actual = recomputed.find((row) => row.workloadId === expected.workloadId && row.policyId === expected.policyId);
    check(Boolean(actual), `${expected.workloadId} ${expected.policyId} summary exists`);
    if (!actual) continue;
    for (const field of ["medianTtftMs", "medianWallMs", "medianTtftDeltaPct", "medianWallDeltaPct", "outputEqualityRate", "traceEqualityRate"]) {
      check(closeEnough(actual[field], expected[field]), `${expected.workloadId} ${expected.policyId} ${field} recomputes`);
    }
  }

  const result = {
    campaignId: payload.campaignId,
    auditedAt: new Date().toISOString(),
    passed: failures.length === 0,
    checkCount: checks.length,
    failures,
    rows: payload.rows.length,
    model: payload.model,
    device: payload.deviceProfile?.gpu?.identity || payload.deviceProfile?.gpu?.architecture || "",
    browser: payload.deviceProfile?.userAgent || "",
    runtimeBuild: payload.runtimeBuild,
  };
  writeFileSync(path.join(directory, "audit.json"), JSON.stringify(result, null, 2) + "\n");
  return result;
}

const results = campaignDirs.map(auditCampaign);
process.stdout.write(JSON.stringify(results, null, 2) + "\n");
if (results.some((result) => !result.passed)) process.exitCode = 1;
