import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildExperimentPolicies,
  EXPERIMENT_DEFINITIONS,
  experimentSupport,
} from "../gpu/controlled-experiment.js";
import { FIXED_MODELS } from "../gpu/models.js";
import { MANUAL_TASK_PROFILES } from "../gpu/task-profiles.js";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const args = process.argv.slice(2);
const argument = (name, fallback) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : fallback;
};
const campaignId = argument(
  "--campaign",
  `manual-${new Date().toISOString().slice(0, 10).replaceAll("-", "")}`
);
const port = Number(argument("--port", "8010"));
const campaignMode = argument("--mode", "confirmatory");
if (!["confirmatory", "rapid", "latency"].includes(campaignMode)) {
  throw new Error(`unknown mode: ${campaignMode}`);
}
const selectedIds = (name, items) => {
  const ids = argument(name, "").split(",").filter(Boolean);
  if (!ids.length) return items;
  const selected = items.filter((item) => ids.includes(item.id));
  const missing = ids.filter((id) => !selected.some((item) => item.id === id));
  if (missing.length) throw new Error(`unknown ${name.slice(2)}: ${missing.join(", ")}`);
  return selected;
};
const campaignModels = selectedIds("--models", FIXED_MODELS);
const campaignTasks = selectedIds("--tasks", MANUAL_TASK_PROFILES);
const campaignDefinitions = selectedIds("--parameters", EXPERIMENT_DEFINITIONS);
if (
  campaignMode === "latency" &&
  campaignTasks.some((task) => task.id !== "chat-latency")
) {
  throw new Error("latency mode requires --tasks chat-latency");
}
const validationSourceId = argument("--validate-candidates-from", "");
const dataDirectory = join(root, "..", "data");
const campaignDirectory = join(dataDirectory, "manual-campaign", campaignId);
const runsDirectory = join(campaignDirectory, "runs");
const errorsDirectory = join(campaignDirectory, "errors");
const manifestPath = join(campaignDirectory, "campaign.json");
const contentTypes = {
  ".css": "text/css",
  ".html": "text/html",
  ".js": "text/javascript",
  ".json": "application/json",
  ".md": "text/markdown",
  ".svg": "image/svg+xml",
};

function sourceRevision() {
  const commit = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: root,
    encoding: "utf8",
  }).trim();
  const status = execFileSync("git", ["status", "--porcelain"], {
    cwd: root,
    encoding: "utf8",
  });
  const diff = execFileSync("git", ["diff", "--binary"], {
    cwd: root,
    encoding: "utf8",
    maxBuffer: 50 * 1024 * 1024,
  });
  const files = execFileSync(
    "git",
    ["ls-files", "--cached", "--others", "--exclude-standard"],
    { cwd: root, encoding: "utf8" }
  )
    .trim()
    .split("\n")
    .filter(Boolean)
    .sort();
  const sourceHash = createHash("sha256");
  for (const path of files) {
    sourceHash.update(path);
    sourceHash.update("\0");
    sourceHash.update(readFileSync(join(root, path)));
    sourceHash.update("\0");
  }
  return {
    commit,
    dirty: Boolean(status),
    diffSha256: createHash("sha256").update(diff).digest("hex"),
    sourceSha256: sourceHash.digest("hex"),
  };
}

function entryId(modelConfigId, taskProfileId, parameterId) {
  return `${modelConfigId}__${taskProfileId}__${parameterId}`;
}

function candidatePlan(sourceCampaignId) {
  const sourceDirectory = join(dataDirectory, "manual-campaign", sourceCampaignId);
  const sourceManifestPath = join(sourceDirectory, "campaign.json");
  if (!existsSync(sourceManifestPath)) {
    throw new Error(`candidate source campaign not found: ${sourceCampaignId}`);
  }
  const sourceManifest = JSON.parse(readFileSync(sourceManifestPath, "utf8"));
  const candidates = [];
  for (const entry of sourceManifest.entries.filter((item) => item.status === "complete")) {
    const payload = JSON.parse(
      readFileSync(join(sourceDirectory, entry.payloadFile), "utf8")
    );
    if (!payload.recommendedPolicyId) continue;
    const policy = payload.policies.find(
      (item) => item.id === payload.recommendedPolicyId
    );
    if (!policy) {
      throw new Error(`recommended policy is missing from ${entry.id}`);
    }
    candidates.push({
      modelConfigId: entry.modelConfigId,
      parameterId: entry.parameterId,
      sourceTaskProfileId: entry.taskProfileId,
      candidatePolicyId: policy.id,
      candidateExperimentValue: policy.experimentValue,
    });
  }
  if (!candidates.length) {
    throw new Error(`candidate source campaign has no recommendations: ${sourceCampaignId}`);
  }
  return candidates;
}

function newManifest() {
  const selectedModelIds = new Set(campaignModels.map((model) => model.id));
  const validationPlan = validationSourceId
    ? candidatePlan(validationSourceId).filter((candidate) =>
      selectedModelIds.has(candidate.modelConfigId)
    )
    : null;
  if (validationSourceId && !validationPlan.length) {
    throw new Error("no source candidates match the selected models");
  }
  if (
    validationPlan &&
    campaignTasks.some((task) => task.cases.length !== campaignTasks[0].cases.length)
  ) {
    throw new Error("candidate validation tasks must have the same case count");
  }
  const validationModelIds = new Set(
    validationPlan?.map((candidate) => candidate.modelConfigId) ?? []
  );
  const manifestModels = validationPlan
    ? campaignModels.filter((model) => validationModelIds.has(model.id))
    : campaignModels;
  const entries = [];
  if (validationPlan) {
    for (const candidate of validationPlan) {
      for (const task of campaignTasks) {
        const support = experimentSupport(
          candidate.parameterId,
          candidate.modelConfigId,
          task.id
        );
        entries.push({
          id: entryId(candidate.modelConfigId, task.id, candidate.parameterId),
          modelConfigId: candidate.modelConfigId,
          taskProfileId: task.id,
          parameterId: candidate.parameterId,
          sourceTaskProfileId: candidate.sourceTaskProfileId,
          candidatePolicyId: candidate.candidatePolicyId,
          candidateExperimentValue: candidate.candidateExperimentValue,
          status: support.supported ? "pending" : "skipped",
          payloadFile: "",
          startedAt: null,
          completedAt: null,
          error: support.reason || null,
        });
      }
    }
  } else {
    for (const model of manifestModels) {
      for (const task of campaignTasks) {
        for (const definition of campaignDefinitions) {
          const support = experimentSupport(definition.id, model.id, task.id);
          const candidate = campaignMode === "latency"
            ? buildExperimentPolicies(definition.id, {
              maxTokens: task.outputTokenLimits.standard,
            }).find((policy) => !policy.baselineType)
            : null;
          entries.push({
            id: entryId(model.id, task.id, definition.id),
            modelConfigId: model.id,
            taskProfileId: task.id,
            parameterId: definition.id,
            ...(candidate
              ? {
                candidatePolicyId: candidate.id,
                candidateExperimentValue: candidate.experimentValue,
              }
              : {}),
            status: support.supported ? "pending" : "skipped",
            payloadFile: "",
            startedAt: null,
            completedAt: null,
            error: support.reason || null,
          });
        }
      }
    }
  }
  const artifactPreflight = manifestModels.flatMap((model) =>
    Object.entries(model.artifacts ?? {})
      .filter(([quantization]) => !validationPlan || quantization === "q4f16_1")
      .map(([quantization, artifactId]) => ({
        modelConfigId: model.id,
        quantization,
        artifactId,
        status: "pending",
        error: null,
        checkedAt: null,
      }))
  );
  return {
    schema: "webllm-manual-campaign-v1",
    campaignId,
    campaignType: validationPlan
      ? "candidate-validation"
      : campaignMode === "latency"
        ? "latency-only"
        : "parameter-screen",
    startedAt: null,
    completedAt: null,
    sourceRevision: sourceRevision(),
    webllmVersion: "0.2.79",
    deviceProfile: null,
    includeLibraryDefault: true,
    execution: validationPlan
      ? {
        mode: "validation",
        blocks: 1,
        warmups: 1,
        warmupEachPolicy: true,
        measuredRunsPerCase: 2,
        taskCaseCount: campaignTasks[0].cases.length,
        completeCounterbalance: false,
      }
      : campaignMode === "rapid"
        ? {
          mode: "rapid",
          blocks: 1,
          warmups: 0,
          measuredRunsPerScenario: 1,
          completeCounterbalance: false,
        }
        : campaignMode === "latency"
          ? {
            mode: "latency",
            blocks: 6,
            warmups: 1,
            warmupEachPolicy: true,
            measuredRunsPerCase: 1,
            taskCaseCount: campaignTasks[0].cases.length,
            completeCounterbalance: true,
            includeLibraryDefault: true,
          }
        : {
          mode: "confirmatory",
          completeCounterbalance: true,
        },
    models: manifestModels.map(({ id, family, model, artifacts }) => ({
      id,
      family,
      model,
      artifacts,
    })),
    tasks: campaignTasks.map(({ id, label, taskType }) => ({ id, label, taskType })),
    parameters: validationPlan
      ? [...new Set(validationPlan.map((candidate) => candidate.parameterId))]
      : campaignDefinitions.map((definition) => definition.id),
    sourceCampaignId: validationSourceId || null,
    reportFile: validationPlan
      ? "CROSS_TASK_VALIDATION.md"
      : campaignMode === "latency"
        ? "LATENCY_ONLY_RESULTS.md"
        : "MANUAL_EXPERIMENTS.md",
    expectedExperimentCount: entries.filter((entry) => entry.status !== "skipped").length,
    artifactPreflight,
    entries,
  };
}

mkdirSync(runsDirectory, { recursive: true });
mkdirSync(errorsDirectory, { recursive: true });
let manifest = existsSync(manifestPath)
  ? JSON.parse(readFileSync(manifestPath, "utf8"))
  : newManifest();
for (const entry of manifest.entries) {
  if (entry.status === "running") entry.status = "pending";
}

function saveManifest() {
  const active = manifest.entries.filter((entry) =>
    ["pending", "running"].includes(entry.status)
  );
  manifest.completedAt = active.length ? null : new Date().toISOString();
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
}

function measuredInferenceProgress() {
  let completed = 0;
  let expected = 0;
  for (const entry of manifest.entries) {
    if (entry.status === "skipped") continue;
    if (entry.status === "complete" && entry.payloadFile) {
      const payload = JSON.parse(readFileSync(join(campaignDirectory, entry.payloadFile), "utf8"));
      completed += payload.results?.length ?? 0;
    }
    const definition = EXPERIMENT_DEFINITIONS.find((item) => item.id === entry.parameterId);
    const policyCount = entry.candidatePolicyId
      ? 2 + (manifest.includeLibraryDefault ? 1 : 0)
      : definition.values.length + (manifest.includeLibraryDefault ? 1 : 0);
    const scenarios = manifest.execution.mode === "latency"
      ? 1
      : definition.workload.scenarios.length;
    if (manifest.execution.mode === "rapid") {
      expected += policyCount * manifest.execution.blocks *
        manifest.execution.measuredRunsPerScenario * scenarios;
    } else if (["validation", "latency"].includes(manifest.execution.mode)) {
      expected += policyCount * manifest.execution.blocks *
        manifest.execution.measuredRunsPerCase * manifest.execution.taskCaseCount * scenarios;
    } else {
      expected = null;
      break;
    }
  }
  return { completed, expected };
}

saveManifest();

function sendJson(response, status, value) {
  response.writeHead(status, { "content-type": "application/json" });
  response.end(JSON.stringify(value));
}

function readJson(request) {
  return new Promise((resolveBody, reject) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => {
      body += chunk;
      if (body.length > 50 * 1024 * 1024) {
        reject(new Error("request body exceeds 50 MB"));
        request.destroy();
      }
    });
    request.on("end", () => {
      try {
        resolveBody(JSON.parse(body || "{}"));
      } catch (error) {
        reject(error);
      }
    });
    request.on("error", reject);
  });
}

function findEntry(id) {
  const entry = manifest.entries.find((candidate) => candidate.id === id);
  if (!entry) throw new Error(`unknown campaign target: ${id}`);
  return entry;
}

function deviceFingerprint(profile) {
  const gpu = profile?.gpu ?? {};
  return JSON.stringify({
    cores: profile?.cores,
    deviceClass: profile?.deviceClass,
    memGB: profile?.memGB,
    platform: profile?.platform,
    userAgent: profile?.userAgent,
    userAgentData: profile?.userAgentData,
    gpu: {
      architecture: gpu.architecture,
      description: gpu.description,
      device: gpu.device,
      features: [...(gpu.features ?? [])].sort(),
      identity: gpu.identity,
      identitySource: gpu.identitySource,
      limits: gpu.limits,
      vendor: gpu.vendor,
      webglRenderer: gpu.webglRenderer,
      webglVendor: gpu.webglVendor,
    },
  });
}

function validatePayload(entry, payload) {
  if (payload?.schema !== "webllm-controlled-experiment-v2") {
    throw new Error("unexpected payload schema");
  }
  if (payload.experiment?.modelConfigId !== entry.modelConfigId) {
    throw new Error("payload model does not match target");
  }
  if (payload.experiment?.taskProfileId !== entry.taskProfileId) {
    throw new Error("payload task does not match target");
  }
  if (payload.experiment?.variedParameter !== entry.parameterId) {
    throw new Error("payload parameter does not match target");
  }
  const expectedLabel =
    `${campaignId}:${entry.modelConfigId}:${entry.taskProfileId}:${entry.parameterId}`;
  if (payload.label !== expectedLabel) {
    throw new Error("payload label does not match target");
  }
  if (
    manifest.includeLibraryDefault &&
    !payload.policies?.some((policy) => policy.baselineType === "library_default")
  ) {
    throw new Error("payload does not include the WebLLM reference");
  }
  if (entry.candidatePolicyId) {
    const policyIds = payload.policies.map((policy) => policy.id).sort();
    const expectedPolicyIds = [
      "controlled-baseline",
      entry.candidatePolicyId,
      ...(manifest.includeLibraryDefault ? ["library-default"] : []),
    ].sort();
    if (JSON.stringify(policyIds) !== JSON.stringify(expectedPolicyIds)) {
      throw new Error("validation payload has unexpected policies");
    }
    if (payload.experiment?.candidatePolicyId !== entry.candidatePolicyId) {
      throw new Error("payload does not match the declared candidate");
    }
    if (payload.experiment?.mode !== manifest.execution.mode) {
      throw new Error("payload mode does not match campaign mode");
    }
    const task = MANUAL_TASK_PROFILES.find(
      (item) => item.id === entry.taskProfileId
    );
    const definition = EXPERIMENT_DEFINITIONS.find(
      (item) => item.id === entry.parameterId
    );
    const repeats =
      manifest.execution.measuredRunsPerCase * manifest.execution.blocks;
    const scenarios = manifest.execution.mode === "latency"
      ? ["single-turn"]
      : definition.workload.scenarios;
    for (const policyId of expectedPolicyIds) {
      for (const scenario of scenarios) {
        for (const taskCase of task.cases) {
          const taskCaseId = scenario === "long-history-required"
            ? `${taskCase.id}-history-required`
            : taskCase.id;
          const count = payload.results.filter((row) =>
            row.policyId === policyId &&
            row.conversationScenario === scenario &&
            row.taskCaseId === taskCaseId
          ).length;
          if (count !== repeats) {
            throw new Error(
              `validation coverage mismatch: ${policyId}/${taskCaseId}/${scenario}`
            );
          }
        }
      }
    }
  }
}

async function campaignRequest(request, response, route) {
  if (request.method === "GET" && route === "manifest") {
    sendJson(response, 200, { ...manifest, inferenceProgress: measuredInferenceProgress() });
    return;
  }
  const body = await readJson(request);
  if (route === "configure") {
    if (manifest.startedAt || manifest.entries.some((entry) => entry.status === "complete")) {
      throw new Error("the queue cannot be changed after collection has started");
    }
    const modelIds = new Set(body.modelIds ?? []);
    const taskIds = new Set(body.taskIds ?? []);
    const parameterIds = new Set(body.parameterIds ?? []);
    if (!modelIds.size || !taskIds.size || !parameterIds.size) {
      throw new Error("select at least one model, task, and parameter");
    }
    const entries = manifest.entries.filter((entry) =>
      modelIds.has(entry.modelConfigId) &&
      taskIds.has(entry.taskProfileId) &&
      parameterIds.has(entry.parameterId)
    );
    if (!entries.length) throw new Error("the selection produced no queue targets");
    manifest.entries = entries;
    manifest.models = manifest.models.filter((model) => modelIds.has(model.id));
    manifest.tasks = manifest.tasks.filter((task) => taskIds.has(task.id));
    manifest.parameters = manifest.parameters.filter((id) => parameterIds.has(id));
    manifest.artifactPreflight = manifest.artifactPreflight.filter((artifact) =>
      modelIds.has(artifact.modelConfigId) &&
      (artifact.quantization === "q4f16_1" || parameterIds.has("weightQuantization"))
    );
    manifest.expectedExperimentCount = entries.filter(
      (entry) => entry.status !== "skipped"
    ).length;
    saveManifest();
    sendJson(response, 200, manifest);
    return;
  }
  if (route === "start") {
    if (manifest.deviceProfile) {
      const current = deviceFingerprint(body.deviceProfile);
      const original = deviceFingerprint(manifest.deviceProfile);
      if (current !== original) throw new Error("device profile changed during campaign");
    } else {
      manifest.deviceProfile = body.deviceProfile;
      manifest.startedAt = new Date().toISOString();
      saveManifest();
    }
    sendJson(response, 200, manifest);
    return;
  }
  if (route === "preflight") {
    const artifact = manifest.artifactPreflight.find((candidate) =>
      candidate.modelConfigId === body.modelConfigId &&
      candidate.quantization === body.quantization
    );
    if (!artifact) throw new Error("unknown model artifact");
    if (!["available", "failed"].includes(body.status)) {
      throw new Error("invalid preflight status");
    }
    artifact.status = body.status;
    artifact.error = body.error || null;
    artifact.checkedAt = new Date().toISOString();
    if (body.status === "failed") {
      for (const target of manifest.entries) {
        const dependsOnArtifact = body.quantization === "q4f16_1"
          ? target.modelConfigId === body.modelConfigId
          : target.modelConfigId === body.modelConfigId &&
            target.parameterId === "weightQuantization";
        if (dependsOnArtifact && target.status === "pending") {
          target.status = "skipped";
          target.error =
            `${body.quantization} preflight failed: ${body.error || "load failed"}`;
        }
      }
    }
    saveManifest();
    sendJson(response, 200, artifact);
    return;
  }
  const entry = findEntry(body.id);
  if (route === "claim") {
    if (entry.status === "complete") throw new Error(`${entry.id} is already complete`);
    entry.status = "running";
    entry.startedAt = new Date().toISOString();
    entry.error = null;
    saveManifest();
    sendJson(response, 200, entry);
    return;
  }
  if (route === "result") {
    validatePayload(entry, body.payload);
    const filename = `${entry.id}.json`;
    writeFileSync(join(runsDirectory, filename), `${JSON.stringify(body.payload, null, 2)}\n`);
    entry.status = "complete";
    entry.payloadFile = `runs/${filename}`;
    entry.completedAt = new Date().toISOString();
    entry.error = null;
    saveManifest();
    sendJson(response, 200, entry);
    return;
  }
  if (route === "error") {
    const errorFile = `${entry.id}-attempt-${body.attempt}.json`;
    writeFileSync(join(errorsDirectory, errorFile), `${JSON.stringify(body, null, 2)}\n`);
    entry.status = body.final ? "failed" : "pending";
    entry.error = body.message || "unknown campaign error";
    entry.completedAt = body.final ? new Date().toISOString() : null;
    saveManifest();
    sendJson(response, 200, entry);
    return;
  }
  throw new Error(`unknown campaign route: ${route}`);
}

function staticRequest(request, response) {
  const requestPath = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
  const requested = requestPath === "/" ? "/index.html" : requestPath;
  const path = normalize(join(root, requested));
  if (relative(root, path).startsWith("..") || !existsSync(path) || !statSync(path).isFile()) {
    response.writeHead(404);
    response.end("not found");
    return;
  }
  response.writeHead(200, {
    "content-type": contentTypes[extname(path)] ?? "application/octet-stream",
    "cache-control": "no-store",
  });
  response.end(readFileSync(path));
}

const server = createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, "http://localhost").pathname;
    if (pathname.startsWith("/__campaign/")) {
      await campaignRequest(request, response, pathname.slice("/__campaign/".length));
    } else {
      staticRequest(request, response);
    }
  } catch (error) {
    sendJson(response, 400, { error: error.message });
  }
});

server.listen(port, "0.0.0.0", () => {
  process.stdout.write(
    `Campaign ${campaignId}: http://localhost:${port}/gpu/manual-tune.html\n`
  );
  process.stdout.write(
    `${manifest.expectedExperimentCount} supported targets, ${manifest.entries.length} total entries\n`
  );
});
