#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const WEBLLM_VERSION = "0.2.79";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outputDir = path.join(root, "vendor", "generated");
const outputFile = path.join(outputDir, `webllm-${WEBLLM_VERSION}-runtime-orchestration.js`);

function replaceOnce(source, before, after, label) {
  const first = source.indexOf(before);
  if (first === -1) throw new Error(`WebLLM patch point not found: ${label}`);
  if (source.indexOf(before, first + before.length) !== -1) {
    throw new Error(`WebLLM patch point is ambiguous: ${label}`);
  }
  return source.slice(0, first) + after + source.slice(first + before.length);
}

export function patchWebLlmBundle(input) {
  let source = input.replace(/\r\n/g, "\n");

  source = replaceOnce(
    source,
    `            this.debugLogFinish = false;\n            this.memory = memory;\n            this.device = device;\n        }\n        /**\n         * Dispose context.`,
    `            this.debugLogFinish = false;\n            this.memory = memory;\n            this.device = device;\n            // WebLLM orchestration experiment: baseline remains the default.\n            this.orchestrationPolicy = { dispatchesPerEncoder: 1 };\n            this.orchestrationEncoder = null;\n            this.orchestrationDispatchCount = 0;\n            this.orchestrationBufferPool = [];\n            this.orchestrationBufferPoolSizes = [];\n            this.orchestrationBufferIndex = 0;\n            this.resetOrchestrationStats();\n            globalThis.__WEBLLM_ORCHESTRATION_EXPERIMENT__ = {\n                version: "${WEBLLM_VERSION}",\n                setPolicy: (policy) => __awaiter(this, void 0, void 0, function* () {\n                    yield this.sync();\n                    this.destroyOrchestrationBufferPool();\n                    const requested = policy === null || policy === void 0 ? void 0 : policy.dispatchesPerEncoder;\n                    if (requested !== null && ![1, 4, 16].includes(requested)) {\n                        throw Error("dispatchesPerEncoder must be 1, 4, 16, or null");\n                    }\n                    this.orchestrationPolicy = { dispatchesPerEncoder: requested };\n                    this.resetOrchestrationStats();\n                    return Object.assign({}, this.orchestrationPolicy);\n                }),\n                resetStats: () => this.resetOrchestrationStats(),\n                getStats: () => JSON.parse(JSON.stringify(this.orchestrationStats)),\n                getPolicy: () => Object.assign({}, this.orchestrationPolicy),\n            };\n        }\n        resetOrchestrationStats() {\n            this.orchestrationStats = {\n                dispatchesPerEncoder: this.orchestrationPolicy.dispatchesPerEncoder,\n                dispatches: 0,\n                commandEncoders: 0,\n                commandBuffers: 0,\n                submitCalls: 0,\n                flushes: 0,\n                flushReasons: {},\n                uniformBuffersCreated: 0,\n                recordingMs: 0,\n                submitMs: 0,\n                traceHash: 2166136261,\n            };\n        }\n        recordOrchestrationDispatch(name, x, y, z) {\n            const text = name + ":" + x + ":" + y + ":" + z + ";";\n            let hash = this.orchestrationStats.traceHash >>> 0;\n            for (let i = 0; i < text.length; ++i) {\n                hash ^= text.charCodeAt(i);\n                hash = Math.imul(hash, 16777619) >>> 0;\n            }\n            this.orchestrationStats.traceHash = hash;\n            this.orchestrationStats.dispatches += 1;\n        }\n        createOrchestrationEncoder() {\n            this.orchestrationStats.commandEncoders += 1;\n            return this.device.createCommandEncoder();\n        }\n        getOrchestrationUniformBuffer(nbytes) {\n            const index = this.orchestrationBufferIndex++;\n            let buffer = this.orchestrationBufferPool[index];\n            if (buffer !== undefined && this.orchestrationBufferPoolSizes[index] < nbytes) {\n                buffer.destroy();\n                buffer = undefined;\n            }\n            if (buffer === undefined) {\n                let allocSize = 16;\n                while (allocSize < nbytes) allocSize *= 2;\n                buffer = tryCreateBuffer(this.device, {\n                    size: allocSize,\n                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,\n                });\n                this.orchestrationBufferPool[index] = buffer;\n                this.orchestrationBufferPoolSizes[index] = allocSize;\n                this.orchestrationStats.uniformBuffersCreated += 1;\n            }\n            return buffer;\n        }\n        destroyOrchestrationBufferPool() {\n            while (this.orchestrationBufferPool.length !== 0) {\n                this.orchestrationBufferPool.pop().destroy();\n            }\n            this.orchestrationBufferPoolSizes = [];\n            this.orchestrationBufferIndex = 0;\n        }\n        flushOrchestrationCommands(reason) {\n            if (this.orchestrationEncoder === null) return;\n            const finishStart = performance.now();\n            const command = this.orchestrationEncoder.finish();\n            this.orchestrationStats.recordingMs += performance.now() - finishStart;\n            this.orchestrationStats.commandBuffers += 1;\n            const submitStart = performance.now();\n            this.device.queue.submit([command]);\n            this.orchestrationStats.submitMs += performance.now() - submitStart;\n            this.orchestrationStats.submitCalls += 1;\n            this.orchestrationStats.flushes += 1;\n            this.orchestrationStats.flushReasons[reason] = (this.orchestrationStats.flushReasons[reason] || 0) + 1;\n            this.orchestrationEncoder = null;\n            this.orchestrationDispatchCount = 0;\n            this.orchestrationBufferIndex = 0;\n        }\n        /**\n         * Dispose context.`,
    "WebGPUContext constructor"
  );
  source = source.replace(
    "                    yield this.sync();\n                    this.destroyOrchestrationBufferPool();\n                    const requested",
    "                    yield this.sync();\n                    const requested"
  );
  source = replaceOnce(
    source,
    "this.orchestrationPolicy = { dispatchesPerEncoder: 1 };",
    "this.orchestrationPolicy = { dispatchesPerEncoder: 1, reuseUniformBuffers: false };",
    "default reuse policy"
  );
  source = replaceOnce(
    source,
    `                    if (requested !== null && ![1, 4, 16].includes(requested)) {\n                        throw Error("dispatchesPerEncoder must be 1, 4, 16, or null");\n                    }\n                    this.orchestrationPolicy = { dispatchesPerEncoder: requested };`,
    `                    if (requested !== null && ![1, 2, 4, 8, 16, 32, 64].includes(requested)) {\n                        throw Error("dispatchesPerEncoder must be 1, 2, 4, 8, 16, 32, 64, or null");\n                    }\n                    this.orchestrationPolicy = {\n                        dispatchesPerEncoder: requested,\n                        reuseUniformBuffers: Boolean(policy === null || policy === void 0 ? void 0 : policy.reuseUniformBuffers),\n                    };`,
    "policy validation and reuse"
  );
  source = replaceOnce(
    source,
    `                dispatchesPerEncoder: this.orchestrationPolicy.dispatchesPerEncoder,\n                dispatches: 0,`,
    `                dispatchesPerEncoder: this.orchestrationPolicy.dispatchesPerEncoder,\n                reuseUniformBuffers: this.orchestrationPolicy.reuseUniformBuffers,\n                dispatches: 0,`,
    "reuse stats"
  );

  source = replaceOnce(
    source,
    `        dispose() {\n            var _a, _b, _c;`,
    `        dispose() {\n            var _a, _b, _c;\n            this.flushOrchestrationCommands("dispose");\n            this.destroyOrchestrationBufferPool();`,
    "dispose flush"
  );

  source = replaceOnce(
    source,
    `        sync() {\n            return __awaiter(this, void 0, void 0, function* () {\n                yield this.device.queue.onSubmittedWorkDone();`,
    `        sync() {\n            return __awaiter(this, void 0, void 0, function* () {\n                this.flushOrchestrationCommands("sync");\n                yield this.device.queue.onSubmittedWorkDone();`,
    "sync flush"
  );

  source = replaceOnce(
    source,
    `        copyRawBytesToBuffer(rawBytes, toPtr, toOffset, nbytes) {\n            // Perhaps it would be more useful to use a staging buffer?`,
    `        copyRawBytesToBuffer(rawBytes, toPtr, toOffset, nbytes) {\n            this.flushOrchestrationCommands("raw-write");\n            // Perhaps it would be more useful to use a staging buffer?`,
    "raw write flush"
  );

  source = replaceOnce(
    source,
    `                    const commandEncoder = this.device.createCommandEncoder();\n                    const compute = commandEncoder.beginComputePass();`,
    `                    const recordingStart = performance.now();\n                    const batching = this.orchestrationPolicy.dispatchesPerEncoder !== 1;\n                    const commandEncoder = batching\n                        ? (this.orchestrationEncoder || (this.orchestrationEncoder = this.createOrchestrationEncoder()))\n                        : this.createOrchestrationEncoder();\n                    const compute = commandEncoder.beginComputePass();`,
    "compute encoder"
  );
  source = replaceOnce(
    source,
    `                    const batching = this.orchestrationPolicy.dispatchesPerEncoder !== 1;\n                    const commandEncoder`,
    `                    const batching = this.orchestrationPolicy.dispatchesPerEncoder !== 1;\n                    const reuseUniformBuffers = this.orchestrationPolicy.reuseUniformBuffers;\n                    const commandEncoder`,
    "reuse dispatch switch"
  );

  source = replaceOnce(
    source,
    `                    const podArgBuffer = this.getPodArgsBuffer((podArgIndices.length + 1) * sizeOfI32);`,
    `                    const podArgBytes = (podArgIndices.length + 1) * sizeOfI32;\n                    const podArgBuffer = batching\n                        ? this.getOrchestrationUniformBuffer(podArgBytes)\n                        : this.getPodArgsBuffer(podArgBytes);`,
    "uniform buffer selection"
  );
  source = replaceOnce(
    source,
    `                    const podArgBuffer = batching\n                        ? this.getOrchestrationUniformBuffer(podArgBytes)`,
    `                    const podArgBuffer = reuseUniformBuffers\n                        ? this.getOrchestrationUniformBuffer(podArgBytes)`,
    "independent uniform reuse"
  );

  source = replaceOnce(
    source,
    `                    compute.dispatchWorkgroups(workDim[0], workDim[1], workDim[2]);\n                    compute.end();\n                    const command = commandEncoder.finish();\n                    this.device.queue.submit([command]);\n                    if (this.debugLogFinish) {`,
    `                    compute.dispatchWorkgroups(workDim[0], workDim[1], workDim[2]);\n                    compute.end();\n                    this.recordOrchestrationDispatch(finfo.name, workDim[0], workDim[1], workDim[2]);\n                    this.orchestrationStats.recordingMs += performance.now() - recordingStart;\n                    if (batching) {\n                        this.orchestrationDispatchCount += 1;\n                        const limit = this.orchestrationPolicy.dispatchesPerEncoder;\n                        if (limit !== null && this.orchestrationDispatchCount >= limit) {\n                            this.flushOrchestrationCommands("batch-limit");\n                        }\n                    }\n                    else {\n                        const command = commandEncoder.finish();\n                        this.orchestrationStats.commandBuffers += 1;\n                        const submitStart = performance.now();\n                        this.device.queue.submit([command]);\n                        this.orchestrationStats.submitMs += performance.now() - submitStart;\n                        this.orchestrationStats.submitCalls += 1;\n                    }\n                    if (this.debugLogFinish) {\n                        this.flushOrchestrationCommands("debug-finish");`,
    "compute submission"
  );
  source = source.replace(
    `                    else {\n                        const command = commandEncoder.finish();\n                        this.orchestrationStats.commandBuffers += 1;`,
    `                    else {\n                        const finishStart = performance.now();\n                        const command = commandEncoder.finish();\n                        this.orchestrationStats.recordingMs += performance.now() - finishStart;\n                        this.orchestrationStats.commandBuffers += 1;`
  );
  source = replaceOnce(
    source,
    `                        this.orchestrationStats.submitCalls += 1;\n                    }\n                    if (this.debugLogFinish) {`,
    `                        this.orchestrationStats.submitCalls += 1;\n                        if (reuseUniformBuffers) this.orchestrationBufferIndex = 0;\n                    }\n                    if (this.debugLogFinish) {`,
    "reuse-only slot retirement"
  );

  source = replaceOnce(
    source,
    `        deviceFreeDataSpace(ptr) {\n            const idx = ptr;`,
    `        deviceFreeDataSpace(ptr) {\n            this.flushOrchestrationCommands("buffer-free");\n            const idx = ptr;`,
    "free flush"
  );

  source = replaceOnce(
    source,
    `        deviceCopyToGPU(from, to, toOffset, nbytes) {\n            // Perhaps it would be more useful to use a staging buffer?`,
    `        deviceCopyToGPU(from, to, toOffset, nbytes) {\n            this.flushOrchestrationCommands("cpu-to-gpu");\n            // Perhaps it would be more useful to use a staging buffer?`,
    "CPU to GPU flush"
  );

  source = replaceOnce(
    source,
    `        deviceCopyFromGPU(from, fromOffset, to, nbytes) {\n            // Perhaps it would be more useful to resuse a staging buffer?`,
    `        deviceCopyFromGPU(from, fromOffset, to, nbytes) {\n            this.flushOrchestrationCommands("gpu-to-cpu");\n            // Perhaps it would be more useful to resuse a staging buffer?`,
    "GPU to CPU flush"
  );

  source = replaceOnce(
    source,
    `        deviceCopyWithinGPU(from, fromOffset, to, toOffset, nbytes) {\n            const copyEncoder = this.device.createCommandEncoder();`,
    `        deviceCopyWithinGPU(from, fromOffset, to, toOffset, nbytes) {\n            this.flushOrchestrationCommands("gpu-to-gpu");\n            const copyEncoder = this.device.createCommandEncoder();`,
    "GPU to GPU flush"
  );

  source = source.replace("//# sourceMappingURL=index.js.map", "// Patched locally by scripts/build-runtime-orchestration.mjs");
  return source;
}

export function buildRuntimeOrchestration() {
  const work = mkdtempSync(path.join(tmpdir(), "webllm-orchestration-"));
  try {
    const tarball = execFileSync(
      "npm",
      ["pack", `@mlc-ai/web-llm@${WEBLLM_VERSION}`, "--silent"],
      { cwd: work, encoding: "utf8" }
    ).trim().split(/\r?\n/).at(-1);
    execFileSync("tar", ["-xzf", tarball], { cwd: work });
    const upstream = readFileSync(path.join(work, "package", "lib", "index.js"), "utf8");
    const patched = patchWebLlmBundle(upstream);
    mkdirSync(outputDir, { recursive: true });
    writeFileSync(outputFile, patched);
    const metadata = {
      webLlmVersion: WEBLLM_VERSION,
      upstreamSha256: createHash("sha256").update(upstream).digest("hex"),
      patchedSha256: createHash("sha256").update(patched).digest("hex"),
      output: path.relative(root, outputFile),
    };
    writeFileSync(path.join(outputDir, "runtime-orchestration-build.json"), JSON.stringify(metadata, null, 2) + "\n");
    return metadata;
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.stdout.write(JSON.stringify(buildRuntimeOrchestration(), null, 2) + "\n");
}
