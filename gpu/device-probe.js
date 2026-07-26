import { getWebGpuProfile } from "./device-profile.js";
import { median, quantile } from "../shared/benchmark-utils.js";

const WORKLOADS = [
  { workloadType: "synthetic-matmul", kernel: "matmul", M: 128, N: 128, K: 128, dtype: "f32" },
  { workloadType: "synthetic-matmul", kernel: "matmul", M: 256, N: 256, K: 256, dtype: "f32" },
  { workloadType: "synthetic-matmul", kernel: "matmul", M: 512, N: 512, K: 512, dtype: "f32" },
];

const PROBE_SHAPES = [
  { id: "wg8x8-f32", kernelVariant: "one-output-per-thread", workgroupSizeX: 8, workgroupSizeY: 8, workgroupSizeZ: 1, tileM: 8, tileN: 8, tileK: 1, dtype: "f32", shaderF16: false },
  { id: "wg16x8-f32", kernelVariant: "one-output-per-thread", workgroupSizeX: 16, workgroupSizeY: 8, workgroupSizeZ: 1, tileM: 16, tileN: 8, tileK: 1, dtype: "f32", shaderF16: false },
  { id: "wg8x16-f32", kernelVariant: "one-output-per-thread", workgroupSizeX: 8, workgroupSizeY: 16, workgroupSizeZ: 1, tileM: 8, tileN: 16, tileK: 1, dtype: "f32", shaderF16: false },
  { id: "wg16x16-f32", kernelVariant: "one-output-per-thread", workgroupSizeX: 16, workgroupSizeY: 16, workgroupSizeZ: 1, tileM: 16, tileN: 16, tileK: 1, dtype: "f32", shaderF16: false },
];

const WARMUP_RUNS = 2;
const MEASURED_RUNS = 8;

function classifyProbeError(err, stage = "probe") {
  const message = err?.message ?? String(err);
  const lower = message.toLowerCase();
  if (
    lower.includes("failed to fetch") ||
    lower.includes("err_timed_out") ||
    lower.includes("networkerror") ||
    lower.includes("fetch")
  ) {
    return { errorStage: stage, failureClass: "network_unstable", failureKeep: false };
  }
  if (lower.includes("out of memory") || lower.includes("oom") || lower.includes("memory")) {
    return { errorStage: stage, failureClass: "oom", failureKeep: true };
  }
  if (lower.includes("webgpu") || lower.includes("adapter") || lower.includes("gpu")) {
    return { errorStage: stage, failureClass: "webgpu_unsupported", failureKeep: true };
  }
  return { errorStage: stage, failureClass: "runtime_exception", failureKeep: true };
}

function stats(samples) {
  const sum = samples.reduce((a, b) => a + b, 0);
  return {
    medianMs: median(samples),
    p95Ms: quantile(samples, 0.95),
    meanMs: sum / samples.length,
    minMs: Math.min(...samples),
    maxMs: Math.max(...samples),
  };
}

function makeShader(config) {
  return `
struct Matrix {
  data: array<f32>,
};

struct Dims {
  m: u32,
  n: u32,
  k: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read> b: Matrix;
@group(0) @binding(2) var<storage, read_write> c: Matrix;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(${config.workgroupSizeX}, ${config.workgroupSizeY}, ${config.workgroupSizeZ})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= dims.m || col >= dims.n) {
    return;
  }

  var sum = 0.0;
  for (var kk = 0u; kk < dims.k; kk = kk + 1u) {
    sum = sum + a.data[row * dims.k + kk] * b.data[kk * dims.n + col];
  }
  c.data[row * dims.n + col] = sum;
}
`;
}

function makeInput(size, seed) {
  const out = new Float32Array(size);
  let x = seed >>> 0;
  for (let i = 0; i < out.length; i++) {
    x = (1664525 * x + 1013904223) >>> 0;
    out[i] = ((x & 0xffff) / 0xffff) - 0.5;
  }
  return out;
}

function sampledMaxAbsError(a, b, output, workload) {
  const checks = [
    [0, 0],
    [0, workload.N - 1],
    [Math.floor(workload.M / 2), Math.floor(workload.N / 2)],
    [workload.M - 1, 0],
    [workload.M - 1, workload.N - 1],
  ];
  let maxErr = 0;
  for (const [row, col] of checks) {
    let expected = 0;
    for (let kk = 0; kk < workload.K; kk++) {
      expected += a[row * workload.K + kk] * b[kk * workload.N + col];
    }
    const actual = output[row * workload.N + col];
    if (!Number.isFinite(actual)) return Infinity;
    maxErr = Math.max(maxErr, Math.abs(actual - expected));
  }
  return maxErr;
}

function createBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: align4(data.byteLength),
    usage,
    mappedAtCreation: true,
  });
  new data.constructor(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

function align4(n) {
  return Math.ceil(n / 4) * 4;
}

async function runGpuMatmul(device, pipeline, bindGroup, workload, config) {
  const commandEncoder = device.createCommandEncoder();
  const pass = commandEncoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(
    Math.ceil(workload.N / config.workgroupSizeX),
    Math.ceil(workload.M / config.workgroupSizeY),
    1
  );
  pass.end();

  const start = performance.now();
  device.queue.submit([commandEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - start;
}

async function readOutput(device, outputBuffer, byteLength) {
  const readBuffer = device.createBuffer({
    size: align4(byteLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, align4(byteLength));
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  readBuffer.destroy();
  return out;
}

async function runCandidate(device, workload, config) {
  const a = makeInput(workload.M * workload.K, 17);
  const b = makeInput(workload.K * workload.N, 29);
  const outputBytes = workload.M * workload.N * Float32Array.BYTES_PER_ELEMENT;
  const dims = new Uint32Array([workload.M, workload.N, workload.K, 0]);

  const inputUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const outputUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const uniformUsage = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

  let aBuffer;
  let bBuffer;
  let cBuffer;
  let dimsBuffer;
  try {
    aBuffer = createBuffer(device, a, inputUsage);
    bBuffer = createBuffer(device, b, inputUsage);
    cBuffer = device.createBuffer({ size: align4(outputBytes), usage: outputUsage });
    dimsBuffer = createBuffer(device, dims, uniformUsage);

    const compileStart = performance.now();
    const module = device.createShaderModule({ code: makeShader(config) });
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const pipelineCreateMs = performance.now() - compileStart;

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: cBuffer } },
        { binding: 3, resource: { buffer: dimsBuffer } },
      ],
    });

    for (let i = 0; i < WARMUP_RUNS; i++) {
      await runGpuMatmul(device, pipeline, bindGroup, workload, config);
    }

    const samples = [];
    for (let i = 0; i < MEASURED_RUNS; i++) {
      samples.push(await runGpuMatmul(device, pipeline, bindGroup, workload, config));
    }

    const output = await readOutput(device, cBuffer, outputBytes);
    const maxAbsError = sampledMaxAbsError(a, b, output, workload);
    const s = stats(samples);
    const gflops = (2 * workload.M * workload.N * workload.K) / (s.medianMs / 1000) / 1e9;

    return {
      success: maxAbsError < 1e-2,
      errorType: maxAbsError < 1e-2 ? "" : "validation_failed",
      pipelineCreateMs,
      ...s,
      gflops,
      maxAbsError,
    };
  } finally {
    aBuffer?.destroy();
    bBuffer?.destroy();
    cBuffer?.destroy();
    dimsBuffer?.destroy();
  }
}

export async function runDeviceProbeCore({
  label = "",
  deviceProfile = null,
  logFn = () => {},
  statusFn = () => {},
  progressFn = () => {},
} = {}) {
  statusFn("profiling device...");
  const timestamp = new Date().toISOString();
  const profile = deviceProfile ?? await getWebGpuProfile();
  if (!profile.webgpuAvailable || profile.gpu?.error) {
    throw new Error(profile.gpu?.error ?? "WebGPU is not available");
  }

  logFn(`device: ${profile.gpu.vendor || "?"}/${profile.gpu.architecture || "?"}`);
  logFn(`features: ${profile.gpu.features.join(", ") || "(none reported)"}`);

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter");
  const device = await adapter.requestDevice();
  const rows = [];
  const totalCandidates = WORKLOADS.length * PROBE_SHAPES.length;
  let completedCandidates = 0;

  for (const workload of WORKLOADS) {
    logFn(`\nworkload: ${workload.M}x${workload.N}x${workload.K}`);
    for (const config of PROBE_SHAPES) {
      progressFn({
        phase: "probe",
        completed: completedCandidates,
        total: totalCandidates,
        label: `${workload.M}x${workload.N} ${config.id}`,
      });
      statusFn(`probing ${workload.M}x${workload.N}x${workload.K} ${config.id}...`);
      const startedAt = performance.now();
      try {
        const result = await runCandidate(device, workload, config);
        rows.push({ workload, config, result });
        logFn(
          `${config.id}: ${result.medianMs.toFixed(3)}ms median, ` +
          `${result.p95Ms.toFixed(3)}ms p95, ${result.gflops.toFixed(2)} GFLOP/s, ` +
          `err=${result.maxAbsError.toExponential(2)}`
        );
      } catch (err) {
        const classification = classifyProbeError(err);
        rows.push({
          workload,
          config,
          result: {
            success: false,
            errorType: err.message,
            errorDetail: err.stack ?? String(err),
            errorStage: classification.errorStage,
            failureClass: classification.failureClass,
            failureKeep: classification.failureKeep,
            pipelineCreateMs: performance.now() - startedAt,
            medianMs: null,
            p95Ms: null,
            meanMs: null,
            minMs: null,
            maxMs: null,
            gflops: null,
            maxAbsError: null,
          },
        });
        logFn(`${config.id}: ERROR ${err.message}`);
      }
      completedCandidates += 1;
      progressFn({
        phase: "probe",
        completed: completedCandidates,
        total: totalCandidates,
        label: `${workload.M}x${workload.N} ${config.id}`,
      });
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  device.destroy();

  const probeSummary = deriveProbeSummary(rows);
  const payload = {
    timestamp,
    label,
    schema: "webgpu-device-probe-v2",
    warmupRuns: WARMUP_RUNS,
    measuredRuns: MEASURED_RUNS,
    deviceProfile: profile,
    results: rows,
    probeSummary,
  };

  return payload;
}

function deriveProbeSummary(rows) {
  const workloads = WORKLOADS.map((workload) => {
    const candidates = rows
      .filter((row) => sameWorkload(row.workload, workload) && row.result.success)
      .sort((a, b) => a.result.medianMs - b.result.medianMs);
    const defaultRow = rows.find(
      (row) => sameWorkload(row.workload, workload) && row.config.id === PROBE_SHAPES[0].id
    );
    const best = candidates[0] ?? null;
    return {
      workload,
      bestShapeId: best?.config.id ?? "",
      bestMedianMs: best?.result.medianMs ?? null,
      bestGflops: best?.result.gflops ?? null,
      defaultShapeId: PROBE_SHAPES[0].id,
      defaultMedianMs: defaultRow?.result.medianMs ?? null,
      speedupVsDefault:
        best?.result.medianMs && defaultRow?.result.medianMs
          ? defaultRow.result.medianMs / best.result.medianMs
          : null,
    };
  });
  const bySize = Object.fromEntries(
    workloads.map((item) => [
      String(item.workload.M),
      {
        bestShapeId: item.bestShapeId,
        gflops: item.bestGflops,
        medianMs: item.bestMedianMs,
        speedupVsDefault: item.speedupVsDefault,
      },
    ])
  );
  return {
    schema: "webgpu-device-probe-summary-v2",
    timestamp: new Date().toISOString(),
    success: workloads.every((item) => Boolean(item.bestShapeId)),
    errorType: workloads.every((item) => Boolean(item.bestShapeId)) ? "" : "missing_successful_probe",
    bySize,
    workloads,
  };
}

function sameWorkload(a, b) {
  return a.M === b.M && a.N === b.N && a.K === b.K && a.kernel === b.kernel;
}
