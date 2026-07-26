import * as np from "https://esm.sh/numpy-ts@1.4.0";
import {
  LOOP_ORDER_KERNELS,
  matmulIJK,
  matmulIKJ,
  matmulIKJUnrolled,
  matmulTiled,
  matmulTransposedB,
  transposeB,
} from "./matmul.js";
import {
  matvecFloat32,
  matvecInt4,
  matvecInt8,
  quantizeInt4,
  quantizeInt8,
  residualRmsNormFused,
  residualRmsNormSeparate,
  rmsNormBuffered,
  rmsNormDirect,
  rmsNormUnrolled4,
  softmaxBuffered,
  softmaxOnline,
  softmaxStable,
} from "./llm-ops.js";
import { MatmulWorkerPool } from "./worker-pool.js";
import {
  calibrateIters,
  stats,
  timeBatched,
} from "./bench.js";

const MEASURED_ROUNDS = 6;
const TARGET_BATCH_MS = 40;
const TILE_SIZES = [8, 16, 32, 64, 128];
const UNROLL_FACTORS = [1, 2, 4, 8];
const SHAPES = {
  "square-128": { M: 128, K: 128, N: 128, workload: "square" },
  "square-256": { M: 256, K: 256, N: 256, workload: "square" },
  "square-512": { M: 512, K: 512, N: 512, workload: "square" },
  "decode-1024": { M: 1, K: 1024, N: 1024, workload: "decode-like" },
  "prefill-32": { M: 32, K: 512, N: 512, workload: "short-prefill" },
  "prefill-128": { M: 128, K: 512, N: 512, workload: "long-prefill" },
  wide: { M: 64, K: 256, N: 1024, workload: "wide-output" },
  tall: { M: 64, K: 1024, N: 256, workload: "large-reduction" },
  "matvec-512": { M: 1, K: 512, N: 512, workload: "decode-matvec" },
  "matvec-1024": { M: 1, K: 1024, N: 1024, workload: "decode-matvec" },
  "norm-decode": { M: 1, K: 1024, N: 1, workload: "decode-normalization" },
  "norm-prefill": { M: 128, K: 1024, N: 1, workload: "prefill-normalization" },
  "softmax-512": { M: 1, K: 512, N: 1, workload: "short-context-softmax" },
  "softmax-2048": { M: 1, K: 2048, N: 1, workload: "long-context-softmax" },
  "softmax-batch": { M: 32, K: 1024, N: 1, workload: "batched-softmax" },
};
const FAMILY_BASELINES = {
  "legacy-baseline": "handrolled-ijk",
  "loop-order": "ijk",
  "tile-size": "direct-ikj",
  "weight-layout": "direct-ikj",
  "loop-unroll": "direct-ikj",
  "output-allocation": "allocate-output",
  "numeric-type": "float32",
  "worker-count": "main-thread",
  "quantized-matvec": "float32",
  rmsnorm: "direct",
  softmax: "stable-recompute",
  "fused-residual-rmsnorm": "separate",
};

let checksum = 0;

function seededValues(length, seed, ArrayType = Float32Array, integer = false) {
  let state = seed >>> 0;
  const values = new ArrayType(length);
  for (let index = 0; index < length; index++) {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    values[index] = integer ? (state % 7) - 3 : (state >>> 0) / 0xffffffff - 0.5;
  }
  return values;
}

function shapeData(shape, seed = 42, ArrayType = Float32Array, integer = false) {
  return {
    A: seededValues(shape.M * shape.K, seed, ArrayType, integer),
    B: seededValues(shape.K * shape.N, seed + 1, ArrayType, integer),
  };
}

function maximumDifference(actual, expected) {
  let maximum = 0;
  for (let index = 0; index < actual.length; index++) {
    maximum = Math.max(maximum, Math.abs(actual[index] - expected[index]));
  }
  return maximum;
}

function rotatingOrder(variants, round) {
  const source = round % 2 ? [...variants].reverse() : variants;
  const rotation = Math.floor(round / 2) % source.length;
  return source.slice(rotation).concat(source.slice(0, rotation));
}

function resultRow({
  family,
  shapeId,
  shape,
  variant,
  samples,
  iters,
  maxDiff,
}) {
  const summary = stats(samples);
  const operations = variant.operations ?? 2 * shape.M * shape.K * shape.N;
  const operationUnit = variant.operationUnit ?? "flop";
  const seconds = summary.median / 1000;
  return {
    family,
    shapeId,
    workload: shape.workload,
    M: shape.M,
    K: shape.K,
    N: shape.N,
    variant: variant.id,
    parameter: variant.parameter ?? "",
    value: variant.value ?? "",
    numericType: variant.numericType ?? "float32",
    allocationMode: variant.allocationMode ?? "reuse",
    workerCount: variant.workerCount ?? 0,
    setupMs: variant.setupMs ?? 0,
    iters,
    measuredRounds: samples.length,
    maxDiff,
    samples,
    operations,
    operationUnit,
    medianGflops: operationUnit === "flop" ? operations / seconds / 1e9 : null,
    medianMillionUnitsPerSecond: operations / seconds / 1e6,
    ...summary,
  };
}

function summarizeResults(results) {
  const grouped = new Map();
  for (const result of results) {
    const key = `${result.family}:${result.shapeId}`;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(result);
  }
  return [...grouped.entries()].map(([id, rows]) => {
    const ranked = [...rows].sort((left, right) => left.median - right.median);
    const winner = ranked[0];
    const baseline = rows.find((row) => row.variant === FAMILY_BASELINES[row.family]) ?? winner;
    const deltaVsBaselinePct = baseline.median > 0
      ? (winner.median / baseline.median - 1) * 100
      : null;
    const minimumGainPct = Math.max(10, (baseline.iqrPct ?? 0) * 100, (winner.iqrPct ?? 0) * 100);
    const perCallSavingsMs = baseline.median - winner.median;
    const extraSetupMs = Math.max(0, winner.setupMs - baseline.setupMs);
    const breakEvenCalls = perCallSavingsMs > 0 && extraSetupMs > 0
      ? Math.ceil(extraSetupMs / perCallSavingsMs)
      : 0;
    return {
      id,
      family: winner.family,
      shapeId: winner.shapeId,
      baselineVariant: baseline.variant,
      winnerVariant: winner.variant,
      baselineMedianMs: baseline.median,
      winnerMedianMs: winner.median,
      deltaVsBaselinePct,
      minimumRequiredGainPct: minimumGainPct,
      breakEvenCalls,
      recommendedVariant: deltaVsBaselinePct <= -minimumGainPct ? winner.variant : "",
    };
  });
}

async function benchmarkSyncFamily({
  family,
  shapeId,
  variants,
  log: reportLog,
  status: reportStatus,
}) {
  const shape = SHAPES[shapeId];
  const reference = variants[0].reference;
  const calibration = new Map();
  const differences = new Map();
  const samples = new Map(variants.map((variant) => [variant.id, []]));

  for (const variant of variants) {
    reportStatus(`checking ${family}/${shapeId}/${variant.id}...`);
    const output = variant.run();
    checksum += Number(output[0] ?? 0);
    const maxDiff = maximumDifference(output, variant.reference ?? reference);
    if (maxDiff > (variant.tolerance ?? 1e-3)) {
      throw new Error(`${family}/${shapeId}/${variant.id} failed correctness: ${maxDiff}`);
    }
    differences.set(variant.id, maxDiff);
    const timed = () => {
      const result = variant.run();
      checksum += Number(result[0] ?? 0);
    };
    calibration.set(variant.id, { timed, iters: calibrateIters(timed, TARGET_BATCH_MS).iters });
  }

  for (let round = 0; round < MEASURED_ROUNDS; round++) {
    for (const variant of rotatingOrder(variants, round)) {
      reportStatus(`${family}/${shapeId}: round ${round + 1}/${MEASURED_ROUNDS}, ${variant.id}`);
      const { timed, iters } = calibration.get(variant.id);
      samples.get(variant.id).push(timeBatched(timed, iters));
      if (globalThis.scheduler?.yield) await globalThis.scheduler.yield();
      else await Promise.resolve();
    }
  }

  return variants.map((variant) => {
    const row = resultRow({
      family,
      shapeId,
      shape,
      variant,
      samples: samples.get(variant.id),
      iters: calibration.get(variant.id).iters,
      maxDiff: differences.get(variant.id),
    });
    const throughput = row.operationUnit === "flop"
      ? `${row.medianGflops.toFixed(2)} GFLOP/s`
      : `${row.medianMillionUnitsPerSecond.toFixed(2)}M ${row.operationUnit}s/s`;
    reportLog(`${family}/${shapeId}/${variant.id}: ${row.median.toFixed(3)} ms, ${throughput}`);
    return row;
  });
}

async function benchmarkWorkerFamily(shapeId, logFn, statusFn) {
  const family = "worker-count";
  const shape = SHAPES[shapeId];
  const { A, B } = shapeData(shape, 700);
  const reference = matmulIJK(A, B, shape.M, shape.K, shape.N);
  const counts = [...new Set([1, 2, 4, Math.min(8, navigator.hardwareConcurrency ?? 4)])]
    .filter((count) => count <= shape.M);
  const variants = [{
    id: "main-thread",
    parameter: "workerCount",
    value: 0,
    workerCount: 0,
    setupMs: 0,
    run: async () => matmulIKJ(A, B, shape.M, shape.K, shape.N),
  }];
  const pools = [];
  for (const count of counts) {
    statusFn(`starting ${count} worker${count === 1 ? "" : "s"}...`);
    const started = performance.now();
    const pool = new MatmulWorkerPool(count, B, shape.K, shape.N);
    await pool.ready;
    pools.push(pool);
    variants.push({
      id: `workers-${count}`,
      parameter: "workerCount",
      value: count,
      workerCount: count,
      setupMs: performance.now() - started,
      run: () => pool.multiply(A, shape.M),
    });
  }

  const samples = new Map(variants.map((variant) => [variant.id, []]));
  const differences = new Map();
  try {
    for (const variant of variants) {
      const output = await variant.run();
      const maxDiff = maximumDifference(output, reference);
      if (maxDiff > 1e-3) throw new Error(`${family}/${shapeId}/${variant.id} failed correctness: ${maxDiff}`);
      differences.set(variant.id, maxDiff);
    }
    for (let round = 0; round < MEASURED_ROUNDS; round++) {
      for (const variant of rotatingOrder(variants, round)) {
        statusFn(`${family}/${shapeId}: round ${round + 1}/${MEASURED_ROUNDS}, ${variant.id}`);
        const started = performance.now();
        const output = await variant.run();
        samples.get(variant.id).push(performance.now() - started);
        checksum += Number(output[0] ?? 0);
      }
    }
  } finally {
    for (const pool of pools) pool.close();
  }

  return variants.map((variant) => {
    const row = resultRow({
      family,
      shapeId,
      shape,
      variant,
      samples: samples.get(variant.id),
      iters: 1,
      maxDiff: differences.get(variant.id),
    });
    logFn(`${family}/${shapeId}/${variant.id}: ${row.median.toFixed(3)} ms end-to-end`);
    return row;
  });
}

function standardReference(A, B, shape, ArrayType = Float32Array) {
  return matmulIJK(A, B, shape.M, shape.K, shape.N, new ArrayType(shape.M * shape.N));
}

function loopVariants(shape, seed) {
  const { A, B } = shapeData(shape, seed);
  const reference = standardReference(A, B, shape);
  return Object.entries(LOOP_ORDER_KERNELS).map(([id, kernel]) => {
    const output = new Float32Array(shape.M * shape.N);
    return {
      id,
      parameter: "loopOrder",
      value: id,
      reference,
      run: () => kernel(A, B, shape.M, shape.K, shape.N, output),
    };
  });
}

function tileVariants(shape, seed) {
  const { A, B } = shapeData(shape, seed);
  const reference = standardReference(A, B, shape);
  const directOutput = new Float32Array(shape.M * shape.N);
  return [{
    id: "direct-ikj",
    parameter: "tileSize",
    value: "none",
    reference,
    run: () => matmulIKJ(A, B, shape.M, shape.K, shape.N, directOutput),
  }, ...TILE_SIZES.map((tileSize) => {
    const output = new Float32Array(shape.M * shape.N);
    return {
      id: `tile-${tileSize}`,
      parameter: "tileSize",
      value: tileSize,
      reference,
      run: () => matmulTiled(A, B, shape.M, shape.K, shape.N, tileSize, output),
    };
  })];
}

function transposeVariants(shape, seed) {
  const { A, B } = shapeData(shape, seed);
  const reference = standardReference(A, B, shape);
  const started = performance.now();
  const transposed = transposeB(B, shape.K, shape.N);
  const setupMs = performance.now() - started;
  const directOutput = new Float32Array(shape.M * shape.N);
  const transposedOutput = new Float32Array(shape.M * shape.N);
  return [
    {
      id: "direct-ikj",
      parameter: "weightLayout",
      value: "row-major",
      reference,
      run: () => matmulIKJ(A, B, shape.M, shape.K, shape.N, directOutput),
    },
    {
      id: "transposed-b",
      parameter: "weightLayout",
      value: "transposed",
      setupMs,
      reference,
      run: () => matmulTransposedB(A, transposed, shape.M, shape.K, shape.N, transposedOutput),
    },
  ];
}

function unrollVariants(shape, seed) {
  const { A, B } = shapeData(shape, seed);
  const reference = standardReference(A, B, shape);
  const directOutput = new Float32Array(shape.M * shape.N);
  return [{
    id: "direct-ikj",
    parameter: "unrollFactor",
    value: "none",
    reference,
    run: () => matmulIKJ(A, B, shape.M, shape.K, shape.N, directOutput),
  }, ...UNROLL_FACTORS.map((factor) => {
    const output = new Float32Array(shape.M * shape.N);
    return {
      id: `unroll-${factor}`,
      parameter: "unrollFactor",
      value: factor,
      reference,
      run: () => matmulIKJUnrolled(A, B, shape.M, shape.K, shape.N, factor, output),
    };
  })];
}

function allocationVariants(shape, seed) {
  const { A, B } = shapeData(shape, seed);
  const reference = standardReference(A, B, shape);
  const reused = new Float32Array(shape.M * shape.N);
  return [
    {
      id: "allocate-output",
      parameter: "outputAllocation",
      value: "allocate",
      allocationMode: "allocate",
      reference,
      run: () => matmulIKJ(A, B, shape.M, shape.K, shape.N),
    },
    {
      id: "reuse-output",
      parameter: "outputAllocation",
      value: "reuse",
      allocationMode: "reuse",
      reference,
      run: () => matmulIKJ(A, B, shape.M, shape.K, shape.N, reused),
    },
  ];
}

function numericVariants(shape, seed) {
  const float32 = shapeData(shape, seed, Float32Array);
  const float64 = shapeData(shape, seed, Float64Array);
  const int8 = shapeData(shape, seed, Int8Array, true);
  const output32 = new Float32Array(shape.M * shape.N);
  const output64 = new Float64Array(shape.M * shape.N);
  const outputInt = new Int32Array(shape.M * shape.N);
  return [
    {
      id: "float32",
      parameter: "numericType",
      value: "float32",
      numericType: "float32",
      reference: standardReference(float32.A, float32.B, shape),
      run: () => matmulIKJ(float32.A, float32.B, shape.M, shape.K, shape.N, output32),
    },
    {
      id: "float64",
      parameter: "numericType",
      value: "float64",
      numericType: "float64",
      tolerance: 1e-9,
      reference: standardReference(float64.A, float64.B, shape, Float64Array),
      run: () => matmulIKJ(float64.A, float64.B, shape.M, shape.K, shape.N, output64),
    },
    {
      id: "int8-int32",
      parameter: "numericType",
      value: "int8-int32",
      numericType: "int8-int32",
      tolerance: 0,
      reference: standardReference(int8.A, int8.B, shape, Int32Array),
      run: () => matmulIKJ(int8.A, int8.B, shape.M, shape.K, shape.N, outputInt),
    },
  ];
}

function libraryVariants(shape, seed) {
  const { A, B } = shapeData(shape, seed);
  const libraryA = np.array(A).reshape([shape.M, shape.K]).astype("float32");
  const libraryB = np.array(B).reshape([shape.K, shape.N]).astype("float32");
  const reference = standardReference(A, B, shape);
  const ijkOutput = new Float32Array(shape.M * shape.N);
  const ikjOutput = new Float32Array(shape.M * shape.N);
  return [
    {
      id: "numpy-ts.dot",
      parameter: "implementation",
      value: "numpy-ts.dot",
      reference,
      allocationMode: "library",
      run: () => np.dot(libraryA, libraryB).data,
    },
    {
      id: "handrolled-ijk",
      parameter: "implementation",
      value: "ijk",
      reference,
      run: () => matmulIJK(A, B, shape.M, shape.K, shape.N, ijkOutput),
    },
    {
      id: "handrolled-ikj",
      parameter: "implementation",
      value: "ikj",
      reference,
      run: () => matmulIKJ(A, B, shape.M, shape.K, shape.N, ikjOutput),
    },
  ];
}

function quantizedMatvecVariants(shape, seed) {
  const { A: vector, B: weights } = shapeData(shape, seed);
  const reference = matvecFloat32(vector, weights, shape.K, shape.N);
  let started = performance.now();
  const int8 = quantizeInt8(weights);
  const int8SetupMs = performance.now() - started;
  started = performance.now();
  const int4 = quantizeInt4(weights);
  const int4SetupMs = performance.now() - started;
  const output32 = new Float32Array(shape.N);
  const output8 = new Float32Array(shape.N);
  const output4 = new Float32Array(shape.N);
  return [
    {
      id: "float32",
      parameter: "weightRepresentation",
      value: "float32",
      reference,
      operations: 2 * shape.K * shape.N,
      run: () => matvecFloat32(vector, weights, shape.K, shape.N, output32),
    },
    {
      id: "int8-inline-dequant",
      parameter: "weightRepresentation",
      value: "int8",
      numericType: "float32-int8",
      tolerance: 0.1,
      reference,
      operations: 2 * shape.K * shape.N,
      setupMs: int8SetupMs,
      run: () => matvecInt8(vector, int8.data, int8.scale, shape.K, shape.N, output8),
    },
    {
      id: "int4-inline-dequant",
      parameter: "weightRepresentation",
      value: "int4",
      numericType: "float32-int4",
      tolerance: 1,
      reference,
      operations: 2 * shape.K * shape.N,
      setupMs: int4SetupMs,
      run: () => matvecInt4(vector, int4.data, int4.scale, shape.K, shape.N, output4),
    },
  ];
}

function rmsNormVariants(shape, seed) {
  const input = seededValues(shape.M * shape.K, seed);
  const reference = rmsNormDirect(input, shape.M, shape.K);
  const directOutput = new Float32Array(input.length);
  const bufferedOutput = new Float32Array(input.length);
  const unrolledOutput = new Float32Array(input.length);
  const squares = new Float32Array(input.length);
  const common = { reference, operations: input.length, operationUnit: "element" };
  return [
    { id: "direct", parameter: "rmsNormImplementation", value: "direct", ...common,
      run: () => rmsNormDirect(input, shape.M, shape.K, 1e-5, directOutput) },
    { id: "buffered-squares", parameter: "rmsNormImplementation", value: "buffered", ...common,
      run: () => rmsNormBuffered(input, shape.M, shape.K, 1e-5, bufferedOutput, squares) },
    { id: "unrolled-4", parameter: "rmsNormImplementation", value: "unrolled-4", ...common,
      run: () => rmsNormUnrolled4(input, shape.M, shape.K, 1e-5, unrolledOutput) },
  ];
}

function softmaxVariants(shape, seed) {
  const input = seededValues(shape.M * shape.K, seed);
  const reference = softmaxStable(input, shape.M, shape.K);
  const stableOutput = new Float32Array(input.length);
  const bufferedOutput = new Float32Array(input.length);
  const onlineOutput = new Float32Array(input.length);
  const exponentials = new Float32Array(input.length);
  const common = { reference, tolerance: 1e-6, operations: input.length, operationUnit: "element" };
  return [
    { id: "stable-recompute", parameter: "softmaxImplementation", value: "stable-recompute", ...common,
      run: () => softmaxStable(input, shape.M, shape.K, stableOutput) },
    { id: "buffered-exp", parameter: "softmaxImplementation", value: "buffered-exp", ...common,
      run: () => softmaxBuffered(input, shape.M, shape.K, bufferedOutput, exponentials) },
    { id: "online", parameter: "softmaxImplementation", value: "online", ...common,
      run: () => softmaxOnline(input, shape.M, shape.K, onlineOutput) },
  ];
}

function fusedResidualNormVariants(shape, seed) {
  const input = seededValues(shape.M * shape.K, seed);
  const residual = seededValues(shape.M * shape.K, seed + 1);
  const reference = residualRmsNormSeparate(input, residual, shape.M, shape.K);
  const separateOutput = new Float32Array(input.length);
  const fusedOutput = new Float32Array(input.length);
  const separateBuffer = new Float32Array(input.length);
  const fusedBuffer = new Float32Array(input.length);
  const common = { reference, operations: input.length, operationUnit: "element" };
  return [
    { id: "separate", parameter: "fusion", value: "separate", ...common,
      run: () => residualRmsNormSeparate(input, residual, shape.M, shape.K, 1e-5, separateOutput, separateBuffer) },
    { id: "fused", parameter: "fusion", value: "fused", ...common,
      run: () => residualRmsNormFused(input, residual, shape.M, shape.K, 1e-5, fusedOutput, fusedBuffer) },
  ];
}

function verifyCorrectness(logFn = console.log) {
  const shape = { M: 9, K: 7, N: 11 };
  const { A, B } = shapeData(shape, 1);
  const reference = standardReference(A, B, shape);
  const checks = [
    ...Object.entries(LOOP_ORDER_KERNELS).map(([id, kernel]) => [id, () => kernel(A, B, shape.M, shape.K, shape.N)]),
    ...TILE_SIZES.map((size) => [`tile-${size}`, () => matmulTiled(A, B, shape.M, shape.K, shape.N, size)]),
    ...UNROLL_FACTORS.map((factor) => [`unroll-${factor}`, () => matmulIKJUnrolled(A, B, shape.M, shape.K, shape.N, factor)]),
    ["transposed-b", () => matmulTransposedB(A, transposeB(B, shape.K, shape.N), shape.M, shape.K, shape.N)],
  ];
  let worst = 0;
  for (const [id, run] of checks) {
    const difference = maximumDifference(run(), reference);
    if (difference > 1e-3) throw new Error(`${id} verification failed: ${difference}`);
    worst = Math.max(worst, difference);
  }
  logFn(`verified ${checks.length} kernels on 9x7x11; worst difference ${worst.toExponential(3)}`);
  return worst;
}

async function getDeviceInfo() {
  const memory = performance.memory ?? {};
  return {
    userAgent: navigator.userAgent,
    platform: navigator.userAgentData?.platform ?? navigator.platform,
    cores: navigator.hardwareConcurrency ?? null,
    memGB: navigator.deviceMemory ?? null,
    jsHeapSizeLimit: memory.jsHeapSizeLimit ?? null,
  };
}

export async function runCpuBenchmark(logFn = console.log, statusFn = () => {}, label = "") {
  checksum = 0;
  verifyCorrectness(logFn);
  const device = await getDeviceInfo();
  const results = [];
  const run = async (family, shapeId, variants) => {
    results.push(...await benchmarkSyncFamily({
      family,
      shapeId,
      variants,
      log: logFn,
      status: statusFn,
    }));
  };

  for (const shapeId of ["square-128", "square-256", "decode-1024", "prefill-128", "square-512"]) {
    await run("legacy-baseline", shapeId, libraryVariants(SHAPES[shapeId], 100));
  }
  for (const shapeId of ["square-256", "prefill-32", "wide", "tall"]) {
    await run("loop-order", shapeId, loopVariants(SHAPES[shapeId], 200));
  }
  for (const shapeId of ["square-512", "prefill-128"]) {
    await run("tile-size", shapeId, tileVariants(SHAPES[shapeId], 300));
  }
  for (const shapeId of ["decode-1024", "prefill-128"]) {
    await run("weight-layout", shapeId, transposeVariants(SHAPES[shapeId], 400));
  }
  await run("loop-unroll", "prefill-128", unrollVariants(SHAPES["prefill-128"], 500));
  await run("output-allocation", "prefill-128", allocationVariants(SHAPES["prefill-128"], 600));
  await run("numeric-type", "prefill-32", numericVariants(SHAPES["prefill-32"], 650));
  for (const shapeId of ["prefill-128", "square-512"]) {
    results.push(...await benchmarkWorkerFamily(shapeId, logFn, statusFn));
  }
  for (const shapeId of ["matvec-512", "matvec-1024"]) {
    await run("quantized-matvec", shapeId, quantizedMatvecVariants(SHAPES[shapeId], 800));
  }
  for (const shapeId of ["norm-decode", "norm-prefill"]) {
    await run("rmsnorm", shapeId, rmsNormVariants(SHAPES[shapeId], 900));
    await run("fused-residual-rmsnorm", shapeId, fusedResidualNormVariants(SHAPES[shapeId], 950));
  }
  for (const shapeId of ["softmax-512", "softmax-2048", "softmax-batch"]) {
    await run("softmax", shapeId, softmaxVariants(SHAPES[shapeId], 1000));
  }

  const summaries = summarizeResults(results);
  logFn("\nrepeatability-qualified recommendations:");
  for (const summary of summaries.filter((item) => item.recommendedVariant)) {
    logFn(
      `${summary.id}: ${summary.recommendedVariant}, ` +
      `${summary.deltaVsBaselinePct.toFixed(1)}% vs ${summary.baselineVariant}, ` +
      `break-even ${summary.breakEvenCalls} call${summary.breakEvenCalls === 1 ? "" : "s"}`
    );
  }
  return {
    schema: "js-cpu-kernel-autotune-v3",
    timestamp: new Date().toISOString(),
    label,
    backend: "cpu-js",
    device,
    protocol: {
      measuredRounds: MEASURED_ROUNDS,
      targetBatchMs: TARGET_BATCH_MS,
      order: "alternating rotated counterbalance",
      correctnessTolerance: 1e-3,
      workerTiming: "persistent pool; row transfer and result collection included; setup separate",
    },
    shapes: SHAPES,
    checksum,
    results,
    summaries,
  };
}
