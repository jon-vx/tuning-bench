import { toCsv } from "../shared/benchmark-utils.js";

export function cpuFileStem(payload) {
  const label = payload.label ? `${payload.label.replace(/[^a-z0-9]+/gi, "-")}-` : "";
  return `bench-cpu-kernels-${label}${payload.timestamp.replace(/[:.]/g, "-")}`;
}

export function cpuCsvText(payload) {
  const header = [
    "timestamp", "label", "user_agent", "platform", "cores", "mem_gb",
    "family", "shape_id", "workload", "M", "K", "N", "variant", "parameter", "value",
    "numeric_type", "allocation_mode", "worker_count", "setup_ms", "iters", "measured_rounds",
    "max_diff", "operations", "operation_unit", "median_ms", "q1_ms", "q3_ms", "iqr_ms", "iqr_pct",
    "p95_ms", "mean_ms", "min_ms", "median_gflops", "median_million_units_per_second",
    "recommended_variant",
  ];
  const summaryById = new Map(payload.summaries.map((summary) => [summary.id, summary]));
  const rows = payload.results.map((result) => [
    payload.timestamp, payload.label, payload.device.userAgent, payload.device.platform,
    payload.device.cores, payload.device.memGB, result.family, result.shapeId, result.workload,
    result.M, result.K, result.N, result.variant, result.parameter, result.value,
    result.numericType, result.allocationMode, result.workerCount, result.setupMs,
    result.iters, result.measuredRounds, result.maxDiff, result.operations, result.operationUnit,
    result.median, result.q1, result.q3, result.iqr, result.iqrPct,
    result.p95, result.mean, result.min, result.medianGflops, result.medianMillionUnitsPerSecond,
    summaryById.get(`${result.family}:${result.shapeId}`)?.recommendedVariant,
  ]);
  return toCsv(header, rows);
}
