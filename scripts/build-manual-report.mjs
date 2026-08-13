import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { toCsv } from "../shared/benchmark-utils.js";
import { EXPERIMENT_DEFINITIONS } from "../gpu/controlled-experiment.js";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const docsDirectory = join(root, "..", "docs");
const campaignDirectory = resolve(process.argv[2] ?? "");
if (!process.argv[2] || !existsSync(campaignDirectory)) {
  throw new Error("pass an existing campaign directory");
}
const manifestPath = join(campaignDirectory, "campaign.json");
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
if (manifest.schema !== "webllm-manual-campaign-v1") {
  throw new Error("unexpected campaign manifest schema");
}
const rapidMode = manifest.execution?.mode === "rapid";
const validationMode = manifest.execution?.mode === "validation";
const latencyMode = manifest.execution?.mode === "latency";
const reportDefinitions = EXPERIMENT_DEFINITIONS.filter((definition) =>
  manifest.parameters?.includes(definition.id)
);

function measuredRunsFor(definition) {
  if (latencyMode) {
    return manifest.execution.blocks
      * manifest.execution.measuredRunsPerCase
      * manifest.execution.taskCaseCount;
  }
  if (validationMode) {
    return manifest.execution.measuredRunsPerCase
      * manifest.execution.taskCaseCount
      * definition.workload.scenarios.length;
  }
  if (!rapidMode) return definition.workload.measuredRuns;
  return (manifest.execution.measuredRunsPerScenario ?? 1)
    * definition.workload.scenarios.length;
}

const liveEntries = manifest.entries.filter((entry) =>
  ["pending", "running"].includes(entry.status)
);
if (liveEntries.length) {
  throw new Error(`${liveEntries.length} campaign targets are not terminal`);
}

const payloads = new Map();
for (const entry of manifest.entries.filter((candidate) => candidate.status === "complete")) {
  const path = join(campaignDirectory, entry.payloadFile);
  if (!existsSync(path)) throw new Error(`missing payload: ${entry.payloadFile}`);
  const payload = JSON.parse(readFileSync(path, "utf8"));
  if (
    payload.schema !== "webllm-controlled-experiment-v2" ||
    payload.experiment?.modelConfigId !== entry.modelConfigId ||
    payload.experiment?.taskProfileId !== entry.taskProfileId ||
    payload.experiment?.variedParameter !== entry.parameterId
  ) {
    throw new Error(`payload does not match manifest entry: ${entry.id}`);
  }
  payloads.set(entry.id, payload);
}

const csvFields = [
  "campaign_id",
  "model_config_id",
  "model",
  "model_family",
  "task_profile_id",
  "parameter_id",
  "parameter_label",
  "experiment_type",
  "objective_metric",
  "experiment_value",
  "policy_id",
  "baseline_type",
  "status",
  "eligible",
  "run_count",
  "success_rate",
  "quality_pass_rate",
  "clean_quality_pass_rate",
  "required_history_quality_pass_rate",
  "median_quality_score",
  "median_prompt_tokens",
  "median_completion_tokens",
  "median_ttft_ms",
  "matched_ttft_delta_pct",
  "median_wall_ms",
  "matched_wall_delta_pct",
  "matched_objective_delta_pct",
  "matched_repeatability_iqr_pct",
  "matched_faster_pair_rate",
  "matched_completion_token_equality_rate",
  "matched_output_equality_rate",
  "matched_prefix_equality_rate",
  "output_truncated",
  "failure_classes",
  "recommended_policy_id",
];

const definitionById = new Map(
  EXPERIMENT_DEFINITIONS.map((definition) => [definition.id, definition])
);
const summaryRows = [];
for (const entry of manifest.entries) {
  const payload = payloads.get(entry.id);
  if (!payload) continue;
  const definition = definitionById.get(entry.parameterId);
  for (const summary of payload.summaries) {
    const result = payload.results.find((row) => row.policyId === summary.policyId);
    summaryRows.push({
      campaign_id: manifest.campaignId,
      model_config_id: entry.modelConfigId,
      model: result?.model ?? payload.experiment.modelConfigId,
      model_family: result?.modelFamily ?? "",
      task_profile_id: entry.taskProfileId,
      parameter_id: entry.parameterId,
      parameter_label: definition?.label ?? entry.parameterId,
      experiment_type: definition?.configurations ? "composite" : "single-parameter",
      objective_metric: payload.criteria.objectiveMetric,
      experiment_value: summary.experimentValue,
      policy_id: summary.policyId,
      baseline_type: summary.baselineType,
      status: summary.status,
      eligible: summary.eligible,
      run_count: summary.runCount,
      success_rate: summary.successRate,
      quality_pass_rate: summary.qualityPassRate,
      clean_quality_pass_rate: summary.cleanQualityPassRate,
      required_history_quality_pass_rate: summary.requiredHistoryQualityPassRate,
      median_quality_score: summary.medianQualityScore,
      median_prompt_tokens: summary.medianPromptTokens,
      median_completion_tokens: summary.medianCompletionTokens,
      median_ttft_ms: summary.medianTtftMs,
      matched_ttft_delta_pct: summary.matchedTtftDeltaPct,
      median_wall_ms: summary.medianWallMs,
      matched_wall_delta_pct: summary.matchedWallDeltaPct,
      matched_objective_delta_pct: summary.matchedObjectiveDeltaPct,
      matched_repeatability_iqr_pct: summary.matchedRepeatabilityIqrPct,
      matched_faster_pair_rate: summary.matchedFasterPairRate,
      matched_completion_token_equality_rate:
        summary.matchedCompletionTokenEqualityRate,
      matched_output_equality_rate: summary.matchedOutputEqualityRate,
      matched_prefix_equality_rate: summary.matchedPrefixEqualityRate,
      output_truncated: summary.anyTruncated,
      failure_classes: summary.failureClasses?.join("|") ?? "",
      recommended_policy_id: payload.recommendedPolicyId,
    });
  }
}

const summaryCsv = toCsv(
  csvFields,
  summaryRows.map((row) => csvFields.map((field) => row[field]))
);
writeFileSync(join(campaignDirectory, "manual-summary.csv"), `${summaryCsv}\n`);

const chartDirectory = join(
  docsDirectory,
  "manual-experiments",
  manifest.campaignId
);
mkdirSync(chartDirectory, { recursive: true });

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function svgDocument(title, description, width, height, body) {
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img">`,
    `<title>${escapeXml(title)}</title>`,
    `<desc>${escapeXml(description)}</desc>`,
    `<rect width="${width}" height="${height}" fill="#0d0f0c"/>`,
    `<style>text{font-family:ui-monospace,monospace;fill:#d9dfd0}.muted{fill:#929b88}.axis{stroke:#596050}.good{fill:#82d173}.bad{fill:#e06c75}.control{fill:#61afef}.reference{fill:#929b88}.missing{fill:#4a4f46}</style>`,
    body,
    "</svg>",
  ].join("");
}

function entryFor(definitionId, modelId, taskId) {
  return manifest.entries.find((entry) =>
    entry.parameterId === definitionId &&
    entry.modelConfigId === modelId &&
    entry.taskProfileId === taskId
  );
}

function recommendationValue(entry) {
  const payload = payloads.get(entry?.id);
  if (!payload?.recommendedPolicyId) return "none";
  return payload.summaries.find(
    (summary) => summary.policyId === payload.recommendedPolicyId
  )?.experimentValue ?? "none";
}

function recommendationMatrix(definition) {
  const width = 920;
  const left = 160;
  const top = 72;
  const cellWidth = 180;
  const cellHeight = 42;
  const height = top + manifest.models.length * cellHeight + 40;
  const parts = [
    `<text x="20" y="28" font-size="17">${escapeXml(definition.label)} recommendations</text>`,
  ];
  manifest.tasks.forEach((task, index) => {
    parts.push(
      `<text x="${left + index * cellWidth + cellWidth / 2}" y="56" text-anchor="middle" class="muted">${escapeXml(task.label)}</text>`
    );
  });
  manifest.models.forEach((model, rowIndex) => {
    const y = top + rowIndex * cellHeight;
    parts.push(
      `<text x="12" y="${y + 26}" class="muted">${escapeXml(model.id)}</text>`
    );
    manifest.tasks.forEach((task, columnIndex) => {
      const entry = entryFor(definition.id, model.id, task.id);
      const x = left + columnIndex * cellWidth;
      let value = "missing";
      let className = "missing";
      if (entry?.status === "skipped") {
        value = "unsupported";
      } else if (entry?.status === "failed") {
        value = "failed";
        className = "bad";
      } else if (entry?.status === "complete") {
        value = recommendationValue(entry);
        className = value === "none" ? "missing" : "good";
      }
      parts.push(
        `<rect x="${x + 2}" y="${y + 2}" width="${cellWidth - 4}" height="${cellHeight - 4}" rx="3" class="${className}" opacity="0.22"/>`,
        `<text x="${x + cellWidth / 2}" y="${y + 26}" text-anchor="middle">${escapeXml(value)}</text>`
      );
    });
  });
  return svgDocument(
    `${definition.label} recommendation matrix`,
    "Recommended value for each supported model and task combination.",
    width,
    height,
    parts.join("")
  );
}

function pairRows(definition) {
  return manifest.entries
    .filter((entry) =>
      entry.parameterId === definition.id && entry.status !== "skipped"
    )
    .map((entry) => ({
      entry,
      label: `${entry.modelConfigId} / ${entry.taskProfileId}`,
      payload: payloads.get(entry.id),
    }));
}

function pointClass(summary) {
  if (summary.baselineType === "library_default") return "reference";
  if (summary.baselineType === "controlled") return "control";
  return summary.status === "candidate" ? "good" : "bad";
}

function scatterChart(definition, metric, title, axisLabel, fixedRange) {
  const rows = pairRows(definition);
  const width = 1100;
  const left = 250;
  const right = 1050;
  const top = 66;
  const rowHeight = 30;
  const height = Math.max(180, top + rows.length * rowHeight + 58);
  const metricValue = (summary) => metric === "qualityPassRate"
    ? summary[metric] * 100
    : summary[metric];
  const values = rows.flatMap(({ payload }) =>
    payload?.summaries.map(metricValue).filter(Number.isFinite) ?? []
  );
  let minimum = fixedRange?.[0] ?? Math.min(0, ...values);
  let maximum = fixedRange?.[1] ?? Math.max(0, ...values);
  if (minimum === maximum) maximum = minimum + 1;
  if (!fixedRange) {
    const padding = Math.max((maximum - minimum) * 0.08, 1);
    minimum -= padding;
    maximum += padding;
  }
  const x = (value) => left + (value - minimum) / (maximum - minimum) * (right - left);
  const parts = [
    `<text x="20" y="28" font-size="17">${escapeXml(title)}</text>`,
    `<line x1="${left}" y1="${top - 12}" x2="${left}" y2="${height - 42}" class="axis"/>`,
    `<line x1="${right}" y1="${top - 12}" x2="${right}" y2="${height - 42}" class="axis"/>`,
  ];
  if (minimum <= 0 && maximum >= 0) {
    parts.push(
      `<line x1="${x(0)}" y1="${top - 12}" x2="${x(0)}" y2="${height - 42}" class="axis" stroke-dasharray="4 4"/>`
    );
  }
  rows.forEach(({ label, payload }, rowIndex) => {
    const y = top + rowIndex * rowHeight;
    parts.push(
      `<text x="${left - 10}" y="${y + 4}" text-anchor="end" class="muted">${escapeXml(label)}</text>`,
      `<line x1="${left}" y1="${y}" x2="${right}" y2="${y}" class="axis" opacity="0.22"/>`
    );
    for (const summary of payload?.summaries ?? []) {
      const value = metricValue(summary);
      if (!Number.isFinite(value)) continue;
      parts.push(
        `<circle cx="${x(value)}" cy="${y}" r="5" class="${pointClass(summary)}"/>`,
        `<text x="${Math.min(right - 2, x(value) + 7)}" y="${y - 7}" font-size="9">${escapeXml(summary.experimentValue)}</text>`
      );
    }
  });
  parts.push(
    `<text x="${(left + right) / 2}" y="${height - 12}" text-anchor="middle" class="muted">${escapeXml(axisLabel)}</text>`,
    `<text x="${left}" y="${height - 28}" text-anchor="middle" class="muted">${minimum.toFixed(1)}</text>`,
    `<text x="${right}" y="${height - 28}" text-anchor="middle" class="muted">${maximum.toFixed(1)}</text>`
  );
  return svgDocument(title, `${axisLabel} by model, task, and tested value.`, width, height, parts.join(""));
}

for (const definition of reportDefinitions) {
  const objectiveMetric = definition.objectiveMetric === "quality"
    ? "qualityPassDeltaPoints"
    : "matchedObjectiveDeltaPct";
  const objectiveLabel = definition.objectiveMetric === "quality"
    ? "quality-pass delta (percentage points; positive is better)"
    : `matched ${definition.objectiveMetric.toUpperCase()} delta (%; negative is faster)`;
  writeFileSync(
    join(chartDirectory, `${definition.id}-recommendations.svg`),
    recommendationMatrix(definition)
  );
  writeFileSync(
    join(chartDirectory, `${definition.id}-objective.svg`),
    scatterChart(
      definition,
      objectiveMetric,
      `${definition.label} objective`,
      objectiveLabel
    )
  );
  writeFileSync(
    join(chartDirectory, `${definition.id}-quality.svg`),
    scatterChart(
      definition,
      "qualityPassRate",
      `${definition.label} strict quality`,
      "strict quality-pass rate (%)",
      [0, 100]
    )
  );
}

const archiveManifest = {
  ...manifest,
  rawArchiveSha256: undefined,
};
writeFileSync(
  join(campaignDirectory, "archive-manifest.json"),
  `${JSON.stringify(archiveManifest, null, 2)}\n`
);
const tarPath = join(campaignDirectory, "raw-results.tar");
const archivePath = join(campaignDirectory, "raw-results.tar.gz");
const tar = spawnSync(
  "tar",
  [
    "--sort=name",
    "--mtime=1970-01-01",
    "--owner=0",
    "--group=0",
    "--numeric-owner",
    "-cf",
    tarPath,
    "archive-manifest.json",
    "runs",
    "errors",
  ],
  { cwd: campaignDirectory, encoding: "utf8" }
);
if (tar.status !== 0) throw new Error(tar.stderr || "tar failed");
const gzip = spawnSync("gzip", ["-n", "-f", tarPath], {
  cwd: campaignDirectory,
  encoding: "utf8",
});
if (gzip.status !== 0) throw new Error(gzip.stderr || "gzip failed");
if (!existsSync(archivePath)) throw new Error("raw archive was not created");
unlinkSync(join(campaignDirectory, "archive-manifest.json"));
const archiveHash = createHash("sha256")
  .update(readFileSync(archivePath))
  .digest("hex");
manifest.rawArchiveSha256 = archiveHash;
writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
writeFileSync(join(campaignDirectory, "raw-results.sha256"), `${archiveHash}  raw-results.tar.gz\n`);

function markdownCell(value) {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}

function percent(value, digits = 1) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "n/a";
}

function number(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

function experimentTable(definition) {
  const lines = [
    "| model | task | recommended | control quality | recommended quality | objective delta | status |",
    "|---|---|---:|---:|---:|---:|---|",
  ];
  for (const row of pairRows(definition)) {
    if (row.entry.status !== "complete") {
      lines.push(
        `| ${markdownCell(row.entry.modelConfigId)} | ${markdownCell(row.entry.taskProfileId)} | n/a | n/a | n/a | n/a | ${markdownCell(row.entry.status)} |`
      );
      continue;
    }
    const controlled = row.payload.summaries.find(
      (summary) => summary.baselineType === "controlled"
    );
    const recommended = row.payload.summaries.find(
      (summary) => summary.policyId === row.payload.recommendedPolicyId
    );
    lines.push(
      `| ${markdownCell(row.entry.modelConfigId)} | ${markdownCell(row.entry.taskProfileId)} | ${markdownCell(recommended?.experimentValue ?? "none")} | ${percent(controlled?.qualityPassRate)} | ${percent(recommended?.qualityPassRate)} | ${number(recommended?.matchedObjectiveDeltaPct)} | ${recommended ? "candidate" : "none"} |`
    );
  }
  return lines.join("\n");
}

const completedCount = manifest.entries.filter((entry) => entry.status === "complete").length;
const failedCount = manifest.entries.filter((entry) => entry.status === "failed").length;
const skippedCount = manifest.entries.filter((entry) => entry.status === "skipped").length;
const inferenceCount = [...payloads.values()].reduce(
  (sum, payload) => sum + payload.results.length,
  0
);
const validationGroups = new Map();
if (validationMode) {
  for (const entry of manifest.entries) {
    const key = `${entry.modelConfigId}|${entry.parameterId}|${entry.candidatePolicyId}`;
    if (!validationGroups.has(key)) {
      validationGroups.set(key, {
        modelConfigId: entry.modelConfigId,
        parameterId: entry.parameterId,
        candidatePolicyId: entry.candidatePolicyId,
        candidateExperimentValue: entry.candidateExperimentValue,
        entries: [],
      });
    }
    validationGroups.get(key).entries.push(entry);
  }
}
const report = [
  validationMode
    ? "# Cross-task WebLLM candidate validation"
    : latencyMode
      ? "# WebLLM latency-only runtime experiment"
    : "# Manual WebLLM experiments",
  "",
  "This is the human-readable record of the controlled manual-tuning campaign. It describes runtime and model-artifact experiments; it does not train or modify model weights.",
  "",
  "## Campaign",
  "",
  `- Campaign: \`${manifest.campaignId}\``,
  `- Started: ${manifest.startedAt ?? "n/a"}`,
  `- Completed: ${manifest.completedAt ?? "n/a"}`,
  `- WebLLM: \`${manifest.webllmVersion}\``,
  `- Mode: \`${manifest.execution?.mode ?? "confirmatory"}\``,
  `- Git commit: \`${manifest.sourceRevision.commit}\``,
  `- Working-tree diff SHA-256: \`${manifest.sourceRevision.diffSha256}\``,
  `- Source SHA-256: \`${manifest.sourceRevision.sourceSha256}\``,
  `- Completed targets: ${completedCount}`,
  `- Failed targets: ${failedCount}`,
  `- Unsupported targets: ${skippedCount}`,
  `- Captured result rows: ${inferenceCount}`,
  `- Raw archive SHA-256: \`${archiveHash}\``,
  "",
  ...(manifest.runNotes?.length
    ? [
      "## Run integrity",
      "",
      ...manifest.runNotes.map((note) => `- ${note}`),
      "",
    ]
    : []),
  ...(manifest.correction
    ? [
      "## Correction lineage",
      "",
      `- Source campaign: \`${manifest.correction.sourceCampaignId}\``,
      `- Reused single-scenario targets: ${manifest.correction.reusedTargetCount}`,
      `- Re-run multi-scenario targets: ${manifest.correction.rerunTargetCount}`,
      `- Corrected parameters: ${manifest.correction.parameterIds.map((id) => `\`${id}\``).join(", ")}`,
      `- Reason: ${manifest.correction.reason}`,
      "",
    ]
    : []),
  "## Device",
  "",
  "```json",
  JSON.stringify(manifest.deviceProfile, null, 2),
  "```",
  "",
  "## Models",
  "",
  "| ID | Family | Default artifact | Alternate artifacts |",
  "|---|---|---|---|",
  ...manifest.models.map((model) =>
    `| ${markdownCell(model.id)} | ${markdownCell(model.family)} | ${markdownCell(model.model)} | ${markdownCell(Object.entries(model.artifacts ?? {}).map(([key, value]) => `${key}: ${value}`).join("; "))} |`
  ),
  "",
  "## Tasks",
  "",
  "| ID | Label | Type |",
  "|---|---|---|",
  ...manifest.tasks.map((task) =>
    `| ${markdownCell(task.id)} | ${markdownCell(task.label)} | ${markdownCell(task.taskType)} |`
  ),
  "",
  "## Method",
  "",
  validationMode
    ? `Each extraction-screen candidate is compared with its controlled baseline and the untouched WebLLM reference on ${manifest.tasks.map((task) => task.label).join(", ")}. Every task case runs twice for each policy. Policy order rotates across targets, and each policy receives one warmup.`
    : latencyMode
    ? "Each direct WebLLM/runtime candidate is compared with its controlled baseline over six ordinary queries and six counterbalanced blocks. Output meaning is not scored. A row is valid only when nonempty generation completes and timing data is available."
    : rapidMode
    ? "Each experiment fixes the device, base model, task, prompts, quality evaluator, and all non-varied settings. Every declared value runs in one rotated-order block."
    : "Each experiment fixes the device, base model, task, prompts, quality evaluator, and all non-varied settings. Every declared value runs in counterbalanced blocks. Results are matched by block, task case, conversation scenario, and run index.",
  "",
  ...(rapidMode
    ? [
      "This campaign is a rapid screen: each value runs once per declared scenario in one rotated-order block, without warmups or complete counterbalancing. Its recommendations are candidates for later replicated confirmation, not final causal estimates.",
      "",
    ]
    : []),
  ...(latencyMode
    ? [
      "This is a performance-characterization campaign. Its recommendations do not establish that answer quality is preserved. Wall-time candidates additionally require matched completion-token counts so shorter answers cannot qualify as faster execution.",
      "",
    ]
    : []),
  validationMode
    ? "A source candidate is retained only when it remains the recommended policy for the new task under the experiment's success, quality, truncation, equality, and performance requirements."
    : latencyMode
    ? "Timing candidates require complete nonempty generation, no unexpected truncation, a gain larger than repeatability noise, and at least two-thirds faster matched pairs. No semantic answer-quality claim is made."
    : rapidMode
    ? "Screening candidates require complete execution, no truncation, retained quality, and the experiment's declared acceptance threshold. The untouched-WebLLM configuration is a reference and cannot become the recommendation."
    : "Timing candidates require complete execution, no truncation, retained quality, a gain larger than repeatability noise, and at least two-thirds faster matched pairs. Quality candidates use their declared TTFT or wall-time guard. The untouched-WebLLM configuration is a reference and cannot become the recommendation.",
  "",
  "`q4f16_1` uses four-bit weights with float16 activations. `q0f16` is unquantized float16. Weight quantization is tested only where WebLLM 0.2.79 provides both artifacts.",
  "",
  "## Artifact preflight",
  "",
  "| Model | Quantization | Artifact | Status | Error |",
  "|---|---|---|---|---|",
  ...(manifest.artifactPreflight ?? []).map((artifact) =>
    `| ${markdownCell(artifact.modelConfigId)} | ${markdownCell(artifact.quantization)} | ${markdownCell(artifact.artifactId)} | ${markdownCell(artifact.status)} | ${markdownCell(artifact.error)} |`
  ),
  "",
  "## Experiment catalog",
  "",
  "| ID | Label | Values | Control | Objective | Scenarios | Runs per value | Classification |",
  "|---|---|---|---|---|---|---:|---|",
  ...reportDefinitions.map((definition) =>
    `| ${markdownCell(definition.id)} | ${markdownCell(definition.label)} | ${markdownCell(definition.values.join(", "))} | ${markdownCell(definition.controlledValue)} | ${markdownCell(definition.objectiveMetric)} | ${markdownCell(latencyMode ? "single-turn" : definition.workload.scenarios.join(", "))} | ${measuredRunsFor(definition)} | ${definition.configurations ? "composite" : "single parameter"} |`
  ),
  "",
  ...(validationMode
    ? [
      "## Cross-task retention",
      "",
      "| Model | Parameter | Candidate | Tasks retained | Result |",
      "|---|---|---|---:|---|",
      ...[...validationGroups.values()].map((group) => {
        const retained = group.entries.filter((entry) =>
          payloads.get(entry.id)?.recommendedPolicyId === group.candidatePolicyId
        ).length;
        return `| ${markdownCell(group.modelConfigId)} | ${markdownCell(group.parameterId)} | ${markdownCell(group.candidateExperimentValue)} | ${retained}/${group.entries.length} | ${retained === group.entries.length ? "retained" : "not universal"} |`;
      }),
      "",
    ]
    : []),
];

for (const definition of reportDefinitions) {
  const relativeChartDirectory = relative(docsDirectory, chartDirectory).replaceAll("\\", "/");
  const candidates = pairRows(definition)
    .map(({ payload }) => payload?.recommendedPolicyId)
    .filter(Boolean).length;
  report.push(
    `## ${definition.label}`,
    "",
    `- Parameter: \`${definition.id}\``,
    `- Values: ${definition.values.map((value) => `\`${value}\``).join(", ")}`,
    `- Controlled value: \`${definition.controlledValue}\``,
    `- Objective: \`${definition.objectiveMetric}\``,
    `- Workload: ${measuredRunsFor(definition)} measured runs per value over ${latencyMode ? "single-turn" : definition.workload.scenarios.join(", ")}`,
    `- Passing recommendations: ${candidates}`,
    "",
    `![${definition.label} recommendation matrix](${relativeChartDirectory}/${definition.id}-recommendations.svg)`,
    "",
    `![${definition.label} objective results](${relativeChartDirectory}/${definition.id}-objective.svg)`,
    "",
    `![${definition.label} quality results](${relativeChartDirectory}/${definition.id}-quality.svg)`,
    "",
    experimentTable(definition),
    ""
  );
}

const campaignRelative = relative(root, campaignDirectory).replaceAll("\\", "/");
report.push(
  "## Deferred experiments",
  "",
  "Temperature, top-p, and nonzero repetition penalties require a matched multi-seed sampling design. Relevance-selector chunk size is application logic rather than a WebLLM setting. Prefill chunk size requires separately prepared artifacts in the pinned browser runtime. Cache-backend comparisons belong to a cold-start lifecycle benchmark.",
  "",
  "## Limitations",
  "",
  "These measurements describe one browser, GPU, driver, and campaign period. Shared-memory pressure, thermal state, background work, browser updates, and model availability can change results. A recommendation for this device is not evidence that the same setting is best elsewhere.",
  "",
  "## Artifacts",
  "",
  `- [Campaign manifest](${campaignRelative}/campaign.json)`,
  `- [Summary CSV](${campaignRelative}/manual-summary.csv)`,
  `- [Raw payload archive](${campaignRelative}/raw-results.tar.gz)`,
  `- [Archive checksum](${campaignRelative}/raw-results.sha256)`,
  ""
);

writeFileSync(
  join(docsDirectory, manifest.reportFile ?? "MANUAL_EXPERIMENTS.md"),
  `${report.join("\n")}\n`
);
process.stdout.write(
  `${manifest.campaignId}: ${completedCount} complete, ${failedCount} failed, ${skippedCount} unsupported\n`
);
process.stdout.write(
  `${summaryRows.length} summary rows and ${reportDefinitions.length * 3} SVG charts written\n`
);
