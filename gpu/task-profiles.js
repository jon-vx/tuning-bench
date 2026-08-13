export const TASK_PROFILES = [
  {
    id: "chat-explain",
    label: "Explain",
    taskType: "chat",
    systemPrompt: "Use the supplied facts. Answer in exactly 4 concise bullet points with no introduction or conclusion.",
    compactSystemPrompt: "Return exactly 4 concise bullet points.",
    outputTokenLimits: { short: 128, standard: 192, expanded: 256 },
    cases: [
      {
        id: "transformer-impact",
        prompt:
          "Using one fact per bullet, explain why transformers changed AI: self-attention models long-range context; removing recurrence enables parallel training; large-scale pretraining supports transfer; and the architecture scales across tasks.",
        quality: {
          threshold: 0.75,
          minBulletCount: 4,
          minimumCheckScores: { concept_coverage: 0.75, bullet_count: 1 },
          conceptGroups: [
            ["attention", "self-attention"],
            ["parallel", "parallelization", "parallelism"],
            ["pretraining", "pre-training", "pre-trained", "scaling", "scalable"],
            ["long-range", "dependency", "context"],
          ],
        },
        historyRequirement: {
          distanceTokens: 288,
          factText:
            "Project Atlas uses WebGPU to avoid server round trips, reuses cached prefixes to avoid repeated prefill, selects relevant history to preserve needed facts, and keeps model weights fixed for comparable quality.",
          prompt: "Using only the earlier conversation, explain Project Atlas's runtime strategy in exactly 4 concise bullet points.",
          quality: {
            threshold: 1,
            minBulletCount: 4,
            conceptGroups: [
              ["webgpu", "server"],
              ["cache", "cached", "prefix", "prefill"],
              ["relevant", "history", "facts"],
              ["weights", "fixed", "quality"],
            ],
          },
        },
      },
      {
        id: "attention-vs-recurrence",
        prompt:
          "Using one fact per bullet, explain why attention improves on recurrence: attention accesses all tokens directly; recurrent networks process sequentially; parallel computation speeds training; and direct access helps long-range dependencies.",
        quality: {
          threshold: 0.75,
          minBulletCount: 4,
          minimumCheckScores: { concept_coverage: 0.75, bullet_count: 1 },
          conceptGroups: [
            ["attention", "self-attention"],
            ["parallel", "parallelization", "parallelism"],
            ["recurrent", "recurrence", "sequential"],
            ["long-range", "dependency", "context"],
          ],
        },
        historyRequirement: {
          distanceTokens: 416,
          factText:
            "Project Beacon profiles adapter limits before loading, warms the model once, measures TTFT separately from decode speed, and rejects any configuration that truncates output.",
          prompt: "Using only the earlier conversation, explain Project Beacon's measurement protocol in exactly 4 concise bullet points.",
          quality: {
            threshold: 1,
            minBulletCount: 4,
            conceptGroups: [
              ["adapter", "limit"],
              ["warm", "model"],
              ["ttft", "decode"],
              ["reject", "truncat"],
            ],
          },
        },
      },
      {
        id: "pretraining-transfer",
        prompt:
          "Using one fact per bullet, explain transformer transfer learning: pretraining learns broad patterns; fine-tuning adapts to a target; shared representations transfer knowledge; and this improves generalization across downstream tasks.",
        quality: {
          threshold: 0.75,
          minBulletCount: 4,
          minimumCheckScores: { concept_coverage: 0.75, bullet_count: 1 },
          conceptGroups: [
            ["pretraining", "pre-training", "pre-trained"],
            ["fine-tuning", "finetuning", "fine tuning"],
            ["transfer", "adapt"],
            ["generalize", "generalization", "many tasks", "downstream"],
          ],
        },
        historyRequirement: {
          distanceTokens: 480,
          factText:
            "Project Comet keeps the Qwen model fixed, tests one runtime parameter at a time, repeats matched prompts in six blocks, and accepts a tune only when quality remains at least 90 percent.",
          prompt: "Using only the earlier conversation, explain Project Comet's experimental design in exactly 4 concise bullet points.",
          quality: {
            threshold: 1,
            minBulletCount: 4,
            conceptGroups: [
              ["qwen", "model", "fixed"],
              ["one", "parameter", "runtime"],
              ["six", "block", "repeat", "matched"],
              ["quality", "90", "percent"],
            ],
          },
        },
      },
    ],
  },
  {
    id: "chat-summarize",
    label: "Summarize",
    taskType: "summarization",
    systemPrompt: "You summarize technical writing clearly and briefly.",
    compactSystemPrompt: "Return an accurate bulleted summary.",
    outputTokenLimits: { short: 128, standard: 256, expanded: 384 },
    cases: [
      {
        id: "transformer-summary",
        prompt:
          "Summarize in 5 bullet points: Transformers replaced recurrence with attention, enabled large-scale pretraining, improved parallel training efficiency, generalized across modalities, and accelerated the modern LLM era.",
        quality: {
          threshold: 0.75,
          minBulletCount: 4,
          conceptGroups: [
            ["attention"],
            ["pretraining", "pre-training"],
            ["parallel", "efficiency"],
            ["modalities", "multimodal"],
            ["modern", "llm", "language model"],
          ],
        },
        historyRequirement: {
          distanceTokens: 288,
          factText:
            "Project Cedar uses Qwen2.5-0.5B, runs in Chromium, targets 500 millisecond TTFT, and keeps model weights fixed.",
          prompt: "Using only the earlier conversation, summarize Project Cedar in 4 bullet points.",
          quality: {
            threshold: 1,
            minBulletCount: 4,
            conceptGroups: [
              ["qwen2.5", "qwen"],
              ["chromium"],
              ["500", "millisecond"],
              ["weights", "fixed"],
            ],
          },
        },
      },
      {
        id: "webgpu-summary",
        prompt:
          "Summarize in 4 bullet points: WebGPU exposes GPU compute in the browser, avoids server round trips, has adapter-specific limits, and can accelerate local machine-learning inference.",
        quality: {
          threshold: 0.75,
          minBulletCount: 4,
          conceptGroups: [
            ["browser"],
            ["server", "round trip"],
            ["adapter", "limit"],
            ["local", "machine-learning", "machine learning", "inference"],
          ],
        },
        historyRequirement: {
          distanceTokens: 416,
          factText:
            "Project Harbor uses Llama-3.2-1B, runs in Firefox, targets 900 millisecond TTFT, and rejects truncated output.",
          prompt: "Using only the earlier conversation, summarize Project Harbor in 4 bullet points.",
          quality: {
            threshold: 1,
            minBulletCount: 4,
            conceptGroups: [
              ["llama-3.2", "llama"],
              ["firefox"],
              ["900", "millisecond"],
              ["truncated", "truncation"],
            ],
          },
        },
      },
      {
        id: "runtime-tuning-summary",
        prompt:
          "Summarize in 4 bullet points: Runtime tuning keeps model weights fixed, tests configuration candidates, measures latency and quality, and selects settings separately for different devices and tasks.",
        quality: {
          threshold: 0.75,
          minBulletCount: 4,
          conceptGroups: [
            ["weights", "model"],
            ["configuration", "candidate", "setting"],
            ["latency", "quality"],
            ["device", "task"],
          ],
        },
        historyRequirement: {
          distanceTokens: 480,
          factText:
            "Project Maple uses SmolLM2-360M, runs in Edge, targets 1200 millisecond TTFT, and stores model files in IndexedDB.",
          prompt: "Using only the earlier conversation, summarize Project Maple in 4 bullet points.",
          quality: {
            threshold: 1,
            minBulletCount: 4,
            conceptGroups: [
              ["smollm2", "smol"],
              ["edge"],
              ["1200", "millisecond"],
              ["indexeddb", "indexed db"],
            ],
          },
        },
      },
    ],
  },
  {
    id: "chat-extract",
    label: "Extract",
    taskType: "extraction",
    systemPrompt: "Return exactly one JSON object matching the requested schema. Do not return markdown, code, or explanations. Copy every supplied value.",
    compactSystemPrompt: "Return only valid JSON.",
    outputTokenLimits: { short: 192, standard: 256, expanded: 384 },
    cases: [
      {
        id: "model-records",
        prompt:
          "Input: SmolLM2 has size 360M and tradeoff speed. Qwen2.5 has size 0.5B and tradeoff balanced. Return {\"records\":[{\"family\":string,\"size\":string,\"tradeoff\":string}]} with exactly two records.",
        quality: {
          threshold: 1,
          expectedJsonKeys: ["family", "size", "tradeoff"],
          expectedJsonRecords: [
            [["smollm2", "smol"], ["360m", "360"], ["speed", "fast"]],
            [["qwen2.5", "qwen"], ["0.5b", "500m"], ["balanced"]],
          ],
        },
        historyRequirement: {
          distanceTokens: 288,
          factText:
            "Project Delta records two models: Cedar has size 0.5B and priority latency. Harbor has size 1B and priority quality.",
          prompt: "Using only the earlier Project Delta data, return {\"records\":[{\"name\":string,\"size\":string,\"priority\":string}]} with exactly two records.",
          quality: {
            threshold: 1,
            expectedJsonKeys: ["name", "size", "priority"],
            expectedJsonRecords: [
              [["cedar"], ["0.5b"], ["latency"]],
              [["harbor"], ["1b"], ["quality"]],
            ],
          },
        },
      },
      {
        id: "device-records",
        prompt:
          "Input: Laptop A uses Intel Xe-LPG with 16GB. Desktop B uses NVIDIA Ada with 24GB. Return {\"records\":[{\"name\":string,\"gpu\":string,\"memory\":string}]} with exactly two records.",
        quality: {
          threshold: 1,
          expectedJsonKeys: ["name", "gpu", "memory"],
          expectedJsonRecords: [
            [["laptop a"], ["intel"], ["xe-lpg", "xe lpg"], ["16gb", "16 gb"]],
            [["desktop b"], ["nvidia"], ["ada"], ["24gb", "24 gb"]],
          ],
        },
        historyRequirement: {
          distanceTokens: 416,
          factText:
            "Project Ember records two devices: Laptop C uses AMD RDNA with 8GB. Desktop D uses Intel Arc with 12GB.",
          prompt: "Using only the earlier Project Ember data, return {\"records\":[{\"name\":string,\"gpu\":string,\"memory\":string}]} with exactly two records.",
          quality: {
            threshold: 1,
            expectedJsonKeys: ["name", "gpu", "memory"],
            expectedJsonRecords: [
              [["laptop c"], ["amd"], ["rdna"], ["8gb", "8 gb"]],
              [["desktop d"], ["intel"], ["arc"], ["12gb", "12 gb"]],
            ],
          },
        },
      },
      {
        id: "policy-records",
        prompt:
          "Input: Controlled has TTFT 800ms and quality 0.80. Responsive has TTFT 250ms and quality 0.85. Return {\"records\":[{\"policy\":string,\"ttft\":string,\"quality\":string}]} with exactly two records.",
        quality: {
          threshold: 1,
          expectedJsonKeys: ["policy", "ttft", "quality"],
          expectedJsonRecords: [
            [["controlled"], ["800ms", "800 ms"], ["0.80", "0.8"]],
            [["responsive"], ["250ms", "250 ms"], ["0.85"]],
          ],
        },
        historyRequirement: {
          distanceTokens: 480,
          factText:
            "Project Frost records two policies: Baseline has TTFT 900ms and quality 0.90. Tuned has TTFT 300ms and quality 0.92.",
          prompt: "Using only the earlier Project Frost data, return {\"records\":[{\"policy\":string,\"ttft\":string,\"quality\":string}]} with exactly two records.",
          quality: {
            threshold: 1,
            expectedJsonKeys: ["policy", "ttft", "quality"],
            expectedJsonRecords: [
              [["baseline"], ["900ms", "900 ms"], ["0.90", "0.9"]],
              [["tuned"], ["300ms", "300 ms"], ["0.92"]],
            ],
          },
        },
      },
    ],
  },
  {
    id: "chat-code",
    label: "Code",
    taskType: "coding",
    systemPrompt: "Return only one short JavaScript function. Use object properties exactly as named and do not mutate the input.",
    compactSystemPrompt: "Return only JavaScript code.",
    outputTokenLimits: { short: 256, standard: 384, expanded: 512 },
    cases: [
      {
        id: "successful-rows",
        prompt:
          "Write function successfulRows(rows) that returns a new array containing only rows where row.success is exactly true.",
        quality: { threshold: 1, codeFunctionName: "successfulRows", codeTestId: "successful-rows" },
        historyRequirement: {
          distanceTokens: 288,
          factText:
            "Project Grove requires function completedRows(rows). It must return a new array containing only rows where row.completed is exactly true, without mutating rows.",
          prompt: "Using only the earlier Project Grove requirement, return the requested JavaScript function.",
          quality: { threshold: 1, codeFunctionName: "completedRows", codeTestId: "completed-rows" },
        },
      },
      {
        id: "total-wall-ms",
        prompt:
          "Write function totalWallMs(rows) that returns the sum of row.wallMs for every row, or 0 for an empty array.",
        quality: { threshold: 1, codeFunctionName: "totalWallMs", codeTestId: "total-wall-ms" },
        historyRequirement: {
          distanceTokens: 416,
          factText:
            "Project Hill requires function totalLatency(rows). It must return the sum of row.latencyMs for every row, or 0 for an empty array, without mutating rows.",
          prompt: "Using only the earlier Project Hill requirement, return the requested JavaScript function.",
          quality: { threshold: 1, codeFunctionName: "totalLatency", codeTestId: "total-latency" },
        },
      },
      {
        id: "policy-ids",
        prompt:
          "Write function policyIds(rows) that returns a new array of each row.policyId in the same order.",
        quality: { threshold: 1, codeFunctionName: "policyIds", codeTestId: "policy-ids" },
        historyRequirement: {
          distanceTokens: 480,
          factText:
            "Project Iris requires function rowLabels(rows). It must return a new array of each row.label in the same order, without mutating rows.",
          prompt: "Using only the earlier Project Iris requirement, return the requested JavaScript function.",
          quality: { threshold: 1, codeFunctionName: "rowLabels", codeTestId: "row-labels" },
        },
      },
    ],
  },
];

export const LATENCY_TASK_PROFILE = {
  id: "chat-latency",
  label: "General query latency",
  taskType: "latency-only",
  systemPrompt: "Answer the user's question directly and concisely.",
  compactSystemPrompt: "Answer concisely.",
  outputTokenLimits: { short: 32, standard: 64, expanded: 128 },
  cases: [
    {
      id: "material-temperature",
      prompt: "Why does metal feel colder than wood at the same temperature?",
      quality: { latencyOnly: true },
    },
    {
      id: "sleep-consistency",
      prompt: "What are three practical ways to improve sleep consistency?",
      quality: { latencyOnly: true },
    },
    {
      id: "memory-storage",
      prompt: "What is the difference between RAM and persistent storage?",
      quality: { latencyOnly: true },
    },
    {
      id: "transport-comparison",
      prompt: "Compare public transportation with private car travel.",
      quality: { latencyOnly: true },
    },
    {
      id: "worker-overhead",
      prompt: "Why can adding more workers sometimes make a program slower?",
      quality: { latencyOnly: true },
    },
    {
      id: "interview-checklist",
      prompt: "Create a short checklist for preparing for a job interview.",
      quality: { latencyOnly: true },
    },
  ],
};

export const MANUAL_TASK_PROFILES = [...TASK_PROFILES, LATENCY_TASK_PROFILE];

export const LATENCY_BUDGETS = [
  { id: "5", label: "Interactive", targetMs: 5000 },
  { id: "10", label: "Balanced", targetMs: 10000 },
  { id: "15", label: "Quality", targetMs: 15000 },
];

export function getTaskProfile(id) {
  return MANUAL_TASK_PROFILES.find((profile) => profile.id === id) ?? TASK_PROFILES[0];
}

export function getTaskCase(taskProfile, index) {
  return taskProfile.cases[index % taskProfile.cases.length];
}
