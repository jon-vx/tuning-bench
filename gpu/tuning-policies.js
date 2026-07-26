export const CONTROLLED_GENERATION = {
  temperature: 0,
  topP: 1,
  repetitionPenalty: 1,
  frequencyPenalty: 0,
  presencePenalty: 0,
  seed: 42,
};

const QUALITY_SAFE_BASELINE = {
  contextWindowSize: 4096,
  historyTokenBudget: 512,
  outputTokenTier: "standard",
  maxTokens: null,
  promptTemplateId: "full",
  historySelectionPolicy: "tail",
  useLibraryGenerationDefaults: false,
  ...CONTROLLED_GENERATION,
};

export const TUNING_POLICIES = [
  {
    id: "library-default",
    label: "Untouched WebLLM reference",
    baselineType: "library_default",
    contextWindowSize: null,
    historyTokenBudget: 512,
    outputTokenTier: "library-default",
    maxTokens: null,
    promptTemplateId: "full",
    historySelectionPolicy: "tail",
    useLibraryGenerationDefaults: true,
  },
  {
    ...QUALITY_SAFE_BASELINE,
    id: "controlled-baseline",
    label: "512-token tail baseline",
    baselineType: "controlled",
  },
  {
    ...QUALITY_SAFE_BASELINE,
    id: "history-96-relevance",
    label: "96-token relevance history",
    baselineType: "",
    historyTokenBudget: 96,
    historySelectionPolicy: "relevance",
  },
];

export const EXPLORATORY_POLICIES = [
  TUNING_POLICIES[0],
  TUNING_POLICIES[1],
  {
    ...QUALITY_SAFE_BASELINE,
    id: "small-kv",
    label: "Small KV cache",
    baselineType: "",
    contextWindowSize: 1024,
  },
  {
    ...QUALITY_SAFE_BASELINE,
    id: "short-history",
    label: "Short history",
    baselineType: "",
    historyTokenBudget: 128,
  },
  {
    ...QUALITY_SAFE_BASELINE,
    id: "short-output",
    label: "Short output",
    baselineType: "",
    outputTokenTier: "short",
  },
  {
    ...QUALITY_SAFE_BASELINE,
    id: "compact-prompt",
    label: "Compact prompt",
    baselineType: "",
    promptTemplateId: "compact",
  },
  {
    ...QUALITY_SAFE_BASELINE,
    id: "responsive",
    label: "Responsive",
    baselineType: "",
    contextWindowSize: 1024,
    historyTokenBudget: 128,
    outputTokenTier: "short",
    promptTemplateId: "compact",
  },
  {
    ...QUALITY_SAFE_BASELINE,
    id: "expanded",
    label: "Expanded quality",
    baselineType: "",
    historyTokenBudget: 768,
    outputTokenTier: "expanded",
  },
];

export const AUTOMATIC_POLICY_GROUPS = [
  {
    id: "focused-history",
    label: "Focused history validation",
    policies: TUNING_POLICIES,
    measuredConversationScenarios: ["long-history-distraction", "long-history-required"],
    requiresHistoryRequirements: true,
  },
  {
    id: "exploratory-matrix",
    label: "Exploratory policy matrix",
    policies: EXPLORATORY_POLICIES,
    measuredConversationScenarios: ["single-turn", "long-history-distraction"],
    requiresHistoryRequirements: false,
  },
];
