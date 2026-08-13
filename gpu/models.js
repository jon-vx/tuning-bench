export const FIXED_MODELS = [
  {
    id: "smol-360m",
    family: "smol-360m",
    model: "SmolLM2-360M-Instruct-q4f16_1-MLC",
    artifacts: {
      q4f16_1: "SmolLM2-360M-Instruct-q4f16_1-MLC",
      q0f16: "SmolLM2-360M-Instruct-q0f16-MLC",
    },
  },
  {
    id: "qwen-0.5b",
    family: "qwen-0.5b",
    model: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
    artifacts: {
      q4f16_1: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
      q0f16: "Qwen2.5-0.5B-Instruct-q0f16-MLC",
    },
  },
  {
    id: "llama-1b",
    family: "llama-1b",
    model: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
    artifacts: {
      q4f16_1: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
      q0f16: "Llama-3.2-1B-Instruct-q0f16-MLC",
    },
  },
  {
    id: "qwen-1.5b",
    family: "qwen-1.5b",
    model: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
    artifacts: {
      q4f16_1: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
    },
  },
  {
    id: "smol-1.7b",
    family: "smol-1.7b",
    model: "SmolLM2-1.7B-Instruct-q4f16_1-MLC",
    artifacts: {
      q4f16_1: "SmolLM2-1.7B-Instruct-q4f16_1-MLC",
    },
  },
];

export function parseQuantization(modelId) {
  return /-(q[^-]+)-MLC$/.exec(modelId)?.[1] ?? "";
}

export function modelArtifactId(modelConfigId, quantization) {
  return FIXED_MODELS.find((model) => model.id === modelConfigId)
    ?.artifacts?.[quantization] ?? "";
}

export function resolveModelArtifact(modelConfigId, artifactId) {
  const model = FIXED_MODELS.find((candidate) => candidate.id === modelConfigId);
  if (!model) throw new Error(`unknown model configuration: ${modelConfigId}`);
  if (!artifactId) return model;
  if (!Object.values(model.artifacts ?? {}).includes(artifactId)) {
    throw new Error(`${artifactId} is not configured for ${modelConfigId}`);
  }
  return { ...model, model: artifactId };
}
