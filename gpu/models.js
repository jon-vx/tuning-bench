export const FIXED_MODELS = [
  {
    id: "smol-360m",
    family: "smol-360m",
    model: "SmolLM2-360M-Instruct-q4f16_1-MLC",
  },
  {
    id: "qwen-0.5b",
    family: "qwen-0.5b",
    model: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
  },
  {
    id: "llama-1b",
    family: "llama-1b",
    model: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
  },
];

export function parseQuantization(modelId) {
  return /-(q[^-]+)-MLC$/.exec(modelId)?.[1] ?? "";
}

