import {
  CreateMLCEngine,
  CreateWebWorkerMLCEngine,
} from "https://esm.run/@mlc-ai/web-llm@0.2.79";

let engine = null;
let worker = null;
let loadedEngineKey = "";

export async function loadWebLlmModel(modelId, policy, reportStatus) {
  const contextKey = policy.contextWindowSize ?? "library-default";
  const penaltyKey = policy.penaltyProcessingMode ?? "default-penalties";
  const kvCacheKey = policy.kvCacheMode ?? "default-kv";
  const slidingKey = policy.slidingWindowSize ?? "default-sliding";
  const sinkKey = policy.attentionSinkSize ?? "default-sink";
  const threadKey = policy.engineThreadMode ?? "main";
  const engineKey =
    `${modelId}:${contextKey}:${penaltyKey}:${kvCacheKey}:${slidingKey}:${sinkKey}:${threadKey}`;

  if (engine && loadedEngineKey === engineKey) {
    return { loadMs: 0, engineCacheState: "already_loaded" };
  }

  await unloadWebLlm(reportStatus);
  const startedAt = performance.now();
  const engineConfig = {
    initProgressCallback: (progress) => reportStatus(progress.text),
  };
  const chatOptions = {};

  if (policy.slidingWindowSize != null) {
    chatOptions.context_window_size = -1;
    chatOptions.sliding_window_size = policy.slidingWindowSize;
    chatOptions.attention_sink_size = policy.attentionSinkSize ?? 0;
  } else if (policy.kvCacheMode === "sliding") {
    chatOptions.context_window_size = -1;
    chatOptions.sliding_window_size = 4096;
    chatOptions.attention_sink_size = policy.attentionSinkSize ?? 0;
  } else if (policy.contextWindowSize != null) {
    chatOptions.context_window_size = policy.contextWindowSize;
  }

  if (policy.penaltyProcessingMode === "bypass-zero") {
    chatOptions.frequency_penalty = undefined;
    chatOptions.presence_penalty = undefined;
    chatOptions.repetition_penalty = 1;
  }

  try {
    if (policy.engineThreadMode === "worker") {
      worker = new Worker(new URL("./webllm-worker.js", import.meta.url), { type: "module" });
      engine = Object.keys(chatOptions).length
        ? await CreateWebWorkerMLCEngine(worker, modelId, engineConfig, chatOptions)
        : await CreateWebWorkerMLCEngine(worker, modelId, engineConfig);
    } else {
      engine = Object.keys(chatOptions).length
        ? await CreateMLCEngine(modelId, engineConfig, chatOptions)
        : await CreateMLCEngine(modelId, engineConfig);
    }
  } catch (error) {
    worker?.terminate();
    worker = null;
    throw error;
  }

  loadedEngineKey = engineKey;
  return {
    loadMs: performance.now() - startedAt,
    engineCacheState: "engine_loaded",
  };
}

export async function resetWebLlmChat() {
  await engine.resetChat();
}

export async function completeWithWebLlm(messages, policy, timeoutMs, onFirstToken = () => {}) {
  let output = "";
  let usage = {};
  let finishReason = "";
  let timeout;

  const request = { messages };
  if (!policy.useLibraryGenerationDefaults) {
    request.stream = policy.responseDeliveryMode !== "non-streaming";
  }
  if (request.stream) request.stream_options = { include_usage: true };
  if (policy.maxTokens != null) request.max_tokens = policy.maxTokens;
  if (policy.logprobsMode === "on") request.logprobs = true;
  if (policy.topLogprobs != null) request.top_logprobs = policy.topLogprobs;
  if (policy.responseFormatMode === "json-object") {
    request.response_format = { type: "json_object" };
  }
  if (!policy.useLibraryGenerationDefaults) {
    request.temperature = policy.temperature;
    request.top_p = policy.topP;
    request.repetition_penalty = policy.repetitionPenalty;
    if (policy.penaltyProcessingMode !== "bypass-zero") {
      request.frequency_penalty = policy.frequencyPenalty;
      request.presence_penalty = policy.presencePenalty;
    }
    request.seed = policy.seed;
  }

  const consumeResponse = async () => {
    const response = await engine.chat.completions.create(request);
    if (!response?.[Symbol.asyncIterator]) {
      output = response.choices?.[0]?.message?.content || "";
      finishReason = response.choices?.[0]?.finish_reason || "";
      usage = response.usage || {};
      return;
    }

    for await (const chunk of response) {
      const choice = chunk.choices?.[0];
      const delta = choice?.delta?.content || "";
      if (delta) onFirstToken();
      output += delta;
      if (choice?.finish_reason) finishReason = choice.finish_reason;
      if (chunk.usage) usage = chunk.usage;
    }
  };

  const timeoutPromise = new Promise((_, reject) => {
    timeout = setTimeout(() => {
      Promise.resolve()
        .then(() => engine.interruptGenerate())
        .catch(() => {});
      reject(new Error(`inference timeout after ${timeoutMs}ms`));
    }, timeoutMs);
  });

  try {
    await Promise.race([consumeResponse(), timeoutPromise]);
  } finally {
    clearTimeout(timeout);
  }

  return { output, usage, finishReason };
}

export async function readWebLlmStats() {
  const text = await engine.runtimeStatsText();
  return {
    prefillTokS: Number(/prefill:\s*([\d.]+)/.exec(text)?.[1]) || null,
    decodeTokS: Number(/decod\w*:\s*([\d.]+)/.exec(text)?.[1]) || null,
  };
}

export async function unloadWebLlm(report) {
  try {
    if (engine) await engine.unload();
  } catch (error) {
    report(`engine unload failed: ${error.message || error}`);
  } finally {
    engine = null;
    worker?.terminate();
    worker = null;
    loadedEngineKey = "";
  }
}

export function classifyWebLlmError(error, stage) {
  const message = error?.message ?? String(error);
  const lower = message.toLowerCase();
  if (lower.includes("timeout")) {
    return { failureClass: "timeout", failureKeep: true, errorStage: stage };
  }
  if (lower.includes("out of memory") || lower.includes("oom") || lower.includes("memory")) {
    return { failureClass: "oom", failureKeep: true, errorStage: stage };
  }
  if (lower.includes("device lost")) {
    return { failureClass: "device_lost", failureKeep: true, errorStage: stage };
  }
  if (lower.includes("fetch") || lower.includes("network") || lower.includes("timed out")) {
    return { failureClass: "network_unstable", failureKeep: false, errorStage: stage };
  }
  return { failureClass: "runtime_exception", failureKeep: true, errorStage: stage };
}
