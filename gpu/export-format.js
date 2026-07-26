import { toCsv } from "../shared/benchmark-utils.js";

export function manualCsvText(payload) {
  const gpu = payload.deviceProfile.gpu ?? {};
  const rows = payload.results.map((row) => ({
    experiment_timestamp: payload.timestamp,
    label: payload.label,
    experiment_parameter: row.experimentParameter,
    experiment_value: row.experimentValue,
    experiment_block: row.experimentBlock,
    experiment_order_index: row.experimentOrderIndex,
    device_class: payload.deviceProfile.deviceClass,
    platform: payload.deviceProfile.platform,
    cores: payload.deviceProfile.cores,
    mem_gb: payload.deviceProfile.memGB,
    gpu_identity: gpu.identity,
    gpu_identity_source: gpu.identitySource,
    gpu_vendor: gpu.vendor,
    gpu_architecture: gpu.architecture,
    model_family: row.modelFamily,
    model: row.model,
    quantization: row.quantization,
    task_profile_id: row.taskProfileId,
    task_case_id: row.taskCaseId,
    conversation_scenario: row.conversationScenario,
    policy_id: row.policyId,
    baseline_type: row.baselineType,
    context_window: row.requestedContextWindowSize,
    effective_context_window: row.effectiveContextWindowSize,
    configured_history_tokens: row.configuredHistoryTokenBudget,
    history_selection_policy: row.historySelectionPolicy,
    conversation_cache_mode: row.conversationCacheMode,
    conversation_prefix_chars: row.conversationPrefixChars,
    conversation_prefix_output: row.conversationPrefixOutput,
    penalty_processing_mode: row.penaltyProcessingMode,
    response_delivery_mode: row.responseDeliveryMode,
    kv_cache_mode: row.kvCacheMode,
    sliding_window_size: row.slidingWindowSize,
    engine_thread_mode: row.engineThreadMode,
    active_history_tokens: row.historyTokenBudget,
    max_tokens: row.maxTokens,
    prompt_template: row.promptTemplateId,
    library_generation_defaults: row.useLibraryGenerationDefaults,
    prompt_tokens: row.promptTokens,
    completion_tokens: row.completionTokens,
    ttft_ms: row.timeToFirstTokenMs,
    reported_ttft_ms: row.reportedTimeToFirstTokenMs,
    wall_ms: row.wallMs,
    prefill_tok_s: row.prefillTokS,
    decode_tok_s: row.decodeTokS,
    quality_score: row.qualityScore,
    content_quality_score: row.contentQualityScore,
    content_quality_passed: row.contentQualityPassed,
    output_complete: row.outputComplete,
    quality_passed: row.qualityPassed,
    quality_detail: row.qualityDetail,
    finish_reason: row.finishReason,
    output_truncated: row.outputTruncated,
    success: row.success,
    failure_class: row.failureClass,
    error_type: row.errorType,
    output: row.output,
  }));
  const header = Object.keys(rows[0] ?? {});
  return toCsv(header, rows.map((row) => header.map((column) => row[column])));
}

export function manualFileStem(payload) {
  const slug = payload.label.replace(/[^a-z0-9]+/gi, "-");
  return `controlled-${payload.experiment.variedParameter}-${slug}-${payload.timestamp.replace(/[:.]/g, "-")}`;
}

function fixedPrecision(value) {
  return value == null ? "" : Number(value).toFixed(3);
}

export function collectionCsvText(payload) {
  const gpu = payload.deviceProfile.gpu ?? {};
  const limits = gpu.limits ?? {};
  const probe = payload.probe.probeSummary?.bySize ?? {};
  const exportedRuns = [
    ...payload.baselineValidation.runs.map((tuning) => ({ phase: "baseline-validation", tuning })),
    ...payload.retentionValidation.runs.map((tuning) => ({ phase: "retention-validation", tuning })),
    ...payload.tuningRuns.map((tuning) => ({ phase: "tuning", tuning })),
  ];
  const rows = exportedRuns.flatMap(({ phase, tuning }, matrixCaseIndex) => {
    const meta = tuning.meta;
    const qualification = payload.baselineValidation.qualifications.find((item) =>
      item.modelId === meta.modelFamily && item.taskProfileId === meta.taskProfileId
    );
    const retentionQualification = payload.retentionValidation.qualifications.find((item) =>
      item.modelId === meta.modelFamily && item.taskProfileId === meta.taskProfileId
    );
    return tuning.results.map((row) => {
      const summary = tuning.summaries.find((item) => item.policyId === row.policyId);
      return {
        timestamp: meta.timestamp,
        label: payload.label,
        collection_phase: phase,
        policy_group: meta.policyGroupId ?? "baseline-validation",
        matrix_case_index: matrixCaseIndex,
        baseline_qualified: qualification?.qualified,
        baseline_exclusion_reasons: qualification?.reasons.join("|"),
        retention_qualified: retentionQualification?.qualified,
        retention_exclusion_reasons: retentionQualification?.reasons.join("|"),
        webllm_version: meta.webllmVersion,
        device_class: payload.deviceProfile.deviceClass,
        platform: payload.deviceProfile.platform,
        cores: payload.deviceProfile.cores,
        mem_gb: payload.deviceProfile.memGB,
        gpu_identity: gpu.identity,
        gpu_identity_source: gpu.identitySource,
        gpu_vendor: gpu.vendor,
        gpu_architecture: gpu.architecture,
        gpu_device: gpu.device,
        gpu_webgl_renderer: gpu.webglRenderer,
        gpu_features: gpu.features?.join("|"),
        gpu_max_buffer_size: limits.maxBufferSize,
        gpu_max_storage_binding_size: limits.maxStorageBufferBindingSize,
        probe_128_gflops: probe["128"]?.gflops,
        probe_256_gflops: probe["256"]?.gflops,
        probe_512_gflops: probe["512"]?.gflops,
        model_family: row.modelFamily,
        model: row.model,
        quantization: row.quantization,
        task_profile_id: row.taskProfileId,
        task_case_id: row.taskCaseId,
        task_type: row.taskType,
        conversation_scenario: row.conversationScenario,
        policy_id: row.policyId,
        policy_label: row.policyLabel,
        baseline_type: row.baselineType,
        execution_index: row.executionIndex,
        run_index: row.runIndex,
        requested_context_window: row.requestedContextWindowSize,
        effective_context_window: row.effectiveContextWindowSize,
        history_token_budget: row.historyTokenBudget,
        configured_history_token_budget: row.configuredHistoryTokenBudget,
        history_selection_policy: row.historySelectionPolicy,
        conversation_cache_mode: row.conversationCacheMode,
        conversation_prefix_chars: row.conversationPrefixChars,
        prompt_template_id: row.promptTemplateId,
        library_generation_defaults: row.useLibraryGenerationDefaults,
        max_tokens: row.maxTokens,
        output_token_tier: row.outputTokenTier,
        temperature: row.requestedTemperature,
        top_p: row.requestedTopP,
        repetition_penalty: row.requestedRepetitionPenalty,
        frequency_penalty: row.requestedFrequencyPenalty,
        presence_penalty: row.requestedPresencePenalty,
        seed: row.requestedSeed,
        estimated_system_tokens: row.estimatedSystemTokens,
        estimated_history_tokens: row.estimatedHistoryTokens,
        estimated_user_tokens: row.estimatedUserTokens,
        estimated_input_tokens: row.estimatedInputTokens,
        prompt_tokens: row.promptTokens,
        completion_tokens: row.completionTokens,
        policy_load_ms: fixedPrecision(row.policyLoadMs),
        engine_cache_state: row.engineCacheState,
        ttft_ms: fixedPrecision(row.timeToFirstTokenMs),
        wall_ms: fixedPrecision(row.wallMs),
        prefill_tok_s: row.prefillTokS,
        decode_tok_s: row.decodeTokS,
        estimated_prefill_ms: fixedPrecision(row.estimatedPrefillMs),
        decode_ms_per_token: fixedPrecision(row.decodeMsPerToken),
        latency_target_ms: row.latencyTargetMs,
        latency_target_met: row.latencyTargetMet,
        quality_score: row.qualityScore,
        content_quality_score: row.contentQualityScore,
        content_quality_passed: row.contentQualityPassed,
        output_complete: row.outputComplete,
        quality_passed: row.qualityPassed,
        quality_detail: row.qualityDetail,
        quality_delta_vs_controlled: summary?.qualityDeltaVsControlled,
        quality_retained_vs_controlled: summary?.qualityRetainedVsControlled,
        matched_output_equality_rate: summary?.matchedOutputEqualityRate,
        matched_prefix_equality_rate: summary?.matchedPrefixEqualityRate,
        finish_reason: row.finishReason,
        output_truncated: row.outputTruncated,
        success: row.success,
        failure_class: row.failureClass,
        failure_keep: row.failureKeep,
        error_stage: row.errorStage,
        error_type: row.errorType,
        timeout: row.timeout,
        recommended_policy_id: tuning.recommendedPolicyId,
        shuffle_seed: meta.shuffleSeed,
        policy_execution_order: meta.policyExecutionOrder.join("|"),
        user_agent: payload.deviceProfile.userAgent,
      };
    });
  });
  const header = Object.keys(rows[0] ?? {});
  return toCsv(header, rows.map((row) => header.map((column) => row[column])));
}

export function collectionFileStem(payload) {
  const slug = payload.label ? `${payload.label.replace(/[^a-z0-9]+/gi, "-")}-` : "";
  return `model-task-matrix-${slug}${payload.timestamp.replace(/[:.]/g, "-")}`;
}
