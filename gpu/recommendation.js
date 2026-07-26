function isQualitySafe(summary, minimumPassRate, qualityTolerance, referenceQuality) {
  return summary.successRate === 1 &&
    summary.qualityPassRate >= minimumPassRate &&
    summary.contentQualityPassRate >= minimumPassRate &&
    summary.outputCompletionRate === 1 &&
    !summary.anyTruncated &&
    summary.medianQualityScore >= referenceQuality - qualityTolerance &&
    Object.values(summary.scenarioMetrics).every(
      (metrics) => metrics.qualityPassRate >= minimumPassRate
    );
}

export function chooseRecommendedPolicy(summaries, criteria) {
  const reference = summaries.find((summary) => summary.baselineType === "library_default");
  const controlled = summaries.find((summary) => summary.baselineType === "controlled");
  if (!Number.isFinite(reference?.medianTtftMs) || !Number.isFinite(controlled?.medianTtftMs)) {
    return null;
  }

  const referenceQuality = Math.min(reference.medianQualityScore, controlled.medianQualityScore);
  const eligible = summaries.filter((summary) => {
    if (summary.baselineType || !isQualitySafe(
      summary,
      criteria.minimumPassRate,
      criteria.qualityTolerance,
      referenceQuality
    )) return false;
    if (summary.conversationCacheMode === "prefix-reuse" &&
        (summary.matchedOutputEqualityRate !== 1 || summary.matchedPrefixEqualityRate !== 1)) {
      return false;
    }
    const requiredFactor = 1 - criteria.minimumTtftImprovement;
    return summary.medianTtftMs <= reference.medianTtftMs * requiredFactor &&
      summary.medianTtftMs <= controlled.medianTtftMs * requiredFactor &&
      summary.latencyTargetHitRate >= reference.latencyTargetHitRate;
  });

  return [...eligible].sort((a, b) =>
    b.latencyTargetHitRate - a.latencyTargetHitRate ||
    a.medianTtftMs - b.medianTtftMs ||
    a.medianWallMs - b.medianWallMs ||
    b.medianQualityScore - a.medianQualityScore ||
    b.historyTokenBudget - a.historyTokenBudget
  )[0] ?? null;
}
