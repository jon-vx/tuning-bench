const STOP_WORDS = new Set([
  "about", "conversation", "earlier", "from", "into", "only", "points",
  "summarize", "summary", "the", "this", "using", "with",
]);

function terms(text) {
  return new Set(
    text.toLowerCase().match(/[a-z0-9][a-z0-9.-]*/g)
      ?.filter((term) => term.length > 2 && !STOP_WORDS.has(term)) ?? []
  );
}

export function selectRelevantHistory(sourceWords, query, wordBudget, chunkSize = 24) {
  if (sourceWords.length <= wordBudget) return sourceWords.join(" ");
  const queryTerms = terms(query);
  const chunks = [];
  for (let index = 0; index < sourceWords.length; index += chunkSize) {
    const words = sourceWords.slice(index, index + chunkSize);
    const chunkTerms = terms(words.join(" "));
    const score = [...queryTerms].filter((term) => chunkTerms.has(term)).length;
    chunks.push({ index, score, words });
  }
  chunks.sort((a, b) => b.score - a.score || a.index - b.index);
  const selected = [];
  for (const chunk of chunks) {
    const remaining = wordBudget - selected.length;
    if (remaining <= 0) break;
    selected.push(...chunk.words.slice(0, remaining));
  }
  return selected.join(" ");
}
