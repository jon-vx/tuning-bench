export function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function quantile(values, fraction) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const index = (sorted.length - 1) * fraction;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

export function matchedRows(rows, referenceRows) {
  const key = (row) => [
    row.experimentBlock ?? 0,
    row.taskCaseId,
    row.conversationScenario,
    row.runIndex,
  ].join("|");
  const references = new Map(
    referenceRows.filter((row) => row.success).map((row) => [key(row), row])
  );
  return rows
    .filter((row) => row.success)
    .map((row) => ({ row, reference: references.get(key(row)) }))
    .filter((pair) => pair.reference);
}

export function matchedEqualityRate(rows, referenceRows, field) {
  const pairs = matchedRows(rows, referenceRows);
  if (!pairs.length) return null;
  return pairs.filter(({ row, reference }) => row[field] === reference[field]).length / pairs.length;
}

function csvCell(value) {
  const text = value == null ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function toCsv(header, rows) {
  return [header, ...rows].map((row) => row.map(csvCell).join(",")).join("\n");
}

function download(text, name, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadJSON(value, name) {
  download(JSON.stringify(value, null, 2), name, "application/json");
}

export function downloadCSV(text, name) {
  download(text, name, "text/csv");
}
