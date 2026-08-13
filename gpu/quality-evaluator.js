function normalizedText(output) {
  return output.toLowerCase().replace(/[^a-z0-9.+-]+/g, " ").trim();
}

function includesAffirmativeTerm(text, term) {
  const normalizedTerm = normalizedText(term);
  const escaped = normalizedTerm.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(`(^|[^a-z0-9])${escaped}(?=$|[^a-z0-9])`, "g");
  const negations = new Set(["no", "not", "never", "without", "isnt", "doesnt", "cannot"]);
  for (const match of text.matchAll(pattern)) {
    const prefix = text.slice(0, match.index).trim().split(/\s+/).slice(-3);
    if (!prefix.some((word) => negations.has(word))) return true;
  }
  return false;
}

function conceptCoverage(output, groups = []) {
  if (!groups.length) return 1;
  const text = normalizedText(output);
  const matched = groups.filter((terms) =>
    terms.some((term) => includesAffirmativeTerm(text, term))
  );
  return matched.length / groups.length;
}

function extractJson(output) {
  const fenced = /```(?:json)?\s*([\s\S]*?)```/i.exec(output)?.[1];
  const candidate = fenced ?? output.slice(output.indexOf("{"), output.lastIndexOf("}") + 1);
  if (!candidate) return null;
  try {
    return JSON.parse(candidate.trim());
  } catch {
    return null;
  }
}

function jsonRecordCoverage(parsed, expectedRecords = [], expectedKeys = []) {
  if (!expectedRecords.length) return 1;
  if (!parsed || !Array.isArray(parsed.records) || parsed.records.length !== expectedRecords.length) {
    return 0;
  }
  if (Object.keys(parsed).length !== 1 || !("records" in parsed)) return 0;
  if (expectedKeys.length && parsed.records.some((record) => {
    if (!record || Array.isArray(record) || typeof record !== "object") return true;
    const keys = Object.keys(record).sort();
    return keys.length !== expectedKeys.length ||
      keys.some((key, index) => key !== [...expectedKeys].sort()[index]);
  })) return 0;
  const candidates = parsed.records.map((record) => normalizedText(JSON.stringify(record)));
  const used = new Set();
  let matched = 0;
  for (const expected of expectedRecords) {
    const index = candidates.findIndex((candidate, candidateIndex) =>
      !used.has(candidateIndex) &&
      expected.every((terms) => terms.some((term) => candidate.includes(normalizedText(term).trim())))
    );
    if (index >= 0) {
      used.add(index);
      matched += 1;
    }
  }
  return matched / expectedRecords.length;
}

function extractCode(output) {
  return /```(?:javascript|js)?\s*([\s\S]*?)```/i.exec(output)?.[1]?.trim() || output.trim();
}

async function testGeneratedCode(output, functionName, testId) {
  if (typeof Worker === "undefined") return { passed: false, detail: "worker_unavailable" };
  const workerSource = `
    self.fetch = undefined;
    self.XMLHttpRequest = undefined;
    self.WebSocket = undefined;
    self.importScripts = undefined;
    self.onmessage = ({ data }) => {
      try {
        const getFunction = new Function(data.code + "\\n; return typeof " + data.functionName + " === 'function' ? " + data.functionName + " : null;");
        const fn = getFunction();
        if (!fn) throw new Error("required function not found");
        const fixtures = {
          "successful-rows": {
            rows: [
              { id: "a", success: true }, { id: "b", success: false },
              { id: "c", success: true }
            ],
            expected: [{ id: "a", success: true }, { id: "c", success: true }],
            requireNewArray: true
          },
          "completed-rows": {
            rows: [{ id: 1, completed: false }, { id: 2, completed: true }],
            expected: [{ id: 2, completed: true }],
            requireNewArray: true
          },
          "total-wall-ms": {
            rows: [{ wallMs: 10 }, { wallMs: 25 }], expected: 35, emptyExpected: 0
          },
          "total-latency": {
            rows: [{ latencyMs: 7 }, { latencyMs: 11 }], expected: 18, emptyExpected: 0
          },
          "policy-ids": {
            rows: [{ policyId: "control" }, { policyId: "tuned" }],
            expected: ["control", "tuned"],
            requireNewArray: true
          },
          "row-labels": {
            rows: [{ label: "a" }, { label: "b" }],
            expected: ["a", "b"],
            requireNewArray: true
          }
        };
        const fixture = fixtures[data.testId];
        if (!fixture) throw new Error("unknown code test");
        const inputBefore = JSON.stringify(fixture.rows);
        const result = fn(fixture.rows);
        const value = result instanceof Map ? Object.fromEntries(result) : result;
        const correct = JSON.stringify(value) === JSON.stringify(fixture.expected);
        const unchanged = JSON.stringify(fixture.rows) === inputBefore;
        const newArray = !fixture.requireNewArray || result !== fixture.rows;
        const emptyCorrect = !("emptyExpected" in fixture) ||
          JSON.stringify(fn([])) === JSON.stringify(fixture.emptyExpected);
        const passed = correct && unchanged && newArray && emptyCorrect;
        self.postMessage({ passed, detail: passed ? "tests_passed" : "wrong_result" });
      } catch (error) {
        self.postMessage({ passed: false, detail: error.message });
      }
    };
  `;
  const url = URL.createObjectURL(new Blob([workerSource], { type: "text/javascript" }));
  const worker = new Worker(url);
  try {
    return await new Promise((resolve) => {
      const timer = setTimeout(() => {
        worker.terminate();
        resolve({ passed: false, detail: "test_timeout" });
      }, 1000);
      worker.onmessage = ({ data }) => {
        clearTimeout(timer);
        resolve(data);
      };
      worker.onerror = (event) => {
        clearTimeout(timer);
        resolve({ passed: false, detail: event.message || "worker_error" });
      };
      worker.postMessage({ code: extractCode(output), functionName, testId });
    });
  } finally {
    worker.terminate();
    URL.revokeObjectURL(url);
  }
}

export async function evaluateTaskOutput(taskProfile, output) {
  const quality = taskProfile.quality ?? {};
  if (!output.trim()) return { score: 0, passed: false, detail: "empty_output" };
  if (quality.latencyOnly) {
    return { score: 1, passed: true, detail: "generation_observed" };
  }

  if (quality.codeFunctionName) {
    const test = await testGeneratedCode(output, quality.codeFunctionName, quality.codeTestId);
    return { score: test.passed ? 1 : 0, passed: test.passed, detail: test.detail };
  }

  const checks = [];
  if (quality.conceptGroups) {
    checks.push({ name: "concept_coverage", score: conceptCoverage(output, quality.conceptGroups) });
  }
  if (quality.expectedJsonRecords) {
    const parsed = extractJson(output);
    checks.push({ name: "valid_json", score: parsed ? 1 : 0 });
    checks.push({
      name: "record_coverage",
      score: jsonRecordCoverage(
        parsed,
        quality.expectedJsonRecords,
        quality.expectedJsonKeys
      ),
    });
  } else if (quality.requireJson) {
    checks.push({ name: "valid_json", score: extractJson(output) ? 1 : 0 });
  }
  if (quality.minBulletCount) {
    const bullets = output.split("\n").filter((line) => /^\s*(?:[-*\u2022]|\d+[.)])\s+/.test(line)).length;
    checks.push({ name: "bullet_count", score: Math.min(1, bullets / quality.minBulletCount) });
  }
  const score = checks.length
    ? checks.reduce((sum, check) => sum + check.score, 0) / checks.length
    : 0;
  const threshold = quality.threshold ?? 0.6;
  const minimumCheckScores = quality.minimumCheckScores ?? {};
  const checksPassed = checks.every((check) =>
    check.score >= (minimumCheckScores[check.name] ?? 0)
  );
  return {
    score,
    passed: score >= threshold && checksPassed,
    detail: checks.map((check) => `${check.name}:${check.score.toFixed(2)}`).join("|"),
  };
}
