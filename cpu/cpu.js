import { downloadCSV, downloadJSON } from "../shared/benchmark-utils.js";
import {
  runCpuBenchmark,
} from "./cpu-benchmark.js";
import { cpuCsvText, cpuFileStem } from "./export-format.js";

const $ = (id) => document.getElementById(id);
const logElement = $("log");
const statusElement = $("status");
const runButton = $("run-btn");
const labelElement = $("label");
const exportButton = $("export-btn");
const exportCsvButton = $("export-csv-btn");

let payload = null;

function setExportsEnabled(enabled) {
  exportButton.disabled = !enabled;
  exportCsvButton.disabled = !enabled;
}

function log(message) {
  logElement.textContent += `${message}\n`;
  logElement.scrollTop = logElement.scrollHeight;
}

async function runBenchmark() {
  runButton.disabled = true;
  setExportsEnabled(false);
  logElement.textContent = "";
  statusElement.textContent = "starting...";

  try {
    payload = await runCpuBenchmark(
      log,
      (message) => {
        statusElement.textContent = message;
      },
      labelElement.value.trim()
    );
    window.__cpuBenchmarkPayload = payload;
    setExportsEnabled(true);
    statusElement.textContent = `done; ${payload.results.length} configurations`;
  } catch (error) {
    statusElement.textContent = `error: ${error.message}`;
    log(error.stack ?? String(error));
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", runBenchmark);
exportButton.addEventListener("click", () => {
  downloadJSON(payload, `${cpuFileStem(payload)}.json`);
});
exportCsvButton.addEventListener("click", () => {
  downloadCSV(cpuCsvText(payload), `${cpuFileStem(payload)}.csv`);
});
