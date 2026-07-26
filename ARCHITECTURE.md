# Code map

This is a static site. There is no build step. The three pages are:

- `gpu/collect.html`: automatic WebLLM data collection
- `gpu/manual-tune.html`: one-variable WebLLM experiments
- `cpu/cpu.html`: JavaScript CPU kernel benchmarks

## Shared

`shared/benchmark-utils.js` has the statistics, CSV escaping, and download code used by more than one page.

`shared/styles.css` has the common page styles. Page-specific CSS stays in the HTML files.

## Automatic collector

`gpu/collect.js` reads the form, displays progress, and exports the finished run.

`gpu/collection-runner.js` runs these steps:

1. Read the device profile.
2. Run the WebGPU probe.
3. Test the controlled baseline for every model/task pair.
4. Test required-history retention when the selected workflow needs it.
5. Run tuning policies for the pairs that passed.
6. Build the JSON payload.

The runner calls the UI functions passed by `collect.js`; it does not access the DOM.

`gpu/collection-matrix.js` builds model/task pairs, counts runs, and decides whether a baseline passed.

`gpu/export-format.js` contains the collector and manual-tuner CSV layouts and filenames.

## WebLLM runs
There is no build step. See [ARCHITECTURE.md](ARCHITECTURE.md) for the file layout and benchmark flow.

`gpu/fixed-model-bench.js` builds prompts, runs policies, scores responses, summarizes measurements, and chooses a recommendation. Both GPU pages call `runFixedModelPoliciesCore`.

`gpu/webllm-runtime.js` holds the loaded engine and worker. It handles model loading, request options, streaming, timeouts, runtime statistics, and unloading.

WebLLM is pinned to `0.2.79` in:

- `gpu/webllm-runtime.js`
- `gpu/webllm-worker.js`
- the exported metadata in `gpu/fixed-model-bench.js`

`gpu/models.js` lists the fixed models.

`gpu/task-profiles.js` contains prompts, expected answers, history fixtures, and task-specific output limits. These values should not change during a data-collection round.

`gpu/tuning-policies.js` contains automatic tuning policies and workflow groups.

`gpu/history-selection.js` selects relevant history chunks.

`gpu/quality-evaluator.js` scores task output.

`gpu/recommendation.js` applies the acceptance criteria to policy summaries.

## Manual tuner

`gpu/manual-tune.js` reads the experiment controls, runs counterbalanced blocks, displays results, and exports the payload.

`gpu/controlled-experiment.js` contains:

- `EXPERIMENT_DEFINITIONS`: values and workloads shown in the parameter menu
- `BASE_SETTINGS`: settings held fixed during a sweep
- policy construction
- counterbalanced ordering
- matched-run analysis and acceptance rules

`gpu/experiment-plots.js` draws the history dose-response plots.

To add a manual parameter, add a definition in `EXPERIMENT_DEFINITIONS`. If the value needs a new WebLLM option, also map it in `gpu/webllm-runtime.js`.

## CPU benchmark

`cpu/cpu.js` handles the page and exports.

`cpu/cpu-benchmark.js` contains the benchmark shapes, variants, run order, correctness checks, and summaries.

`cpu/matmul.js` contains matrix multiplication kernels.

`cpu/llm-ops.js` contains quantized matvec, RMSNorm, softmax, and fused residual normalization.

`cpu/worker-pool.js` and `cpu/matmul-worker.js` run the worker-count variants.

`cpu/bench.js` contains timing calibration and summary statistics.

`cpu/export-format.js` contains the CPU CSV layout and filename.


