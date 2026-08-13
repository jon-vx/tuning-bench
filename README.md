# Tuning Bench

Browser benchmark and experiment bed for LLM based GPU tuning.

## Tools

- [Automatic collector](gpu/collect.html)
- [Controlled tuner](gpu/manual-tune.html)
- [Runtime orchestration experiment](gpu/runtime-orchestration.html)
- [CPU benchmark](cpu/cpu.html)

The GPU pages need Chrome or Edge with WebGPU enabled. 

## Run locally

Serve the repository over HTTP:

```sh
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Runtime orchestration experiment

Build the exact WebLLM 0.2.79 experimental bundle before opening the experiment page:

```sh
node scripts/build-runtime-orchestration.mjs
node scripts/runtime-orchestration-server.mjs 8000
```

Open `http://localhost:8000/gpu/runtime-orchestration.html`.

The experiment server automatically stores each completed campaign under `../data/runtime-orchestration/<campaign-id>/`
