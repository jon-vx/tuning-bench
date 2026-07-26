# WebLLM Runtime Tuning Bench

Browser benchmark and expirment bed for WebLLM and CPU tuning. 

## Tools

- [Automatic collector](gpu/collect.html)
- [Controlled tuner](gpu/manual-tune.html)
- [CPU benchmark](cpu/cpu.html)

The GPU pages need Chrome or Edge with WebGPU enabled. Nothing is uploaded automatically.

## Run locally

Serve the repository over HTTP:

```sh
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

