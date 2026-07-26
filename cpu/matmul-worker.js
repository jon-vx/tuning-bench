import { matmulIKJ } from "./matmul.js";

let weights = null;
let K = 0;
let N = 0;

self.onmessage = ({ data }) => {
  if (data.type === "init") {
    weights = new Float32Array(data.weights);
    K = data.K;
    N = data.N;
    self.postMessage({ type: "ready" });
    return;
  }
  if (data.type !== "multiply" || !weights) return;
  const rows = new Float32Array(data.rows);
  const output = matmulIKJ(rows, weights, data.rowCount, K, N);
  self.postMessage(
    { type: "result", requestId: data.requestId, rowStart: data.rowStart, output: output.buffer },
    [output.buffer]
  );
};
