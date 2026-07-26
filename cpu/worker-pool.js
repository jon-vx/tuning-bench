export class MatmulWorkerPool {
  constructor(workerCount, weights, K, N) {
    this.workerCount = workerCount;
    this.K = K;
    this.N = N;
    this.requestId = 0;
    this.workers = Array.from({ length: workerCount }, () =>
      new Worker(new URL("./matmul-worker.js", import.meta.url), { type: "module" })
    );
    this.ready = Promise.all(this.workers.map((worker) => new Promise((resolve, reject) => {
      const cleanup = () => {
        worker.removeEventListener("message", onMessage);
        worker.removeEventListener("error", onError);
      };
      const onMessage = ({ data }) => {
        if (data.type !== "ready") return;
        cleanup();
        resolve();
      };
      const onError = (error) => {
        cleanup();
        reject(error);
      };
      worker.addEventListener("message", onMessage);
      worker.addEventListener("error", onError);
      worker.postMessage({ type: "init", weights: weights.slice().buffer, K, N });
    })));
  }

  async multiply(A, M) {
    await this.ready;
    const requestId = ++this.requestId;
    const output = new Float32Array(M * this.N);
    const rowsPerWorker = Math.ceil(M / this.workerCount);
    const jobs = this.workers.map((worker, index) => {
      const rowStart = index * rowsPerWorker;
      const rowEnd = Math.min(M, rowStart + rowsPerWorker);
      if (rowStart >= rowEnd) return Promise.resolve();
      const rows = A.slice(rowStart * this.K, rowEnd * this.K);
      return new Promise((resolve, reject) => {
        const cleanup = () => {
          worker.removeEventListener("message", onMessage);
          worker.removeEventListener("error", onError);
        };
        const onMessage = ({ data }) => {
          if (data.type !== "result" || data.requestId !== requestId) return;
          cleanup();
          output.set(new Float32Array(data.output), data.rowStart * this.N);
          resolve();
        };
        const onError = (error) => {
          cleanup();
          reject(error);
        };
        worker.addEventListener("message", onMessage);
        worker.addEventListener("error", onError);
        worker.postMessage({
          type: "multiply",
          requestId,
          rowStart,
          rowCount: rowEnd - rowStart,
          rows: rows.buffer,
        }, [rows.buffer]);
      });
    });
    await Promise.all(jobs);
    return output;
  }

  close() {
    for (const worker of this.workers) worker.terminate();
  }
}
