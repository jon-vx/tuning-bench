// matmul.js

import * as np from "https://esm.sh/numpy-ts@1.4.0";

export function matmulNumpy(A, B) {
  return np.dot(A, B);
}

export function matmulIJK(A, B, N) {
  const C = new Float32Array(N * N);
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let k = 0; k < N; k++) s += A[i * N + k] * B[k * N + j];
      C[i * N + j] = s;
    }
  return C;
}

export function matmulIKJ(A, B, N) {
  const C = new Float32Array(N * N);
  for (let i = 0; i < N; i++)
    for (let k = 0; k < N; k++) {
      const a = A[i * N + k];
      for (let j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
    }
  return C;
}
