function dimensions(M, K, N) {
  return K == null ? { M, K: M, N: M } : { M, K, N };
}

function outputBuffer(output, length, ArrayType = Float32Array) {
  const target = output ?? new ArrayType(length);
  if (target.length !== length) throw new Error(`output length ${target.length} != ${length}`);
  target.fill(0);
  return target;
}

export function matmulIJK(A, B, M, K, N, output) {
  ({ M, K, N } = dimensions(M, K, N));
  const C = outputBuffer(output, M * N, output?.constructor);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j];
      C[i * N + j] = sum;
    }
  }
  return C;
}

export function matmulIKJ(A, B, M, K, N, output) {
  ({ M, K, N } = dimensions(M, K, N));
  const C = outputBuffer(output, M * N, output?.constructor);
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < K; k++) {
      const value = A[i * K + k];
      for (let j = 0; j < N; j++) C[i * N + j] += value * B[k * N + j];
    }
  }
  return C;
}

export function matmulJIK(A, B, M, K, N, output) {
  ({ M, K, N } = dimensions(M, K, N));
  const C = outputBuffer(output, M * N, output?.constructor);
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < M; i++) {
      let sum = 0;
      for (let k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j];
      C[i * N + j] = sum;
    }
  }
  return C;
}

export function matmulJKI(A, B, M, K, N, output) {
  ({ M, K, N } = dimensions(M, K, N));
  const C = outputBuffer(output, M * N, output?.constructor);
  for (let j = 0; j < N; j++) {
    for (let k = 0; k < K; k++) {
      const b = B[k * N + j];
      for (let i = 0; i < M; i++) C[i * N + j] += A[i * K + k] * b;
    }
  }
  return C;
}

export function matmulKIJ(A, B, M, K, N, output) {
  ({ M, K, N } = dimensions(M, K, N));
  const C = outputBuffer(output, M * N, output?.constructor);
  for (let k = 0; k < K; k++) {
    for (let i = 0; i < M; i++) {
      const value = A[i * K + k];
      for (let j = 0; j < N; j++) C[i * N + j] += value * B[k * N + j];
    }
  }
  return C;
}

export function matmulKJI(A, B, M, K, N, output) {
  ({ M, K, N } = dimensions(M, K, N));
  const C = outputBuffer(output, M * N, output?.constructor);
  for (let k = 0; k < K; k++) {
    for (let j = 0; j < N; j++) {
      const b = B[k * N + j];
      for (let i = 0; i < M; i++) C[i * N + j] += A[i * K + k] * b;
    }
  }
  return C;
}

export function matmulTiled(A, B, M, K, N, tileSize, output) {
  const C = outputBuffer(output, M * N, output?.constructor);
  for (let ii = 0; ii < M; ii += tileSize) {
    for (let kk = 0; kk < K; kk += tileSize) {
      for (let jj = 0; jj < N; jj += tileSize) {
        const iEnd = Math.min(ii + tileSize, M);
        const kEnd = Math.min(kk + tileSize, K);
        const jEnd = Math.min(jj + tileSize, N);
        for (let i = ii; i < iEnd; i++) {
          for (let k = kk; k < kEnd; k++) {
            const value = A[i * K + k];
            for (let j = jj; j < jEnd; j++) C[i * N + j] += value * B[k * N + j];
          }
        }
      }
    }
  }
  return C;
}

export function matmulIKJUnrolled(A, B, M, K, N, factor, output) {
  const C = outputBuffer(output, M * N, output?.constructor);
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < K; k++) {
      const value = A[i * K + k];
      const cOffset = i * N;
      const bOffset = k * N;
      let j = 0;
      for (; j + factor <= N; j += factor) {
        for (let offset = 0; offset < factor; offset++) {
          C[cOffset + j + offset] += value * B[bOffset + j + offset];
        }
      }
      for (; j < N; j++) C[cOffset + j] += value * B[bOffset + j];
    }
  }
  return C;
}

export function transposeB(B, K, N) {
  const transposed = new B.constructor(B.length);
  for (let k = 0; k < K; k++) {
    for (let j = 0; j < N; j++) transposed[j * K + k] = B[k * N + j];
  }
  return transposed;
}

export function matmulTransposedB(A, transposedB, M, K, N, output) {
  const C = outputBuffer(output, M * N, output?.constructor);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) sum += A[i * K + k] * transposedB[j * K + k];
      C[i * N + j] = sum;
    }
  }
  return C;
}

export const LOOP_ORDER_KERNELS = {
  ijk: matmulIJK,
  ikj: matmulIKJ,
  jik: matmulJIK,
  jki: matmulJKI,
  kij: matmulKIJ,
  kji: matmulKJI,
};
