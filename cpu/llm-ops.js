function targetBuffer(output, length) {
  const target = output ?? new Float32Array(length);
  if (target.length !== length) throw new Error(`output length ${target.length} != ${length}`);
  return target;
}

export function matvecFloat32(vector, weights, K, N, output) {
  const target = targetBuffer(output, N);
  target.fill(0);
  for (let k = 0; k < K; k++) {
    const value = vector[k];
    for (let n = 0; n < N; n++) target[n] += value * weights[k * N + n];
  }
  return target;
}

export function quantizeInt8(values) {
  let maximum = 0;
  for (const value of values) maximum = Math.max(maximum, Math.abs(value));
  const scale = maximum ? maximum / 127 : 1;
  const data = new Int8Array(values.length);
  for (let index = 0; index < values.length; index++) {
    data[index] = Math.max(-127, Math.min(127, Math.round(values[index] / scale)));
  }
  return { data, scale };
}

export function matvecInt8(vector, weights, scale, K, N, output) {
  const target = targetBuffer(output, N);
  target.fill(0);
  for (let k = 0; k < K; k++) {
    const value = vector[k] * scale;
    for (let n = 0; n < N; n++) target[n] += value * weights[k * N + n];
  }
  return target;
}

export function quantizeInt4(values) {
  let maximum = 0;
  for (const value of values) maximum = Math.max(maximum, Math.abs(value));
  const scale = maximum ? maximum / 7 : 1;
  const data = new Uint8Array(Math.ceil(values.length / 2));
  for (let index = 0; index < values.length; index++) {
    const quantized = Math.max(-7, Math.min(7, Math.round(values[index] / scale)));
    const nibble = quantized & 0x0f;
    const byteIndex = index >> 1;
    if (index & 1) data[byteIndex] |= nibble << 4;
    else data[byteIndex] = nibble;
  }
  return { data, scale, length: values.length };
}

function signedNibble(value) {
  return value & 0x08 ? value - 16 : value;
}

export function matvecInt4(vector, packedWeights, scale, K, N, output) {
  const target = targetBuffer(output, N);
  target.fill(0);
  for (let k = 0; k < K; k++) {
    const value = vector[k] * scale;
    for (let n = 0; n < N; n++) {
      const index = k * N + n;
      const byte = packedWeights[index >> 1];
      const nibble = index & 1 ? byte >> 4 : byte & 0x0f;
      target[n] += value * signedNibble(nibble);
    }
  }
  return target;
}

export function rmsNormDirect(input, rows, width, epsilon = 1e-5, output) {
  const target = targetBuffer(output, input.length);
  for (let row = 0; row < rows; row++) {
    const offset = row * width;
    let sumSquares = 0;
    for (let index = 0; index < width; index++) {
      const value = input[offset + index];
      sumSquares += value * value;
    }
    const scale = 1 / Math.sqrt(sumSquares / width + epsilon);
    for (let index = 0; index < width; index++) {
      target[offset + index] = input[offset + index] * scale;
    }
  }
  return target;
}

export function rmsNormBuffered(input, rows, width, epsilon = 1e-5, output, squareBuffer) {
  const target = targetBuffer(output, input.length);
  const squares = squareBuffer ?? new Float32Array(input.length);
  for (let index = 0; index < input.length; index++) squares[index] = input[index] * input[index];
  for (let row = 0; row < rows; row++) {
    const offset = row * width;
    let sumSquares = 0;
    for (let index = 0; index < width; index++) sumSquares += squares[offset + index];
    const scale = 1 / Math.sqrt(sumSquares / width + epsilon);
    for (let index = 0; index < width; index++) target[offset + index] = input[offset + index] * scale;
  }
  return target;
}

export function rmsNormUnrolled4(input, rows, width, epsilon = 1e-5, output) {
  const target = targetBuffer(output, input.length);
  for (let row = 0; row < rows; row++) {
    const offset = row * width;
    let sumSquares = 0;
    let index = 0;
    for (; index + 4 <= width; index += 4) {
      const a = input[offset + index];
      const b = input[offset + index + 1];
      const c = input[offset + index + 2];
      const d = input[offset + index + 3];
      sumSquares += a * a + b * b + c * c + d * d;
    }
    for (; index < width; index++) {
      const value = input[offset + index];
      sumSquares += value * value;
    }
    const scale = 1 / Math.sqrt(sumSquares / width + epsilon);
    for (index = 0; index < width; index++) target[offset + index] = input[offset + index] * scale;
  }
  return target;
}

export function softmaxStable(input, rows, width, output) {
  const target = targetBuffer(output, input.length);
  for (let row = 0; row < rows; row++) {
    const offset = row * width;
    let maximum = -Infinity;
    for (let index = 0; index < width; index++) maximum = Math.max(maximum, input[offset + index]);
    let sum = 0;
    for (let index = 0; index < width; index++) sum += Math.exp(input[offset + index] - maximum);
    for (let index = 0; index < width; index++) {
      target[offset + index] = Math.exp(input[offset + index] - maximum) / sum;
    }
  }
  return target;
}

export function softmaxBuffered(input, rows, width, output, expBuffer) {
  const target = targetBuffer(output, input.length);
  const exponentials = expBuffer ?? new Float32Array(input.length);
  for (let row = 0; row < rows; row++) {
    const offset = row * width;
    let maximum = -Infinity;
    for (let index = 0; index < width; index++) maximum = Math.max(maximum, input[offset + index]);
    let sum = 0;
    for (let index = 0; index < width; index++) {
      const value = Math.exp(input[offset + index] - maximum);
      exponentials[offset + index] = value;
      sum += value;
    }
    for (let index = 0; index < width; index++) target[offset + index] = exponentials[offset + index] / sum;
  }
  return target;
}

export function softmaxOnline(input, rows, width, output) {
  const target = targetBuffer(output, input.length);
  for (let row = 0; row < rows; row++) {
    const offset = row * width;
    let maximum = -Infinity;
    let sum = 0;
    for (let index = 0; index < width; index++) {
      const value = input[offset + index];
      if (value <= maximum) sum += Math.exp(value - maximum);
      else {
        sum = sum * Math.exp(maximum - value) + 1;
        maximum = value;
      }
    }
    for (let index = 0; index < width; index++) {
      target[offset + index] = Math.exp(input[offset + index] - maximum) / sum;
    }
  }
  return target;
}

export function residualRmsNormSeparate(input, residual, rows, width, epsilon = 1e-5, output, residualBuffer) {
  const combined = residualBuffer ?? new Float32Array(input.length);
  for (let index = 0; index < input.length; index++) combined[index] = input[index] + residual[index];
  return rmsNormDirect(combined, rows, width, epsilon, output);
}

export function residualRmsNormFused(input, residual, rows, width, epsilon = 1e-5, output, residualBuffer) {
  const target = targetBuffer(output, input.length);
  const combined = residualBuffer ?? new Float32Array(input.length);
  for (let row = 0; row < rows; row++) {
    const offset = row * width;
    let sumSquares = 0;
    for (let index = 0; index < width; index++) {
      const value = input[offset + index] + residual[offset + index];
      combined[offset + index] = value;
      sumSquares += value * value;
    }
    const scale = 1 / Math.sqrt(sumSquares / width + epsilon);
    for (let index = 0; index < width; index++) target[offset + index] = combined[offset + index] * scale;
  }
  return target;
}
