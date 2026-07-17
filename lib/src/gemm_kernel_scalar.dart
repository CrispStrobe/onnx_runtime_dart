/// Scalar GEMM micro-kernel: portable fallback used on targets without
/// reliable `Float32x4` support (dart2js / dart2wasm). Same contract as the
/// SIMD kernel in `gemm_kernel_simd.dart` — see there for the layout notes.
library;

import 'dart:typed_data';

/// Computes `out[m×n] += a[m×k] · b[k×n]` (row-major, at the given flat
/// offsets). `out` is assumed zero-initialized (fresh allocation).
void matmulKernel(Float32List a, int aOff, Float32List b, int bOff,
    Float32List out, int outOff, int m, int k, int n) {
  // 4-row unroll: each loaded b value is reused for 4 output rows, which
  // quarters B memory traffic vs the naive i-k-j loop.
  int i = 0;
  for (; i + 4 <= m; i += 4) {
    final a0 = aOff + i * k, a1 = a0 + k, a2 = a1 + k, a3 = a2 + k;
    final o0 = outOff + i * n, o1 = o0 + n, o2 = o1 + n, o3 = o2 + n;
    for (int kk = 0; kk < k; kk++) {
      final v0 = a[a0 + kk], v1 = a[a1 + kk], v2 = a[a2 + kk], v3 = a[a3 + kk];
      final bRow = bOff + kk * n;
      for (int j = 0; j < n; j++) {
        final bv = b[bRow + j];
        out[o0 + j] += v0 * bv;
        out[o1 + j] += v1 * bv;
        out[o2 + j] += v2 * bv;
        out[o3 + j] += v3 * bv;
      }
    }
  }
  for (; i < m; i++) {
    final aRow = aOff + i * k;
    final oRow = outOff + i * n;
    for (int kk = 0; kk < k; kk++) {
      final v = a[aRow + kk];
      final bRow = bOff + kk * n;
      for (int j = 0; j < n; j++) {
        out[oRow + j] += v * b[bRow + j];
      }
    }
  }
}
