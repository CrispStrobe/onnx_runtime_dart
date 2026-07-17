/// SIMD GEMM micro-kernel (`Float32x4`, compiles to SSE/NEON on the native
/// VM and AOT). Selected via conditional import on `dart.library.ffi`; web
/// targets get `gemm_kernel_scalar.dart` instead, where Float32x4 is
/// emulated and slower than plain loops.
///
/// Strategy: pack B into a row-padded buffer whose row stride is a multiple
/// of 4 floats (so a single aligned Float32x4List view covers every row),
/// then compute 4 A-rows at a time — each loaded B vector is reused for 4
/// accumulator rows, and each lane op covers 4 output columns: ~16 scalar
/// multiply-adds per inner iteration.
library;

import 'dart:typed_data';

/// Computes `out[m×n] += a[m×k] · b[k×n]` (row-major, at the given flat
/// offsets). `out` is assumed zero-initialized (fresh allocation).
void matmulKernel(Float32List a, int aOff, Float32List b, int bOff,
    Float32List out, int outOff, int m, int k, int n) {
  final n4 = (n + 3) >> 2; // B/accumulator row stride in Float32x4 lanes

  // Pack B: k rows, each padded to n4*4 floats (zeros in the tail), giving
  // an aligned x4 view. O(k·n) copy vs O(m·k·n) math — amortizes for m ≥ ~4.
  final bp = Float32x4List(k * n4);
  final bpF = bp.buffer.asFloat32List();
  for (int kk = 0; kk < k; kk++) {
    final src = bOff + kk * n, dst = kk * n4 * 4;
    for (int j = 0; j < n; j++) {
      bpF[dst + j] = b[src + j];
    }
  }

  final acc0 = Float32x4List(n4);
  final acc1 = Float32x4List(n4);
  final acc2 = Float32x4List(n4);
  final acc3 = Float32x4List(n4);
  final acc0F = acc0.buffer.asFloat32List();
  final acc1F = acc1.buffer.asFloat32List();
  final acc2F = acc2.buffer.asFloat32List();
  final acc3F = acc3.buffer.asFloat32List();

  int i = 0;
  for (; i + 4 <= m; i += 4) {
    final a0 = aOff + i * k, a1 = a0 + k, a2 = a1 + k, a3 = a2 + k;
    final zero = Float32x4.zero();
    for (int j = 0; j < n4; j++) {
      acc0[j] = zero;
      acc1[j] = zero;
      acc2[j] = zero;
      acc3[j] = zero;
    }
    for (int kk = 0; kk < k; kk++) {
      final v0 = Float32x4.splat(a[a0 + kk]);
      final v1 = Float32x4.splat(a[a1 + kk]);
      final v2 = Float32x4.splat(a[a2 + kk]);
      final v3 = Float32x4.splat(a[a3 + kk]);
      final bRow = kk * n4;
      for (int j = 0; j < n4; j++) {
        final bv = bp[bRow + j];
        acc0[j] += v0 * bv;
        acc1[j] += v1 * bv;
        acc2[j] += v2 * bv;
        acc3[j] += v3 * bv;
      }
    }
    final o0 = outOff + i * n, o1 = o0 + n, o2 = o1 + n, o3 = o2 + n;
    for (int j = 0; j < n; j++) {
      out[o0 + j] = acc0F[j];
      out[o1 + j] = acc1F[j];
      out[o2 + j] = acc2F[j];
      out[o3 + j] = acc3F[j];
    }
  }
  // Remainder rows (m % 4): single-row SIMD.
  for (; i < m; i++) {
    final aRow = aOff + i * k;
    final zero = Float32x4.zero();
    for (int j = 0; j < n4; j++) {
      acc0[j] = zero;
    }
    for (int kk = 0; kk < k; kk++) {
      final v = Float32x4.splat(a[aRow + kk]);
      final bRow = kk * n4;
      for (int j = 0; j < n4; j++) {
        acc0[j] += v * bp[bRow + j];
      }
    }
    final oRow = outOff + i * n;
    for (int j = 0; j < n; j++) {
      out[oRow + j] = acc0F[j];
    }
  }
}
