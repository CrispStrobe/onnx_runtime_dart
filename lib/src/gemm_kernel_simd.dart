/// SIMD GEMM micro-kernel (`Float32x4`, compiles to SSE/NEON on the native
/// VM and AOT). Selected via conditional import on `dart.library.ffi`; web
/// targets get `gemm_kernel_scalar.dart` instead, where Float32x4 is
/// emulated and slower than plain loops.
///
/// Strategy: pack B into a row-padded buffer whose row stride is a multiple
/// of 4 floats (one aligned Float32x4List view covers every row), then
/// compute the output in register-blocked 4-row × 8-column tiles — the 8
/// accumulator vectors live in locals for the whole k loop (16 multiply-adds
/// per 6 loads, no accumulator memory traffic) and are stored exactly once
/// per tile. Row/column remainders fall back to narrower tiles.
library;

import 'dart:typed_data';

/// Computes `out[m×n] += a[m×k] · b[k×n]` (row-major, at the given flat
/// offsets). `out` is assumed zero-initialized (fresh allocation).
void matmulKernel(Float32List a, int aOff, Float32List b, int bOff,
    Float32List out, int outOff, int m, int k, int n) {
  final n4 = (n + 3) >> 2; // packed B row stride in Float32x4 lanes

  final bp = Float32x4List(k * n4);
  final bpF = bp.buffer.asFloat32List();
  for (int kk = 0; kk < k; kk++) {
    final src = bOff + kk * n, dst = kk * n4 * 4;
    for (int j = 0; j < n; j++) {
      bpF[dst + j] = b[src + j];
    }
  }

  // Staging lane for stores into the (unpadded) output.
  final lane = Float32x4List(1);
  final laneF = lane.buffer.asFloat32List();

  void storeLane(Float32x4 v, int outIdx, int cols) {
    lane[0] = v;
    for (int c = 0; c < cols; c++) {
      out[outIdx + c] = laneF[c];
    }
  }

  int i = 0;
  for (; i + 4 <= m; i += 4) {
    final a0 = aOff + i * k, a1 = a0 + k, a2 = a1 + k, a3 = a2 + k;
    final o0 = outOff + i * n, o1 = o0 + n, o2 = o1 + n, o3 = o2 + n;

    int j = 0;
    // Full 4×8 tiles (both lanes entirely inside n).
    for (; (j + 2) * 4 <= n; j += 2) {
      var c00 = Float32x4.zero(), c01 = Float32x4.zero();
      var c10 = Float32x4.zero(), c11 = Float32x4.zero();
      var c20 = Float32x4.zero(), c21 = Float32x4.zero();
      var c30 = Float32x4.zero(), c31 = Float32x4.zero();
      int bRow = j;
      for (int kk = 0; kk < k; kk++) {
        final bv0 = bp[bRow], bv1 = bp[bRow + 1];
        bRow += n4;
        final v0 = Float32x4.splat(a[a0 + kk]);
        c00 += v0 * bv0;
        c01 += v0 * bv1;
        final v1 = Float32x4.splat(a[a1 + kk]);
        c10 += v1 * bv0;
        c11 += v1 * bv1;
        final v2 = Float32x4.splat(a[a2 + kk]);
        c20 += v2 * bv0;
        c21 += v2 * bv1;
        final v3 = Float32x4.splat(a[a3 + kk]);
        c30 += v3 * bv0;
        c31 += v3 * bv1;
      }
      final col = j * 4;
      storeLane(c00, o0 + col, 4);
      storeLane(c01, o0 + col + 4, 4);
      storeLane(c10, o1 + col, 4);
      storeLane(c11, o1 + col + 4, 4);
      storeLane(c20, o2 + col, 4);
      storeLane(c21, o2 + col + 4, 4);
      storeLane(c30, o3 + col, 4);
      storeLane(c31, o3 + col + 4, 4);
    }
    // Remaining lanes (≤ 8 tail columns), one lane at a time.
    for (; j < n4; j++) {
      var c0 = Float32x4.zero(),
          c1 = Float32x4.zero(),
          c2 = Float32x4.zero(),
          c3 = Float32x4.zero();
      int bRow = j;
      for (int kk = 0; kk < k; kk++) {
        final bv = bp[bRow];
        bRow += n4;
        c0 += Float32x4.splat(a[a0 + kk]) * bv;
        c1 += Float32x4.splat(a[a1 + kk]) * bv;
        c2 += Float32x4.splat(a[a2 + kk]) * bv;
        c3 += Float32x4.splat(a[a3 + kk]) * bv;
      }
      final col = j * 4;
      final cols = n - col < 4 ? n - col : 4;
      storeLane(c0, o0 + col, cols);
      storeLane(c1, o1 + col, cols);
      storeLane(c2, o2 + col, cols);
      storeLane(c3, o3 + col, cols);
    }
  }

  // Remainder rows (m % 4): single-row, two lanes at a time.
  for (; i < m; i++) {
    final aRow = aOff + i * k;
    final oRow = outOff + i * n;
    int j = 0;
    for (; (j + 2) * 4 <= n; j += 2) {
      var c0 = Float32x4.zero(), c1 = Float32x4.zero();
      int bRow = j;
      for (int kk = 0; kk < k; kk++) {
        final v = Float32x4.splat(a[aRow + kk]);
        c0 += v * bp[bRow];
        c1 += v * bp[bRow + 1];
        bRow += n4;
      }
      final col = j * 4;
      storeLane(c0, oRow + col, 4);
      storeLane(c1, oRow + col + 4, 4);
    }
    for (; j < n4; j++) {
      var c = Float32x4.zero();
      int bRow = j;
      for (int kk = 0; kk < k; kk++) {
        c += Float32x4.splat(a[aRow + kk]) * bp[bRow];
        bRow += n4;
      }
      final col = j * 4;
      storeLane(c, oRow + col, n - col < 4 ? n - col : 4);
    }
  }
}
