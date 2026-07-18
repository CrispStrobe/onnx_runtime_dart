/// SIMD GEMM micro-kernel (`Float32x4`, compiles to SSE/NEON on the native
/// VM and AOT). Selected via conditional import on `dart.library.ffi`; web
/// targets get `gemm_kernel_scalar.dart` instead, where Float32x4 is
/// emulated and slower than plain loops.
///
/// Strategy: an outer loop over **column panels** of B keeps the packed
/// panel (k × panelWidth) inside L2, so every A-row tile reads B from cache
/// rather than re-streaming the whole matrix from memory (the difference
/// between memory- and compute-bound for large `k`). Within a panel, B is
/// packed to a 4-float-aligned row stride and the output is computed in
/// register-blocked 4-row × 8-column tiles — the 8 accumulators live in
/// locals across the whole k loop and are stored once per tile. Panel width,
/// row and column remainders all fall back to narrower work. The panel loop
/// does not change accumulation order, so results are bitwise identical to a
/// single-panel run (the isolate pool relies on this).
library;

import 'dart:typed_data';

/// Target packed-panel footprint in bytes (~half a typical 512 KB L2, leaving
/// room for the A-row tiles and output). Panel width is derived from this and
/// `k` so the panel stays resident across the full m loop.
const int _panelBytes = 192 * 1024;

/// Computes `out[m×n] += a[m×k] · b[k×n]` (row-major, at the given flat
/// offsets). `out` is assumed zero-initialized (fresh allocation).
void matmulKernel(Float32List a, int aOff, Float32List b, int bOff,
    Float32List out, int outOff, int m, int k, int n) {
  // Column-panel width: as many columns as keep k×panel×4 bytes under the
  // target, rounded up to a multiple of 4 (one Float32x4 lane); never wider
  // than n and at least 4.
  int panel = _panelBytes ~/ (k * 4);
  panel = (panel >> 2) << 2;
  if (panel < 4) panel = 4;
  if (panel > n) panel = n;
  final singlePanel = panel >= n;

  final panelN4 = (panel + 3) >> 2;
  final bp = Float32x4List(k * panelN4);
  final bpF = bp.buffer.asFloat32List();

  final lane = Float32x4List(1);
  final laneF = lane.buffer.asFloat32List();
  void storeLane(Float32x4 v, int outIdx, int cols) {
    lane[0] = v;
    for (int c = 0; c < cols; c++) {
      out[outIdx + c] = laneF[c];
    }
  }

  for (int p0 = 0; p0 < n; p0 += panel) {
    final pw = singlePanel ? n : (p0 + panel <= n ? panel : n - p0);
    final pn4 = (pw + 3) >> 2;

    // Pack this column panel of B (zero-filled tail lanes for alignment).
    for (int kk = 0; kk < k; kk++) {
      final src = bOff + kk * n + p0, dst = kk * pn4 * 4;
      int j = 0;
      for (; j < pw; j++) {
        bpF[dst + j] = b[src + j];
      }
      for (; j < pn4 * 4; j++) {
        bpF[dst + j] = 0;
      }
    }

    int i = 0;
    for (; i + 4 <= m; i += 4) {
      final a0 = aOff + i * k, a1 = a0 + k, a2 = a1 + k, a3 = a2 + k;
      final o0 = outOff + i * n + p0, o1 = o0 + n, o2 = o1 + n, o3 = o2 + n;

      int j = 0;
      // Full 4×8 tiles (two lanes entirely inside the panel width).
      for (; (j + 2) * 4 <= pw; j += 2) {
        var c00 = Float32x4.zero(), c01 = Float32x4.zero();
        var c10 = Float32x4.zero(), c11 = Float32x4.zero();
        var c20 = Float32x4.zero(), c21 = Float32x4.zero();
        var c30 = Float32x4.zero(), c31 = Float32x4.zero();
        int bRow = j;
        for (int kk = 0; kk < k; kk++) {
          final bv0 = bp[bRow], bv1 = bp[bRow + 1];
          bRow += pn4;
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
      // Remaining lanes (≤ 8 tail columns of the panel), one lane at a time.
      for (; j < pn4; j++) {
        var c0 = Float32x4.zero(),
            c1 = Float32x4.zero(),
            c2 = Float32x4.zero(),
            c3 = Float32x4.zero();
        int bRow = j;
        for (int kk = 0; kk < k; kk++) {
          final bv = bp[bRow];
          bRow += pn4;
          c0 += Float32x4.splat(a[a0 + kk]) * bv;
          c1 += Float32x4.splat(a[a1 + kk]) * bv;
          c2 += Float32x4.splat(a[a2 + kk]) * bv;
          c3 += Float32x4.splat(a[a3 + kk]) * bv;
        }
        final col = j * 4;
        final cols = pw - col < 4 ? pw - col : 4;
        storeLane(c0, o0 + col, cols);
        storeLane(c1, o1 + col, cols);
        storeLane(c2, o2 + col, cols);
        storeLane(c3, o3 + col, cols);
      }
    }

    // Remainder rows (m % 4): single-row, two lanes at a time.
    for (; i < m; i++) {
      final aRow = aOff + i * k;
      final oRow = outOff + i * n + p0;
      int j = 0;
      for (; (j + 2) * 4 <= pw; j += 2) {
        var c0 = Float32x4.zero(), c1 = Float32x4.zero();
        int bRow = j;
        for (int kk = 0; kk < k; kk++) {
          final v = Float32x4.splat(a[aRow + kk]);
          c0 += v * bp[bRow];
          c1 += v * bp[bRow + 1];
          bRow += pn4;
        }
        final col = j * 4;
        storeLane(c0, oRow + col, 4);
        storeLane(c1, oRow + col + 4, 4);
      }
      for (; j < pn4; j++) {
        var c = Float32x4.zero();
        int bRow = j;
        for (int kk = 0; kk < k; kk++) {
          c += Float32x4.splat(a[aRow + kk]) * bp[bRow];
          bRow += pn4;
        }
        final col = j * 4;
        storeLane(c, oRow + col, pw - col < 4 ? pw - col : 4);
      }
    }
  }
}

/// SIMD dot product of two contiguous rows (used by einsum kernels whose
/// operand layout is already dot-friendly — no transpose or packing needed).
double dotProduct(Float32List a, int aOff, Float32List b, int bOff, int n) {
  var acc = Float32x4.zero();
  int k = 0;
  for (; k + 4 <= n; k += 4) {
    acc += Float32x4(a[aOff + k], a[aOff + k + 1], a[aOff + k + 2],
            a[aOff + k + 3]) *
        Float32x4(b[bOff + k], b[bOff + k + 1], b[bOff + k + 2],
            b[bOff + k + 3]);
  }
  double sum = acc.x + acc.y + acc.z + acc.w;
  for (; k < n; k++) {
    sum += a[aOff + k] * b[bOff + k];
  }
  return sum;
}
