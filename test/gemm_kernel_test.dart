/// Both GEMM micro-kernels (scalar + SIMD) vs a naive triple-loop reference:
/// exhaustive small shapes (every m,k,n in 1..6 — covers all tile remainder
/// combinations) plus larger odd sizes and offset/batched use.
@TestOn('vm')
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:onnx_runtime_dart/src/gemm_kernel_scalar.dart' as scalar;
import 'package:onnx_runtime_dart/src/gemm_kernel_simd.dart' as simd;
import 'package:test/test.dart';

final _rng = math.Random(11);

Float32List _rand(int n) {
  final v = Float32List(n);
  for (int k = 0; k < n; k++) {
    v[k] = _rng.nextDouble() * 2 - 1;
  }
  return v;
}

Float32List _naive(Float32List a, Float32List b, int m, int k, int n) {
  final out = Float32List(m * n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double s = 0;
      for (int kk = 0; kk < k; kk++) {
        s += a[i * k + kk] * b[kk * n + j];
      }
      out[i * n + j] = s;
    }
  }
  return out;
}

void main() {
  final kernels = {
    'scalar': scalar.matmulKernel,
    'simd': simd.matmulKernel,
  };

  kernels.forEach((name, kern) {
    test('$name kernel: exhaustive 1..6 shapes', () {
      for (int m = 1; m <= 6; m++) {
        for (int k = 1; k <= 6; k++) {
          for (int n = 1; n <= 6; n++) {
            final a = _rand(m * k), b = _rand(k * n);
            final out = Float32List(m * n);
            kern(a, 0, b, 0, out, 0, m, k, n);
            final want = _naive(a, b, m, k, n);
            for (int x = 0; x < want.length; x++) {
              expect(out[x], closeTo(want[x], 1e-5),
                  reason: '$name m=$m k=$k n=$n at $x');
            }
          }
        }
      }
    });

    for (final (m, k, n) in [(1, 384, 384), (33, 65, 17), (7, 128, 5),
        (128, 64, 128), (5, 1, 9), (64, 384, 1)]) {
      test('$name kernel: ${m}x${k}x$n', () {
        final a = _rand(m * k), b = _rand(k * n);
        final out = Float32List(m * n);
        kern(a, 0, b, 0, out, 0, m, k, n);
        final want = _naive(a, b, m, k, n);
        double maxAbs = 0;
        for (int x = 0; x < want.length; x++) {
          maxAbs = math.max(maxAbs, (out[x] - want[x]).abs());
        }
        expect(maxAbs, lessThan(1e-4));
      });
    }

    test('$name kernel: honors offsets (batched layout)', () {
      // Two batches packed into single buffers; kernel run per batch.
      const m = 5, k = 7, n = 6;
      final a = _rand(2 * m * k), b = _rand(2 * k * n);
      final out = Float32List(2 * m * n);
      kern(a, 0, b, 0, out, 0, m, k, n);
      kern(a, m * k, b, k * n, out, m * n, m, k, n);
      for (int batch = 0; batch < 2; batch++) {
        final want = _naive(
            Float32List.sublistView(a, batch * m * k, (batch + 1) * m * k),
            Float32List.sublistView(b, batch * k * n, (batch + 1) * k * n),
            m, k, n);
        for (int x = 0; x < want.length; x++) {
          expect(out[batch * m * n + x], closeTo(want[x], 1e-5),
              reason: '$name batch=$batch at $x');
        }
      }
    });
  });
}
