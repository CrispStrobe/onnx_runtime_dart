/// Property tests for the monomorphic float fast paths in Add/Sub/Mul/Div:
/// every shape-pair class (same-shape, scalar, suffix-tile both ways, and
/// general broadcasts that must fall back) is checked against an independent
/// naive numpy-style broadcast reference implemented here in the test.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:onnx_runtime_dart/src/onnx_ops.dart' as ops;
import 'package:onnx_runtime_dart/src/tensor.dart';
import 'package:test/test.dart';

final _rng = math.Random(7);

Tensor _rand(List<int> shape) {
  final n = shape.fold<int>(1, (a, b) => a * b);
  final v = Float32List(n);
  for (int k = 0; k < n; k++) {
    v[k] = _rng.nextDouble() * 4 - 2;
  }
  return Tensor.float(v, shape);
}

/// Naive reference: full per-element coordinate decomposition, no fast paths.
Tensor _refBinary(Tensor a, Tensor b, double Function(double, double) f) {
  final rank = math.max(a.rank, b.rank);
  List<int> pad(List<int> s) =>
      [...List.filled(rank - s.length, 1), ...s];
  final ap = pad(a.shape), bp = pad(b.shape);
  final outShape = <int>[
    for (int k = 0; k < rank; k++) math.max(ap[k], bp[k])
  ];
  final n = outShape.fold<int>(1, (x, y) => x * y);
  final out = Float32List(n);
  for (int flat = 0; flat < n; flat++) {
    int rem = flat, ai = 0, bi = 0, as_ = 1, bs = 1;
    for (int k = rank - 1; k >= 0; k--) {
      final c = rem % outShape[k];
      rem ~/= outShape[k];
      ai += (ap[k] == 1 ? 0 : c) * as_;
      bi += (bp[k] == 1 ? 0 : c) * bs;
      as_ *= ap[k];
      bs *= bp[k];
    }
    out[flat] = f(a.f![ai], b.f![bi]);
  }
  // Trim leading padded 1s the way ONNX broadcast output shapes come out.
  final trimmed = outShape.sublist(rank - math.max(a.rank, b.rank));
  return Tensor.float(out, trimmed);
}

void main() {
  final shapePairs = <List<List<int>>>[
    [[2, 5, 8], [2, 5, 8]], // same shape
    [[2, 5, 8], []], // scalar rhs (rank 0)
    [[2, 5, 8], [1]], // scalar rhs (rank 1)
    [[], [2, 5, 8]], // scalar lhs
    [[2, 5, 8], [2, 5, 1]], // per-row scalar (LayerNorm mean/std)
    [[2, 5, 1], [2, 5, 8]], // reverse: general path
    [[2, 5, 8], [8]], // bias: suffix of length 1 axis group
    [[2, 5, 8], [5, 8]], // suffix, 2 axes
    [[2, 5, 8], [1, 1, 8]], // suffix with leading 1s
    [[8], [2, 5, 8]], // suffix-tile, a smaller
    [[5, 8], [2, 5, 8]],
    [[2, 1, 4, 3], [5, 1, 3]], // internal broadcast → general path
    [[2, 1, 8], [2, 5, 8]], // middle-dim broadcast → general path
    [[4, 1], [1, 6]], // outer product style → general path
  ];

  final fns = <String, (Tensor Function(Tensor, Tensor), double Function(double, double))>{
    'Add': (ops.opAdd, (x, y) => x + y),
    'Sub': (ops.opSub, (x, y) => x - y),
    'Mul': (ops.opMul, (x, y) => x * y),
    'Div': (ops.opDiv, (x, y) => x / y),
  };

  test('Pow scalar-exponent fast path (incl. square)', () {
    final a = _rand([3, 4, 5]);
    for (final e in [2.0, 3.0, 0.5]) {
      // 0.5 exponent needs positive bases: square the inputs first.
      final base = e == 0.5 ? ops.opMul(a, a) : a;
      final got = ops.opPow(base, Tensor.scalarFloat(e));
      final want =
          _refBinary(base, Tensor.scalarFloat(e), (x, y) => math.pow(x, y).toDouble());
      expect(got.shape, want.shape);
      final g = got.asFloatList(), w = want.asFloatList();
      for (int k = 0; k < w.length; k++) {
        expect(g[k], closeTo(w[k], 1e-5), reason: 'e=$e at $k');
      }
    }
  });

  fns.forEach((name, pair) {
    final (op, ref) = pair;
    for (final shapes in shapePairs) {
      test('$name ${shapes[0]} vs ${shapes[1]} matches naive broadcast', () {
        final a = _rand(shapes[0]), b = _rand(shapes[1]);
        final got = op(a, b);
        final want = _refBinary(a, b, ref);
        expect(got.shape, want.shape);
        final g = got.asFloatList(), w = want.asFloatList();
        for (int k = 0; k < w.length; k++) {
          expect(g[k], closeTo(w[k], 1e-6),
              reason: '$name ${shapes[0]}x${shapes[1]} at $k');
        }
      });
    }
  });
}
