/// Micro-benchmark for the non-Conv ops that dominate Spotify Basic Pitch's
/// Dart profile, at the shapes the model actually uses.
///
///   dart run tool/op_bench.dart [--iters N]
///
/// Prints min-of-N per op. On a loaded machine compare mins across
/// alternating runs of the two revisions, not a single pair.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/src/onnx_ops.dart' as ops;

Tensor _rand(List<int> shape, int seed) {
  final n = shape.fold<int>(1, (a, b) => a * b);
  final r = math.Random(seed);
  final v = Float32List(n);
  for (int i = 0; i < n; i++) {
    v[i] = r.nextDouble() * 2 - 1;
  }
  return Tensor.float(v, shape);
}

void main(List<String> args) {
  final iters = args.contains('--iters')
      ? int.parse(args[args.indexOf('--iters') + 1])
      : 15;

  final sig = _rand([1, 1, 43844], 1);
  final hs = _rand([1, 172, 309, 1], 2);
  final hs8 = [
    for (int k = 0; k < 8; k++) _rand([1, 172, 309, 1], 10 + k)
  ];
  final feat = _rand([1, 32, 172, 88], 3);
  final feat1 = _rand([1, 1, 172, 88], 4);
  final cqt = _rand([1, 309, 172, 2], 5);
  final scale = _rand([309, 1, 1], 6);
  final nchw = _rand([1, 172, 264, 8], 7);
  final logmag = _rand([1, 172, 309], 8);

  final cases = <String, Tensor Function()>{
    'Pad reflect [1,1,43844] +128': () =>
        ops.opPad(sig, [0, 0, 128, 0, 0, 128], mode: 'reflect'),
    'Pad const   [1,1,43844] +128': () =>
        ops.opPad(sig, [0, 0, 128, 0, 0, 128]),
    'Pad const   [1,172,273,1]': () =>
        ops.opPad(hs, [0, 0, 18, 0, 0, 0, 18, 0]),
    'Concat x8   axis -1': () => ops.opConcat(hs8, -1),
    'Concat x2   axis 1': () => ops.opConcat([feat1, feat], 1),
    'Transpose   [1,172,264,8] 0312': () => ops.opTranspose(nchw, [0, 3, 1, 2]),
    'ReduceSum   [1,309,172,2] ax3': () => ops.opReduceSum(cqt, [3], false),
    'ReduceMin   [1,172,309] ax12': () =>
        ops.opReduceMinMax(logmag, [1, 2], true, isMax: false),
    'Mul bcast   [1,309,172,2]x[309,1,1]': () => ops.opMul(cqt, scale),
    'Relu        [1,32,172,88]': () => ops.opRelu(feat),
    'Sigmoid     [1,32,172,88]': () => ops.opSigmoid(feat),
  };

  cases.forEach((name, fn) {
    final out = fn();
    double sum = 0;
    for (final v in out.asFloatList()) {
      sum += v;
    }
    int best = 1 << 30;
    for (int i = 0; i < iters; i++) {
      final sw = Stopwatch()..start();
      fn();
      final us = sw.elapsedMicroseconds;
      if (us < best) best = us;
    }
    print(
        '${name.padRight(38)} min=${(best / 1000).toStringAsFixed(2).padLeft(8)} ms   '
        'sum=${sum.toStringAsFixed(3)}');
  });
}
