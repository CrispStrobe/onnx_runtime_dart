/// Micro-benchmark for the Conv shapes that dominate Spotify Basic Pitch.
///
///   dart run tool/conv_bench.dart [--iters N]
///
/// Reports min-of-N wall time per shape. Absolute numbers are noisy on a
/// loaded machine; the min over enough iterations is the usable signal.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/src/onnx_nn_ops.dart' as nn;

class Case {
  final String name;
  final List<int> x;
  final List<int> w;
  final List<int> strides;
  final List<int> pads;
  const Case(this.name, this.x, this.w, this.strides, this.pads);
}

const cases = <Case>[
  Case('conv2d_1   m=8  3x39', [1, 8, 172, 264], [8, 8, 3, 39], [1, 1],
      [1, 19, 1, 19]),
  Case('conv2d_4   m=32 5x5', [1, 8, 172, 264], [32, 8, 5, 5], [1, 3],
      [2, 1, 2, 1]),
  Case('conv2d_2   m=32 7x7', [1, 1, 172, 264], [32, 1, 7, 7], [1, 3],
      [3, 2, 3, 2]),
  Case('contours   m=1  5x5', [1, 8, 172, 264], [1, 8, 5, 5], [1, 1],
      [2, 2, 2, 2]),
  Case('conv2d_3   m=1  7x3', [1, 32, 172, 88], [1, 32, 7, 3], [1, 1],
      [3, 1, 3, 1]),
  Case('conv2d_5   m=1  3x3', [1, 33, 172, 88], [1, 33, 3, 3], [1, 1],
      [1, 1, 1, 1]),
  Case('cqt conv1d_2 m=1 1x256', [1, 1, 1, 44098], [1, 1, 1, 256], [1, 2],
      [0, 0, 0, 0]),
  Case('cqt conv1d   m=36 1x256', [1, 1, 1, 44100], [36, 1, 1, 256], [1, 256],
      [0, 0, 0, 0]),
  Case('cqt conv1d_25 m=36 1x256', [1, 1, 1, 22528], [36, 1, 1, 256], [1, 1],
      [0, 0, 0, 0]),
  // Vision-shaped controls: these must not regress when the direct-path
  // channel threshold is raised.
  Case('vision m=64 c=64 3x3 56', [1, 64, 56, 56], [64, 64, 3, 3], [1, 1],
      [1, 1, 1, 1]),
  Case('vision m=128 c=128 3x3 28', [1, 128, 28, 28], [128, 128, 3, 3], [1, 1],
      [1, 1, 1, 1]),
  Case('vision m=256 c=256 3x3 14', [1, 256, 14, 14], [256, 256, 3, 3], [1, 1],
      [1, 1, 1, 1]),
  Case('vision m=512 c=512 3x3 7', [1, 512, 7, 7], [512, 512, 3, 3], [1, 1],
      [1, 1, 1, 1]),
];

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
      : 7;
  for (final c in cases) {
    final x = _rand(c.x, 1), w = _rand(c.w, 2), b = _rand([c.w[0]], 3);
    Tensor run() =>
        nn.opConv(x, w, b, strides: c.strides, pads: c.pads, dilations: [1, 1]);
    final out = run();
    double checksum = 0;
    for (final v in out.asFloatList()) {
      checksum += v;
    }
    final macs = out.length * c.w[1] * c.w[2] * c.w[3];
    // A/B interleaved in one process so load drift hits both arms equally:
    // arm 0 = im2col + GEMM (threshold 0), arm 1 = direct path.
    final best = [1 << 30, 1 << 30];
    for (int i = 0; i < iters; i++) {
      for (int arm = 0; arm < 2; arm++) {
        nn.directConvMaxChannels = arm == 0 ? 0 : 4096;
        final sw = Stopwatch()..start();
        run();
        final us = sw.elapsedMicroseconds;
        if (us < best[arm]) best[arm] = us;
      }
    }
    nn.directConvMaxChannels = 64;
    final gemmMs = best[0] / 1000, dirMs = best[1] / 1000;
    print('${c.name.padRight(24)} gemm=${gemmMs.toStringAsFixed(1).padLeft(8)}'
        ' direct=${dirMs.toStringAsFixed(1).padLeft(8)} ms  '
        '${(gemmMs / dirMs).toStringAsFixed(2).padLeft(5)}x  '
        '${(macs / 1e6 / dirMs).toStringAsFixed(2).padLeft(6)} GMAC/s  '
        'sum=${checksum.toStringAsFixed(4)}');
  }
}
