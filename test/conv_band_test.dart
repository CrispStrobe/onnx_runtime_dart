/// Banded conv (the primitive behind the isolate-pool conv fan-out):
/// computing output rows in slabs and concatenating must equal the full conv
/// bitwise, for every 2-D code path (im2col, pointwise-fallback, depthwise).
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:onnx_runtime_dart/src/onnx_nn_ops.dart' as nn;
import 'package:onnx_runtime_dart/src/tensor.dart';
import 'package:test/test.dart';

final _rng = math.Random(31);

Tensor _rand(List<int> shape) {
  final n = shape.fold<int>(1, (a, b) => a * b);
  final v = Float32List(n);
  for (int k = 0; k < n; k++) {
    v[k] = _rng.nextDouble() * 2 - 1;
  }
  return Tensor.float(v, shape);
}

void main() {
  final cases = <(String, Tensor, Tensor, Map<String, dynamic>)>[
    ('3x3 pads', _rand([1, 3, 9, 9]), _rand([4, 3, 3, 3]),
        {'pads': [1, 1, 1, 1]}),
    ('strided', _rand([2, 2, 11, 9]), _rand([3, 2, 3, 3]),
        {'strides': [2, 2], 'pads': [1, 0, 1, 0]}),
    ('pointwise', _rand([1, 4, 7, 7]), _rand([5, 4, 1, 1]), {}),
    ('depthwise', _rand([1, 4, 8, 8]), _rand([4, 1, 3, 3]),
        {'group': 4, 'pads': [1, 1, 1, 1]}),
    ('grouped', _rand([1, 4, 6, 6]), _rand([6, 2, 3, 3]), {'group': 2}),
    ('dilated', _rand([1, 2, 10, 10]), _rand([2, 2, 3, 3]),
        {'dilations': [2, 2]}),
  ];

  for (final (name, x, w, opts) in cases) {
    test('banded slabs reassemble full conv: $name', () {
      Tensor conv({int? from, int? to}) => nn.opConv(
            x,
            w,
            null,
            strides: (opts['strides'] as List<int>?),
            pads: (opts['pads'] as List<int>?),
            dilations: (opts['dilations'] as List<int>?),
            group: (opts['group'] as int?) ?? 1,
            bandStart: from,
            bandEnd: to,
          );
      final full = conv();
      final oh = full.shape[2];
      // Split into 3 uneven bands.
      final cuts = [0, oh ~/ 3, oh ~/ 3 + (oh - oh ~/ 3) ~/ 2, oh];
      final stitched = Float32List(full.length);
      final n = full.shape[0], m = full.shape[1], ow = full.shape[3];
      for (int b = 0; b < 3; b++) {
        if (cuts[b] == cuts[b + 1]) continue;
        final slab = conv(from: cuts[b], to: cuts[b + 1]);
        final rows = cuts[b + 1] - cuts[b];
        expect(slab.shape, [n, m, rows, ow]);
        for (int img = 0; img < n * m; img++) {
          stitched.setRange(
              (img * oh + cuts[b]) * ow,
              (img * oh + cuts[b + 1]) * ow,
              slab.f!,
              img * rows * ow);
        }
      }
      expect(stitched, full.asFloatList(),
          reason: '$name: banded result must be bitwise-identical');
    });
  }
}
