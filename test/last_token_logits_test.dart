/// The `lastTokenLogits` graph rewrite: a `logits = MatMul(hidden, W[h,vocab])`
/// output is sliced to only its final sequence position, so autoregressive
/// prefill skips the wasted `seq-1` vocab-projection rows. The last row must
/// stay bitwise identical to the full run.
@TestOn('vm')
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:test/test.dart';

void main() {
  final bytes = File('test/data/tiny_lm_head.onnx').readAsBytesSync();
  const seq = 4, h = 16;
  final hidden = Float32List(seq * h);
  final rng = math.Random(1);
  for (var i = 0; i < hidden.length; i++) {
    hidden[i] = rng.nextDouble() * 2 - 1;
  }
  final inputs = {'hidden': Tensor.float(hidden, [1, seq, h])};

  test('lastTokenLogits slices logits to the final position, bit-exact', () {
    final full = OnnxModel.fromBytes(bytes);
    final fullLogits = full.run(inputs, ['logits'])['logits']!;
    final v = fullLogits.shape[2];
    expect(fullLogits.shape, [1, seq, v]);

    final fast = OnnxModel.fromBytes(bytes, lastTokenLogits: true);
    final fastLogits = fast.run(inputs, ['logits'])['logits']!;
    expect(fastLogits.shape, [1, 1, v], reason: 'only the last row is computed');

    final fa = fullLogits.asFloatList(), ga = fastLogits.asFloatList();
    for (var i = 0; i < v; i++) {
      expect(ga[i], fa[(seq - 1) * v + i], reason: 'logit $i diverged');
    }
  });

  test('lastTokenLogits is a no-op on a single-position input', () {
    final one = {'hidden': Tensor.float(Float32List(h), [1, 1, h])};
    final full = OnnxModel.fromBytes(bytes).run(one, ['logits'])['logits']!;
    final fast =
        OnnxModel.fromBytes(bytes, lastTokenLogits: true).run(one, ['logits'])[
            'logits']!;
    expect(fast.shape, full.shape);
    expect(fast.asFloatList(), full.asFloatList());
  });
}
