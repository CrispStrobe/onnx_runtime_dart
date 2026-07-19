/// Covers the OnnxModel.inputSpecs / outputNames introspection API, which
/// KV-cache decoders rely on to size their empty `past_key_values.*` feeds.
@TestOn('vm')
library;

import 'dart:io';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:test/test.dart';

void main() {
  test('inputSpecs / outputNames report the GQA KV-cache graph I/O', () {
    final model = OnnxModel.fromBytes(
        File('test/fixtures/gqa_kvcache/model.onnx').readAsBytesSync());

    final byName = {for (final s in model.inputSpecs) s.name: s};
    // Runtime inputs are present; the seqlens/total are baked as initializers
    // so they must NOT appear as required inputs.
    expect(byName.keys, containsAll(['q', 'k', 'v', 'past_key', 'past_value']));
    expect(byName.containsKey('seqlens'), isFalse);
    expect(byName.containsKey('total'), isFalse);

    // past_key is [batch, kv_heads, past_seq, head_size]; kv_heads/head_size
    // are concrete, the symbolic dims come back as -1.
    final pk = byName['past_key']!;
    expect(pk.shape.length, 4);
    expect(pk.shape[1], 2); // KVh in the fixture
    expect(pk.shape[3], 8); // head_size
    expect(pk.isInt, isFalse); // float cache

    // All three GQA outputs are declared, in order.
    expect(model.outputNames, ['out0', 'out1', 'out2']);
  });
}
