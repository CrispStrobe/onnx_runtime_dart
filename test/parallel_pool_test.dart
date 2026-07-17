/// The isolate GEMM pool must produce bitwise-identical results to the sync
/// single-threaded path — column slicing computes the exact same dot
/// products, so even float rounding matches.
@TestOn('vm')
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:fixnum/fixnum.dart';
import 'package:onnx_runtime_dart/onnx_proto.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:test/test.dart';

final _rng = math.Random(23);

TensorProto _floatInit(String name, List<int> dims) {
  final n = dims.fold<int>(1, (a, b) => a * b);
  return TensorProto()
    ..name = name
    ..dataType = TensorProto_DataType.FLOAT.value
    ..dims.addAll(dims.map(Int64.new))
    ..floatData.addAll(List.generate(n, (_) => _rng.nextDouble() * 2 - 1));
}

void main() {
  test('runAsync with pool == run, bitwise', () async {
    // x[8,96] -> MatMul W1[96,64] -> Relu -> MatMul W2[64,80] -> Add bias.
    // W1/W2 sized above the test threshold so both get partitioned; the
    // bias Add and Relu stay on the main isolate.
    final g = GraphProto()
      ..input.add(ValueInfoProto()..name = 'X')
      ..output.add(ValueInfoProto()..name = 'Y')
      ..initializer.addAll([
        _floatInit('W1', [96, 64]),
        _floatInit('W2', [64, 80]),
        _floatInit('B', [80]),
      ])
      ..node.addAll([
        NodeProto()
          ..opType = 'MatMul'
          ..input.addAll(['X', 'W1'])
          ..output.add('H1'),
        NodeProto()
          ..opType = 'Relu'
          ..input.add('H1')
          ..output.add('H2'),
        NodeProto()
          ..opType = 'MatMul'
          ..input.addAll(['H2', 'W2'])
          ..output.add('H3'),
        NodeProto()
          ..opType = 'Add'
          ..input.addAll(['H3', 'B'])
          ..output.add('Y'),
      ]);
    final model =
        OnnxModel.fromBytes((ModelProto()..graph = g).writeToBuffer());
    final x = Tensor.float(
        Float32List.fromList(
            List.generate(8 * 96, (_) => _rng.nextDouble() * 2 - 1)),
        [8, 96]);

    final want = model.run({'X': x}, ['Y'])['Y']!;

    await model.parallelize(workers: 3, minWeightElements: 1000);
    try {
      final got = (await model.runAsync({'X': x}, ['Y']))['Y']!;
      expect(got.shape, want.shape);
      expect(got.asFloatList(), want.asFloatList(),
          reason: 'pooled matmul must be bitwise-identical');

      // Batched (3-D) activation through the same 2-D weight.
      final g3 = GraphProto()
        ..input.add(ValueInfoProto()..name = 'X')
        ..output.add(ValueInfoProto()..name = 'Y')
        ..initializer.add(_floatInit('W', [96, 64]))
        ..node.add(NodeProto()
          ..opType = 'MatMul'
          ..input.addAll(['X', 'W'])
          ..output.add('Y'));
      final m3 =
          OnnxModel.fromBytes((ModelProto()..graph = g3).writeToBuffer());
      final x3 = Tensor.float(
          Float32List.fromList(
              List.generate(2 * 5 * 96, (_) => _rng.nextDouble())),
          [2, 5, 96]);
      final want3 = m3.run({'X': x3}, ['Y'])['Y']!;
      await m3.parallelize(workers: 2, minWeightElements: 1000);
      try {
        final got3 = (await m3.runAsync({'X': x3}, ['Y']))['Y']!;
        expect(got3.shape, [2, 5, 64]);
        expect(got3.asFloatList(), want3.asFloatList());
      } finally {
        m3.dispose();
      }
    } finally {
      model.dispose();
    }
  });

  test('pooled conv fan-out == sync conv, bitwise', () async {
    final g = GraphProto()
      ..input.add(ValueInfoProto()..name = 'X')
      ..output.add(ValueInfoProto()..name = 'Y')
      ..initializer.addAll([
        _floatInit('W', [8, 3, 3, 3]),
        _floatInit('B', [8]),
      ])
      ..node.addAll([
        NodeProto()
          ..opType = 'Conv'
          ..input.addAll(['X', 'W', 'B'])
          ..output.add('C')
          ..attribute.add(AttributeProto()
            ..name = 'pads'
            ..ints.addAll([Int64(1), Int64(1), Int64(1), Int64(1)])),
        NodeProto()
          ..opType = 'Relu'
          ..input.add('C')
          ..output.add('Y'),
      ]);
    final model =
        OnnxModel.fromBytes((ModelProto()..graph = g).writeToBuffer());
    final x = Tensor.float(
        Float32List.fromList(
            List.generate(3 * 16 * 12, (_) => _rng.nextDouble() * 2 - 1)),
        [1, 3, 16, 12]);
    final want = model.run({'X': x}, ['Y'])['Y']!;
    await model.parallelize(workers: 3, poolConv: true);
    try {
      final got = (await model.runAsync({'X': x}, ['Y']))['Y']!;
      expect(got.shape, want.shape);
      expect(got.asFloatList(), want.asFloatList(),
          reason: 'pooled conv must be bitwise-identical');
    } finally {
      model.dispose();
    }
  });

  test('runAsync without parallelize matches run', () async {
    final g = GraphProto()
      ..input.add(ValueInfoProto()..name = 'X')
      ..output.add(ValueInfoProto()..name = 'Y')
      ..node.add(NodeProto()
        ..opType = 'Neg'
        ..input.add('X')
        ..output.add('Y'));
    final model =
        OnnxModel.fromBytes((ModelProto()..graph = g).writeToBuffer());
    final x = Tensor.float(Float32List.fromList([1, -2, 3]), [3]);
    expect((await model.runAsync({'X': x}, ['Y']))['Y']!.asFloatList(),
        model.run({'X': x}, ['Y'])['Y']!.asFloatList());
  });
}
