/// INT4 tiles must preserve FP32 reconstruction and dot-product ordering.
@TestOn('vm')
library;

import 'dart:typed_data';

import 'package:fixnum/fixnum.dart';
import 'package:onnx_runtime_dart/onnx_proto.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/src/onnx_ops.dart' as ops;
import 'package:onnx_runtime_dart/src/onnx_qlinear_ops.dart' as quant;
import 'package:test/test.dart';

void main() {
  for (final (shape, n, blockSize, zeroPoints) in [
    ([1, 7], 35, 16, false),
    ([1, 7, 40], 65, 16, true),
    ([2, 3, 131], 33, 128, true),
    ([2, 5, 33], 127, 16, false),
    ([4, 64], 64, 32, true),
    ([3, 16], 1, 16, true),
    ([5, 37], 31, 16, false),
    ([4, 48], 32, 16, true),
  ]) {
    test('exact FP32 result: A=$shape N=$n block=$blockSize zp=$zeroPoints',
        () {
      final data = _fixture(shape, n, blockSize, zeroPoints: zeroPoints);
      // weights is constructed from the original logical quantized values,
      // independently of reading the packed bytes. opMatMul uses the same
      // FP32 GEMM as the former full-matrix dequantization implementation.
      final expected = ops.opMatMul(data.a, data.weights);
      for (final narrowDirect in [false, true]) {
        final actual = quant.opMatMulNBits(data.a, data.b, data.scales, data.zp,
            k: shape.last,
            n: n,
            bits: 4,
            blockSize: blockSize,
            narrowDirect: narrowDirect);
        expect(actual.shape, [...shape.take(shape.length - 1), n]);
        expect(actual.asFloatList(), orderedEquals(expected.asFloatList()),
            reason: 'narrowDirect=$narrowDirect');
      }
    });
  }

  test('graph dispatch supports default and narrowDirectGemm experiment', () {
    final data = _fixture([1, 4, 40], 65, 16, zeroPoints: true);
    final graph = GraphProto()
      ..input.add(ValueInfoProto()..name = 'A')
      ..output.add(ValueInfoProto()..name = 'Y')
      ..initializer.addAll([
        TensorProto()
          ..name = 'B'
          ..dataType = 2
          ..dims.addAll(data.b.shape.map(Int64.new))
          ..rawData = data.b.u8!,
        TensorProto()
          ..name = 'S'
          ..dataType = 1
          ..dims.addAll(data.scales.shape.map(Int64.new))
          ..floatData.addAll(data.scales.asFloatList()),
        TensorProto()
          ..name = 'Z'
          ..dataType = 2
          ..dims.addAll(data.zp!.shape.map(Int64.new))
          ..rawData = data.zp!.u8!,
      ])
      ..node.add(NodeProto()
        ..opType = 'MatMulNBits'
        ..domain = 'com.microsoft'
        ..input.addAll(['A', 'B', 'S', 'Z'])
        ..output.add('Y')
        ..attribute.addAll([
          for (final entry
              in {'K': 40, 'N': 65, 'bits': 4, 'block_size': 16}.entries)
            AttributeProto()
              ..name = entry.key
              ..i = Int64(entry.value),
        ]));
    final bytes = (ModelProto()..graph = graph).writeToBuffer();
    final expected = ops.opMatMul(data.a, data.weights);
    for (final experiments in [
      <OnnxExperiment>{},
      {OnnxExperiment.narrowDirectGemm}
    ]) {
      final model = OnnxModel.fromBytes(bytes, experiments: experiments);
      final actual = model.run({'A': data.a}, ['Y'])['Y']!;
      expect(actual.shape, expected.shape);
      expect(actual.asFloatList(), orderedEquals(expected.asFloatList()));
    }
  });

  test('packed zero-point rows are byte-aligned when block count is odd', () {
    final data = _fixture([1, 35], 3, 16, zeroPoints: true);
    expect(data.zp!.shape, [3, 2]); // three blocks + a padding nibble per row
    final actual = quant.opMatMulNBits(data.a, data.b, data.scales, data.zp,
        k: 35, n: 3, bits: 4, blockSize: 16);
    expect(actual.asFloatList(),
        orderedEquals(ops.opMatMul(data.a, data.weights).asFloatList()));
  });

  test('legacy widened integer weight storage remains supported', () {
    final data = _fixture([2, 39], 67, 32);
    final widened = Tensor.int64(Int64List.fromList(data.b.u8!), data.b.shape);
    final actual = quant.opMatMulNBits(data.a, widened, data.scales, null,
        k: 39, n: 67, bits: 4, blockSize: 32);
    expect(actual.asFloatList(),
        orderedEquals(ops.opMatMul(data.a, data.weights).asFloatList()));
  });

  test('empty activation rows and output columns keep their shapes', () {
    for (final (shape, n) in [
      ([2, 0, 17], 65),
      ([1, 3, 17], 0)
    ]) {
      final data = _fixture(shape, n, 16);
      final actual = quant.opMatMulNBits(data.a, data.b, data.scales, null,
          k: 17, n: n, bits: 4, blockSize: 16);
      expect(actual.shape, [...shape.take(shape.length - 1), n]);
      expect(actual.asFloatList(), isEmpty);
    }
  });

  test('unpacked float zero points fail instead of silently using eight', () {
    final data = _fixture([1, 16], 2, 16);
    expect(
        () => quant.opMatMulNBits(data.a, data.b, data.scales,
            Tensor.float(Float32List.fromList([1, 2]), [2, 1]),
            k: 16, n: 2, bits: 4, blockSize: 16),
        throwsA(isA<UnsupportedError>()));
  });

  test('truncated packed weights, scales and zero points fail clearly', () {
    final data = _fixture([2, 33], 35, 16, zeroPoints: true);
    for (final (b, scales, zp) in [
      (Tensor.uint8(Uint8List(1), [1]), data.scales, data.zp),
      (data.b, Tensor.float(Float32List(1), [1]), data.zp),
      (data.b, data.scales, Tensor.uint8(Uint8List(1), [1])),
    ]) {
      expect(
          () => quant.opMatMulNBits(data.a, b, scales, zp,
              k: 33, n: 35, bits: 4, blockSize: 16),
          throwsA(isA<ArgumentError>()));
    }
  });

  for (final extra in ['g_idx', 'bias']) {
    test('$extra input remains explicitly unsupported by graph dispatch', () {
      final model = _unsupportedModel(extra);
      expect(
          () => model.run({
                'A': Tensor.float(Float32List(16), [1, 16])
              }, [
                'Y'
              ]),
          throwsA(isA<StateError>().having((e) => e.message, 'message',
              contains('g_idx/bias inputs not supported'))));
    });
  }
}

({Tensor a, Tensor b, Tensor scales, Tensor? zp, Tensor weights}) _fixture(
    List<int> shape, int n, int blockSize,
    {bool zeroPoints = false}) {
  final k = shape.last;
  final blocks = (k + blockSize - 1) ~/ blockSize;
  final blobBytes = blockSize ~/ 2;
  final zpStride = (blocks + 1) ~/ 2;
  final bytes = Uint8List(n * blocks * blobBytes)
    ..fillRange(0, n * blocks * blobBytes, 0xA5);
  final scales = Float32List(n * blocks);
  final zp = Uint8List(n * zpStride)..fillRange(0, n * zpStride, 0xF0);
  final weights = Float32List(k * n);
  for (var col = 0; col < n; col++) {
    for (var block = 0; block < blocks; block++) {
      final index = col * blocks + block;
      // Include zero, negative and unequal scales, stored with FP32 rounding.
      scales[index] = index % 11 == 0 ? 0 : ((index % 7) - 3) * 0.071;
      final zero = zeroPoints ? (col * 3 + block * 5 + 1) % 16 : 8;
      final zeroByte = col * zpStride + block ~/ 2;
      final zeroShift = (block % 2) * 4;
      zp[zeroByte] = (zp[zeroByte] & ~(15 << zeroShift)) | (zero << zeroShift);
      for (var row = block * blockSize;
          row < k && row < (block + 1) * blockSize;
          row++) {
        final value = (row * 7 + col * 3 + block * 5) % 16;
        weights[row * n + col] = (value - zero) * scales[index];
        final inside = row - block * blockSize;
        final offset = (col * blocks + block) * blobBytes + inside ~/ 2;
        final shift = (inside % 2) * 4;
        bytes[offset] = (bytes[offset] & ~(15 << shift)) | (value << shift);
      }
    }
  }
  final count = shape.fold<int>(1, (a, b) => a * b);
  return (
    a: Tensor.float(
        Float32List.fromList([
          for (var i = 0; i < count; i++) ((i % 19) - 9) / 7,
        ]),
        shape),
    b: Tensor.uint8(bytes, [n, blocks, blobBytes]),
    scales: Tensor.float(scales, [n, blocks]),
    zp: zeroPoints ? Tensor.uint8(zp, [n, zpStride]) : null,
    weights: Tensor.float(weights, [k, n]),
  );
}

OnnxModel _unsupportedModel(String extra) {
  final graph = GraphProto()
    ..input.add(ValueInfoProto()..name = 'A')
    ..output.add(ValueInfoProto()..name = 'Y')
    ..initializer.addAll([
      TensorProto()
        ..name = 'B'
        ..dataType = 2
        ..dims.addAll([Int64(1), Int64(1), Int64(8)])
        ..rawData = Uint8List(8),
      TensorProto()
        ..name = 'S'
        ..dataType = 1
        ..dims.addAll([Int64(1), Int64(1)])
        ..floatData.add(1),
      TensorProto()
        ..name = extra
        ..dataType = 1
        ..dims.add(Int64(1))
        ..floatData.add(1),
    ])
    ..node.add(NodeProto()
      ..opType = 'MatMulNBits'
      ..domain = 'com.microsoft'
      ..input.addAll(extra == 'bias'
          ? ['A', 'B', 'S', '', '', extra]
          : ['A', 'B', 'S', '', extra])
      ..output.add('Y')
      ..attribute.addAll([
        for (final entry
            in {'K': 16, 'N': 1, 'bits': 4, 'block_size': 16}.entries)
          AttributeProto()
            ..name = entry.key
            ..i = Int64(entry.value),
      ]));
  return OnnxModel.fromBytes((ModelProto()..graph = graph).writeToBuffer());
}
