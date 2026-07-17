import 'dart:typed_data';

import 'package:fixnum/fixnum.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_proto.dart';
import 'package:onnx_runtime_dart/src/onnx_proto_loader.dart'
    show halfToFloat32Bits, tensorFromProto;
import 'package:test/test.dart';

void main() {
  group('tensorFromProto rejects malformed tensors cleanly', () {
    // A malformed / truncated .onnx model must fail with FormatException, not
    // an opaque RangeError/IndexError leaking from a per-element read or a
    // typed-list allocation. Regression guard for the asymmetric length check
    // (only BOOL used to be guarded).
    final tooShort = <String, TensorProto>{
      'float32 raw_data': TensorProto(
          name: 'w', dataType: 1, dims: [Int64(100)], rawData: [0, 0, 0, 0]),
      'int64 raw_data': TensorProto(
          name: 'w', dataType: 7, dims: [Int64(100)], rawData: List.filled(8, 0)),
      'int32 raw_data':
          TensorProto(name: 'w', dataType: 6, dims: [Int64(100)], rawData: []),
      'double raw_data':
          TensorProto(name: 'w', dataType: 11, dims: [Int64(100)], rawData: []),
      'float16 raw_data':
          TensorProto(name: 'w', dataType: 10, dims: [Int64(100)], rawData: []),
      'int8 raw_data':
          TensorProto(name: 'w', dataType: 3, dims: [Int64(100)], rawData: []),
      'bool raw_data':
          TensorProto(name: 'w', dataType: 9, dims: [Int64(100)], rawData: []),
    };
    tooShort.forEach((label, t) {
      test('truncated $label -> FormatException, not RangeError', () {
        expect(
          () => tensorFromProto(t),
          throwsA(isA<FormatException>()),
          reason: '$label truncation must reject cleanly',
        );
        // Guard against the RangeError/IndexError subclass sneaking through.
        expect(() => tensorFromProto(t), throwsA(isNot(isA<RangeError>())));
      });
    });

    test('negative dimension -> FormatException', () {
      final t = TensorProto(
          name: 'w', dataType: 1, dims: [Int64(-8)], rawData: [0, 0, 0, 0]);
      expect(() => tensorFromProto(t),
          throwsA(isA<FormatException>().having((e) => '$e', 'msg',
              contains('negative dimension'))));
    });

    final inlineMismatch = <String, TensorProto>{
      'float_data': TensorProto(
          name: 'w', dataType: 1, dims: [Int64(100)], floatData: [1.5]),
      'int64_data': TensorProto(
          name: 'w', dataType: 7, dims: [Int64(100)], int64Data: [Int64(3)]),
      'double_data': TensorProto(
          name: 'w', dataType: 11, dims: [Int64(100)], doubleData: [1.5]),
    };
    inlineMismatch.forEach((label, t) {
      test('inline $label length != shape product -> FormatException', () {
        expect(() => tensorFromProto(t), throwsA(isA<FormatException>()));
      });
    });

    test('well-formed tensors still load (no false positives)', () {
      final f = TensorProto(
          name: 'ok',
          dataType: 1,
          dims: [Int64(2)],
          rawData: Uint8List.sublistView(Float32List.fromList([1, 2])));
      expect(tensorFromProto(f).asFloatList(), [1.0, 2.0]);
      final inline = TensorProto(
          name: 'ok', dataType: 1, dims: [Int64(2)], floatData: [3, 4]);
      expect(tensorFromProto(inline).asFloatList(), [3.0, 4.0]);
    });
  });
  test('halfToFloat32Bits decodes representative fp16 values', () {
    double half(int bits) {
      final b = ByteData(4)..setUint32(0, halfToFloat32Bits(bits));
      return b.getFloat32(0);
    }

    expect(half(0x0000), 0.0); // +0
    expect(half(0x3C00), 1.0); // 1.0
    expect(half(0xC000), -2.0); // -2.0
    expect(half(0x3555), closeTo(0.333, 1e-3)); // ~1/3
  });

  test('CumSum accumulates along the axis', () {
    final g = GraphProto()
      ..input.addAll([
        ValueInfoProto()..name = 'X',
        ValueInfoProto()..name = 'axis',
      ])
      ..output.add(ValueInfoProto()..name = 'Y')
      ..node.add(NodeProto()
        ..opType = 'CumSum'
        ..input.addAll(['X', 'axis'])
        ..output.add('Y'));
    final x = Tensor.int64(Int64List.fromList([1, 1, 0, 1]), [4]);
    final axis = Tensor.int64(Int64List.fromList([0]), const []);
    final y = OnnxModel.fromBytes((ModelProto()..graph = g).writeToBuffer())
        .run({'X': x, 'axis': axis}, ['Y'])['Y']!;
    expect(y.asIntList(), [1, 2, 2, 3]); // running sum (position-id pattern)
  });
}
