import 'dart:typed_data';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_proto.dart';
import 'package:onnx_runtime_dart/src/onnx_proto_loader.dart'
    show halfToFloat32Bits;
import 'package:test/test.dart';

void main() {
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
