/// Gemm transB weight prepacking: the transposed initializer is cached after
/// the first run, so repeated runs must produce identical results (and match
/// the unprepacked math done by hand).
library;

import 'dart:typed_data';

import 'package:fixnum/fixnum.dart';
import 'package:onnx_runtime_dart/onnx_proto.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:test/test.dart';

void main() {
  test('Gemm transB=1 with initializer B is stable across runs', () {
    // Y = X (2x3) * W^T (W is 4x3, so W^T is 3x4) + C (4)
    final w = [for (int k = 0; k < 12; k++) (k + 1).toDouble()];
    final g = GraphProto()
      ..input.add(ValueInfoProto()..name = 'X')
      ..output.add(ValueInfoProto()..name = 'Y')
      ..initializer.addAll([
        TensorProto()
          ..name = 'W'
          ..dataType = TensorProto_DataType.FLOAT.value
          ..dims.addAll([Int64(4), Int64(3)])
          ..floatData.addAll(w),
        TensorProto()
          ..name = 'C'
          ..dataType = TensorProto_DataType.FLOAT.value
          ..dims.add(Int64(4))
          ..floatData.addAll([10, 20, 30, 40]),
      ])
      ..node.add(NodeProto()
        ..opType = 'Gemm'
        ..input.addAll(['X', 'W', 'C'])
        ..output.add('Y')
        ..attribute.add(AttributeProto()
          ..name = 'transB'
          ..i = Int64(1)));

    final model =
        OnnxModel.fromBytes((ModelProto()..graph = g).writeToBuffer());
    final x = Tensor.float(Float32List.fromList([1, 2, 3, 4, 5, 6]), [2, 3]);

    // Hand-computed: row0 = [1,2,3]·W[r,:] + C[r].
    // W rows: [1,2,3],[4,5,6],[7,8,9],[10,11,12]
    // x0: [14, 32, 50, 68] + C = [24, 52, 80, 108]
    // x1: [32, 77, 122, 167] + C = [42, 97, 152, 207]
    const want = [24.0, 52.0, 80.0, 108.0, 42.0, 97.0, 152.0, 207.0];

    final first = model.run({'X': x}, ['Y'])['Y']!;
    expect(first.shape, [2, 4]);
    expect(first.asFloatList(), want);

    // Second run exercises the cached prepacked weight.
    final second = model.run({'X': x}, ['Y'])['Y']!;
    expect(second.asFloatList(), want);
  });
}
