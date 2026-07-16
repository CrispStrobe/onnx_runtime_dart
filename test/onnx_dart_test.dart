import 'dart:typed_data';

import 'package:fixnum/fixnum.dart';
import 'package:onnx_dart/onnx_dart.dart';
import 'package:onnx_dart/onnx_proto.dart';
import 'package:test/test.dart';

/// Builds a serialized ONNX model computing `Y = Relu(X + W)` with a constant
/// `W` initializer, then returns its bytes — exercising initializers, a
/// two-node graph, and the protobuf round-trip.
Uint8List buildReluAddModel(List<double> w, List<int> shape) {
  final model = ModelProto()
    ..irVersion = Int64(7)
    ..graph = (GraphProto()
      ..name = 'relu_add'
      ..input.add(ValueInfoProto()..name = 'X')
      ..output.add(ValueInfoProto()..name = 'Y')
      ..initializer.add(TensorProto()
        ..name = 'W'
        ..dataType = 1 // FLOAT
        ..dims.addAll(shape.map((d) => Int64(d)))
        ..floatData.addAll(w))
      ..node.add(NodeProto()
        ..opType = 'Add'
        ..input.addAll(['X', 'W'])
        ..output.add('S'))
      ..node.add(NodeProto()
        ..opType = 'Relu'
        ..input.add('S')
        ..output.add('Y')));
  return model.writeToBuffer();
}

void main() {
  test('runs Relu(X + W) end-to-end from serialized bytes', () {
    final bytes = buildReluAddModel([0.5, 0.5, 0.5, 0.5], [2, 2]);
    final model = OnnxModel.fromBytes(bytes);

    final x = Tensor.float(Float32List.fromList([1, -2, 3, -4]), [2, 2]);
    final out = model.run({'X': x}, ['Y']);

    final y = out['Y']!;
    expect(y.shape, [2, 2]);
    // X + W = [1.5, -1.5, 3.5, -3.5]; Relu clamps negatives to 0.
    expect(y.asFloatList(), [1.5, 0.0, 3.5, 0.0]);
  });

  test('MatMul produces the expected product', () {
    // Y = X . W, with W the 2x2 identity → Y == X.
    final model = ModelProto()
      ..irVersion = Int64(7)
      ..graph = (GraphProto()
        ..name = 'matmul'
        ..input.add(ValueInfoProto()..name = 'X')
        ..output.add(ValueInfoProto()..name = 'Y')
        ..initializer.add(TensorProto()
          ..name = 'W'
          ..dataType = 1
          ..dims.addAll([Int64(2), Int64(2)])
          ..floatData.addAll([1, 0, 0, 1]))
        ..node.add(NodeProto()
          ..opType = 'MatMul'
          ..input.addAll(['X', 'W'])
          ..output.add('Y')));

    final onnx = OnnxModel.fromBytes(model.writeToBuffer());
    final x = Tensor.float(Float32List.fromList([5, 6, 7, 8]), [2, 2]);
    final y = onnx.run({'X': x}, ['Y'])['Y']!;
    expect(y.asFloatList(), [5.0, 6.0, 7.0, 8.0]);
  });

  test('unknown output name surfaces a clear error', () {
    final bytes = buildReluAddModel([0, 0, 0, 0], [2, 2]);
    final model = OnnxModel.fromBytes(bytes);
    final x = Tensor.float(Float32List.fromList([1, 2, 3, 4]), [2, 2]);
    expect(() => model.run({'X': x}, ['nope']), throwsA(anything));
  });
}
