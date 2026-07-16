// Runs a tiny ONNX graph — `Y = Relu(X + W)` — entirely in Dart, no FFI.
//
// In real use you would load the bytes of a `.onnx` file
// (`OnnxModel.fromBytes(File('model.onnx').readAsBytesSync())`); here we build a
// small model programmatically with the ONNX protobuf types so the example is
// self-contained. Run with:
//
//   dart run example/onnx_runtime_dart_example.dart
import 'dart:typed_data';

import 'package:fixnum/fixnum.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_proto.dart';

void main() {
  // Build a model: S = X + W (W constant), Y = Relu(S).
  final model = ModelProto()
    ..irVersion = Int64(7)
    ..graph = (GraphProto()
      ..name = 'relu_add'
      ..input.add(ValueInfoProto()..name = 'X')
      ..output.add(ValueInfoProto()..name = 'Y')
      ..initializer.add(TensorProto()
        ..name = 'W'
        ..dataType = 1 // FLOAT
        ..dims.addAll([Int64(2), Int64(2)])
        ..floatData.addAll([0.5, 0.5, 0.5, 0.5]))
      ..node.add(NodeProto()
        ..opType = 'Add'
        ..input.addAll(['X', 'W'])
        ..output.add('S'))
      ..node.add(NodeProto()
        ..opType = 'Relu'
        ..input.add('S')
        ..output.add('Y')));

  // Parse it back from bytes, exactly as a real `.onnx` file would be loaded.
  final onnx = OnnxModel.fromBytes(model.writeToBuffer());

  final x = Tensor.float(Float32List.fromList([1, -2, 3, -4]), [2, 2]);
  final y = onnx.run({'X': x}, ['Y'])['Y']!;

  print('X      = ${x.asFloatList()}');
  print('Y = Relu(X + 0.5) = ${y.asFloatList()}'); // [1.5, 0.0, 3.5, 0.0]
}
