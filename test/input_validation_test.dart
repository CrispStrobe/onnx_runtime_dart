/// Input-signature validation: fixed-dimension mismatches and missing
/// inputs fail loudly at run() instead of computing silently wrong numbers
/// (the hazard class: batch-fixed exports fed batched inputs).
library;

import 'dart:typed_data';

import 'package:fixnum/fixnum.dart';
import 'package:onnx_runtime_dart/onnx_proto.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:test/test.dart';

OnnxModel _model() {
  TensorShapeProto_Dimension fixed(int v) =>
      TensorShapeProto_Dimension()..dimValue = Int64(v);
  TensorShapeProto_Dimension dynamic_() =>
      TensorShapeProto_Dimension()..dimParam = 'batch';
  final g = GraphProto()
    ..input.add(ValueInfoProto()
      ..name = 'X'
      ..type = (TypeProto()
        ..tensorType = (TypeProto_Tensor()
          ..elemType = TensorProto_DataType.FLOAT.value
          ..shape = (TensorShapeProto()
            ..dim.addAll([dynamic_(), fixed(3)])))))
    ..output.add(ValueInfoProto()..name = 'Y')
    ..node.add(NodeProto()
      ..opType = 'Neg'
      ..input.add('X')
      ..output.add('Y'));
  return OnnxModel.fromBytes((ModelProto()..graph = g).writeToBuffer());
}

void main() {
  test('dynamic dims accept any size; fixed dims are enforced', () {
    final model = _model();
    // batch is dynamic: [5, 3] fine.
    final ok = model.run(
        {'X': Tensor.float(Float32List(15), [5, 3])}, ['Y'])['Y']!;
    expect(ok.shape, [5, 3]);
    // fixed dim mismatch: [5, 4] must throw, naming input and sizes.
    expect(
        () => model.run({'X': Tensor.float(Float32List(20), [5, 4])}, ['Y']),
        throwsA(isA<ArgumentError>().having(
            (e) => e.message, 'message', contains('fixed size 3'))));
    // rank mismatch.
    expect(
        () => model.run({'X': Tensor.float(Float32List(3), [3])}, ['Y']),
        throwsA(isA<ArgumentError>()));
  });

  test('missing required input throws by name', () {
    expect(() => _model().run({}, ['Y']),
        throwsA(isA<ArgumentError>().having(
            (e) => e.message, 'message', contains('"X"'))));
  });

  test('a malformed protobuf rejects with FormatException, not RangeError', () {
    // Untrusted .onnx bytes with a corrupt length-delimited field: unknown
    // field #100 (wire type 2) declaring an oversized length. The protobuf
    // decoder leaks a RangeError from Uint8List.view; fromBytes must normalize
    // it. (Found by coverage-guided fuzzing of OnnxModel.fromBytes.)
    final bad = Uint8List.fromList([0xA2, 0x06, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F]);
    expect(() => OnnxModel.fromBytes(bad), throwsFormatException);
    expect(() => OnnxModel.fromBytes(bad), throwsA(isNot(isA<RangeError>())));
  });
}
