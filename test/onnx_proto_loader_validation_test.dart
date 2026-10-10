/// A malicious or just broken ONNX file can contain setup or runtime instructions that DDOS resources
/// Most of this file is constructing minimal risky examples of ONNX structures as ModelProto
@TestOn('vm')
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:fixnum/fixnum.dart';
import 'package:onnx_runtime_dart/onnx_proto.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/src/onnx_proto_loader_validation.dart'
    show bytesPerElement;
import 'package:test/test.dart';

const hostileWindowMillis = 1000;

const onnxFloatDtype = 1;
const onnxInt64Dtype = 7;
const onnxUint8Dtype = 2;
const onnxBoolDtype = 9;

final floatByteWidth = bytesPerElement(onnxFloatDtype)!;
final int64ByteWidth = bytesPerElement(onnxInt64Dtype)!;
final uint8ByteWidth = bytesPerElement(onnxUint8Dtype)!;

const ddosLoopTrips = kMaxFoldIters << 1;
const ddosScanElems = kMaxFoldIters << 1;
final ddosConstantElems = (kMaxFoldBytes ~/ floatByteWidth) << 1;
final ddosTileRepeats = (kMaxFoldBytes ~/ floatByteWidth) << 1;
final ddosExpandElems = (kMaxFoldBytes ~/ floatByteWidth) << 1;
final ddosRangeLimit = (kMaxFoldBytes ~/ int64ByteWidth) << 1;
final ddosDeclaredElems = (kMaxTensorBytes ~/ uint8ByteWidth) << 1;
const nestedOuterTrips = kMaxFoldIters ~/ 2;
const nestedInnerTrips = kMaxFoldIters ~/ 2;

final benignOnnxShapes = <String, ModelProto>{
  'constant of shape 1024 floats': constantOfShapeModel(1024),
  'range 0 to 8 int64': rangeModel(8),
  'tile 8 repeats': tileModel(8),
  'expand to 8': expandModel(8),
  'scan 4 elements': scanModel(4),
  'uint8 1024 declared with payload':
      overloadModel(1024, payload: List.filled(1024, 0)),
};

final ddosOnnxShapes = <String, ModelProto>{
  'constant of shape $ddosConstantElems floats':
      constantOfShapeModel(ddosConstantElems),
  'range $ddosRangeLimit int64': rangeModel(ddosRangeLimit),
  'tile $ddosTileRepeats repeats': tileModel(ddosTileRepeats),
  'expand to $ddosExpandElems': expandModel(ddosExpandElems),
  'scan over range $ddosScanElems': scanModel(ddosScanElems),
  'uint8 $ddosDeclaredElems declared': overloadModel(ddosDeclaredElems),
};

ModelProto graphModel(List<NodeProto> nodes, List<TensorProto> constants) =>
    ModelProto()
      ..irVersion = Int64(7)
      ..graph = (GraphProto()
        ..name = 'hostile'
        ..initializer.addAll(constants)
        ..node.addAll(nodes)
        ..output.add(ValueInfoProto()..name = 'y'));

NodeProto identity(String input, String output) => NodeProto()
  ..opType = 'Identity'
  ..input.add(input)
  ..output.add(output);

NodeProto rangeNode(String output) => NodeProto()
  ..opType = 'Range'
  ..input.addAll(['start', 'limit', 'delta'])
  ..output.add(output);

TensorProto intScalar(String name, int value) => TensorProto()
  ..name = name
  ..dataType = onnxInt64Dtype
  ..dims.add(Int64(1))
  ..int64Data.add(Int64(value));

TensorProto floatScalar(String name) => TensorProto()
  ..name = name
  ..dataType = onnxFloatDtype
  ..dims.add(Int64(1))
  ..floatData.add(0.0);

TensorProto boolTrue(String name) => TensorProto()
  ..name = name
  ..dataType = onnxBoolDtype
  ..dims.add(Int64(1))
  ..rawData = [1];

TensorProto uint8Declared(String name, int elems, {List<int>? payload}) {
  final tensor = TensorProto()
    ..name = name
    ..dataType = onnxUint8Dtype
    ..dims.add(Int64(elems));
  if (payload != null) {
    tensor.rawData = payload;
  }
  return tensor;
}

GraphProto identityBody() => GraphProto()
  ..input.addAll([
    ValueInfoProto()..name = 'keep_in',
    ValueInfoProto()..name = 'iter_in',
    ValueInfoProto()..name = 'x_in',
  ])
  ..output.addAll([
    ValueInfoProto()..name = 'keep_out',
    ValueInfoProto()..name = 'x_out',
  ])
  ..node.addAll([
    identity('keep_in', 'keep_out'),
    identity('x_in', 'x_out'),
  ]);

NodeProto loopNode(String tripName, GraphProto body) => NodeProto()
  ..opType = 'Loop'
  ..input.addAll([tripName, 'keep', 'x'])
  ..output.add('y')
  ..attribute.add(AttributeProto()
    ..name = 'body'
    ..g = body);

ModelProto loopModel(int trips) => graphModel(
      [loopNode('trip', identityBody())],
      [intScalar('trip', trips), boolTrue('keep'), floatScalar('x')],
    );

ModelProto nestedLoopModel(int outerTrips, int innerTrips) {
  final innerLoop = loopNode('trip2', identityBody());
  final outerBody = GraphProto()
    ..input.addAll([
      ValueInfoProto()..name = 'keep_in',
      ValueInfoProto()..name = 'iter_in',
      ValueInfoProto()..name = 'x_in',
    ])
    ..output.addAll([
      ValueInfoProto()..name = 'keep_out',
      ValueInfoProto()..name = 'x_out',
    ])
    ..node.addAll([
      identity('keep_in', 'keep_out'),
      innerLoop,
      identity('y2', 'x_out'),
    ]);
  return graphModel(
    [loopNode('trip', outerBody)],
    [
      intScalar('trip', outerTrips),
      intScalar('trip2', innerTrips),
      boolTrue('keep'),
      floatScalar('x'),
    ],
  );
}

ModelProto scanModel(int elems) {
  final scan = NodeProto()
    ..opType = 'Scan'
    ..input.addAll(['s', 'seq'])
    ..output.addAll(['y', 'sc'])
    ..attribute.addAll([
      AttributeProto()
        ..name = 'body'
        ..g = identityBody(),
      AttributeProto()
        ..name = 'num_scan_inputs'
        ..i = Int64(1),
    ]);
  return graphModel([
    rangeNode('seq'),
    scan,
  ], [
    intScalar('start', 0),
    intScalar('limit', elems),
    intScalar('delta', 1),
    floatScalar('s'),
  ]);
}

ModelProto constantOfShapeModel(int elems) => graphModel(
      [
        NodeProto()
          ..opType = 'ConstantOfShape'
          ..input.add('n')
          ..output.add('y'),
      ],
      [intScalar('n', elems)],
    );

ModelProto tileModel(int repeats) => graphModel(
      [
        NodeProto()
          ..opType = 'Tile'
          ..input.addAll(['x', 'rep'])
          ..output.add('y'),
      ],
      [floatScalar('x'), intScalar('rep', repeats)],
    );

ModelProto expandModel(int elems) => graphModel(
      [
        NodeProto()
          ..opType = 'Expand'
          ..input.addAll(['x', 'shape'])
          ..output.add('y'),
      ],
      [floatScalar('x'), intScalar('shape', elems)],
    );

ModelProto rangeModel(int limit) => graphModel(
      [rangeNode('y')],
      [intScalar('start', 0), intScalar('limit', limit), intScalar('delta', 1)],
    );

ModelProto overloadModel(int elems, {List<int>? payload}) => graphModel(
      [identity('w', 'y')],
      [uint8Declared('w', elems, payload: payload)],
    );

class IsolatedRequest {
  final Uint8List bytes;
  final SendPort replyTo;
  IsolatedRequest(this.bytes, this.replyTo);
}

class IsolatedResult {
  final bool timedOut;
  final String? rejectType;
  const IsolatedResult(this.timedOut, this.rejectType);
}

void runIsolatedRequest(IsolatedRequest req) {
  Object? outcome;
  try {
    OnnxModel.fromBytes(req.bytes);
    outcome = {'ok': true};
  } catch (e) {
    outcome = {'type': '${e.runtimeType}'};
  }
  req.replyTo.send(outcome);
}

Future<IsolatedResult> isolatedLoad(ModelProto model) async {
  final replyChannel = ReceivePort();
  final isolate = await Isolate.spawn(runIsolatedRequest,
      IsolatedRequest(model.writeToBuffer(), replyChannel.sendPort));
  final finished = Completer<Object?>();
  final subscription = replyChannel.listen(finished.complete);
  final message = await finished.future.timeout(
      Duration(milliseconds: hostileWindowMillis),
      onTimeout: () => 'timeout');
  await subscription.cancel();
  replyChannel.close();
  isolate.kill(priority: Isolate.immediate);
  if (message == 'timeout') {
    return const IsolatedResult(true, null);
  }
  final outcome = message as Map;
  return IsolatedResult(false, outcome['type'] as String?);
}

void main() {
  ddosOnnxShapes.forEach((label, model) {
    test('$label rejects with FormatException without hanging', () async {
      final outcome = await isolatedLoad(model);
      expect(outcome.timedOut, isFalse,
          reason: 'probe $label hung past $hostileWindowMillis milliseconds');
      expect(outcome.rejectType, 'FormatException',
          reason: 'probe $label rejected with an unexpected type');
    });
  });

  test('loop trips over the iteration cap reject', () async {
    final outcome = await isolatedLoad(loopModel(ddosLoopTrips));
    expect(outcome.timedOut, isFalse, reason: 'probe loop trips hung');
    expect(outcome.rejectType, 'FormatException',
        reason: 'probe loop trips rejected with an unexpected type');
  });

  test('nested loop trips over the cumulative iteration cap reject', () async {
    final outcome =
        await isolatedLoad(nestedLoopModel(nestedOuterTrips, nestedInnerTrips));
    expect(outcome.timedOut, isFalse, reason: 'probe nested loop trips hung');
    expect(outcome.rejectType, 'FormatException',
        reason: 'probe nested loop trips rejected with an unexpected type');
  });

  benignOnnxShapes.forEach((label, model) {
    test('$label loads under the fold byte cap', () {
      expect(() => OnnxModel.fromBytes(model.writeToBuffer()), returnsNormally,
          reason: 'probe $label failed at load');
    });
  });

  test('small trips load', () {
    expect(() => OnnxModel.fromBytes(loopModel(3).writeToBuffer()),
        returnsNormally);
  });

  test('small nested trips load', () {
    expect(() => OnnxModel.fromBytes(nestedLoopModel(2, 3).writeToBuffer()),
        returnsNormally);
  });

  test('byte overloads reject synchronously at load', () {
    expect(
        () => OnnxModel.fromBytes(
            constantOfShapeModel(ddosConstantElems).writeToBuffer()),
        throwsA(isA<FormatException>()),
        reason: 'constant of shape overload did not reject at load');
    expect(
        () => OnnxModel.fromBytes(
            overloadModel(ddosDeclaredElems).writeToBuffer()),
        throwsA(isA<FormatException>()),
        reason: 'declared shape overload did not reject at load');
  });

  test('initializers with inline payload load', () {
    final model = OnnxModel.fromBytes(graphModel([
      identity('w', 'y'),
    ], [
      TensorProto()
        ..name = 'w'
        ..dataType = onnxUint8Dtype
        ..dims.add(Int64(1024))
        ..int32Data.addAll(List.filled(1024, 0)),
    ]).writeToBuffer());
    expect(model.outputNames, contains('y'));
  });

  test('initializers with assigned raw_data load', () {
    final model = OnnxModel.fromBytes(graphModel([
      identity('w', 'y'),
    ], [
      uint8Declared('w', 1024, payload: List.filled(1024, 0)),
    ]).writeToBuffer());
    expect(model.outputNames, contains('y'));
  });

  test('raw_data addAll serializes empty', () {
    final serialized = ModelProto.fromBuffer(graphModel(const [], [
      TensorProto()
        ..name = 'w'
        ..dataType = onnxUint8Dtype
        ..dims.add(Int64(1024))
        ..rawData.addAll(List.filled(1024, 0)),
    ]).writeToBuffer());
    expect(serialized.graph.initializer.first.rawData, isEmpty);
  });

  test('raw_data add serializes empty', () {
    final serialized = ModelProto.fromBuffer(graphModel(const [], [
      TensorProto()
        ..name = 'w'
        ..dataType = onnxUint8Dtype
        ..dims.add(Int64(1))
        ..rawData.add(1),
    ]).writeToBuffer());
    expect(serialized.graph.initializer.first.rawData, isEmpty);
  });
}
