import 'dart:typed_data';

import 'package:fixnum/fixnum.dart';
import 'package:onnx_dart/onnx_dart.dart';
import 'package:onnx_dart/onnx_proto.dart';
import 'package:test/test.dart';

NodeProto node(String op, List<String> ins, String out,
        [List<AttributeProto> attrs = const []]) =>
    NodeProto()
      ..opType = op
      ..input.addAll(ins)
      ..output.add(out)
      ..attribute.addAll(attrs);

AttributeProto intsAttr(String name, List<int> v) => AttributeProto()
  ..name = name
  ..ints.addAll(v.map((e) => Int64(e)));
AttributeProto intAttr(String name, int v) => AttributeProto()
  ..name = name
  ..i = Int64(v);

OnnxModel modelOf(GraphProto g) =>
    OnnxModel.fromBytes((ModelProto()..graph = g).writeToBuffer());

void main() {
  // Regression: ReduceMean's `axes` given as an ATTRIBUTE (older opsets, as in
  // BERT LayerNorm) must reduce only that axis — not collapse to a scalar. This
  // was the bug that broke jina-v2 parity.
  test('ReduceMean reduces the axes attribute, not all axes', () {
    final g = GraphProto()
      ..input.add(ValueInfoProto()..name = 'X')
      ..output.add(ValueInfoProto()..name = 'Y')
      ..node.add(node(
          'ReduceMean',
          ['X'],
          'Y',
          [
            intsAttr('axes', [-1]),
            intAttr('keepdims', 1)
          ]));
    // X = [[1,2,3],[10,20,30]] → per-row mean over last axis = [[2],[20]].
    final x = Tensor.float(Float32List.fromList([1, 2, 3, 10, 20, 30]), [2, 3]);
    final y = modelOf(g).run({'X': x}, ['Y'])['Y']!;
    expect(y.shape, [2, 1]);
    expect(y.asFloatList(), [2.0, 20.0]);
  });

  test('Where selects element-wise on a bool-ish condition', () {
    final g = GraphProto()
      ..input.addAll([
        ValueInfoProto()..name = 'C',
        ValueInfoProto()..name = 'A',
        ValueInfoProto()..name = 'B',
      ])
      ..output.add(ValueInfoProto()..name = 'Y')
      ..node.add(node('Where', ['C', 'A', 'B'], 'Y'));
    final c = Tensor.int64(Int64List.fromList([1, 0, 1, 0]), [4]);
    final a = Tensor.float(Float32List.fromList([1, 2, 3, 4]), [4]);
    final b = Tensor.float(Float32List.fromList([-1, -2, -3, -4]), [4]);
    final y = modelOf(g).run({'C': c, 'A': a, 'B': b}, ['Y'])['Y']!;
    expect(y.asFloatList(), [1.0, -2.0, 3.0, -4.0]);
  });

  test('Range + Equal + Cast build a simple mask', () {
    final g = GraphProto()
      ..input.addAll([
        ValueInfoProto()..name = 'start',
        ValueInfoProto()..name = 'limit',
        ValueInfoProto()..name = 'delta',
      ])
      ..output.add(ValueInfoProto()..name = 'R')
      ..node.add(node('Range', ['start', 'limit', 'delta'], 'R'));
    Tensor s(int v) => Tensor.int64(Int64List.fromList([v]), const []);
    final r = modelOf(g)
        .run({'start': s(0), 'limit': s(5), 'delta': s(1)}, ['R'])['R']!;
    expect(r.shape, [5]);
    expect(r.asIntList(), [0, 1, 2, 3, 4]);
  });

  test('Sigmoid, Tanh, Abs, Neg', () {
    for (final entry in {
      'Abs': [-3.0, 3.0],
      'Neg': [3.0, -3.0],
    }.entries) {
      final g = GraphProto()
        ..input.add(ValueInfoProto()..name = 'X')
        ..output.add(ValueInfoProto()..name = 'Y')
        ..node.add(node(entry.key, ['X'], 'Y'));
      final y = modelOf(g).run({
        'X': Tensor.float(Float32List.fromList([entry.value[0]]), [1])
      }, [
        'Y'
      ])['Y']!;
      expect(y.asFloatList()[0], closeTo(entry.value[1], 1e-6),
          reason: entry.key);
    }
  });
}
