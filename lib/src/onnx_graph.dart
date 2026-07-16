/// Executes a parsed ONNX graph node-by-node. ONNX graphs are required by
/// the spec to be in topological order already, so this is a single linear
/// pass with a name -> Tensor value cache — no separate topo-sort needed.
library;

import 'dart:typed_data';

import 'onnx.pb.dart';
import 'onnx_ops.dart' as ops;
import 'onnx_proto_loader.dart';
import 'tensor.dart';

class OnnxGraphExecutor {
  final GraphProto _graph;
  final Map<String, Tensor> _initializers = {};

  OnnxGraphExecutor(ModelProto model, {ExternalDataResolver? externalData})
      : _graph = model.graph {
    for (final t in _graph.initializer) {
      _initializers[t.name] = tensorFromProto(t, ext: externalData);
    }
  }

  /// Runs the graph with the given named runtime inputs, returns the
  /// requested named outputs.
  Map<String, Tensor> run(
      Map<String, Tensor> inputs, List<String> outputNames) {
    final values = <String, Tensor>{..._initializers, ...inputs};

    for (final node in _graph.node) {
      final attrs = _AttrMap(node.attribute);
      final ins = [
        for (final name in node.input) name.isEmpty ? null : values[name]
      ];
      List<Tensor> outs;
      try {
        outs = _dispatch(node.opType, ins, attrs);
      } catch (e) {
        throw StateError('ONNX graph execution failed at node "${node.name}" '
            '(op=${node.opType}): $e');
      }
      for (int k = 0; k < node.output.length && k < outs.length; k++) {
        if (node.output[k].isNotEmpty) values[node.output[k]] = outs[k];
      }
    }

    return {for (final name in outputNames) name: values[name]!};
  }

  List<Tensor> _dispatch(String opType, List<Tensor?> ins, _AttrMap attrs) {
    Tensor need(int i) => ins[i]!;
    switch (opType) {
      case 'Add':
        return [ops.opAdd(need(0), need(1))];
      case 'Sub':
        return [ops.opSub(need(0), need(1))];
      case 'Mul':
        return [ops.opMul(need(0), need(1))];
      case 'Div':
        return [ops.opDiv(need(0), need(1))];
      case 'Pow':
        return [ops.opPow(need(0), need(1))];
      case 'Sqrt':
        return [ops.opSqrt(need(0))];
      case 'Reciprocal':
        return [ops.opReciprocal(need(0))];
      case 'Relu':
        return [ops.opRelu(need(0))];
      case 'Erf':
        return [ops.opErf(need(0))];
      case 'Clip':
        return [
          ops.opClip(need(0), ins.length > 1 ? ins[1] : null,
              ins.length > 2 ? ins[2] : null)
        ];
      case 'Cast':
        return [ops.opCast(need(0), attrs.getInt('to')!)];
      case 'Shape':
        return [ops.opShape(need(0))];
      case 'Reshape':
        return [ops.opReshape(need(0), need(1))];
      case 'Transpose':
        return [ops.opTranspose(need(0), attrs.getInts('perm')!)];
      case 'Squeeze':
        return [
          ops.opSqueeze(
              need(0),
              ins.length > 1 && ins[1] != null
                  ? ins[1]!.asIntList()
                  : attrs.getInts('axes'))
        ];
      case 'Unsqueeze':
        return [
          ops.opUnsqueeze(
              need(0),
              ins.length > 1 && ins[1] != null
                  ? ins[1]!.asIntList()
                  : attrs.getInts('axes')!)
        ];
      case 'Concat':
        return [
          ops.opConcat([for (final t in ins) t!], attrs.getInt('axis')!)
        ];
      case 'Gather':
        return [ops.opGather(need(0), need(1), attrs.getInt('axis') ?? 0)];
      case 'GatherND':
        return [
          ops.opGatherND(need(0), need(1), attrs.getInt('batch_dims') ?? 0)
        ];
      case 'Expand':
        return [ops.opExpand(need(0), need(1))];
      case 'Slice':
        return [
          ops.opSlice(
            need(0),
            ins[1]!.asIntList(),
            ins[2]!.asIntList(),
            ins.length > 3 && ins[3] != null ? ins[3]!.asIntList() : null,
            ins.length > 4 && ins[4] != null ? ins[4]!.asIntList() : null,
          )
        ];
      case 'ReduceMean':
        return [
          ops.opReduceMean(
            need(0),
            ins.length > 1 && ins[1] != null
                ? ins[1]!.asIntList()
                : attrs.getInts('axes'),
            (attrs.getInt('keepdims') ?? 1) != 0,
          )
        ];
      case 'Softmax':
        return [ops.opSoftmax(need(0), attrs.getInt('axis') ?? -1)];
      case 'LayerNormalization':
        return [
          ops.opLayerNormalization(
            need(0),
            need(1),
            need(2),
            attrs.getInt('axis') ?? -1,
            attrs.getFloat('epsilon') ?? 1e-5,
          )
        ];
      case 'MatMul':
        return [ops.opMatMul(need(0), need(1))];
      case 'Gemm':
        return [
          ops.opGemm(
            need(0),
            need(1),
            ins.length > 2 ? ins[2] : null,
            alpha: attrs.getFloat('alpha') ?? 1.0,
            beta: attrs.getFloat('beta') ?? 1.0,
            transA: (attrs.getInt('transA') ?? 0) != 0,
            transB: (attrs.getInt('transB') ?? 0) != 0,
          )
        ];
      case 'Einsum':
        return [ops.opEinsum(attrs.getString('equation')!, need(0), need(1))];
      // --- extended op set ---
      case 'Constant':
        return [attrs.constantTensor()];
      case 'ConstantOfShape':
        return [ops.opConstantOfShape(need(0), attrs.getTensor('value'))];
      case 'Range':
        return [ops.opRange(need(0), need(1), need(2))];
      case 'Abs':
        return [ops.opAbs(need(0))];
      case 'Neg':
        return [ops.opNeg(need(0))];
      case 'Sigmoid':
        return [ops.opSigmoid(need(0))];
      case 'Tanh':
        return [ops.opTanh(need(0))];
      case 'Cos':
        return [ops.opCos(need(0))];
      case 'Sin':
        return [ops.opSin(need(0))];
      case 'Exp':
        return [ops.opExp(need(0))];
      case 'Log':
        return [ops.opLog(need(0))];
      case 'Not':
        return [ops.opNot(need(0))];
      case 'Equal':
        return [ops.opEqual(need(0), need(1))];
      case 'Greater':
        return [ops.opGreater(need(0), need(1))];
      case 'Less':
        return [ops.opLess(need(0), need(1))];
      case 'GreaterOrEqual':
        return [ops.opGreaterOrEqual(need(0), need(1))];
      case 'LessOrEqual':
        return [ops.opLessOrEqual(need(0), need(1))];
      case 'And':
        return [ops.opAnd(need(0), need(1))];
      case 'Or':
        return [ops.opOr(need(0), need(1))];
      case 'Max':
        return [
          ops.opMax([for (final t in ins) t!])
        ];
      case 'Min':
        return [
          ops.opMin([for (final t in ins) t!])
        ];
      case 'Where':
        return [ops.opWhere(need(0), need(1), need(2))];
      case 'ReduceSum':
        return [
          ops.opReduceSum(
            need(0),
            ins.length > 1 && ins[1] != null
                ? ins[1]!.asIntList()
                : attrs.getInts('axes'),
            (attrs.getInt('keepdims') ?? 1) != 0,
          )
        ];
      case 'CumSum':
        return [
          ops.opCumSum(need(0), need(1).getI(0),
              exclusive: (attrs.getInt('exclusive') ?? 0) != 0,
              reverse: (attrs.getInt('reverse') ?? 0) != 0)
        ];
      case 'GatherElements':
        return [
          ops.opGatherElements(need(0), need(1), attrs.getInt('axis') ?? 0)
        ];
      default:
        throw UnsupportedError(
            'ONNX op "$opType" not implemented in the Dart interpreter');
    }
  }
}

/// Convenience accessor over a NodeProto's AttributeProto list.
class _AttrMap {
  final Map<String, AttributeProto> _byName;
  _AttrMap(Iterable<AttributeProto> attrs)
      : _byName = {for (final a in attrs) a.name: a};

  int? getInt(String name) =>
      _byName.containsKey(name) ? _byName[name]!.i.toInt() : null;
  double? getFloat(String name) =>
      _byName.containsKey(name) ? _byName[name]!.f : null;
  String? getString(String name) =>
      _byName.containsKey(name) ? String.fromCharCodes(_byName[name]!.s) : null;
  List<int>? getInts(String name) => _byName.containsKey(name)
      ? _byName[name]!.ints.map((v) => v.toInt()).toList()
      : null;

  /// The tensor value of a TENSOR-typed attribute (e.g. `ConstantOfShape`'s
  /// `value`), or null if absent.
  Tensor? getTensor(String name) =>
      _byName.containsKey(name) ? tensorFromProto(_byName[name]!.t) : null;

  /// The value produced by a `Constant` node — either a `value` tensor or one
  /// of the scalar/list attribute forms the op allows.
  Tensor constantTensor() {
    if (_byName.containsKey('value')) {
      return tensorFromProto(_byName['value']!.t);
    }
    final ints = _byName['value_ints'];
    if (ints != null) {
      return Tensor.int64(
          Int64List.fromList(ints.ints.map((v) => v.toInt()).toList()),
          [ints.ints.length]);
    }
    final i = _byName['value_int'];
    if (i != null) {
      return Tensor.int64(Int64List.fromList([i.i.toInt()]), const []);
    }
    final floats = _byName['value_floats'];
    if (floats != null) {
      return Tensor.float(
          Float32List.fromList(floats.floats), [floats.floats.length]);
    }
    final f = _byName['value_float'];
    if (f != null) {
      return Tensor.float(Float32List.fromList([f.f]), const []);
    }
    throw StateError('Constant node has no recognized value attribute');
  }
}
