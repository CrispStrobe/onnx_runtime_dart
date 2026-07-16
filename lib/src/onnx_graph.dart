/// Executes a parsed ONNX graph node-by-node. ONNX graphs are required by
/// the spec to be in topological order already, so this is a single linear
/// pass with a name -> Tensor value cache — no separate topo-sort needed.
library;

import 'onnx.pb.dart';
import 'onnx_ops.dart' as ops;
import 'onnx_proto_loader.dart';
import 'tensor.dart';

class OnnxGraphExecutor {
  final GraphProto _graph;
  final Map<String, Tensor> _initializers = {};

  OnnxGraphExecutor(ModelProto model) : _graph = model.graph {
    for (final t in _graph.initializer) {
      _initializers[t.name] = tensorFromProto(t);
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
          ops.opSqueeze(need(0), ins.length > 1 ? ins[1]!.asIntList() : null)
        ];
      case 'Unsqueeze':
        return [ops.opUnsqueeze(need(0), need(1).asIntList())];
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
            ins.length > 1 && ins[1] != null ? ins[1]!.asIntList() : null,
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
}
