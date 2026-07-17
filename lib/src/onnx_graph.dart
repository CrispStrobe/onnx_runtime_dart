/// Executes a parsed ONNX graph node-by-node. ONNX graphs are required by
/// the spec to be in topological order already, so this is a single linear
/// pass with a name -> Tensor value cache — no separate topo-sort needed.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'onnx.pb.dart';
import 'onnx_nn_ops.dart' as nn;
import 'onnx_ops.dart' as ops;
import 'onnx_qlinear_ops.dart' as ql;
import 'onnx_rnn_ops.dart' as rnn;
import 'onnx_proto_loader.dart';
import 'parallel_pool_stub.dart' if (dart.library.ffi) 'parallel_pool.dart';
import 'tensor.dart';

/// Cumulative per-op-type timing for one or more [OnnxGraphExecutor.run]
/// calls. Pass an instance to `run(profile: ...)` and read [report] after.
class ExecutionProfile {
  final Map<String, int> callsByOp = {};
  final Map<String, int> microsByOp = {};

  int get totalMicros => microsByOp.values.fold(0, (a, b) => a + b);

  /// Op types sorted by cumulative time, one line each:
  /// `opType  calls  total-ms  share%`.
  String report() {
    final ops = microsByOp.keys.toList()
      ..sort((a, b) => microsByOp[b]!.compareTo(microsByOp[a]!));
    final total = totalMicros;
    final b = StringBuffer(
        '${'op'.padRight(22)}${'calls'.padLeft(7)}${'ms'.padLeft(10)}'
        '${'share'.padLeft(8)}\n');
    for (final op in ops) {
      final us = microsByOp[op]!;
      b.writeln('${op.padRight(22)}${callsByOp[op]!.toString().padLeft(7)}'
          '${(us / 1000).toStringAsFixed(1).padLeft(10)}'
          '${total == 0 ? '' : '${(us * 100 / total).toStringAsFixed(1)}%'.padLeft(8)}');
    }
    b.write('${'total'.padRight(29)}'
        '${(total / 1000).toStringAsFixed(1).padLeft(10)}');
    return b.toString();
  }
}

class OnnxGraphExecutor {
  final GraphProto _graph;
  final Map<String, Tensor> _initializers = {};

  /// Transposed copies of initializer weights fed to `Gemm` with `transB=1`,
  /// built once on first use (ORT-style weight prepacking) — otherwise every
  /// call re-materializes the transpose.
  final Map<String, Tensor> _prepackedGemmB = {};

  /// Nodes remaining after load-time constant folding, in execution order.
  late final List<NodeProto> _nodes;

  /// Isolate GEMM pool (native only), populated by [parallelize].
  GemmPool? _pool;

  /// Minimum activation rows before a matmul is worth the isolate round-trip.
  static const _minPoolRows = 4;

  /// Original TensorProto element types of the initializers — widening int8/
  /// uint8 to int64 loses signedness, which QuantizeLinear's saturation
  /// bounds still need (3 = INT8 per the proto enum).
  final Map<String, int> _initializerElemType = {};

  OnnxGraphExecutor(ModelProto model, {ExternalDataResolver? externalData})
      : _graph = model.graph {
    for (final t in _graph.initializer) {
      _initializers[t.name] = tensorFromProto(t, ext: externalData);
      _initializerElemType[t.name] = t.dataType;
    }
    _nodes = _fusePatterns(_foldConstants());
  }

  /// Rewrites known multi-node patterns into single fused nodes (synthetic
  /// op types prefixed `_Fused`), after constant folding so scalar constants
  /// are initializers. Patterns only fuse when every intermediate value has
  /// exactly one consumer and is not a graph output.
  ///
  /// - erf-GELU: `0.5 * x * (1 + Erf(x / sqrt(2)))` (5 nodes -> _FusedGelu)
  /// - scaled-dot-product attention epilogue:
  ///   `MatMul(Softmax(MatMul(A,B)/c + mask), V)` -> _FusedSDPA, folding the
  ///   scale + mask + softmax into one pass over the attention matrix.
  List<NodeProto> _fusePatterns(List<NodeProto> nodes) {
    final uses = <String, int>{};
    final consumers = <String, List<int>>{};
    for (int i = 0; i < nodes.length; i++) {
      for (final name in nodes[i].input) {
        uses.update(name, (v) => v + 1, ifAbsent: () => 1);
        (consumers[name] ??= []).add(i);
      }
    }
    for (final o in _graph.output) {
      uses.update(o.name, (v) => v + 1000, ifAbsent: () => 1000);
    }
    final producerIdx = <String, int>{};
    for (int i = 0; i < nodes.length; i++) {
      for (final o in nodes[i].output) {
        producerIdx[o] = i;
      }
    }

    double? scalarInit(String name) {
      final t = _initializers[name];
      return t != null && t.length == 1 ? t.getD(0) : null;
    }

    /// The single consumer of [name] if it has exactly one use, else null.
    NodeProto? soleConsumer(String name) {
      if (uses[name] != 1) return null;
      final c = consumers[name];
      return c == null || c.isEmpty ? null : nodes[c.first];
    }

    NodeProto? producerOf(String name, String opType) {
      final i = producerIdx[name];
      return i != null && nodes[i].opType == opType ? nodes[i] : null;
    }

    final removed = <NodeProto>{};
    final replaceAt = <NodeProto, NodeProto>{}; // pattern tail -> fused node

    for (final erf in nodes) {
      if (erf.opType != 'Erf') continue;
      // x / sqrt(2)  (also matches x * (1/sqrt(2)))
      final div = producerOf(erf.input[0], 'Div');
      final mulIn = div == null ? producerOf(erf.input[0], 'Mul') : null;
      final pre = div ?? mulIn;
      if (pre == null || uses[erf.input[0]] != 1) continue;
      final c = scalarInit(pre.input[1]);
      final okScale = c != null &&
          (div != null
              ? (c - math.sqrt2).abs() < 1e-4
              : (c - 1 / math.sqrt2).abs() < 1e-4);
      if (!okScale) continue;
      final x = pre.input[0];
      // (1 + erf)
      final add = soleConsumer(erf.output[0]);
      if (add == null || add.opType != 'Add') continue;
      final one = scalarInit(add.input[0] == erf.output[0]
          ? add.input[1]
          : add.input[0]);
      if (one == null || (one - 1).abs() > 1e-6) continue;
      // * x, then * 0.5 (either order)
      final mul1 = soleConsumer(add.output[0]);
      if (mul1 == null || mul1.opType != 'Mul') continue;
      final mul1Other =
          mul1.input[0] == add.output[0] ? mul1.input[1] : mul1.input[0];
      final mul2 = soleConsumer(mul1.output[0]);
      if (mul2 == null || mul2.opType != 'Mul') continue;
      final mul2Other =
          mul2.input[0] == mul1.output[0] ? mul2.input[1] : mul2.input[0];
      final bool xThenHalf =
          mul1Other == x && (scalarInit(mul2Other) ?? 0) == 0.5;
      final bool halfThenX =
          (scalarInit(mul1Other) ?? 0) == 0.5 && mul2Other == x;
      if (!xThenHalf && !halfThenX) continue;

      removed.addAll([pre, erf, add, mul1]);
      replaceAt[mul2] = NodeProto()
        ..opType = '_FusedGelu'
        ..name = 'fused_gelu_${mul2.output[0]}'
        ..input.add(x)
        ..output.add(mul2.output[0]);
    }

    for (final sm in nodes) {
      if (sm.opType != 'Softmax') continue;
      final axis = _AttrMap(sm.attribute).getInt('axis') ?? -1;
      if (axis != -1) continue; // row softmax only (mask/scale fold per row)
      // scores + mask. The scores operand is either MatMul directly (exports
      // that pre-scale Q/K) or MatMul followed by a scalar Div/Mul.
      final add = producerOf(sm.input[0], 'Add');
      if (add == null || uses[sm.input[0]] != 1) continue;
      NodeProto? scale;
      NodeProto? mm1;
      String? mask;
      for (final cand in [add.input[0], add.input[1]]) {
        if (uses[cand] != 1) continue;
        final direct = producerOf(cand, 'MatMul');
        if (direct != null) {
          mm1 = direct;
          mask = add.input[0] == cand ? add.input[1] : add.input[0];
          break;
        }
        final p = producerOf(cand, 'Div') ?? producerOf(cand, 'Mul');
        if (p != null &&
            scalarInit(p.input[1]) != null &&
            uses[p.input[0]] == 1) {
          final mm = producerOf(p.input[0], 'MatMul');
          if (mm != null) {
            scale = p;
            mm1 = mm;
            mask = add.input[0] == cand ? add.input[1] : add.input[0];
            break;
          }
        }
      }
      if (mm1 == null) continue;
      final mm2 = soleConsumer(sm.output[0]);
      if (mm2 == null ||
          mm2.opType != 'MatMul' ||
          mm2.input[0] != sm.output[0]) {
        continue;
      }
      final double scaleVal;
      if (scale == null) {
        scaleVal = 1.0;
      } else {
        final c = scalarInit(scale.input[1])!;
        scaleVal = scale.opType == 'Div' ? 1.0 / c : c;
      }

      removed.addAll([mm1, if (scale != null) scale, add, sm]);
      replaceAt[mm2] = NodeProto()
        ..opType = '_FusedSDPA'
        ..name = 'fused_sdpa_${mm2.output[0]}'
        ..input.addAll([mm1.input[0], mm1.input[1], mm2.input[1], mask!])
        ..output.add(mm2.output[0])
        ..attribute.add(AttributeProto()
          ..name = 'scale'
          ..f = scaleVal);
    }

    return [
      for (final n in nodes)
        if (replaceAt.containsKey(n))
          replaceAt[n]!
        else if (!removed.contains(n))
          n
    ];
  }

  /// Executes every node whose inputs are all compile-time constants once,
  /// storing the results as initializers and dropping the node — transformer
  /// exports are full of `Constant`/`Shape`-arithmetic chains that would
  /// otherwise be recomputed identically on every run.
  List<NodeProto> _foldConstants() {
    // An initializer that is also a graph input is only a *default* the
    // caller may override at run time — never fold through those.
    final overridable = {for (final vi in _graph.input) vi.name};
    final constNames = {
      for (final name in _initializers.keys)
        if (!overridable.contains(name)) name
    };
    final kept = <NodeProto>[];
    for (final node in _graph.node) {
      final wantedOutputs = node.output.where((o) => o.isNotEmpty).length;
      if (wantedOutputs == 0 ||
          !node.input.every((s) => s.isEmpty || constNames.contains(s))) {
        kept.add(node);
        continue;
      }
      List<Tensor> outs;
      try {
        final ins = [
          for (final s in node.input) s.isEmpty ? null : _initializers[s]
        ];
        outs = _dispatch(node, ins, _AttrMap(node.attribute));
      } catch (_) {
        kept.add(node); // unsupported/failed op: leave it for run time
        continue;
      }
      if (outs.length < node.output.length &&
          node.output.skip(outs.length).any((o) => o.isNotEmpty)) {
        kept.add(node); // op produced fewer outputs than the graph consumes
        continue;
      }
      for (int k = 0; k < node.output.length && k < outs.length; k++) {
        if (node.output[k].isNotEmpty) {
          _initializers[node.output[k]] = outs[k];
          constNames.add(node.output[k]);
        }
      }
    }
    return kept;
  }

  /// Runs the graph with the given named runtime inputs, returns the
  /// requested named outputs.
  ///
  /// Pass a [profile] to accumulate per-op-type wall time across the run
  /// (adds one Stopwatch read per node — negligible next to the op work).
  Map<String, Tensor> run(Map<String, Tensor> inputs, List<String> outputNames,
      {ExecutionProfile? profile}) {
    final values = <String, Tensor>{..._initializers, ...inputs};
    _execNodes(_nodes, values, profile);
    // Debug escape hatch: ['*'] returns every value produced (note that
    // constant folding and fusion remove some of the file's tensor names).
    if (outputNames.length == 1 && outputNames.first == '*') return values;
    return {for (final name in outputNames) name: values[name]!};
  }

  /// Spawns [workers] isolate workers and partitions every 2-D float
  /// initializer fed to a top-level `MatMul` (with at least
  /// [minWeightElements] elements) across them by output column. After this,
  /// [runAsync] executes those matmuls on the pool; [run] stays
  /// single-threaded. Native targets only.
  Future<void> parallelize(
      {required int workers, int minWeightElements = 65536}) async {
    final toPartition = <String, (Float32List, int, int)>{};
    for (final node in _nodes) {
      if (node.opType != 'MatMul' || node.input.length < 2) continue;
      final name = node.input[1];
      final t = _initializers[name];
      if (t == null || !t.isFloat || t.rank != 2) continue;
      if (t.length < minWeightElements) continue;
      toPartition[name] = (t.f!, t.shape[0], t.shape[1]);
    }
    _pool = await GemmPool.spawn(workers, toPartition);
  }

  void dispose() {
    _pool?.dispose();
    _pool = null;
  }

  /// Async variant of [run]: identical semantics and (bitwise) results, but
  /// pool-partitioned matmuls execute across the worker isolates.
  Future<Map<String, Tensor>> runAsync(
      Map<String, Tensor> inputs, List<String> outputNames,
      {ExecutionProfile? profile}) async {
    final values = <String, Tensor>{..._initializers, ...inputs};
    final sw = profile == null ? null : Stopwatch();
    final pool = _pool;

    for (final node in _nodes) {
      final attrs = _AttrMap(node.attribute);
      final ins = [
        for (final name in node.input) name.isEmpty ? null : values[name]
      ];
      List<Tensor> outs;
      sw?..reset()..start();
      try {
        final part = pool != null && node.opType == 'MatMul'
            ? pool.weights[node.input[1]]
            : null;
        final a = part == null ? null : ins[0];
        if (part != null &&
            a != null &&
            a.isFloat &&
            a.rank >= 2 &&
            a.shape[a.rank - 1] == part.k &&
            a.length ~/ part.k >= _minPoolRows) {
          final m = a.length ~/ part.k;
          final out = await pool!.matmul(node.input[1], a.f!, m);
          outs = [
            Tensor.float(out, [...a.shape.sublist(0, a.rank - 1), part.n])
          ];
        } else {
          switch (node.opType) {
            case 'If':
              final branch =
                  ins[0]!.getI(0) != 0 ? 'then_branch' : 'else_branch';
              outs = _execSubgraph(
                  attrs.getGraph(branch)!, values, const [], profile);
            case 'Loop':
              outs = _runLoop(node, ins, attrs, values, profile);
            case 'Scan':
              outs = _runScan(node, ins, attrs, values, profile);
            default:
              outs = _dispatch(node, ins, attrs);
          }
        }
      } catch (e) {
        throw StateError('ONNX graph execution failed at node "${node.name}" '
            '(op=${node.opType}): $e');
      }
      if (profile != null) {
        sw!.stop();
        profile.callsByOp.update(node.opType, (v) => v + 1, ifAbsent: () => 1);
        profile.microsByOp.update(
            node.opType, (v) => v + sw.elapsedMicroseconds,
            ifAbsent: () => sw.elapsedMicroseconds);
      }
      for (int k = 0; k < node.output.length && k < outs.length; k++) {
        if (node.output[k].isNotEmpty) values[node.output[k]] = outs[k];
      }
    }
    return {for (final name in outputNames) name: values[name]!};
  }

  void _execNodes(List<NodeProto> nodes, Map<String, Tensor> values,
      ExecutionProfile? profile) {
    final sw = profile == null ? null : Stopwatch();
    for (final node in nodes) {
      final attrs = _AttrMap(node.attribute);
      final ins = [
        for (final name in node.input) name.isEmpty ? null : values[name]
      ];
      List<Tensor> outs;
      sw?..reset()..start();
      try {
        // Control-flow ops run subgraphs against the current scope, so they
        // are handled here rather than in the pure-function dispatch.
        switch (node.opType) {
          case 'If':
            final branch = ins[0]!.getI(0) != 0 ? 'then_branch' : 'else_branch';
            outs = _execSubgraph(attrs.getGraph(branch)!, values, const [],
                profile);
          case 'Loop':
            outs = _runLoop(node, ins, attrs, values, profile);
          case 'Scan':
            outs = _runScan(node, ins, attrs, values, profile);
          default:
            outs = _dispatch(node, ins, attrs);
        }
      } catch (e) {
        throw StateError('ONNX graph execution failed at node "${node.name}" '
            '(op=${node.opType}): $e');
      }
      if (profile != null) {
        sw!.stop();
        profile.callsByOp.update(node.opType, (v) => v + 1, ifAbsent: () => 1);
        profile.microsByOp.update(
            node.opType, (v) => v + sw.elapsedMicroseconds,
            ifAbsent: () => sw.elapsedMicroseconds);
      }
      for (int k = 0; k < node.output.length && k < outs.length; k++) {
        if (node.output[k].isNotEmpty) values[node.output[k]] = outs[k];
      }
    }
  }

  /// Executes a subgraph attribute with the enclosing scope visible
  /// (ONNX subgraphs capture outer values by name), binding [boundInputs]
  /// positionally to the subgraph's declared inputs. Returns the subgraph's
  /// outputs in declaration order.
  List<Tensor> _execSubgraph(GraphProto g, Map<String, Tensor> outer,
      List<Tensor?> boundInputs, ExecutionProfile? profile) {
    final values = Map.of(outer);
    for (final t in g.initializer) {
      values[t.name] = tensorFromProto(t);
    }
    for (int k = 0; k < g.input.length && k < boundInputs.length; k++) {
      if (boundInputs[k] != null) values[g.input[k].name] = boundInputs[k]!;
    }
    _execNodes(g.node, values, profile);
    return [for (final o in g.output) values[o.name]!];
  }

  /// `Scan` — inputs `[state_1..state_M, scan_1..scan_N]`; body graph maps
  /// `[states, scan slices] -> [states, scan output slices]`. Scan inputs
  /// are sliced along axis 0, scan outputs stacked along a new axis 0
  /// (only the default axes/directions are supported).
  List<Tensor> _runScan(NodeProto node, List<Tensor?> ins, _AttrMap attrs,
      Map<String, Tensor> values, ExecutionProfile? profile) {
    final body = attrs.getGraph('body')!;
    final nScan = attrs.getInt('num_scan_inputs')!;
    for (final attr in [
      'scan_input_axes',
      'scan_input_directions',
      'scan_output_axes',
      'scan_output_directions'
    ]) {
      final v = attrs.getInts(attr);
      if (v != null && v.any((x) => x != 0)) {
        throw UnsupportedError(
            'Scan: only default $attr (all zero) supported, got $v');
      }
    }
    final nState = node.input.length - nScan;
    var states = [for (int k = 0; k < nState; k++) ins[k]!];
    final scanIns = [for (int k = nState; k < ins.length; k++) ins[k]!];
    final iters = scanIns.isEmpty ? 0 : scanIns.first.shape[0];
    final nScanOut = body.output.length - nState;
    final scans = List.generate(nScanOut, (_) => <Tensor>[]);

    Tensor sliceRow(Tensor t, int i) {
      final rowLen = t.length ~/ t.shape[0];
      final shape = t.shape.sublist(1);
      return t.isFloat
          ? Tensor.float(
              Float32List.sublistView(t.f!, i * rowLen, (i + 1) * rowLen),
              shape)
          : Tensor.int64(
              Int64List.sublistView(t.i!, i * rowLen, (i + 1) * rowLen),
              shape);
    }

    for (int i = 0; i < iters; i++) {
      final outs = _execSubgraph(body, values,
          [...states, for (final t in scanIns) sliceRow(t, i)], profile);
      states = outs.sublist(0, nState);
      for (int k = 0; k < nScanOut; k++) {
        scans[k].add(outs[nState + k]);
      }
    }
    return [...states, for (final steps in scans) _stack(steps)];
  }

  /// `Loop` — inputs `[M?, cond?, v_1..v_N]`; body graph inputs
  /// `[iter, cond, v_1..v_N]`, outputs `[cond, v_1..v_N, scan_1..scan_K]`.
  /// Node outputs are the final `v` values, then each scan output stacked
  /// along a new leading axis.
  List<Tensor> _runLoop(NodeProto node, List<Tensor?> ins, _AttrMap attrs,
      Map<String, Tensor> values, ExecutionProfile? profile) {
    final body = attrs.getGraph('body')!;
    final maxTrips = ins.isNotEmpty && ins[0] != null ? ins[0]!.getI(0) : null;
    bool cond = ins.length < 2 || ins[1] == null || ins[1]!.getI(0) != 0;
    var carried = [for (int k = 2; k < ins.length; k++) ins[k]!];
    final nCarried = carried.length;
    final nScan = body.output.length - 1 - nCarried;
    final scans = List.generate(nScan, (_) => <Tensor>[]);

    int iter = 0;
    while (cond && (maxTrips == null || iter < maxTrips)) {
      final outs = _execSubgraph(
          body,
          values,
          [Tensor.scalarInt(iter), Tensor.scalarInt(cond ? 1 : 0), ...carried],
          profile);
      cond = outs[0].getI(0) != 0;
      carried = outs.sublist(1, 1 + nCarried);
      for (int k = 0; k < nScan; k++) {
        scans[k].add(outs[1 + nCarried + k]);
      }
      iter++;
    }

    return [
      ...carried,
      for (final steps in scans) _stack(steps),
    ];
  }

  /// Stacks per-iteration tensors along a new leading axis (scan outputs).
  static Tensor _stack(List<Tensor> steps) {
    if (steps.isEmpty) return Tensor.float(Float32List(0), const [0]);
    final inner = steps.first.shape;
    final shape = [steps.length, ...inner];
    if (steps.first.isFloat) {
      final out = Float32List(steps.length * steps.first.length);
      for (int k = 0; k < steps.length; k++) {
        out.setRange(k * steps.first.length, (k + 1) * steps.first.length,
            steps[k].f!);
      }
      return Tensor.float(out, shape);
    }
    final out = Int64List(steps.length * steps.first.length);
    for (int k = 0; k < steps.length; k++) {
      out.setRange(k * steps.first.length, (k + 1) * steps.first.length,
          steps[k].asIntList());
    }
    return Tensor.int64(out, shape);
  }

  List<Tensor> _dispatch(NodeProto node, List<Tensor?> ins, _AttrMap attrs) {
    Tensor need(int i) => ins[i]!;
    switch (node.opType) {
      case 'Identity':
        return [need(0)];
      // --- synthetic fused ops (created by _fusePatterns, never in files) ---
      case '_FusedGelu':
        return [ops.opGelu(need(0))];
      case '_FusedSDPA':
        return [
          ops.opFusedSDPA(need(0), need(1), need(2), need(3),
              attrs.getFloat('scale')!)
        ];
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
        return [
          ops.opShape(need(0),
              start: attrs.getInt('start'), end: attrs.getInt('end'))
        ];
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
        // Opset 10+ takes starts/ends/axes/steps as inputs; opset 1-9 as
        // attributes (no steps).
        if (ins.length == 1) {
          return [
            ops.opSlice(need(0), attrs.getInts('starts')!,
                attrs.getInts('ends')!, attrs.getInts('axes'), null)
          ];
        }
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
        var b = need(1);
        var transB = (attrs.getInt('transB') ?? 0) != 0;
        if (transB && _initializers.containsKey(node.input[1])) {
          b = _prepackedGemmB[node.input[1]] ??= ops.opTranspose(b, [1, 0]);
          transB = false;
        }
        return [
          ops.opGemm(
            need(0),
            b,
            ins.length > 2 ? ins[2] : null,
            alpha: attrs.getFloat('alpha') ?? 1.0,
            beta: attrs.getFloat('beta') ?? 1.0,
            transA: (attrs.getInt('transA') ?? 0) != 0,
            transB: transB,
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
      case 'ReduceProd':
        return [
          ops.opReduceProd(
            need(0),
            ins.length > 1 && ins[1] != null
                ? ins[1]!.asIntList()
                : attrs.getInts('axes'),
            (attrs.getInt('keepdims') ?? 1) != 0,
          )
        ];
      case 'ReduceMax':
      case 'ReduceMin':
        return [
          ops.opReduceMinMax(
            need(0),
            ins.length > 1 && ins[1] != null
                ? ins[1]!.asIntList()
                : attrs.getInts('axes'),
            (attrs.getInt('keepdims') ?? 1) != 0,
            isMax: node.opType == 'ReduceMax',
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
      // --- convolution / pooling / normalization family ---
      case 'Conv':
        return [
          nn.opConv(
            need(0),
            need(1),
            ins.length > 2 ? ins[2] : null,
            strides: attrs.getInts('strides'),
            pads: attrs.getInts('pads'),
            dilations: attrs.getInts('dilations'),
            group: attrs.getInt('group') ?? 1,
            autoPad: attrs.getString('auto_pad') ?? 'NOTSET',
          )
        ];
      case 'ConvTranspose':
        return [
          nn.opConvTranspose(
            need(0),
            need(1),
            ins.length > 2 ? ins[2] : null,
            strides: attrs.getInts('strides'),
            pads: attrs.getInts('pads'),
            dilations: attrs.getInts('dilations'),
            outputPadding: attrs.getInts('output_padding'),
            outputShape: attrs.getInts('output_shape'),
            group: attrs.getInt('group') ?? 1,
          )
        ];
      case 'MaxPool':
        return [
          nn.opMaxPool(
            need(0),
            kernel: attrs.getInts('kernel_shape')!,
            strides: attrs.getInts('strides'),
            pads: attrs.getInts('pads'),
            dilations: attrs.getInts('dilations'),
            autoPad: attrs.getString('auto_pad') ?? 'NOTSET',
            ceilMode: (attrs.getInt('ceil_mode') ?? 0) != 0,
          )
        ];
      case 'AveragePool':
        return [
          nn.opAveragePool(
            need(0),
            kernel: attrs.getInts('kernel_shape')!,
            strides: attrs.getInts('strides'),
            pads: attrs.getInts('pads'),
            autoPad: attrs.getString('auto_pad') ?? 'NOTSET',
            ceilMode: (attrs.getInt('ceil_mode') ?? 0) != 0,
            countIncludePad: (attrs.getInt('count_include_pad') ?? 0) != 0,
          )
        ];
      case 'GlobalAveragePool':
        return [nn.opGlobalPool(need(0), isMax: false)];
      case 'GlobalMaxPool':
        return [nn.opGlobalPool(need(0), isMax: true)];
      case 'BatchNormalization':
        return [
          nn.opBatchNormalization(need(0), need(1), need(2), need(3), need(4),
              attrs.getFloat('epsilon') ?? 1e-5)
        ];
      case 'InstanceNormalization':
        return [
          nn.opInstanceNormalization(
              need(0), need(1), need(2), attrs.getFloat('epsilon') ?? 1e-5)
        ];
      case 'Resize':
        // Opset 10: (X, scales). Opset 11+: (X, roi, scales, sizes).
        final scalesT = ins.length == 2 ? ins[1] : (ins.length > 2 ? ins[2] : null);
        final sizesT = ins.length > 3 ? ins[3] : null;
        return [
          nn.opResize(
            need(0),
            scales: scalesT == null || scalesT.length == 0
                ? null
                : scalesT.asFloatList().toList(),
            sizes: sizesT?.asIntList().toList(),
            mode: attrs.getString('mode') ?? 'nearest',
            coordMode: attrs.getString('coordinate_transformation_mode') ??
                'half_pixel',
            nearestMode:
                attrs.getString('nearest_mode') ?? 'round_prefer_floor',
          )
        ];
      case 'Flatten':
        return [nn.opFlatten(need(0), attrs.getInt('axis') ?? 1)];
      case 'LeakyRelu':
        return [ops.opLeakyRelu(need(0), attrs.getFloat('alpha') ?? 0.01)];
      case 'Elu':
        return [ops.opElu(need(0), attrs.getFloat('alpha') ?? 1.0)];
      case 'HardSigmoid':
        return [
          ops.opHardSigmoid(need(0), attrs.getFloat('alpha') ?? 0.2,
              attrs.getFloat('beta') ?? 0.5)
        ];
      case 'HardSwish':
        return [ops.opHardSwish(need(0))];
      case 'Softplus':
        return [ops.opSoftplus(need(0))];
      case 'Gelu':
        return [
          ops.opGelu(need(0),
              tanhApprox: attrs.getString('approximate') == 'tanh')
        ];
      case 'PRelu':
        return [ops.opPRelu(need(0), need(1))];
      case 'Size':
        return [ops.opSize(need(0))];
      case 'QuantizeLinear':
        // Saturation bounds come from the zero-point tensor's declared
        // dtype; absent zero point means uint8 per the spec.
        final int8 = node.input.length > 2 &&
            _initializerElemType[node.input[2]] == 3;
        return [
          ops.opQuantizeLinear(need(0), need(1),
              ins.length > 2 ? ins[2] : null,
              axis: attrs.getInt('axis') ?? 1,
              lo: int8 ? -128 : 0,
              hi: int8 ? 127 : 255)
        ];
      case 'DequantizeLinear':
        return [
          ops.opDequantizeLinear(need(0), need(1),
              ins.length > 2 ? ins[2] : null,
              axis: attrs.getInt('axis') ?? 1)
        ];
      case 'DynamicQuantizeLinear':
        return ops.opDynamicQuantizeLinear(need(0));
      case 'MatMulInteger':
        return [
          ql.opMatMulInteger(need(0), need(1),
              ins.length > 2 ? ins[2] : null, ins.length > 3 ? ins[3] : null)
        ];
      case 'ConvInteger':
        return [
          ql.opConvInteger(need(0), need(1), ins.length > 2 ? ins[2] : null,
              ins.length > 3 ? ins[3] : null,
              strides: attrs.getInts('strides'),
              pads: attrs.getInts('pads'),
              dilations: attrs.getInts('dilations'),
              group: attrs.getInt('group') ?? 1)
        ];
      case 'QLinearMatMul':
        final qmInt8 = _initializerElemType[node.input[7]] == 3;
        return [
          ql.opQLinearMatMul(need(0), need(1), ins[2], need(3), need(4),
              ins[5], need(6), ins.length > 7 ? ins[7] : null,
              lo: qmInt8 ? -128 : 0, hi: qmInt8 ? 127 : 255)
        ];
      case 'QLinearConv':
        final qcInt8 = _initializerElemType[node.input[7]] == 3;
        return [
          ql.opQLinearConv(need(0), need(1), ins[2], need(3), need(4),
              ins[5], need(6), ins.length > 7 ? ins[7] : null,
              ins.length > 8 ? ins[8] : null,
              strides: attrs.getInts('strides'),
              pads: attrs.getInts('pads'),
              dilations: attrs.getInts('dilations'),
              group: attrs.getInt('group') ?? 1,
              lo: qcInt8 ? -128 : 0,
              hi: qcInt8 ? 127 : 255)
        ];
      case 'Pad':
        // Opset 11+ takes pads/value/axes as inputs; opset 2 as attributes.
        final padsList = ins.length > 1 && ins[1] != null
            ? ins[1]!.asIntList().toList()
            : attrs.getInts('pads')!;
        return [
          ops.opPad(
            need(0),
            padsList,
            mode: attrs.getString('mode') ?? 'constant',
            constantValue: ins.length > 2 && ins[2] != null
                ? ins[2]!.getD(0)
                : attrs.getFloat('value') ?? 0,
            axes: ins.length > 3 && ins[3] != null
                ? ins[3]!.asIntList().toList()
                : null,
          )
        ];
      // --- recurrent family ---
      case 'LSTM':
        return rnn.opLSTM(
          need(0),
          need(1),
          need(2),
          b: ins.length > 3 ? ins[3] : null,
          sequenceLens: ins.length > 4 ? ins[4] : null,
          initialH: ins.length > 5 ? ins[5] : null,
          initialC: ins.length > 6 ? ins[6] : null,
          peepholes: ins.length > 7 ? ins[7] : null,
          hiddenSize: attrs.getInt('hidden_size')!,
          direction: attrs.getString('direction') ?? 'forward',
          activations: attrs.getStrings('activations'),
          clip: attrs.getFloat('clip'),
        );
      case 'GRU':
        return rnn.opGRU(
          need(0),
          need(1),
          need(2),
          b: ins.length > 3 ? ins[3] : null,
          sequenceLens: ins.length > 4 ? ins[4] : null,
          initialH: ins.length > 5 ? ins[5] : null,
          hiddenSize: attrs.getInt('hidden_size')!,
          direction: attrs.getString('direction') ?? 'forward',
          activations: attrs.getStrings('activations'),
          linearBeforeReset: (attrs.getInt('linear_before_reset') ?? 0) != 0,
          clip: attrs.getFloat('clip'),
        );
      case 'RNN':
        return rnn.opRNN(
          need(0),
          need(1),
          need(2),
          b: ins.length > 3 ? ins[3] : null,
          sequenceLens: ins.length > 4 ? ins[4] : null,
          initialH: ins.length > 5 ? ins[5] : null,
          hiddenSize: attrs.getInt('hidden_size')!,
          direction: attrs.getString('direction') ?? 'forward',
          activations: attrs.getStrings('activations'),
          clip: attrs.getFloat('clip'),
        );
      default:
        throw UnsupportedError(
            'ONNX op "${node.opType}" not implemented in the Dart interpreter');
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
  List<String>? getStrings(String name) => _byName.containsKey(name)
      ? _byName[name]!
          .strings
          .map((s) => String.fromCharCodes(s))
          .toList()
      : null;
  GraphProto? getGraph(String name) =>
      _byName.containsKey(name) ? _byName[name]!.g : null;

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
