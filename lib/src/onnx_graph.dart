/// Executes a parsed ONNX graph node-by-node. ONNX graphs are required by
/// the spec to be in topological order already, so this is a single linear
/// pass with a name -> Tensor value cache — no separate topo-sort needed.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:fixnum/fixnum.dart';

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

  /// External-data resolver, kept for tensors that live in node
  /// attributes (e.g. large `Constant` values) rather than initializers.
  final ExternalDataResolver? _ext;

  /// Default-domain opset version of the loaded model (some op defaults are
  /// opset-sensitive, e.g. RoiAlign's coordinate transform).
  late final int _opset;

  OnnxGraphExecutor(ModelProto model,
      {ExternalDataResolver? externalData, bool fuse = true})
      : _graph = model.graph,
        _ext = externalData {
    _opset = model.opsetImport
        .where((o) => o.domain.isEmpty)
        .map((o) => o.version.toInt())
        .fold(0, math.max);
    for (final t in _graph.initializer) {
      _initializers[t.name] = tensorFromProto(t, ext: externalData);
      _initializerElemType[t.name] = t.dataType;
    }
    final folded = _foldConstants();
    _nodes = fuse ? _fusePatterns(folded) : folded;
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
    // Names captured by Loop/If/Scan body subgraphs are consumers the
    // top-level scan above cannot see — protect them like graph outputs.
    void protectSubgraphCaptures(GraphProto g) {
      for (final n in g.node) {
        for (final i in n.input) {
          if (i.isNotEmpty) {
            uses.update(i, (v) => v + 1000, ifAbsent: () => 1000);
          }
        }
        for (final a in n.attribute) {
          if (a.hasG()) protectSubgraphCaptures(a.g);
          for (final sg in a.graphs) {
            protectSubgraphCaptures(sg);
          }
        }
      }
    }

    for (final n in nodes) {
      for (final a in n.attribute) {
        if (a.hasG()) protectSubgraphCaptures(a.g);
        for (final sg in a.graphs) {
          protectSubgraphCaptures(sg);
        }
      }
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

    // RMSNorm: x -> Pow(2) -> ReduceMean(axis) -> Add(eps) -> Sqrt ->
    // Reciprocal -> Mul(x) -> Mul(gamma). Qwen-style transformers and Maia3
    // carry 16-113 of these chains.
    for (final pw in nodes) {
      if (pw.opType != 'Pow') continue;
      final exp = scalarInit(pw.input[1]);
      if (exp == null || exp != 2.0) continue;
      final x = pw.input[0];
      final rm = soleConsumer(pw.output[0]);
      if (rm == null || rm.opType != 'ReduceMean') continue;
      // Single reduce axis, from attribute or initializer input.
      final axesAttr = _AttrMap(rm.attribute, _ext).getInts('axes');
      List<int>? axes = axesAttr;
      if (axes == null && rm.input.length > 1) {
        final t = _initializers[rm.input[1]];
        if (t != null) axes = t.asIntList().toList();
      }
      if (axes == null || axes.length != 1) continue;
      if ((_AttrMap(rm.attribute, _ext).getInt('keepdims') ?? 1) != 1) {
        continue;
      }
      final addEps = soleConsumer(rm.output[0]);
      if (addEps == null || addEps.opType != 'Add') continue;
      final eps = scalarInit(addEps.input[0] == rm.output[0]
          ? addEps.input[1]
          : addEps.input[0]);
      if (eps == null) continue;
      final sqrt = soleConsumer(addEps.output[0]);
      if (sqrt == null || sqrt.opType != 'Sqrt') continue;
      final rec = soleConsumer(sqrt.output[0]);
      if (rec == null || rec.opType != 'Reciprocal') continue;
      final mul1 = soleConsumer(rec.output[0]);
      if (mul1 == null || mul1.opType != 'Mul') continue;
      final mul1Other =
          mul1.input[0] == rec.output[0] ? mul1.input[1] : mul1.input[0];
      if (mul1Other != x) continue;
      final mul2 = soleConsumer(mul1.output[0]);
      if (mul2 == null || mul2.opType != 'Mul') continue;
      final gamma =
          mul2.input[0] == mul1.output[0] ? mul2.input[1] : mul2.input[0];
      final gammaT = _initializers[gamma];
      if (gammaT == null || gammaT.rank != 1) continue;

      removed.addAll([pw, rm, addEps, sqrt, rec, mul1]);
      replaceAt[mul2] = NodeProto()
        ..opType = '_FusedRMSNorm'
        ..name = 'fused_rmsnorm_${mul2.output[0]}'
        ..input.addAll([x, gamma])
        ..output.add(mul2.output[0])
        ..attribute.addAll([
          AttributeProto()
            ..name = 'axis'
            ..i = Int64(axes.first),
          AttributeProto()
            ..name = 'epsilon'
            ..f = eps,
        ]);
    }

    for (final sm in nodes) {
      if (sm.opType != 'Softmax') continue;
      final axis = _AttrMap(sm.attribute, _ext).getInt('axis') ?? -1;
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
        outs = _dispatch(node, ins, _AttrMap(node.attribute, _ext));
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
  /// Validates provided inputs against the graph's declared signatures:
  /// missing required inputs and fixed-dimension mismatches fail loudly here
  /// instead of producing silently wrong numbers downstream (batch-fixed
  /// exports are a real hazard — ORT rejects such feeds too).
  void _validateInputs(Map<String, Tensor> inputs) {
    for (final vi in _graph.input) {
      final t = inputs[vi.name];
      if (t == null) {
        if (_initializers.containsKey(vi.name)) continue; // default present
        throw ArgumentError('Missing required input "${vi.name}"');
      }
      final dims = vi.type.tensorType.shape.dim;
      if (dims.isEmpty) continue; // no declared shape
      if (dims.length != t.rank) {
        throw ArgumentError('Input "${vi.name}": rank ${t.rank} provided, '
            'model declares rank ${dims.length}');
      }
      for (int i = 0; i < dims.length; i++) {
        final want = dims[i].dimValue.toInt();
        if (want > 0 && want != t.shape[i]) {
          throw ArgumentError('Input "${vi.name}": dim $i is ${t.shape[i]}, '
              'model declares fixed size $want — this export does not '
              'support that shape');
        }
      }
    }
  }

  Map<String, Tensor> run(Map<String, Tensor> inputs, List<String> outputNames,
      {ExecutionProfile? profile}) {
    _validateInputs(inputs);
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
  /// [poolConv] additionally fans 2-D convolutions out across the workers by
  /// output-row bands. Off by default: conv messages carry the whole input
  /// activation to every worker, and for CNN workloads measured so far that
  /// copying costs more than the banded compute saves (unlike matmuls, whose
  /// per-call messages are small). Worth trying for large batches or very
  /// high-resolution inputs.
  Future<void> parallelize(
      {required int workers,
      int minWeightElements = 65536,
      bool poolConv = false}) async {
    final toPartition = <String, (Float32List, int, int)>{};
    final convToReplicate = <String, (Float32List, List<int>, Float32List?)>{};
    for (final node in _nodes) {
      if (node.opType == 'Conv' && !poolConv) continue;
      if (node.opType == 'MatMul' && node.input.length >= 2) {
        final name = node.input[1];
        final t = _initializers[name];
        if (t == null || !t.isFloat || t.rank != 2) continue;
        if (t.length < minWeightElements) continue;
        toPartition[name] = (t.f!, t.shape[0], t.shape[1]);
      } else if (node.opType == 'Gemm' && node.input.length >= 2) {
        // Partition Gemm's B in its effective (possibly transposed)
        // orientation, keyed so the runAsync path can find it; alpha/beta/
        // bias/transA stay on the main isolate.
        final name = node.input[1];
        final t = _initializers[name];
        if (t == null || !t.isFloat || t.rank != 2) continue;
        if (t.length < minWeightElements) continue;
        final transB =
            (_AttrMap(node.attribute, _ext).getInt('transB') ?? 0) != 0;
        final b = transB ? ops.opTranspose(t, [1, 0]) : t;
        toPartition['gemm:$name'] = (b.f!, b.shape[0], b.shape[1]);
      } else if (node.opType == 'Conv' && node.input.length >= 2) {
        var w = _initializers[node.input[1]];
        if (w == null || !w.isFloat || (w.rank != 4 && w.rank != 3)) {
          continue;
        }
        // 1-D convs pool as 2-D with a singleton trailing axis, so the
        // output-row band split runs along time.
        if (w.rank == 3) w = w.reshape([...w.shape, 1]);
        final biasT =
            node.input.length > 2 ? _initializers[node.input[2]] : null;
        if (node.input.length > 2 &&
            node.input[2].isNotEmpty &&
            biasT == null) {
          continue; // runtime bias — keep local
        }
        convToReplicate[node.input[1]] = (w.f!, w.shape, biasT?.f);
      }
    }
    _pool = await GemmPool.spawn(workers, toPartition, convToReplicate);
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
    _validateInputs(inputs);
    final values = <String, Tensor>{..._initializers, ...inputs};
    final sw = profile == null ? null : Stopwatch();
    final pool = _pool;

    for (final node in _nodes) {
      final attrs = _AttrMap(node.attribute, _ext);
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
        final gemmPart = pool != null && node.opType == 'Gemm'
            ? pool.weights['gemm:${node.input[1]}']
            : null;
        final convX = pool != null &&
                node.opType == 'Conv' &&
                pool.convWeights.contains(node.input[1])
            ? ins[0]
            : null;
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
        } else if (gemmPart != null &&
            ins[0] != null &&
            ins[0]!.isFloat &&
            ins[0]!.rank == 2 &&
            (attrs.getInt('transA') ?? 0) == 0 &&
            ins[0]!.shape[1] == gemmPart.k &&
            ins[0]!.shape[0] >= _minPoolRows) {
          final aG = ins[0]!;
          final m = aG.shape[0];
          final raw =
              await pool!.matmul('gemm:${node.input[1]}', aG.f!, m);
          var y = Tensor.float(raw, [m, gemmPart.n]);
          final alpha = attrs.getFloat('alpha') ?? 1.0;
          if (alpha != 1.0) {
            for (int idx = 0; idx < y.f!.length; idx++) {
              y.f![idx] *= alpha;
            }
          }
          final c = ins.length > 2 ? ins[2] : null;
          if (c != null) {
            final beta = attrs.getFloat('beta') ?? 1.0;
            y = ops.opAdd(
                y, beta == 1.0 ? c : ops.opMul(c, Tensor.scalarFloat(beta)));
          }
          outs = [y];
        } else if (convX != null &&
            convX.isFloat &&
            (convX.rank == 4 || convX.rank == 3)) {
          final attrsC = attrs;
          final oneD = convX.rank == 3;
          final x4 = oneD ? convX.reshape([...convX.shape, 1]) : convX;
          final w0 = _initializers[node.input[1]]!;
          final w = oneD ? w0.reshape([...w0.shape, 1]) : w0;
          final strides0 = attrsC.getInts('strides');
          final pads0 = attrsC.getInts('pads');
          final dil0 = attrsC.getInts('dilations');
          final strides = oneD ? [strides0?.first ?? 1, 1] : strides0;
          final padsAttr =
              oneD ? [pads0?[0] ?? 0, 0, pads0?[1] ?? 0, 0] : pads0;
          final dilations = oneD ? [dil0?.first ?? 1, 1] : dil0;
          final autoPad = attrsC.getString('auto_pad') ?? 'NOTSET';
          final outSp = nn.convOutputSpatial(x4.shape.sublist(2),
              w.shape.sublist(2), strides, padsAttr, dilations, autoPad);
          if (outSp[0] < pool!.workerCount * 2) {
            outs = _dispatch(node, ins, attrs); // too few rows to split
          } else {
            final out = await pool.conv(
              node.input[1],
              x4.f!,
              x4.shape,
              strides: strides,
              pads: padsAttr,
              dilations: dilations,
              group: attrsC.getInt('group') ?? 1,
              autoPad: autoPad,
              n: x4.shape[0],
              m: w.shape[0],
              oh: outSp[0],
              ow: outSp[1],
            );
            final y =
                Tensor.float(out, [x4.shape[0], w.shape[0], ...outSp]);
            outs = [
              oneD ? y.reshape([y.shape[0], y.shape[1], y.shape[2]]) : y
            ];
          }
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
      final attrs = _AttrMap(node.attribute, _ext);
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
        final shapes = [
          for (final t in ins) t == null ? 'null' : t.shape.toString()
        ].join(', ');
        throw StateError('ONNX graph execution failed at node "${node.name}" '
            '(op=${node.opType}, input shapes: $shapes): $e');
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
      values[t.name] = tensorFromProto(t, ext: _ext);
    }
    for (int k = 0; k < g.input.length && k < boundInputs.length; k++) {
      if (boundInputs[k] != null) values[g.input[k].name] = boundInputs[k]!;
    }
    _execNodes(g.node, values, profile);
    return [for (final o in g.output) values[o.name]!];
  }

  /// `Scan` — inputs `[state_1..state_M, scan_1..scan_N]`; body graph maps
  /// `[states, scan slices] -> [states, scan output slices]`. Scan inputs
  /// are sliced along their `scan_input_axes` (default 0), optionally
  /// reversed; scan outputs stack along `scan_output_axes` (default a new
  /// axis 0), optionally reversed.
  List<Tensor> _runScan(NodeProto node, List<Tensor?> ins, _AttrMap attrs,
      Map<String, Tensor> values, ExecutionProfile? profile) {
    final body = attrs.getGraph('body')!;
    final nScan = attrs.getInt('num_scan_inputs')!;
    final inAxesAttr = attrs.getInts('scan_input_axes');
    final inDirs = attrs.getInts('scan_input_directions');
    final outAxesAttr = attrs.getInts('scan_output_axes');
    final outDirs = attrs.getInts('scan_output_directions');

    final nState = node.input.length - nScan;
    var states = [for (int k = 0; k < nState; k++) ins[k]!];
    final scanIns = [for (int k = nState; k < ins.length; k++) ins[k]!];
    final inAxes = [
      for (int j = 0; j < nScan; j++)
        () {
          final a = inAxesAttr == null ? 0 : inAxesAttr[j];
          return a < 0 ? a + scanIns[j].rank : a;
        }()
    ];
    final iters = scanIns.isEmpty ? 0 : scanIns.first.shape[inAxes.first];
    final nScanOut = body.output.length - nState;
    final scans = List.generate(nScanOut, (_) => <Tensor>[]);

    Tensor sliceAt(Tensor t, int axis, int i) {
      if (axis == 0) {
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
      return ops.opSqueeze(
          ops.opSlice(t, [i], [i + 1], [axis], null), [axis]);
    }

    for (int i = 0; i < iters; i++) {
      final sliced = [
        for (int j = 0; j < nScan; j++)
          sliceAt(
              scanIns[j],
              inAxes[j],
              inDirs != null && inDirs[j] == 1 ? iters - 1 - i : i)
      ];
      final outs =
          _execSubgraph(body, values, [...states, ...sliced], profile);
      states = outs.sublist(0, nState);
      for (int k = 0; k < nScanOut; k++) {
        scans[k].add(outs[nState + k]);
      }
    }

    Tensor stackOut(int k) {
      var steps = scans[k];
      if (outDirs != null && outDirs[k] == 1) {
        steps = steps.reversed.toList();
      }
      final axis = outAxesAttr == null ? 0 : outAxesAttr[k];
      if (axis == 0) return _stack(steps);
      final rank = steps.first.rank + 1;
      final ax = axis < 0 ? axis + rank : axis;
      return ops.opConcat(
          [for (final s in steps) ops.opUnsqueeze(s, [ax])], ax);
    }

    return [
      ...states,
      for (int k = 0; k < nScanOut; k++) stackOut(k),
    ];
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
      case '_FusedRMSNorm':
        return [
          ops.opRMSNorm(need(0), need(1), attrs.getInt('axis')!,
              attrs.getFloat('epsilon')!)
        ];
      case 'GroupQueryAttention':
        return [
          ops.opGroupQueryAttention(need(0), need(1), need(2),
              numHeads: attrs.getInt('num_heads')!,
              kvNumHeads: attrs.getInt('kv_num_heads')!,
              scale: attrs.getFloat('scale'),
              doRotary: (attrs.getInt('do_rotary') ?? 0) != 0,
              rotaryInterleaved: (attrs.getInt('rotary_interleaved') ?? 0) != 0,
              cosCache: ins.length > 7 ? ins[7] : null,
              sinCache: ins.length > 8 ? ins[8] : null,
              attentionBias: ins.length > 10 ? ins[10] : null,
              softcap: attrs.getFloat('softcap') ?? 0,
              localWindow: attrs.getInt('local_window_size') ?? -1),
          // present_k / present_v (dummy — no KV-cache support).
          Tensor.scalarFloat(0),
          Tensor.scalarFloat(0),
        ];
      case 'MultiHeadAttention':
        return [
          ops.opMultiHeadAttention(need(0), need(1), need(2),
              ins.length > 5 ? ins[5] : null,
              numHeads: attrs.getInt('num_heads')!,
              scale: attrs.getFloat('scale'))
        ];
      case 'RotaryEmbedding':
        return [
          ops.opRotaryEmbedding(need(0), need(1), need(2), need(3),
              interleaved: (attrs.getInt('interleaved') ?? 0) != 0,
              numHeads: attrs.getInt('num_heads') ?? 0,
              rotaryEmbeddingDim: attrs.getInt('rotary_embedding_dim') ?? 0)
        ];
      case 'SimplifiedLayerNormalization':
        // com.microsoft RMSNorm: x * rsqrt(mean(x^2, axis) + eps) * scale.
        return [
          ops.opRMSNorm(need(0), need(1), attrs.getInt('axis') ?? -1,
              attrs.getFloat('epsilon') ?? 1e-5)
        ];
      case 'SkipSimplifiedLayerNormalization':
        // Fused residual add + RMSNorm: inputs (x, skip, scale[, bias]);
        // outputs (rmsnorm(x+skip)[, mean, inv_std, x+skip]).
        final sum = ops.opAdd(need(0), need(1));
        final normed = ops.opRMSNorm(sum, need(2), -1,
            attrs.getFloat('epsilon') ?? 1e-5);
        final withBias = ins.length > 3 && ins[3] != null
            ? ops.opAdd(normed, ins[3]!)
            : normed;
        return [
          withBias,
          if (node.output.length > 1) Tensor.scalarFloat(0),
          if (node.output.length > 2) Tensor.scalarFloat(0),
          if (node.output.length > 3) sum,
        ];
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
        // Opset 11+ takes min/max as inputs; opset 6-10 as attributes.
        final minAttr = attrs.getFloat('min'), maxAttr = attrs.getFloat('max');
        return [
          ops.opClip(
              need(0),
              ins.length > 1 && ins[1] != null
                  ? ins[1]
                  : (minAttr != null ? Tensor.scalarFloat(minAttr) : null),
              ins.length > 2 && ins[2] != null
                  ? ins[2]
                  : (maxAttr != null ? Tensor.scalarFloat(maxAttr) : null))
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
        // perm is optional: default reverses all axes.
        return [
          ops.opTranspose(
              need(0),
              attrs.getInts('perm') ??
                  List.generate(need(0).rank, (k) => need(0).rank - 1 - k))
        ];
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
      case 'LogSoftmax':
        return [ops.opLogSoftmax(need(0), attrs.getInt('axis') ?? -1)];
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
        return [
          ops.opEinsum(attrs.getString('equation')!,
              [for (final t in ins) t!])
        ];
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
      case 'IsNaN':
        return [ops.opIsNaN(need(0))];
      case 'IsInf':
        return [
          ops.opIsInf(need(0),
              detectPositive: (attrs.getInt('detect_positive') ?? 1) != 0,
              detectNegative: (attrs.getInt('detect_negative') ?? 1) != 0)
        ];
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
      case 'ReduceSumSquare':
        return [
          ops.opReduceSumSquare(
            need(0),
            ins.length > 1 && ins[1] != null
                ? ins[1]!.asIntList()
                : attrs.getInts('axes'),
            (attrs.getInt('keepdims') ?? 1) != 0,
          )
        ];
      case 'Split':
        // Opset 13+ takes split sizes as an input; opset 11 as an attribute.
        return ops.opSplit(
          need(0),
          attrs.getInt('axis') ?? 0,
          node.output.length,
          ins.length > 1 && ins[1] != null
              ? ins[1]!.asIntList().toList()
              : attrs.getInts('split'),
        );
      case 'STFT':
        return [
          ops.opSTFT(
            need(0),
            need(1).getI(0),
            ins.length > 2 ? ins[2] : null,
            ins.length > 3 && ins[3] != null ? ins[3]!.getI(0) : null,
            onesided: (attrs.getInt('onesided') ?? 1) != 0,
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
      case 'GroupNormalization':
        return [
          nn.opGroupNormalization(need(0), need(1), need(2),
              attrs.getInt('num_groups')!, attrs.getFloat('epsilon') ?? 1e-5)
        ];
      case 'GridSample':
        return [
          nn.opGridSample(need(0), need(1),
              mode: attrs.getString('mode') ?? 'linear',
              paddingMode: attrs.getString('padding_mode') ?? 'zeros',
              alignCorners: (attrs.getInt('align_corners') ?? 0) != 0)
        ];
      case 'RoiAlign':
        return [
          nn.opRoiAlign(need(0), need(1), need(2),
              outH: attrs.getInt('output_height') ?? 1,
              outW: attrs.getInt('output_width') ?? 1,
              spatialScale: attrs.getFloat('spatial_scale') ?? 1.0,
              samplingRatio: attrs.getInt('sampling_ratio') ?? 0,
              isMax: attrs.getString('mode') == 'max',
              // Opset < 16 has no such attribute and behaves as
              // output_half_pixel (no -0.5 shift).
              halfPixel: (attrs.getString('coordinate_transformation_mode') ??
                      (_opset >= 16 ? 'half_pixel' : 'output_half_pixel')) ==
                  'half_pixel')
        ];
      case 'ArgMax':
      case 'ArgMin':
        return [
          ops.opArgMinMax(need(0), attrs.getInt('axis') ?? 0,
              (attrs.getInt('keepdims') ?? 1) != 0,
              isMax: node.opType == 'ArgMax',
              selectLastIndex: (attrs.getInt('select_last_index') ?? 0) != 0)
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
      case 'Upsample':
        // Deprecated pre-Resize op (opset <= 9): asymmetric coordinates,
        // floor rounding — exactly Resize-10's semantics.
        return [
          nn.opResize(
            need(0),
            scales: ins.length > 1 && ins[1] != null
                ? ins[1]!.asFloatList().toList()
                : attrs.getFloats('scales'),
            mode: attrs.getString('mode') ?? 'nearest',
            coordMode: 'asymmetric',
            nearestMode: 'floor',
          )
        ];
      case 'Dropout':
        // Inference: identity; the optional mask output is all-true.
        return [
          need(0),
          if (node.output.length > 1 && node.output[1].isNotEmpty)
            Tensor.int64(Int64List(need(0).length)..fillRange(0, need(0).length, 1),
                need(0).shape),
        ];
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
      case 'Sign':
        return [ops.opSign(need(0))];
      case 'Atan':
        return [ops.opAtan(need(0))];
      case 'Floor':
        return [ops.opFloor(need(0))];
      case 'Ceil':
        return [ops.opCeil(need(0))];
      case 'Round':
        return [ops.opRound(need(0))];
      case 'Gelu':
        return [
          ops.opGelu(need(0),
              tanhApprox: attrs.getString('approximate') == 'tanh')
        ];
      case 'PRelu':
        return [ops.opPRelu(need(0), need(1))];
      case 'Size':
        return [ops.opSize(need(0))];
      case 'RandomNormalLike':
        return [
          ops.opRandomNormalFill(need(0).shape, attrs.getFloat('mean') ?? 0.0)
        ];
      case 'RandomNormal':
        return [
          ops.opRandomNormalFill(
              attrs.getInts('shape')!, attrs.getFloat('mean') ?? 0.0)
        ];
      case 'Tile':
        return [ops.opTile(need(0), need(1).asIntList().toList())];
      case 'NonZero':
        return [ops.opNonZero(need(0))];
      case 'TopK':
        return ops.opTopK(need(0), need(1).getI(0),
            axis: attrs.getInt('axis') ?? -1,
            largest: (attrs.getInt('largest') ?? 1) != 0);
      case 'NonMaxSuppression':
        return [
          ops.opNonMaxSuppression(
            need(0),
            need(1),
            maxOutputBoxesPerClass:
                ins.length > 2 && ins[2] != null ? ins[2]!.getI(0) : 0,
            iouThreshold:
                ins.length > 3 && ins[3] != null ? ins[3]!.getD(0) : 0,
            scoreThreshold:
                ins.length > 4 && ins[4] != null ? ins[4]!.getD(0) : null,
            centerPointBox: (attrs.getInt('center_point_box') ?? 0) != 0,
          )
        ];
      case 'Trilu':
        return [
          ops.opTrilu(need(0),
              upper: (attrs.getInt('upper') ?? 1) != 0,
              k: ins.length > 1 && ins[1] != null ? ins[1]!.getI(0) : 0)
        ];
      case 'ScatterND':
        if ((attrs.getString('reduction') ?? 'none') != 'none') {
          throw UnsupportedError('ScatterND: only reduction=none supported');
        }
        return [ops.opScatterND(need(0), need(1), need(2))];
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
      case 'MatMulNBits':
        if (node.input.length > 4) {
          throw UnsupportedError(
              'MatMulNBits: g_idx/bias inputs not supported');
        }
        return [
          ql.opMatMulNBits(need(0), need(1), need(2),
              ins.length > 3 ? ins[3] : null,
              k: attrs.getInt('K')!,
              n: attrs.getInt('N')!,
              bits: attrs.getInt('bits') ?? 4,
              blockSize: attrs.getInt('block_size')!)
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
  final ExternalDataResolver? _ext;
  _AttrMap(Iterable<AttributeProto> attrs, [this._ext])
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
  List<double>? getFloats(String name) => _byName.containsKey(name)
      ? _byName[name]!.floats.toList()
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
      _byName.containsKey(name) ? tensorFromProto(_byName[name]!.t, ext: _ext) : null;

  /// The value produced by a `Constant` node — either a `value` tensor or one
  /// of the scalar/list attribute forms the op allows.
  Tensor constantTensor() {
    if (_byName.containsKey('value')) {
      return tensorFromProto(_byName['value']!.t, ext: _ext);
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
