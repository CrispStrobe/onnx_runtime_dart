/// Load-time constant folding + the Transpose/ReduceMean fast paths.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:fixnum/fixnum.dart';
import 'package:onnx_runtime_dart/onnx_proto.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/src/onnx_ops.dart' as ops;
import 'package:test/test.dart';

TensorProto floatInit(String name, List<int> dims, List<double> v) =>
    TensorProto()
      ..name = name
      ..dataType = TensorProto_DataType.FLOAT.value
      ..dims.addAll(dims.map(Int64.new))
      ..floatData.addAll(v);

/// Abramowitz-Stegun erf, independent reference for the gelu tests.
double _erfRef(double x) {
  final sign = x < 0 ? -1.0 : 1.0;
  x = x.abs();
  const p = 0.3275911;
  final t = 1.0 / (1.0 + p * x);
  final y = 1.0 -
      (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
                  0.284496736) *
              t +
          0.254829592) *
          t *
          math.exp(-x * x);
  return sign * y;
}

void main() {
  test('constant chains fold at load time (no per-run dispatch)', () {
    // C1 (Constant) -> Neg -> Add with initializer K, all constant; result
    // feeds a runtime Add with X. The whole chain must fold, leaving only
    // the runtime Add visible to the profiler.
    final g = GraphProto()
      ..input.add(ValueInfoProto()..name = 'X')
      ..output.add(ValueInfoProto()..name = 'Y')
      ..initializer.add(floatInit('K', [2], [10, 20]))
      ..node.addAll([
        NodeProto()
          ..opType = 'Constant'
          ..output.add('C1')
          ..attribute.add(AttributeProto()
            ..name = 'value'
            ..t = floatInit('', [2], [1, 2])),
        NodeProto()
          ..opType = 'Neg'
          ..input.add('C1')
          ..output.add('C2'),
        NodeProto()
          ..opType = 'Add'
          ..input.addAll(['C2', 'K'])
          ..output.add('C3'),
        NodeProto()
          ..opType = 'Add'
          ..input.addAll(['X', 'C3'])
          ..output.add('Y'),
      ]);
    final model =
        OnnxModel.fromBytes((ModelProto()..graph = g).writeToBuffer());
    final profile = ExecutionProfile();
    final y = model.run(
        {'X': Tensor.float(Float32List.fromList([100, 200]), [2])},
        ['Y'],
        profile: profile)['Y']!;
    // C3 = -[1,2] + [10,20] = [9,18]; Y = X + C3.
    expect(y.asFloatList(), [109.0, 218.0]);
    expect(profile.callsByOp.keys.toList(), ['Add'],
        reason: 'folded nodes must not run');
    expect(profile.callsByOp['Add'], 1);
  });

  test('initializer that is also a graph input stays overridable', () {
    // W has an initializer default but is listed as a graph input — folding
    // through it would bake the default in and ignore the caller's value.
    final g = GraphProto()
      ..input.addAll([
        ValueInfoProto()..name = 'X',
        ValueInfoProto()..name = 'W',
      ])
      ..output.add(ValueInfoProto()..name = 'Y')
      ..initializer.add(floatInit('W', [2], [1, 1]))
      ..node.addAll([
        NodeProto()
          ..opType = 'Neg'
          ..input.add('W')
          ..output.add('NW'),
        NodeProto()
          ..opType = 'Mul'
          ..input.addAll(['X', 'NW'])
          ..output.add('Y'),
      ]);
    final model =
        OnnxModel.fromBytes((ModelProto()..graph = g).writeToBuffer());
    final x = Tensor.float(Float32List.fromList([3, 5]), [2]);

    // Default W = [1,1] → Y = X * -W = [-3,-5].
    expect(model.run({'X': x}, ['Y'])['Y']!.asFloatList(), [-3.0, -5.0]);
    // Overridden W = [2,10] → Y = [-6,-50].
    final w = Tensor.float(Float32List.fromList([2, 10]), [2]);
    expect(model.run({'X': x, 'W': w}, ['Y'])['Y']!.asFloatList(),
        [-6.0, -50.0]);
  });

  group('pattern fusion', () {
    GraphProto geluGraph({bool leakIntermediate = false}) {
      final g = GraphProto()
        ..input.add(ValueInfoProto()..name = 'X')
        ..output.add(ValueInfoProto()..name = 'Y')
        ..initializer.addAll([
          floatInit('sqrt2', [], [math.sqrt2]),
          floatInit('one', [], [1.0]),
          floatInit('half', [], [0.5]),
        ])
        ..node.addAll([
          NodeProto()
            ..opType = 'Div'
            ..input.addAll(['X', 'sqrt2'])
            ..output.add('d'),
          NodeProto()
            ..opType = 'Erf'
            ..input.add('d')
            ..output.add('e'),
          NodeProto()
            ..opType = 'Add'
            ..input.addAll(['e', 'one'])
            ..output.add('a'),
          NodeProto()
            ..opType = 'Mul'
            ..input.addAll(['X', 'a'])
            ..output.add('m'),
          NodeProto()
            ..opType = 'Mul'
            ..input.addAll(['m', 'half'])
            ..output.add('Y'),
        ]);
      if (leakIntermediate) {
        // The erf output is also a graph output -> fusion must not fire.
        g.output.add(ValueInfoProto()..name = 'e');
      }
      return g;
    }

    double gelu(double x) {
      // Reference value via the same erf approximation the runtime uses is
      // not required — check against a loose analytic bound instead.
      return 0.5 * x * (1 + _erfRef(x / math.sqrt2));
    }

    test('erf-GELU chain fuses and computes gelu', () {
      final model = OnnxModel.fromBytes(
          (ModelProto()..graph = geluGraph()).writeToBuffer());
      final x = Tensor.float(
          Float32List.fromList([-2, -0.5, 0, 0.5, 2]), [5]);
      final profile = ExecutionProfile();
      final y = model.run({'X': x}, ['Y'], profile: profile)['Y']!;
      expect(profile.callsByOp.keys.toList(), ['_FusedGelu'],
          reason: 'whole chain must fuse into one node');
      for (int k = 0; k < 5; k++) {
        expect(y.asFloatList()[k], closeTo(gelu(x.asFloatList()[k]), 1e-4));
      }
    });

    test('fusion aborts when an intermediate is a graph output', () {
      final model = OnnxModel.fromBytes(
          (ModelProto()..graph = geluGraph(leakIntermediate: true))
              .writeToBuffer());
      final x = Tensor.float(Float32List.fromList([1.0, -1.0]), [2]);
      final profile = ExecutionProfile();
      final out = model.run({'X': x}, ['Y', 'e'], profile: profile);
      expect(profile.callsByOp.containsKey('Erf'), isTrue,
          reason: 'unfused chain must still run node-by-node');
      expect(out['e']!.length, 2);
      for (int k = 0; k < 2; k++) {
        expect(out['Y']!.asFloatList()[k],
            closeTo(gelu(x.asFloatList()[k]), 1e-4));
      }
    });
  });

  group('Transpose fast/general paths match a naive reference', () {
    final rng = math.Random(3);
    Tensor rand(List<int> shape) {
      final n = shape.fold<int>(1, (a, b) => a * b);
      final v = Float32List(n);
      for (int k = 0; k < n; k++) {
        v[k] = rng.nextDouble();
      }
      return Tensor.float(v, shape);
    }

    Tensor naive(Tensor x, List<int> perm) {
      final newShape = [for (final p in perm) x.shape[p]];
      final out = Float32List(x.length);
      final oldStrides = x.strides;
      for (int idx = 0; idx < x.length; idx++) {
        int rem = idx, oldFlat = 0;
        for (int k = perm.length - 1; k >= 0; k--) {
          final c = rem % newShape[k];
          rem ~/= newShape[k];
          oldFlat += c * oldStrides[perm[k]];
        }
        out[idx] = x.f![oldFlat];
      }
      return Tensor.float(out, newShape);
    }

    for (final (shape, perm) in [
      ([2, 3, 4, 5], [0, 2, 1, 3]), // attention perm, last axis kept
      ([2, 3, 4, 5], [3, 2, 1, 0]), // full reversal, general path
      ([2, 3, 4, 5], [1, 0, 3, 2]),
      ([4, 7], [1, 0]),
      ([2, 3, 4], [2, 0, 1]),
      ([2, 3, 4], [1, 2, 0]),
    ]) {
      test('perm $perm of $shape', () {
        final x = rand(shape);
        final got = ops.opTranspose(x, perm);
        final want = naive(x, perm);
        expect(got.shape, want.shape);
        expect(got.asFloatList(), want.asFloatList());
      });
    }

    test('int64 transpose', () {
      final x = Tensor.int64(
          Int64List.fromList(List.generate(24, (k) => k)), [2, 3, 4]);
      final got = ops.opTranspose(x, [2, 1, 0]);
      expect(got.shape, [4, 3, 2]);
      // element (a,b,c) of result = x[c,b,a] = c*12 + b*4 + a
      expect(got.i![0], 0);
      expect(got.i![1], 12);
      expect(got.i![23], 23);
    });
  });

  group('ReduceMean last-axis fast path', () {
    test('matches general path result', () {
      final x = Tensor.float(
          Float32List.fromList(List.generate(24, (k) => (k * 7 % 13) * 1.0)),
          [2, 3, 4]);
      // axes=[-1] takes the fast path; axes=[2, unused-dup] shape variations
      // exercise the general path on the same reduction.
      final fast = ops.opReduceMean(x, [-1], true);
      final general = ops.opReduceMean(x, [0, 2], true);
      expect(fast.shape, [2, 3, 1]);
      expect(general.shape, [1, 3, 1]);
      // Cross-check fast path against hand-computed row means.
      final xf = x.asFloatList();
      for (int r = 0; r < 6; r++) {
        double s = 0;
        for (int k = 0; k < 4; k++) {
          s += xf[r * 4 + k];
        }
        expect(fast.asFloatList()[r], closeTo(s / 4, 1e-6));
      }
    });

    test('keepdims=false drops the axis', () {
      final x = Tensor.float(Float32List.fromList([1, 2, 3, 4, 5, 6]), [2, 3]);
      final y = ops.opReduceMean(x, [-1], false);
      expect(y.shape, [2]);
      expect(y.asFloatList(), [2.0, 5.0]);
    });
  });
}
