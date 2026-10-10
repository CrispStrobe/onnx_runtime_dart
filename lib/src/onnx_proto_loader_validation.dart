/// Load-time resource caps for untrusted `.onnx` input: fold outputs may
/// materialize at most [kMaxFoldBytes] bytes, fold-time `Loop`/`Scan` bodies
/// at most [kMaxFoldIters] cumulative body iterations, declared tensor shapes
/// at most [kMaxTensorBytes] bytes. [rejectUnsafeFolds] runs before any fold
/// executes and throws FormatException. Projections that inline proto data
/// cannot resolve fail open and leave load behavior unchanged. Verified by
/// `test/onnx_proto_loader_validation_test.dart`.
library;

import 'dart:typed_data';

import 'onnx.pb.dart';

const int kMaxFoldBytes = 1 << 40;

const int kMaxFoldIters = 1 << 30;

const int kMaxTensorBytes = 1 << 40;

const int _kFloat = 1;
const int _kUint8 = 2;
const int _kInt8 = 3;
const int _kInt32 = 6;
const int _kInt64 = 7;
const int _kBool = 9;
const int _kFloat16 = 10;
const int _kDouble = 11;
const int _kUint4 = 21;
const int _kInt4 = 22;

/// Byte width of one element for dtypes this runtime decodes, null otherwise.
/// Four-bit dtypes count as one byte per element, conservative by a factor 2.
int? bytesPerElement(int dataType) {
  switch (dataType) {
    case _kFloat:
    case _kInt32:
      return 4;
    case _kInt64:
      return 8;
    case _kUint8:
    case _kInt8:
    case _kBool:
      return 1;
    case _kFloat16:
      return 2;
    case _kDouble:
      return 8;
    case _kUint4:
    case _kInt4:
      return 1;
    default:
      return null;
  }
}

/// Shape and inline integer metadata of one value, read without materializing
/// tensor bytes. Null dims mean unknown, which makes byte caps fail open.
class _ProjectionMeta {
  final List<int>? dims;
  final int? dataType;
  final List<int>? intValues;
  final int? scalarInt;

  const _ProjectionMeta(this.dims, this.dataType,
      {this.intValues, this.scalarInt});

  static const _ProjectionMeta unknown = _ProjectionMeta(null, null);
}

_ProjectionMeta _initializerMeta(TensorProto t) {
  final dims = t.dims.map((d) => d.toInt()).toList();
  List<int>? ints;
  if (t.int64Data.isNotEmpty) {
    ints = t.int64Data.map((v) => v.toInt()).toList();
  } else if (t.dataType == _kInt64 && t.rawData.isNotEmpty) {
    final full = t.rawData.length <= 1 << 16 && t.rawData.length % 8 == 0;
    final b = ByteData.sublistView(Uint8List.fromList(t.rawData));
    if (full) {
      ints = [
        for (var i = 0; i < t.rawData.length ~/ 8; i++)
          b.getInt64(i * 8, Endian.little),
      ];
    } else if (t.rawData.length >= 8) {
      ints = [b.getInt64(0, Endian.little)];
    }
  }
  int? scalar;
  if (ints != null && ints.length == 1) scalar = ints.first;
  return _ProjectionMeta(dims, t.dataType, intValues: ints, scalarInt: scalar);
}

/// Parent-chained name table mirroring ONNX subgraph scoping, where a body
/// may reference outer constants.
class _Scope {
  final Map<String, _ProjectionMeta> values;
  final _Scope? parent;

  _Scope(this.values, this.parent);

  _ProjectionMeta? lookup(String name) => values[name] ?? parent?.lookup(name);
}

/// Throws FormatException when foldable nodes breach the caps of this library.
void rejectUnsafeFolds(ModelProto model) {
  final overridable = {for (final vi in model.graph.input) vi.name};
  final values = <String, _ProjectionMeta>{
    for (final t in model.graph.initializer)
      if (!overridable.contains(t.name)) t.name: _initializerMeta(t),
  };
  _walk(model.graph, _Scope(values, null), 0);
}

bool _allResolved(List<_ProjectionMeta?> metas, List<String> names) {
  for (var i = 0; i < names.length; i++) {
    if (names[i].isEmpty) continue;
    if (metas[i] == null) return false;
  }
  return true;
}

GraphProto? _bodyOf(NodeProto node) {
  for (final a in node.attribute) {
    if (a.name == 'body' && a.hasG()) return a.g;
  }
  return null;
}

_Scope _childScope(GraphProto body, _Scope parent) => _Scope(
      {for (final vi in body.input) vi.name: _ProjectionMeta.unknown},
      parent,
    );

void _registerOutputs(NodeProto node, _Scope scope, _ProjectionMeta meta) {
  for (final o in node.output) {
    if (o.isNotEmpty) scope.values[o] = meta;
  }
}

void _checkByteCap(List<int> dims, int width, String what) {
  var acc = 1;
  final allowed = kMaxFoldBytes ~/ width;
  for (final d in dims) {
    if (d <= 0) return; // invalid extents reject in loader or dispatch
    if (acc > allowed ~/ d) {
      throw FormatException('$what exceeds the fold cap of $kMaxFoldBytes bytes');
    }
    acc *= d;
  }
}

/// Mirrors `_foldConstants` node order and constant visibility, and returns
/// the cumulative foldable `Loop`/`Scan` body iterations. Throws
/// FormatException on any cap breach.
int _walk(GraphProto g, _Scope scope, int depth) {
  if (depth > 64) {
    throw FormatException('fold-time subgraph nesting exceeds 64 levels');
  }
  var iterations = 0;
  for (final node in g.node) {
    if (node.output.every((o) => o.isEmpty)) continue; // nothing folded away
    final names = node.input;
    final metas = <_ProjectionMeta?>[
      for (final name in names) name.isEmpty ? null : scope.lookup(name),
    ];
    final foldable = _allResolved(metas, names);

    switch (node.opType) {
      case 'Loop':
        if (foldable && metas.isNotEmpty && metas[0]?.scalarInt != null) {
          final trips = metas[0]!.scalarInt!;
          if (trips > kMaxFoldIters) {
            throw FormatException(
                'fold-time Loop requests $trips iterations, cap $kMaxFoldIters');
          }
          var bodyWork = 0;
          final body = _bodyOf(node);
          if (body != null) {
            bodyWork = _walk(body, _childScope(body, scope), depth + 1);
          }
          final total = trips * (1 + bodyWork);
          if (total > kMaxFoldIters) {
            throw FormatException(
                'fold-time Loop executes $total iterations, cap $kMaxFoldIters');
          }
          iterations += total;
          if (iterations > kMaxFoldIters) {
            throw FormatException(
                'fold-time iteration total $iterations exceeds cap $kMaxFoldIters');
          }
        }
        _registerOutputs(node, scope, _ProjectionMeta.unknown);

      case 'Scan':
        if (foldable) {
          var numScan = 1;
          for (final a in node.attribute) {
            if (a.name == 'num_scan_inputs' && a.hasI()) numScan = a.i.toInt();
          }
          if (numScan >= 1 && names.length >= numScan) {
            var seqLen = -1;
            var widthsKnown = true;
            for (var i = names.length - numScan; i < names.length; i++) {
              final dims = metas[i]?.dims;
              if (dims == null || dims.isEmpty) {
                widthsKnown = false;
                break;
              }
              if (dims.first > seqLen) seqLen = dims.first;
            }
            if (widthsKnown && seqLen >= 0) {
              if (seqLen > kMaxFoldIters) {
                throw FormatException(
                    'fold-time Scan runs $seqLen iterations, cap $kMaxFoldIters');
              }
              var bodyWork = 0;
              final body = _bodyOf(node);
              if (body != null) {
                bodyWork = _walk(body, _childScope(body, scope), depth + 1);
              }
              final total = seqLen * (1 + bodyWork);
              if (total > kMaxFoldIters) {
                throw FormatException(
                    'fold-time Scan executes $total iterations, cap $kMaxFoldIters');
              }
              iterations += total;
              if (iterations > kMaxFoldIters) {
                throw FormatException(
                    'fold-time iteration total $iterations exceeds cap $kMaxFoldIters');
              }
            }
          }
        }
        _registerOutputs(node, scope, _ProjectionMeta.unknown);

      case 'Range':
        if (foldable && metas.length >= 3) {
          final start = metas[0]?.scalarInt;
          final limit = metas[1]?.scalarInt;
          final delta = metas[2]?.scalarInt;
          final width = bytesPerElement(metas[0]?.dataType ?? -1);
          if (start != null && limit != null && delta != null && width != null) {
            var span = -1;
            var step = 0;
            if (delta > 0 && start >= 0 && limit >= start) {
              span = limit - start;
              step = delta;
            } else if (delta < 0 && limit >= 0 && start >= limit) {
              span = start - limit;
              step = -delta;
            }
            if (span >= 0) {
              _checkByteCap([span], width, 'fold-time Range');
              final q = span ~/ step;
              final elems = span % step == 0 ? q : q + 1;
              _registerOutputs(node, scope, _ProjectionMeta([elems], metas[0]!.dataType));
            } else {
              _registerOutputs(node, scope, _ProjectionMeta.unknown);
            }
          } else {
            _registerOutputs(node, scope, _ProjectionMeta.unknown);
          }
        } else {
          _registerOutputs(node, scope, _ProjectionMeta.unknown);
        }

      case 'ConstantOfShape':
        if (foldable && metas.isNotEmpty) {
          final sVals = metas[0]?.intValues;
          if (sVals != null) {
            var dtype = _kFloat; // the ONNX default value is 0.0f
            for (final a in node.attribute) {
              if (a.name == 'value' && a.hasT()) dtype = a.t.dataType;
            }
            final width = bytesPerElement(dtype);
            if (width != null) {
              _checkByteCap(sVals, width, 'fold-time ConstantOfShape');
              _registerOutputs(node, scope, _ProjectionMeta(sVals, dtype));
            } else {
              _registerOutputs(node, scope, _ProjectionMeta.unknown);
            }
          } else {
            _registerOutputs(node, scope, _ProjectionMeta.unknown);
          }
        } else {
          _registerOutputs(node, scope, _ProjectionMeta.unknown);
        }

      case 'Tile':
        if (foldable && metas.length >= 2) {
          final data = metas[0];
          final reps = metas[1];
          final width = bytesPerElement(data?.dataType ?? -1);
          if (data?.dims != null && reps?.intValues != null && width != null) {
            final dims = data!.dims!;
            final repVals = reps!.intValues!;
            final allowed = kMaxFoldBytes ~/ width;
            var valid = true;
            for (final d in dims) {
              if (d <= 0) {
                valid = false;
                break;
              }
            }
            if (valid) {
              _checkByteCap(dims, width, 'fold-time Tile input');
              final outDims = <int>[];
              var outElems = 1;
              var fail = false;
              for (var i = 0; i < dims.length; i++) {
                final r = i < repVals.length ? repVals[i] : 1;
                if (r <= 0) {
                  fail = true;
                  break;
                }
                final dim = dims[i];
                if (dim > allowed ~/ r) {
                  throw FormatException(
                      'fold-time Tile output exceeds the fold cap of $kMaxFoldBytes bytes');
                }
                final pd = dim * r;
                if (outElems > allowed ~/ pd) {
                  throw FormatException(
                      'fold-time Tile output exceeds the fold cap of $kMaxFoldBytes bytes');
                }
                outElems *= pd;
                outDims.add(pd);
              }
              _registerOutputs(node, scope,
                  fail ? _ProjectionMeta.unknown : _ProjectionMeta(outDims, data.dataType));
            } else {
              _registerOutputs(node, scope, _ProjectionMeta.unknown);
            }
          } else {
            _registerOutputs(node, scope, _ProjectionMeta.unknown);
          }
        } else {
          _registerOutputs(node, scope, _ProjectionMeta.unknown);
        }

      case 'Expand':
        if (foldable && metas.length >= 2) {
          final x = metas[0];
          final shape = metas[1];
          final width = bytesPerElement(x?.dataType ?? -1);
          if (x?.dims != null && shape?.intValues != null && width != null) {
            final inDims = x!.dims!;
            final sVals = shape!.intValues!;
            final allowed = kMaxFoldBytes ~/ width;
            final rank =
                inDims.length > sVals.length ? inDims.length : sVals.length;
            final outDims = <int>[];
            var outElems = 1;
            var fail = false;
            for (var k = 0; k < rank; k++) {
              // ONNX broadcast right-aligns both shapes.
              final ai = inDims.length - rank + k;
              final si = sVals.length - rank + k;
              final a = ai >= 0 ? inDims[ai] : 1;
              final b = si >= 0 ? sVals[si] : 1;
              final m = a > b ? a : b;
              if (m <= 0) {
                fail = true;
                break;
              }
              if (outElems > allowed ~/ m) {
                throw FormatException(
                    'fold-time Expand output exceeds the fold cap of $kMaxFoldBytes bytes');
              }
              outElems *= m;
              outDims.add(m);
            }
            _registerOutputs(node, scope,
                fail ? _ProjectionMeta.unknown : _ProjectionMeta(outDims, x.dataType));
          } else {
            _registerOutputs(node, scope, _ProjectionMeta.unknown);
          }
        } else {
          _registerOutputs(node, scope, _ProjectionMeta.unknown);
        }

      case 'Concat':
        if (foldable) {
          var total = 0;
          for (final m in metas) {
            final width = bytesPerElement(m?.dataType ?? -1);
            if (m == null || m.dims == null || width == null) {
              break;
            }
            var valid = true;
            for (final d in m.dims!) {
              if (d <= 0) {
                valid = false;
                break;
              }
            }
            if (!valid) {
              break;
            }
            _checkByteCap(m.dims!, width, 'fold-time Concat input');
            var elems = 1;
            for (final d in m.dims!) {
              elems *= d; // safe: _checkByteCap passed, product <= cap
            }
            final bytes = elems * width;
            if (total > kMaxFoldBytes - bytes) {
              throw FormatException(
                  'fold-time Concat output exceeds the fold cap of $kMaxFoldBytes bytes');
            }
            total += bytes;
          }
        }
        _registerOutputs(node, scope, _ProjectionMeta.unknown);

      default:
        _registerOutputs(node, scope, _ProjectionMeta.unknown);
    }
  }
  return iterations;
}
