/// Implementations of the standard ONNX operators (per the public ONNX
/// operator specification, https://onnx.ai/onnx/operators/) used by
/// transformer embedding / reranking graphs. Mechanical execution only — each
/// op is run the same way any ONNX runtime would, with no knowledge of why the
/// graph is shaped the way it is.
library;

import 'dart:math' as math;
import 'dart:typed_data';

// SIMD GEMM kernel on native targets (VM/AOT, where Float32x4 maps to real
// SSE/NEON); portable scalar kernel on web targets where it's emulated.
import 'gemm_kernel_scalar.dart' if (dart.library.ffi) 'gemm_kernel_simd.dart'
    as gemm;
import 'onnx_proto_loader.dart' show float32ToHalfBits, halfToFloat32Bits;
import 'tensor.dart';

// ---------------------------------------------------------------------------
// Broadcasting helpers (numpy-style, as used throughout the ONNX spec)
// ---------------------------------------------------------------------------

List<int> _broadcastShape(List<int> a, List<int> b) {
  final rank = math.max(a.length, b.length);
  final ap = List<int>.filled(rank - a.length, 1, growable: true)..addAll(a);
  final bp = List<int>.filled(rank - b.length, 1, growable: true)..addAll(b);
  final out = List<int>.filled(rank, 1);
  for (int k = 0; k < rank; k++) {
    final ad = ap[k], bd = bp[k];
    if (ad == bd) {
      out[k] = ad;
    } else if (ad == 1) {
      out[k] = bd;
    } else if (bd == 1) {
      out[k] = ad;
    } else {
      throw ArgumentError('Cannot broadcast shapes $a and $b');
    }
  }
  return out;
}

List<int> _unflatten(int flat, List<int> shape) {
  final coords = List<int>.filled(shape.length, 0);
  int rem = flat;
  for (int k = shape.length - 1; k >= 0; k--) {
    final d = shape[k] == 0 ? 1 : shape[k];
    coords[k] = rem % d;
    rem ~/= d;
  }
  return coords;
}

int _flattenBroadcast(List<int> outCoords, List<int> srcShape) {
  final rank = outCoords.length;
  final srcRank = srcShape.length;
  int flat = 0;
  int stride = 1;
  for (int k = srcRank - 1; k >= 0; k--) {
    final outK = k + (rank - srcRank);
    final dim = srcShape[k];
    final coord = dim == 1 ? 0 : outCoords[outK];
    flat += coord * stride;
    stride *= dim;
  }
  return flat;
}

bool _shapeEq(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (int k = 0; k < a.length; k++) {
    if (a[k] != b[k]) return false;
  }
  return true;
}

enum _Arith { add, sub, mul, div }

/// True if [small] (leading 1-dims ignored) equals the trailing dims of [big]
/// — i.e. broadcasting [small] against [big] just tiles it along the leading
/// axes, so no per-element coordinate math is needed.
bool _isTrailingSuffix(List<int> small, List<int> big) {
  int s = 0;
  while (s < small.length && small[s] == 1) {
    s++;
  }
  final n = small.length - s;
  if (n > big.length) return false;
  for (int k = 0; k < n; k++) {
    if (small[small.length - 1 - k] != big[big.length - 1 - k]) return false;
  }
  return true;
}

/// Monomorphic float32 paths for Add/Sub/Mul/Div — the hottest elementwise
/// ops. Direct Float32List indexing (no per-element closure or dtype branch)
/// for the three layouts that cover essentially all transformer traffic:
/// same-shape, scalar, and suffix-tile (bias adds like `[B,T,N] + [N]`).
/// Returns null when the layout (or dtype) needs the general path.
Tensor? _arithFloatFast(Tensor a, Tensor b, _Arith op) {
  final af = a.f, bf = b.f;
  if (af == null || bf == null) return null;
  final an = a.length, bn = b.length;

  if (_shapeEq(a.shape, b.shape)) {
    final out = Float32List(an);
    switch (op) {
      case _Arith.add:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] + bf[k];
        }
      case _Arith.sub:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] - bf[k];
        }
      case _Arith.mul:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] * bf[k];
        }
      case _Arith.div:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] / bf[k];
        }
    }
    return Tensor.float(out, a.shape);
  }

  if (bn == 1 && a.rank >= b.rank) {
    final s = bf[0];
    final out = Float32List(an);
    switch (op) {
      case _Arith.add:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] + s;
        }
      case _Arith.sub:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] - s;
        }
      case _Arith.mul:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] * s;
        }
      case _Arith.div:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] / s;
        }
    }
    return Tensor.float(out, a.shape);
  }
  if (an == 1 && b.rank >= a.rank) {
    final s = af[0];
    final out = Float32List(bn);
    switch (op) {
      case _Arith.add:
        for (int k = 0; k < bn; k++) {
          out[k] = s + bf[k];
        }
      case _Arith.sub:
        for (int k = 0; k < bn; k++) {
          out[k] = s - bf[k];
        }
      case _Arith.mul:
        for (int k = 0; k < bn; k++) {
          out[k] = s * bf[k];
        }
      case _Arith.div:
        for (int k = 0; k < bn; k++) {
          out[k] = s / bf[k];
        }
    }
    return Tensor.float(out, b.shape);
  }

  // Per-row scalar: b matches a except the last axis is 1 (LayerNorm's
  // `x - mean` / `x / std` pattern, [B,T,D] op [B,T,1]).
  if (a.rank >= 1 &&
      b.rank == a.rank &&
      b.shape[a.rank - 1] == 1 &&
      a.shape[a.rank - 1] > 1 &&
      _shapeEq(a.shape.sublist(0, a.rank - 1), b.shape.sublist(0, b.rank - 1))) {
    final d = a.shape[a.rank - 1];
    final rows = bn; // = an ~/ d
    final out = Float32List(an);
    switch (op) {
      case _Arith.add:
        for (int r = 0; r < rows; r++) {
          final s = bf[r], base = r * d;
          for (int j = 0; j < d; j++) {
            out[base + j] = af[base + j] + s;
          }
        }
      case _Arith.sub:
        for (int r = 0; r < rows; r++) {
          final s = bf[r], base = r * d;
          for (int j = 0; j < d; j++) {
            out[base + j] = af[base + j] - s;
          }
        }
      case _Arith.mul:
        for (int r = 0; r < rows; r++) {
          final s = bf[r], base = r * d;
          for (int j = 0; j < d; j++) {
            out[base + j] = af[base + j] * s;
          }
        }
      case _Arith.div:
        for (int r = 0; r < rows; r++) {
          final s = bf[r], base = r * d;
          for (int j = 0; j < d; j++) {
            out[base + j] = af[base + j] / s;
          }
        }
    }
    return Tensor.float(out, a.shape);
  }

  // Suffix tile, b smaller: out[off+j] = a[off+j] op b[j]. Output shape is
  // the broadcast shape, which here is a.shape (possibly with b's extra
  // leading 1-dims — only possible when b.rank <= a.rank, so require that).
  if (bn < an && b.rank <= a.rank && _isTrailingSuffix(b.shape, a.shape)) {
    final out = Float32List(an);
    switch (op) {
      case _Arith.add:
        for (int off = 0; off < an; off += bn) {
          for (int j = 0; j < bn; j++) {
            out[off + j] = af[off + j] + bf[j];
          }
        }
      case _Arith.sub:
        for (int off = 0; off < an; off += bn) {
          for (int j = 0; j < bn; j++) {
            out[off + j] = af[off + j] - bf[j];
          }
        }
      case _Arith.mul:
        for (int off = 0; off < an; off += bn) {
          for (int j = 0; j < bn; j++) {
            out[off + j] = af[off + j] * bf[j];
          }
        }
      case _Arith.div:
        for (int off = 0; off < an; off += bn) {
          for (int j = 0; j < bn; j++) {
            out[off + j] = af[off + j] / bf[j];
          }
        }
    }
    return Tensor.float(out, a.shape);
  }
  // Suffix tile, a smaller (kept separate: Sub/Div aren't commutative).
  if (an < bn && a.rank <= b.rank && _isTrailingSuffix(a.shape, b.shape)) {
    final out = Float32List(bn);
    switch (op) {
      case _Arith.add:
        for (int off = 0; off < bn; off += an) {
          for (int j = 0; j < an; j++) {
            out[off + j] = af[j] + bf[off + j];
          }
        }
      case _Arith.sub:
        for (int off = 0; off < bn; off += an) {
          for (int j = 0; j < an; j++) {
            out[off + j] = af[j] - bf[off + j];
          }
        }
      case _Arith.mul:
        for (int off = 0; off < bn; off += an) {
          for (int j = 0; j < an; j++) {
            out[off + j] = af[j] * bf[off + j];
          }
        }
      case _Arith.div:
        for (int off = 0; off < bn; off += an) {
          for (int j = 0; j < an; j++) {
            out[off + j] = af[j] / bf[off + j];
          }
        }
    }
    return Tensor.float(out, b.shape);
  }

  return null;
}

Tensor _elementwiseBinary(
    Tensor a, Tensor b, double Function(double, double) op) {
  final bothInt = !a.isFloat && !b.isFloat;

  // Fast path: identical shapes (the common case for residual adds etc.) —
  // no coordinate decomposition needed, just walk both flat buffers in lockstep.
  if (_shapeEq(a.shape, b.shape)) {
    final n = a.length;
    if (bothInt) {
      final out = Int64List(n);
      for (int k = 0; k < n; k++) {
        out[k] = op(a.getD(k), b.getD(k)).round();
      }
      return Tensor.int64(out, a.shape);
    }
    final out = Float32List(n);
    for (int k = 0; k < n; k++) {
      out[k] = op(a.getD(k), b.getD(k));
    }
    return Tensor.float(out, a.shape);
  }

  // Fast path: b is a scalar (very common for bias/constant ops). Only valid
  // when a's rank is >= b's — otherwise the broadcast output shape is b's,
  // not a's (e.g. a: rank-0 `[]` vs b: `[1]` broadcasts to `[1]`, not `[]`).
  if (b.length == 1 && a.rank >= b.rank) {
    final n = a.length;
    final scalar = b.getD(0);
    if (bothInt) {
      final out = Int64List(n);
      for (int k = 0; k < n; k++) {
        out[k] = op(a.getD(k), scalar).round();
      }
      return Tensor.int64(out, a.shape);
    }
    final out = Float32List(n);
    for (int k = 0; k < n; k++) {
      out[k] = op(a.getD(k), scalar);
    }
    return Tensor.float(out, a.shape);
  }
  if (a.length == 1 && b.rank >= a.rank) {
    final n = b.length;
    final scalar = a.getD(0);
    if (bothInt) {
      final out = Int64List(n);
      for (int k = 0; k < n; k++) {
        out[k] = op(scalar, b.getD(k)).round();
      }
      return Tensor.int64(out, b.shape);
    }
    final out = Float32List(n);
    for (int k = 0; k < n; k++) {
      out[k] = op(scalar, b.getD(k));
    }
    return Tensor.float(out, b.shape);
  }

  // General broadcasting path: incremental odometer with per-axis source
  // strides (stride 0 on broadcast axes) — no per-element coordinate
  // decomposition.
  final outShape = _broadcastShape(a.shape, b.shape);
  final n = outShape.fold<int>(1, (x, y) => x * y);
  final rank = outShape.length;
  final aStridesFull = List<int>.filled(rank, 0);
  final bStridesFull = List<int>.filled(rank, 0);
  final aStrides = a.strides, bStrides = b.strides;
  for (int k = 0; k < a.rank; k++) {
    if (a.shape[k] != 1) aStridesFull[k + rank - a.rank] = aStrides[k];
  }
  for (int k = 0; k < b.rank; k++) {
    if (b.shape[k] != 1) bStridesFull[k + rank - b.rank] = bStrides[k];
  }
  final coords = List<int>.filled(rank, 0);
  int aOff = 0, bOff = 0;

  void advance() {
    for (int k = rank - 1; k >= 0; k--) {
      aOff += aStridesFull[k];
      bOff += bStridesFull[k];
      if (++coords[k] < outShape[k]) return;
      coords[k] = 0;
      aOff -= outShape[k] * aStridesFull[k];
      bOff -= outShape[k] * bStridesFull[k];
    }
  }

  if (bothInt) {
    final out = Int64List(n);
    for (int idx = 0; idx < n; idx++) {
      out[idx] = op(a.getD(aOff), b.getD(bOff)).round();
      advance();
    }
    return Tensor.int64(out, outShape);
  }
  final out = Float32List(n);
  for (int idx = 0; idx < n; idx++) {
    out[idx] = op(a.getD(aOff), b.getD(bOff));
    advance();
  }
  return Tensor.float(out, outShape);
}

// ---------------------------------------------------------------------------
// Elementwise ops
// ---------------------------------------------------------------------------

Tensor opAdd(Tensor a, Tensor b) =>
    _arithFloatFast(a, b, _Arith.add) ??
    _elementwiseBinary(a, b, (x, y) => x + y);
Tensor opSub(Tensor a, Tensor b) =>
    _arithFloatFast(a, b, _Arith.sub) ??
    _elementwiseBinary(a, b, (x, y) => x - y);
Tensor opMul(Tensor a, Tensor b) =>
    _arithFloatFast(a, b, _Arith.mul) ??
    _elementwiseBinary(a, b, (x, y) => x * y);
Tensor opDiv(Tensor a, Tensor b) {
  // Integer division truncates toward zero per the ONNX spec (the generic
  // path would round the float quotient — off by one on length arithmetic
  // like ceil-div chains).
  if (!a.isFloat && !b.isFloat) {
    if (_shapeEq(a.shape, b.shape) || (b.length == 1 && a.rank >= b.rank)) {
      final n = a.length;
      final ai = a.intData, bi = b.intData;
      final scalar = b.length == 1 ? bi[0] : 0;
      final out = Int64List(n);
      for (int k = 0; k < n; k++) {
        out[k] = ai[k] ~/ (b.length == 1 ? scalar : bi[k]);
      }
      return Tensor.int64(out, a.shape);
    }
    // General broadcast: divide via coordinates but truncate.
    final r = _elementwiseBinary(a, b, (x, y) => (x / y).truncateToDouble());
    return r;
  }
  return _arithFloatFast(a, b, _Arith.div) ??
      _elementwiseBinary(a, b, (x, y) => x / y);
}
Tensor opPow(Tensor a, Tensor b) {
  // Scalar exponent (LayerNorm's `(x-mean)^2`, sqrt-as-pow etc.) — direct
  // loop, and plain multiply for the ubiquitous square.
  if (a.f != null && b.length == 1 && a.rank >= b.rank) {
    final e = b.getD(0);
    final af = a.f!;
    final out = Float32List(af.length);
    if (e == 2.0) {
      for (int k = 0; k < af.length; k++) {
        out[k] = af[k] * af[k];
      }
    } else {
      for (int k = 0; k < af.length; k++) {
        out[k] = math.pow(af[k], e).toDouble();
      }
    }
    return Tensor.float(out, a.shape);
  }
  return _elementwiseBinary(a, b, (x, y) => math.pow(x, y).toDouble());
}

Tensor _elementwiseUnary(Tensor a, double Function(double) op) {
  // Hoist the length (Tensor.length is a computed getter — in the loop
  // condition it would re-fold the shape every iteration) and skip getD's
  // per-element dtype branch.
  final n = a.length;
  final out = Float32List(n);
  final af = a.f;
  if (af != null) {
    for (int k = 0; k < n; k++) {
      out[k] = op(af[k]);
    }
  } else {
    final ai = a.intData;
    for (int k = 0; k < n; k++) {
      out[k] = op(ai[k].toDouble());
    }
  }
  return Tensor.float(out, a.shape);
}

Tensor opSqrt(Tensor a) => _elementwiseUnary(a, (x) => math.sqrt(x));
Tensor opReciprocal(Tensor a) => _elementwiseUnary(a, (x) => 1.0 / x);
// Direct loop: Relu runs over every activation map in CNNs.
Tensor opRelu(Tensor a) {
  final af = a.f;
  if (af == null) return _elementwiseUnary(a, (x) => x < 0 ? 0.0 : x);
  final n = af.length;
  final out = Float32List(n);
  for (int k = 0; k < n; k++) {
    final v = af[k];
    out[k] = v < 0 ? 0.0 : v;
  }
  return Tensor.float(out, a.shape);
}
Tensor opLeakyRelu(Tensor a, double alpha) =>
    _elementwiseUnary(a, (x) => x < 0 ? alpha * x : x);
Tensor opElu(Tensor a, double alpha) =>
    _elementwiseUnary(a, (x) => x < 0 ? alpha * (math.exp(x) - 1) : x);
Tensor opHardSigmoid(Tensor a, double alpha, double beta) =>
    _elementwiseUnary(a, (x) => (alpha * x + beta).clamp(0.0, 1.0));

/// `HardSwish`: `x * hardSigmoid(x)` with the spec-fixed alpha=1/6, beta=0.5.
Tensor opHardSwish(Tensor a) =>
    _elementwiseUnary(a, (x) => x * (x / 6 + 0.5).clamp(0.0, 1.0));
Tensor opSoftplus(Tensor a) =>
    _elementwiseUnary(a, (x) => math.log(1 + math.exp(x)));
Tensor opAtan(Tensor a) => _elementwiseUnary(a, math.atan);
Tensor opSign(Tensor a) => a.isFloat
    ? _elementwiseUnary(a, (x) => x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0))
    : _unaryInt(a, (x) => x > 0 ? 1 : (x < 0 ? -1 : 0));
Tensor opFloor(Tensor a) =>
    a.isFloat ? _elementwiseUnary(a, (x) => x.floorToDouble()) : a;
Tensor opCeil(Tensor a) =>
    a.isFloat ? _elementwiseUnary(a, (x) => x.ceilToDouble()) : a;

/// `Round` — half to even, per the ONNX spec.
Tensor opRound(Tensor a) => a.isFloat
    ? _elementwiseUnary(a, (x) => _roundEven(x).toDouble())
    : a;

/// `Gelu`: exact (erf) form, or the tanh approximation when
/// `approximate="tanh"`.
Tensor opGelu(Tensor a, {bool tanhApprox = false}) => tanhApprox
    ? _elementwiseUnary(a, (x) {
        const c = 0.7978845608028654; // sqrt(2/pi)
        return 0.5 * x * (1 + _tanh(c * (x + 0.044715 * x * x * x)));
      })
    : _elementwiseUnary(a, (x) => 0.5 * x * (1 + _erf(x / math.sqrt2)));

/// `PRelu`: `x < 0 ? slope * x : x`, with [slope] broadcast onto [x].
Tensor opPRelu(Tensor x, Tensor slope) =>
    _elementwiseBinary(x, slope, (v, s) => v < 0 ? s * v : v);

// Erf via the Abramowitz & Stegun 7.1.26 approximation (public numerical
// analysis formula, max abs error ~1.5e-7) — needed for the GELU activation
// (gelu(x) = 0.5*x*(1+erf(x/sqrt(2)))), which the graph expresses directly
// as an Erf node rather than a fused GELU op.
double _erf(double x) {
  final sign = x < 0 ? -1.0 : 1.0;
  x = x.abs();
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  final t = 1.0 / (1.0 + p * x);
  final y = 1.0 -
      (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x);
  return sign * y;
}

// Direct loop rather than _elementwiseUnary: Erf is hot (GELU runs it over
// the whole FFN activation) and the per-element closure call costs as much
// as the math at this size.
Tensor opErf(Tensor a) {
  final af = a.f;
  if (af == null) return _elementwiseUnary(a, _erf);
  final out = Float32List(af.length);
  for (int k = 0; k < af.length; k++) {
    out[k] = _erf(af[k]);
  }
  return Tensor.float(out, a.shape);
}

Tensor opClip(Tensor x, Tensor? min, Tensor? max) {
  final lo = min != null ? min.getD(0) : double.negativeInfinity;
  final hi = max != null ? max.getD(0) : double.infinity;
  return _elementwiseUnary(x, (v) => v.clamp(lo, hi));
}

// ---------------------------------------------------------------------------
// Cast
// ---------------------------------------------------------------------------

Tensor opCast(Tensor x, int to) {
  // ONNX TensorProto.DataType: 1=FLOAT, 6=INT32, 7=INT64, 9=BOOL,
  // 10=FLOAT16, 11=DOUBLE. int32/int64 share our int64 tensor, bool is
  // int64 0/1, double is carried as float32.
  if (to == 9) {
    final n = x.length;
    final out = Int64List(n);
    for (int k = 0; k < n; k++) {
      out[k] = x.getD(k) != 0 ? 1 : 0;
    }
    return Tensor.int64(out, x.shape);
  }
  if (to == 6 || to == 7) return Tensor.int64(x.asIntList(), x.shape);
  if (to == 10) {
    // fp16: values must round through half precision (fp16-casting exports
    // depend on it; treating this as identity drifts vs ORT), even though
    // the result is carried in a float32 tensor.
    final src = x.asFloatList();
    final out = Float32List(src.length);
    final bits = out.buffer.asUint32List();
    final srcBits = src.buffer.asUint32List(src.offsetInBytes, src.length);
    for (int k = 0; k < src.length; k++) {
      bits[k] = halfToFloat32Bits(float32ToHalfBits(srcBits[k]));
    }
    return Tensor.float(out, x.shape);
  }
  return Tensor.float(x.asFloatList(), x.shape);
}

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

/// `Shape(x)` — the dimensions of [x] as an int64 vector. The optional
/// [start] / [end] (opset 15+) return a slice of the shape, e.g.
/// `Shape(x, start: 1)` drops the batch dim.
Tensor opShape(Tensor x, {int? start, int? end}) {
  final rank = x.shape.length;
  var s = start ?? 0;
  var e = end ?? rank;
  if (s < 0) s += rank;
  if (e < 0) e += rank;
  s = s.clamp(0, rank);
  e = e.clamp(0, rank);
  final slice = e > s ? x.shape.sublist(s, e) : <int>[];
  return Tensor.int64(Int64List.fromList(slice), [slice.length]);
}

Tensor opReshape(Tensor x, Tensor shapeT) => x.reshape(shapeT.asIntList());

Tensor opTranspose(Tensor x, List<int> perm) {
  final newShape = [for (final p in perm) x.shape[p]];
  final n = x.length;
  final oldStrides = x.strides;
  final rank = perm.length;

  // Permuted source strides in output-axis order, so walking the output in
  // order just adds permStrides[k] when output coordinate k increments.
  final permStrides = [for (final p in perm) oldStrides[p]];

  // Fast path: the last axis stays last (e.g. attention's [B,T,H,D] ->
  // [B,H,T,D]) — both source and destination rows of length `last` are
  // contiguous, so copy row-blocks instead of single elements.
  if (x.isFloat && rank >= 2 && perm[rank - 1] == rank - 1) {
    final last = newShape[rank - 1];
    final out = Float32List(n);
    final xf = x.f!;
    final coords = List<int>.filled(rank - 1, 0);
    int srcOff = 0;
    for (int dst = 0; dst < n; dst += last) {
      out.setRange(dst, dst + last, xf, srcOff);
      for (int k = rank - 2; k >= 0; k--) {
        srcOff += permStrides[k];
        if (++coords[k] < newShape[k]) break;
        coords[k] = 0;
        srcOff -= newShape[k] * permStrides[k];
      }
    }
    return Tensor.float(out, newShape);
  }

  // General path: incremental coordinate walk, no per-element allocation.
  final coords = List<int>.filled(rank, 0);
  int srcOff = 0;
  if (x.isFloat) {
    final out = Float32List(n);
    final xf = x.f!;
    for (int idx = 0; idx < n; idx++) {
      out[idx] = xf[srcOff];
      for (int k = rank - 1; k >= 0; k--) {
        srcOff += permStrides[k];
        if (++coords[k] < newShape[k]) break;
        coords[k] = 0;
        srcOff -= newShape[k] * permStrides[k];
      }
    }
    return Tensor.float(out, newShape);
  }
  final out = Int64List(n);
  final xi = x.intData;
  for (int idx = 0; idx < n; idx++) {
    out[idx] = xi[srcOff];
    for (int k = rank - 1; k >= 0; k--) {
      srcOff += permStrides[k];
      if (++coords[k] < newShape[k]) break;
      coords[k] = 0;
      srcOff -= newShape[k] * permStrides[k];
    }
  }
  return Tensor.int64(out, newShape);
}

Tensor opSqueeze(Tensor x, List<int>? axes) {
  final ax = (axes ??
          [
            for (int k = 0; k < x.shape.length; k++)
              if (x.shape[k] == 1) k
          ])
      .map((a) => a < 0 ? a + x.shape.length : a)
      .toSet();
  final newShape = [
    for (int k = 0; k < x.shape.length; k++)
      if (!ax.contains(k)) x.shape[k]
  ];
  return x.reshape(newShape);
}

Tensor opUnsqueeze(Tensor x, List<int> axes) {
  final outRank = x.shape.length + axes.length;
  final normAxes = axes.map((a) => a < 0 ? a + outRank : a).toSet();
  final newShape = <int>[];
  int srcIdx = 0;
  for (int k = 0; k < outRank; k++) {
    if (normAxes.contains(k)) {
      newShape.add(1);
    } else {
      newShape.add(x.shape[srcIdx++]);
    }
  }
  return x.reshape(newShape);
}

Tensor opConcat(List<Tensor> inputs, int axis) {
  final rank = inputs.first.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final outShape = List<int>.from(inputs.first.shape);
  outShape[ax] = inputs.fold<int>(0, (sum, t) => sum + t.shape[ax]);

  final isFloat = inputs.first.isFloat;
  final n = outShape.fold<int>(1, (a, b) => a * b);
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);

  final outerSize = outShape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final innerSize = outShape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);

  for (int outer = 0; outer < outerSize; outer++) {
    int axOffset = 0;
    for (final t in inputs) {
      final tAx = t.shape[ax];
      final blockSize = tAx * innerSize;
      final srcStart = outer * blockSize;
      final dstStart = outer * outShape[ax] * innerSize + axOffset * innerSize;
      for (int k = 0; k < blockSize; k++) {
        if (isFloat) {
          outF![dstStart + k] = t.f![srcStart + k];
        } else {
          outI![dstStart + k] = t.intData[srcStart + k];
        }
      }
      axOffset += tAx;
    }
  }
  return isFloat
      ? Tensor.float(outF!, outShape)
      : Tensor.int64(outI!, outShape);
}

// ---------------------------------------------------------------------------
// Gather / GatherND
// ---------------------------------------------------------------------------

Tensor opGather(Tensor data, Tensor indices, int axis) {
  final rank = data.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final idxShape = indices.shape;
  final outShape = [
    ...data.shape.sublist(0, ax),
    ...idxShape,
    ...data.shape.sublist(ax + 1)
  ];

  final outerSize = data.shape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final axisSize = data.shape[ax];
  final innerSize = data.shape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);
  final idxCount = idxShape.fold<int>(1, (a, b) => a * b);

  final n = outShape.fold<int>(1, (a, b) => a * b);
  final isFloat = data.isFloat;
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);

  int outPos = 0;
  for (int outer = 0; outer < outerSize; outer++) {
    for (int ii = 0; ii < idxCount; ii++) {
      int idx = indices.getI(ii);
      if (idx < 0) idx += axisSize;
      final srcStart = outer * axisSize * innerSize + idx * innerSize;
      for (int k = 0; k < innerSize; k++) {
        if (isFloat) {
          outF![outPos + k] = data.f![srcStart + k];
        } else {
          outI![outPos + k] = data.intData[srcStart + k];
        }
      }
      outPos += innerSize;
    }
  }
  return isFloat
      ? Tensor.float(outF!, outShape)
      : Tensor.int64(outI!, outShape);
}

Tensor opGatherND(Tensor data, Tensor indices, int batchDims) {
  // batch_dims == 0 is the only case this graph uses.
  assert(batchDims == 0);
  final idxRank = indices.shape.length;
  final k = indices.shape[idxRank - 1];
  final batchIdxShape = indices.shape.sublist(0, idxRank - 1);
  final remainingDataShape = data.shape.sublist(k);
  final outShape = [...batchIdxShape, ...remainingDataShape];

  final dataStrides = data.strides;
  final numIdxTuples = batchIdxShape.fold<int>(1, (a, b) => a * b);
  final innerSize = remainingDataShape.fold<int>(1, (a, b) => a * b);

  final isFloat = data.isFloat;
  final n = outShape.fold<int>(1, (a, b) => a * b);
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);

  for (int t = 0; t < numIdxTuples; t++) {
    int dataFlatStart = 0;
    for (int d = 0; d < k; d++) {
      final idxVal = indices.getI(t * k + d);
      dataFlatStart += idxVal * dataStrides[d];
    }
    for (int j = 0; j < innerSize; j++) {
      if (isFloat) {
        outF![t * innerSize + j] = data.f![dataFlatStart + j];
      } else {
        outI![t * innerSize + j] = data.intData[dataFlatStart + j];
      }
    }
  }
  return isFloat
      ? Tensor.float(outF!, outShape)
      : Tensor.int64(outI!, outShape);
}

// ---------------------------------------------------------------------------
// Expand / Slice
// ---------------------------------------------------------------------------

Tensor opExpand(Tensor x, Tensor shapeT) {
  final targetShape = shapeT.asIntList().toList();
  final outShape = _broadcastShape(x.shape, targetShape);
  final n = outShape.fold<int>(1, (a, b) => a * b);
  if (x.isFloat) {
    final out = Float32List(n);
    for (int idx = 0; idx < n; idx++) {
      out[idx] = x.f![_flattenBroadcast(_unflatten(idx, outShape), x.shape)];
    }
    return Tensor.float(out, outShape);
  } else {
    final out = Int64List(n);
    for (int idx = 0; idx < n; idx++) {
      out[idx] = x.intData[_flattenBroadcast(_unflatten(idx, outShape), x.shape)];
    }
    return Tensor.int64(out, outShape);
  }
}

Tensor opSlice(Tensor x, List<int> starts, List<int> ends, List<int>? axes,
    List<int>? steps) {
  final rank = x.shape.length;
  final ax = axes ?? List<int>.generate(starts.length, (k) => k);
  final st = steps ?? List<int>.filled(starts.length, 1);

  final normStart = List<int>.from(x.shape.map((d) => 0));
  final normEnd = List<int>.from(x.shape);
  final normStep = List<int>.filled(rank, 1);

  for (int j = 0; j < ax.length; j++) {
    int a = ax[j];
    if (a < 0) a += rank;
    final dim = x.shape[a];
    int s = starts[j];
    int e = ends[j];
    final step = st[j];
    if (s < 0) s += dim;
    if (e < 0) e += dim;
    s = s.clamp(0, step < 0 ? dim - 1 : dim);
    e = e.clamp(step < 0 ? -1 : 0, dim);
    normStart[a] = s;
    normEnd[a] = e;
    normStep[a] = step;
  }

  final outShape = <int>[];
  for (int a = 0; a < rank; a++) {
    final cnt = normStep[a] > 0
        ? math.max(
            0, (normEnd[a] - normStart[a] + normStep[a] - 1) ~/ normStep[a])
        : math.max(
            0, (normStart[a] - normEnd[a] - normStep[a] - 1) ~/ (-normStep[a]));
    outShape.add(cnt);
  }

  final n = outShape.fold<int>(1, (a, b) => a * b);
  final isFloat = x.isFloat;
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);
  final srcStrides = x.strides;

  for (int idx = 0; idx < n; idx++) {
    final outCoords = _unflatten(idx, outShape);
    int srcFlat = 0;
    for (int a = 0; a < rank; a++) {
      final coord = normStart[a] + outCoords[a] * normStep[a];
      srcFlat += coord * srcStrides[a];
    }
    if (isFloat) {
      outF![idx] = x.f![srcFlat];
    } else {
      outI![idx] = x.intData[srcFlat];
    }
  }
  return isFloat
      ? Tensor.float(outF!, outShape)
      : Tensor.int64(outI!, outShape);
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

Tensor opReduceMean(Tensor x, List<int>? axes, bool keepdims) {
  final rank = x.shape.length;
  final ax = (axes ?? List<int>.generate(rank, (k) => k))
      .map((a) => a < 0 ? a + rank : a)
      .toSet();

  // Fast path: reducing only the last axis (LayerNorm's mean) — each output
  // is the mean of one contiguous row.
  if (x.isFloat && ax.length == 1 && ax.first == rank - 1 && rank >= 1) {
    final d = x.shape[rank - 1];
    final rows = d == 0 ? 0 : x.length ~/ d;
    final out = Float32List(rows);
    final xf = x.f!;
    for (int r = 0; r < rows; r++) {
      final base = r * d;
      double sum = 0;
      for (int k = 0; k < d; k++) {
        sum += xf[base + k];
      }
      out[r] = sum / d;
    }
    final shape = keepdims
        ? [...x.shape.sublist(0, rank - 1), 1]
        : x.shape.sublist(0, rank - 1);
    return Tensor.float(out, shape);
  }

  final outShapeFull = [
    for (int k = 0; k < rank; k++) ax.contains(k) ? 1 : x.shape[k]
  ];
  final reducedCount = ax.fold<int>(1, (acc, k) => acc * x.shape[k]);

  final n = outShapeFull.fold<int>(1, (a, b) => a * b);
  final sums = Float64List(n);
  final outStridesFull = Tensor.filledFloat(outShapeFull, 0).strides;

  for (int idx = 0; idx < x.length; idx++) {
    final coords = _unflatten(idx, x.shape);
    int outFlat = 0;
    for (int k = 0; k < rank; k++) {
      final c = ax.contains(k) ? 0 : coords[k];
      outFlat += c * outStridesFull[k];
    }
    sums[outFlat] += x.getD(idx);
  }
  final out = Float32List(n);
  for (int k = 0; k < n; k++) {
    out[k] = sums[k] / reducedCount;
  }

  if (keepdims) return Tensor.float(out, outShapeFull);
  final squeezedShape = [
    for (int k = 0; k < rank; k++)
      if (!ax.contains(k)) x.shape[k]
  ];
  return Tensor.float(out, squeezedShape);
}

/// `ReduceProd` — product over the given axes.
Tensor opReduceProd(Tensor x, List<int>? axes, bool keepdims) {
  final rank = x.shape.length;
  final ax = (axes ?? List<int>.generate(rank, (k) => k))
      .map((a) => a < 0 ? a + rank : a)
      .toSet();
  final outShapeFull = [
    for (int k = 0; k < rank; k++) ax.contains(k) ? 1 : x.shape[k]
  ];
  final n = outShapeFull.fold<int>(1, (a, b) => a * b);
  final prods = Float64List(n)..fillRange(0, n, 1);
  final outStridesFull = Tensor.filledFloat(outShapeFull, 0).strides;
  final coords = List<int>.filled(rank, 0);
  for (int idx = 0; idx < x.length; idx++) {
    int outFlat = 0;
    for (int k = 0; k < rank; k++) {
      if (!ax.contains(k)) outFlat += coords[k] * outStridesFull[k];
    }
    prods[outFlat] *= x.getD(idx);
    for (int k = rank - 1; k >= 0; k--) {
      if (++coords[k] < x.shape[k]) break;
      coords[k] = 0;
    }
  }
  final shape = keepdims
      ? outShapeFull
      : [
          for (int k = 0; k < rank; k++)
            if (!ax.contains(k)) x.shape[k]
        ];
  if (x.isFloat) {
    final out = Float32List(n);
    for (int k = 0; k < n; k++) {
      out[k] = prods[k];
    }
    return Tensor.float(out, shape);
  }
  final out = Int64List(n);
  for (int k = 0; k < n; k++) {
    out[k] = prods[k].toInt();
  }
  return Tensor.int64(out, shape);
}

/// `ReduceSumSquare` — sum of squares over the given axes.
Tensor opReduceSumSquare(Tensor x, List<int>? axes, bool keepdims) {
  final squared = opMul(x, x);
  return opReduceSum(squared, axes, keepdims);
}

/// `Split` — slices [x] along [axis] into [numOutputs] parts, sized by
/// [splitSizes] when given, else evenly (ceil-divided, last part smaller —
/// the opset-18 `num_outputs` rule, which reduces to equal parts when the
/// axis divides evenly).
List<Tensor> opSplit(Tensor x, int axis, int numOutputs,
    [List<int>? splitSizes]) {
  final ax = axis < 0 ? axis + x.rank : axis;
  final dim = x.shape[ax];
  final sizes = splitSizes ??
      () {
        final chunk = (dim + numOutputs - 1) ~/ numOutputs;
        return [
          for (int i = 0; i < numOutputs; i++)
            i < numOutputs - 1 ? chunk : dim - chunk * (numOutputs - 1)
        ];
      }();
  final outs = <Tensor>[];
  int start = 0;
  for (final size in sizes) {
    outs.add(opSlice(x, [start], [start + size], [ax], null));
    start += size;
  }
  return outs;
}

/// GLU gate — the fused form of `Split`-in-half + `Sigmoid(second)` +
/// `Mul(first, ·)`: splits [x] in half along [axis] into `(a, b)` and returns
/// `a * sigmoid(b)` in one pass (one memory read/write instead of three).
/// Bitwise identical to the decomposition (same `1/(1+exp(-x))` sigmoid).
Tensor opGlu(Tensor x, int axis) {
  final rank = x.rank;
  final ax = axis < 0 ? axis + rank : axis;
  final axisSize = x.shape[ax];
  final half = axisSize ~/ 2;
  final inner = x.shape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);
  final outer = x.shape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final xf = x.f ?? x.asFloatList();
  final out = Float32List(outer * half * inner);
  for (int o = 0; o < outer; o++) {
    for (int h = 0; h < half; h++) {
      final aBase = (o * axisSize + h) * inner;
      final bBase = (o * axisSize + half + h) * inner;
      final oBase = (o * half + h) * inner;
      for (int i = 0; i < inner; i++) {
        // s = Sigmoid(b) then a * s — matches the decomposed op bit-for-bit.
        final s = 1.0 / (1.0 + math.exp(-xf[bBase + i]));
        out[oBase + i] = xf[aBase + i] * s;
      }
    }
  }
  return Tensor.float(out, [...x.shape]..[ax] = half);
}

/// `STFT` (opset 17) — real input only, frame-major output
/// `[batch, frames, bins, 2]` (re/im). No centering or padding: frames start
/// at multiples of [frameStep] and must fit entirely inside the signal (the
/// ONNX graph does its own `Pad` beforehand). [onesided] keeps
/// `dftSize/2 + 1` bins. Odd or non-power-of-2 frame lengths are fine — bins
/// come from a direct DFT over a single-period twiddle table.
Tensor opSTFT(Tensor signal, int frameStep, Tensor? window, int? frameLength,
    {bool onesided = true}) {
  // Signal is [batch, len] or [batch, len, 1].
  assert(
      signal.rank == 2 || (signal.rank == 3 && signal.shape[2] == 1),
      'STFT supports real signals only');
  final batch = signal.shape[0], len = signal.shape[1];
  final n = frameLength ?? window!.length;
  final win = window?.asFloatList();
  final frames = (len - n) ~/ frameStep + 1;
  final bins = onesided ? n ~/ 2 + 1 : n;

  final cosT = Float64List(n), sinT = Float64List(n);
  for (int i = 0; i < n; i++) {
    final a = -2 * math.pi * i / n;
    cosT[i] = math.cos(a);
    sinT[i] = math.sin(a);
  }

  final sf = signal.asFloatList();
  final out = Float32List(batch * frames * bins * 2);
  final frame = Float64List(n);
  for (int b = 0; b < batch; b++) {
    for (int f = 0; f < frames; f++) {
      final src = b * len + f * frameStep;
      for (int i = 0; i < n; i++) {
        frame[i] = win == null ? sf[src + i] : sf[src + i] * win[i];
      }
      final outBase = ((b * frames) + f) * bins * 2;
      for (int k = 0; k < bins; k++) {
        double re = 0, im = 0;
        int idx = 0;
        for (int i = 0; i < n; i++) {
          final v = frame[i];
          re += v * cosT[idx];
          im += v * sinT[idx];
          idx += k;
          if (idx >= n) idx -= n;
        }
        out[outBase + 2 * k] = re;
        out[outBase + 2 * k + 1] = im;
      }
    }
  }
  return Tensor.float(out, [batch, frames, bins, 2]);
}

/// `ReduceMax` / `ReduceMin` (dtype-preserving, unlike the mean/sum
/// reductions which are float by definition).
Tensor opReduceMinMax(Tensor x, List<int>? axes, bool keepdims,
    {required bool isMax}) {
  final rank = x.shape.length;
  final ax = (axes == null || axes.isEmpty
          ? List<int>.generate(rank, (k) => k)
          : axes)
      .map((a) => a < 0 ? a + rank : a)
      .toSet();
  final outShapeFull = [
    for (int k = 0; k < rank; k++) ax.contains(k) ? 1 : x.shape[k]
  ];
  final n = outShapeFull.fold<int>(1, (a, b) => a * b);
  final outStridesFull = Tensor.filledFloat(outShapeFull, 0).strides;
  final best = Float64List(n)
    ..fillRange(0, n, isMax ? double.negativeInfinity : double.infinity);

  final coords = List<int>.filled(rank, 0);
  final total = x.length;
  for (int idx = 0; idx < total; idx++) {
    int outFlat = 0;
    for (int k = 0; k < rank; k++) {
      if (!ax.contains(k)) outFlat += coords[k] * outStridesFull[k];
    }
    final v = x.getD(idx);
    if (isMax ? v > best[outFlat] : v < best[outFlat]) best[outFlat] = v;
    for (int k = rank - 1; k >= 0; k--) {
      if (++coords[k] < x.shape[k]) break;
      coords[k] = 0;
    }
  }
  final shape = keepdims
      ? outShapeFull
      : [
          for (int k = 0; k < rank; k++)
            if (!ax.contains(k)) x.shape[k]
        ];
  if (x.isFloat) {
    final out = Float32List(n);
    for (int k = 0; k < n; k++) {
      out[k] = best[k];
    }
    return Tensor.float(out, shape);
  }
  final out = Int64List(n);
  for (int k = 0; k < n; k++) {
    out[k] = best[k].toInt();
  }
  return Tensor.int64(out, shape);
}

Tensor opSoftmax(Tensor x, int axis) {
  final rank = x.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final outerSize = x.shape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final axisSize = x.shape[ax];
  final innerSize = x.shape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);

  final out = Float32List(x.length);
  final xf = x.f ?? x.asFloatList(); // attention scores are always float
  // Contiguous last-axis softmax (the overwhelmingly common case: attention
  // rows, classifier heads) — walk each row linearly instead of striding.
  if (innerSize == 1) {
    final rows = x.length ~/ axisSize;
    for (int r = 0; r < rows; r++) {
      final base = r * axisSize;
      double maxV = double.negativeInfinity;
      for (int a = 0; a < axisSize; a++) {
        final v = xf[base + a];
        if (v > maxV) maxV = v;
      }
      double sum = 0;
      for (int a = 0; a < axisSize; a++) {
        final e = math.exp(xf[base + a] - maxV);
        out[base + a] = e;
        sum += e;
      }
      final inv = 1.0 / sum;
      for (int a = 0; a < axisSize; a++) {
        out[base + a] *= inv;
      }
    }
    return Tensor.float(out, x.shape);
  }
  for (int outer = 0; outer < outerSize; outer++) {
    for (int inner = 0; inner < innerSize; inner++) {
      final b0 = outer * axisSize * innerSize + inner;
      double maxV = double.negativeInfinity;
      for (int a = 0; a < axisSize; a++) {
        final v = xf[b0 + a * innerSize];
        if (v > maxV) maxV = v;
      }
      double sum = 0;
      for (int a = 0; a < axisSize; a++) {
        final e = math.exp(xf[b0 + a * innerSize] - maxV);
        out[b0 + a * innerSize] = e;
        sum += e;
      }
      final inv = 1.0 / sum;
      for (int a = 0; a < axisSize; a++) {
        out[b0 + a * innerSize] *= inv;
      }
    }
  }
  return Tensor.float(out, x.shape);
}

/// `LogSoftmax(x) = x - max - log(sum(exp(x - max)))` — numerically stable
/// via the same max-shift as Softmax, but keeping the result in log space
/// (CTC / LID heads use this).
Tensor opLogSoftmax(Tensor x, int axis) {
  final rank = x.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final outerSize = x.shape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final axisSize = x.shape[ax];
  final innerSize = x.shape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);

  final out = Float32List(x.length);
  for (int outer = 0; outer < outerSize; outer++) {
    for (int inner = 0; inner < innerSize; inner++) {
      double maxV = double.negativeInfinity;
      for (int a = 0; a < axisSize; a++) {
        final v = x.getD(outer * axisSize * innerSize + a * innerSize + inner);
        if (v > maxV) maxV = v;
      }
      double sum = 0;
      for (int a = 0; a < axisSize; a++) {
        sum += math.exp(
            x.getD(outer * axisSize * innerSize + a * innerSize + inner) -
                maxV);
      }
      final logSum = maxV + math.log(sum);
      for (int a = 0; a < axisSize; a++) {
        final idx = outer * axisSize * innerSize + a * innerSize + inner;
        out[idx] = x.getD(idx) - logSum;
      }
    }
  }
  return Tensor.float(out, x.shape);
}

Tensor opLayerNormalization(
    Tensor x, Tensor scale, Tensor bias, int axis, double epsilon) {
  final rank = x.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final outerSize = x.shape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final normSize = x.shape.sublist(ax).fold<int>(1, (a, b) => a * b);

  final out = Float32List(x.length);
  // LayerNorm inputs are always float; direct-index the backing buffer to
  // skip getD's per-element dtype branch (this runs per token per layer).
  final xf = x.f ?? x.asFloatList();
  final sf = scale.asFloatList(), bf = bias.asFloatList();
  for (int outer = 0; outer < outerSize; outer++) {
    final base = outer * normSize;
    double mean = 0;
    for (int k = 0; k < normSize; k++) {
      mean += xf[base + k];
    }
    mean /= normSize;
    double variance = 0;
    for (int k = 0; k < normSize; k++) {
      final d = xf[base + k] - mean;
      variance += d * d;
    }
    variance /= normSize;
    final invStd = 1.0 / math.sqrt(variance + epsilon);
    for (int k = 0; k < normSize; k++) {
      out[base + k] = (xf[base + k] - mean) * invStd * sf[k] + bf[k];
    }
  }
  return Tensor.float(out, x.shape);
}

// ---------------------------------------------------------------------------
// MatMul / Gemm
// ---------------------------------------------------------------------------

/// numpy-style matmul: last two dims are the matrix dims, leading dims batch
/// (broadcastable).
Tensor opMatMul(Tensor a, Tensor b) {
  final aRank = a.shape.length, bRank = b.shape.length;
  assert(aRank >= 2 && bRank >= 2);
  final m = a.shape[aRank - 2], k = a.shape[aRank - 1];
  final k2 = b.shape[bRank - 2], n = b.shape[bRank - 1];
  assert(k == k2, 'MatMul inner dims mismatch: $k vs $k2');

  final aBatch = a.shape.sublist(0, aRank - 2);
  final bBatch = b.shape.sublist(0, bRank - 2);
  final outBatch = _broadcastShape(aBatch, bBatch);
  final outShape = [...outBatch, m, n];

  final batchCount = outBatch.fold<int>(1, (x, y) => x * y);

  // B without batch dims (a shared weight): every batch of A multiplies the
  // same matrix, and A's batches are contiguous rows — collapse them into a
  // single GEMM. Otherwise a [64,1,k] @ [k,n] would re-pack the weight 64
  // times to compute one row each (observed: 3x whole-model slowdowns).
  if (a.f != null &&
      b.f != null &&
      batchCount > 1 &&
      bBatch.fold<int>(1, (x, y) => x * y) == 1) {
    final out = Float32List(batchCount * m * n);
    gemm.matmulKernel(a.f!, 0, b.f!, 0, out, 0, batchCount * m, k, n);
    return Tensor.float(out, outShape);
  }

  final out = Float32List(batchCount * m * n);

  final aMatSize = m * k, bMatSize = k2 * n;
  final af = a.f,
      bf = b.f; // non-null in the (only actually occurring) all-float case
  for (int batch = 0; batch < batchCount; batch++) {
    final batchCoords = _unflatten(batch, outBatch);
    final aBatchFlat = _flattenBroadcast(batchCoords, aBatch);
    final bBatchFlat = _flattenBroadcast(batchCoords, bBatch);
    final aOff = aBatchFlat * aMatSize;
    final bOff = bBatchFlat * bMatSize;
    final outOff = batch * m * n;
    if (af != null && bf != null) {
      // Tiled (and, on native targets, SIMD) micro-kernel — see
      // gemm_kernel_simd.dart for the strategy.
      gemm.matmulKernel(af, aOff, bf, bOff, out, outOff, m, k, n);
    } else {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          double sum = 0;
          for (int kk = 0; kk < k; kk++) {
            sum += a.getD(aOff + i * k + kk) * b.getD(bOff + kk * n + j);
          }
          out[outOff + i * n + j] = sum;
        }
      }
    }
  }
  return Tensor.float(out, outShape);
}

/// Fused RMSNorm: `y = x * gamma / sqrt(mean(x², axis) + eps)` — the
/// 7-node Pow/ReduceMean/Add/Sqrt/Reciprocal/Mul/Mul chain emitted by
/// RMSNorm exports (Qwen-style transformers, Maia3).
Tensor opRMSNorm(Tensor x, Tensor gamma, int axis, double eps) {
  final ax = axis < 0 ? axis + x.rank : axis;
  final xf = x.f!;
  final gf = gamma.asFloatList();
  // The original chain's rank-1 Mul(gamma) broadcasts along the LAST axis
  // regardless of the reduce axis — enforce the shapes that makes valid.
  if (gf.length != x.shape[x.rank - 1]) {
    throw ArgumentError('RMSNorm gamma length ${gf.length} != last dim '
        '${x.shape[x.rank - 1]}');
  }
  final out = Float32List(xf.length);
  if (ax == x.rank - 1) {
    final d = x.shape[ax];
    final rows = xf.length ~/ d;
    for (int r = 0; r < rows; r++) {
      final base = r * d;
      double ss = 0;
      for (int j = 0; j < d; j++) {
        final v = xf[base + j];
        ss += v * v;
      }
      final inv = 1.0 / math.sqrt(ss / d + eps);
      for (int j = 0; j < d; j++) {
        out[base + j] = xf[base + j] * inv * gf[j];
      }
    }
    return Tensor.float(out, x.shape);
  }
  // General axis: strided walk.
  final dim = x.shape[ax];
  int outer = 1, inner = 1;
  for (int a = 0; a < ax; a++) {
    outer *= x.shape[a];
  }
  for (int a = ax + 1; a < x.rank; a++) {
    inner *= x.shape[a];
  }
  final lastDim = x.shape[x.rank - 1];
  for (int o = 0; o < outer; o++) {
    for (int i = 0; i < inner; i++) {
      final base = o * dim * inner + i;
      double ss = 0;
      for (int j = 0; j < dim; j++) {
        final v = xf[base + j * inner];
        ss += v * v;
      }
      final inv = 1.0 / math.sqrt(ss / dim + eps);
      for (int j = 0; j < dim; j++) {
        final idx = base + j * inner;
        out[idx] = xf[idx] * inv * gf[idx % lastDim];
      }
    }
  }
  return Tensor.float(out, x.shape);
}

/// `GroupQueryAttention` (com.microsoft) — grouped-query attention over
/// already-projected `[batch, seq, *]` Q/K/V, the fused op most modern LLM
/// exports use (Llama3 / Qwen2 / Gemma3). [numHeads] query heads share
/// [kvNumHeads] KV heads (each query head `h` attends to KV head
/// `h ~/ (numHeads/kvNumHeads)`). When [doRotary], RoPE is applied to Q and K
/// inside the op from [cosCache]/[sinCache] at positions `pastLen..pastLen+seq`.
/// [attentionBias] is the additive mask (typically the causal mask).
/// `softcap` and `local_window` are not supported.
///
/// KV cache: [pastKey]/[pastValue] (`[batch, kvNumHeads, pastLen, headSize]`)
/// are prepended to the current K/V; the op returns
/// `[output, presentKey, presentValue]` where present KV is the full
/// `[batch, kvNumHeads, pastLen+seq, headSize]` concatenation (the RoPE'd K and
/// the raw V), ready to feed back as the next step's past. Query token `i` has
/// absolute position `pastLen+i` and attends causally to keys `0..pastLen+i`.
List<Tensor> opGroupQueryAttention(Tensor q, Tensor k, Tensor v,
    {required int numHeads,
    required int kvNumHeads,
    double? scale,
    bool doRotary = false,
    bool rotaryInterleaved = false,
    Tensor? cosCache,
    Tensor? sinCache,
    Tensor? pastKey,
    Tensor? pastValue,
    Tensor? attentionBias,
    double softcap = 0,
    int localWindow = -1}) {
  if (softcap != 0 || localWindow > 0) {
    throw UnsupportedError(
        'GroupQueryAttention: softcap / local_window not supported');
  }
  final batch = q.shape[0], seq = q.shape[1];
  final headSize = q.shape[2] ~/ numHeads;
  final group = numHeads ~/ kvNumHeads;
  final kvHidden = kvNumHeads * headSize;
  final pastLen =
      (pastKey != null && pastKey.shape.length == 4) ? pastKey.shape[2] : 0;
  final totalLen = pastLen + seq;

  var qq = q, kk = k;
  if (doRotary) {
    final pos = Tensor.int64(
        Int64List.fromList([for (int s = 0; s < seq; s++) pastLen + s]),
        [1, seq]);
    qq = opRotaryEmbedding(q, pos, cosCache!, sinCache!,
        interleaved: rotaryInterleaved, numHeads: numHeads);
    kk = opRotaryEmbedding(k, pos, cosCache, sinCache,
        interleaved: rotaryInterleaved, numHeads: kvNumHeads);
  }
  final s = scale ?? (1.0 / math.sqrt(headSize));
  final qf = qq.f!, kf = kk.f!, vf = v.f ?? v.asFloatList();
  final pkf = pastKey?.f ?? pastKey?.asFloatList();
  final pvf = pastValue?.f ?? pastValue?.asFloatList();
  final bias = attentionBias?.asFloatList();
  final biasHeads = attentionBias == null ? 0 : attentionBias.shape[1];

  // present_key / present_value: [batch, kvNumHeads, totalLen, headSize],
  // the past cache (if any) followed by the current step's K/V.
  final presentK = Float32List(batch * kvNumHeads * totalLen * headSize);
  final presentV = Float32List(batch * kvNumHeads * totalLen * headSize);
  for (int b = 0; b < batch; b++) {
    for (int kvh = 0; kvh < kvNumHeads; kvh++) {
      final baseP = ((b * kvNumHeads + kvh) * totalLen) * headSize;
      for (int j = 0; j < pastLen; j++) {
        final src = ((b * kvNumHeads + kvh) * pastLen + j) * headSize;
        final dst = baseP + j * headSize;
        for (int d = 0; d < headSize; d++) {
          presentK[dst + d] = pkf![src + d];
          presentV[dst + d] = pvf![src + d];
        }
      }
      for (int j = 0; j < seq; j++) {
        final src = (b * seq + j) * kvHidden + kvh * headSize;
        final dst = baseP + (pastLen + j) * headSize;
        for (int d = 0; d < headSize; d++) {
          presentK[dst + d] = kf[src + d];
          presentV[dst + d] = vf[src + d];
        }
      }
    }
  }

  final out = Float32List(batch * seq * numHeads * headSize);
  final qh = Float32List(seq * headSize);
  final kt = Float32List(headSize * totalLen);
  final scores = Float32List(seq * totalLen);
  final ctx = Float32List(seq * headSize);

  for (int b = 0; b < batch; b++) {
    for (int h = 0; h < numHeads; h++) {
      final kvh = h ~/ group;
      final baseP = ((b * kvNumHeads + kvh) * totalLen) * headSize;
      for (int i = 0; i < seq; i++) {
        final qs = (b * seq + i) * numHeads * headSize + h * headSize;
        for (int d = 0; d < headSize; d++) {
          qh[i * headSize + d] = qf[qs + d];
        }
      }
      // K^T for this KV head: [headSize, totalLen], read from present cache.
      for (int j = 0; j < totalLen; j++) {
        final ks = baseP + j * headSize;
        for (int d = 0; d < headSize; d++) {
          kt[d * totalLen + j] = presentK[ks + d];
        }
      }
      scores.fillRange(0, seq * totalLen, 0);
      gemm.matmulKernel(qh, 0, kt, 0, scores, 0, seq, headSize, totalLen);
      final biasBase = bias == null
          ? 0
          : (b * biasHeads + (h % biasHeads)) * seq * totalLen;
      for (int i = 0; i < seq; i++) {
        final row = i * totalLen;
        // token i (absolute position pastLen+i) attends causally to keys
        // 0..pastLen+i, plus any additive bias.
        final lastKey = pastLen + i;
        double maxV = double.negativeInfinity;
        for (int j = 0; j <= lastKey; j++) {
          var sc = scores[row + j] * s;
          if (bias != null) sc += bias[biasBase + row + j];
          scores[row + j] = sc;
          if (sc > maxV) maxV = sc;
        }
        double sum = 0;
        for (int j = 0; j <= lastKey; j++) {
          final e = math.exp(scores[row + j] - maxV);
          scores[row + j] = e;
          sum += e;
        }
        final inv = 1.0 / sum;
        for (int j = 0; j <= lastKey; j++) {
          scores[row + j] *= inv;
        }
        for (int j = lastKey + 1; j < totalLen; j++) {
          scores[row + j] = 0; // masked
        }
      }
      // ctx = scores · present_V  ([seq,totalLen] · [totalLen,headSize]); the
      // per-head V slice is contiguous at baseP.
      ctx.fillRange(0, seq * headSize, 0);
      gemm.matmulKernel(scores, 0, presentV, baseP, ctx, 0, seq, totalLen,
          headSize);
      for (int i = 0; i < seq; i++) {
        final dst = (b * seq + i) * numHeads * headSize + h * headSize;
        for (int d = 0; d < headSize; d++) {
          out[dst + d] = ctx[i * headSize + d];
        }
      }
    }
  }
  return [
    Tensor.float(out, [batch, seq, numHeads * headSize]),
    Tensor.float(presentK, [batch, kvNumHeads, totalLen, headSize]),
    Tensor.float(presentV, [batch, kvNumHeads, totalLen, headSize]),
  ];
}

/// `MultiHeadAttention` (com.microsoft) — fused multi-head attention over
/// already-projected `[batch, seq, hidden]` Q/K/V. [attentionBias] (optional)
/// is an additive `[batch, heads, seq, kvSeq]` mask added to the scaled
/// scores before softmax. Q/K/V may carry different head counts already
/// expanded to match (grouped-query attention is expanded upstream here).
Tensor opMultiHeadAttention(Tensor q, Tensor k, Tensor v, Tensor? attentionBias,
    {required int numHeads, double? scale}) {
  final batch = q.shape[0], seq = q.shape[1], hidden = q.shape[2];
  final headSize = hidden ~/ numHeads;
  final kvSeq = k.shape[1];
  final s = scale ?? (1.0 / math.sqrt(headSize));
  final qf = q.f ?? q.asFloatList();
  final kf = k.f ?? k.asFloatList();
  final vf = v.f ?? v.asFloatList();
  final bias = attentionBias?.asFloatList();
  final biasHeads = attentionBias == null ? 0 : attentionBias.shape[1];

  final out = Float32List(batch * seq * hidden);
  final qh = Float32List(seq * headSize);
  final kt = Float32List(headSize * kvSeq); // K_h transposed [d, kvSeq]
  final vh = Float32List(kvSeq * headSize);
  final scores = Float32List(seq * kvSeq);
  final ctx = Float32List(seq * headSize);

  for (int b = 0; b < batch; b++) {
    for (int h = 0; h < numHeads; h++) {
      // Gather this head's Q [seq,d], Kᵀ [d,kvSeq], V [kvSeq,d].
      for (int i = 0; i < seq; i++) {
        final src = (b * seq + i) * hidden + h * headSize;
        for (int d = 0; d < headSize; d++) {
          qh[i * headSize + d] = qf[src + d];
        }
      }
      for (int j = 0; j < kvSeq; j++) {
        final src = (b * kvSeq + j) * hidden + h * headSize;
        for (int d = 0; d < headSize; d++) {
          kt[d * kvSeq + j] = kf[src + d];
          vh[j * headSize + d] = vf[src + d];
        }
      }
      // scores = Q·Kᵀ, then scale + bias + row softmax.
      scores.fillRange(0, seq * kvSeq, 0);
      gemm.matmulKernel(qh, 0, kt, 0, scores, 0, seq, headSize, kvSeq);
      final biasBase =
          bias == null ? 0 : (b * biasHeads + (h % biasHeads)) * seq * kvSeq;
      for (int i = 0; i < seq; i++) {
        final row = i * kvSeq;
        double maxV = double.negativeInfinity;
        for (int j = 0; j < kvSeq; j++) {
          var sc = scores[row + j] * s;
          if (bias != null) sc += bias[biasBase + row + j];
          scores[row + j] = sc;
          if (sc > maxV) maxV = sc;
        }
        double sum = 0;
        for (int j = 0; j < kvSeq; j++) {
          final e = math.exp(scores[row + j] - maxV);
          scores[row + j] = e;
          sum += e;
        }
        final inv = 1.0 / sum;
        for (int j = 0; j < kvSeq; j++) {
          scores[row + j] *= inv;
        }
      }
      // context = P·V, scattered back to [b, seq, h*d].
      ctx.fillRange(0, seq * headSize, 0);
      gemm.matmulKernel(scores, 0, vh, 0, ctx, 0, seq, kvSeq, headSize);
      for (int i = 0; i < seq; i++) {
        final dst = (b * seq + i) * hidden + h * headSize;
        for (int d = 0; d < headSize; d++) {
          out[dst + d] = ctx[i * headSize + d];
        }
      }
    }
  }
  return Tensor.float(out, [batch, seq, hidden]);
}

/// Fused scaled-dot-product-attention epilogue:
/// `MatMul(Softmax(MatMul(a, b) * scale + mask, axis: -1), v)`.
///
/// The scale, mask-add and row softmax happen in one pass over the attention
/// matrix (no intermediate tensors); [mask] broadcasts against the score
/// shape numpy-style.
Tensor opFusedSDPA(Tensor a, Tensor b, Tensor v, Tensor mask, double scale) {
  final s = opMatMul(a, b);
  final sf = s.f!;
  final t = s.shape.last;
  final rows = s.length ~/ t;

  final mf = mask.asFloatList();
  // Per-row mask offset via broadcast; within a row the mask stride is 1
  // (mask last dim == t) or 0 (mask last dim == 1).
  final rowShape = s.shape.sublist(0, s.rank - 1);
  final maskLastStride =
      mask.shape.isNotEmpty && mask.shape.last == t ? 1 : 0;
  final rowCoords = List<int>.filled(rowShape.length, 0);
  // Mask shape aligned against the full score shape, minus its last axis.
  final maskRowShape = mask.rank == 0
      ? const <int>[]
      : mask.shape.sublist(0, mask.rank - 1);

  for (int r = 0; r < rows; r++) {
    final base = r * t;
    final mBase = _flattenBroadcast(rowCoords, maskRowShape) *
        (mask.rank == 0 ? 0 : mask.shape.last);
    double max = double.negativeInfinity;
    for (int j = 0; j < t; j++) {
      final val = sf[base + j] * scale + mf[mBase + j * maskLastStride];
      sf[base + j] = val;
      if (val > max) max = val;
    }
    double sum = 0;
    for (int j = 0; j < t; j++) {
      final e = math.exp(sf[base + j] - max);
      sf[base + j] = e;
      sum += e;
    }
    for (int j = 0; j < t; j++) {
      sf[base + j] /= sum;
    }
    for (int k = rowShape.length - 1; k >= 0; k--) {
      if (++rowCoords[k] < rowShape[k]) break;
      rowCoords[k] = 0;
    }
  }
  return opMatMul(s, v);
}

/// Gemm: Y = alpha * A' * B' + beta * C  (A'/B' optionally transposed)
Tensor opGemm(Tensor a, Tensor b, Tensor? c,
    {double alpha = 1.0,
    double beta = 1.0,
    bool transA = false,
    bool transB = false}) {
  final at = transA ? opTranspose(a, [1, 0]) : a;
  final bt = transB ? opTranspose(b, [1, 0]) : b;
  final mm = opMatMul(at, bt);
  if (alpha != 1.0) {
    for (int k = 0; k < mm.f!.length; k++) {
      mm.f![k] *= alpha;
    }
  }
  if (c == null) return mm;
  final scaledC = beta == 1.0 ? c : _elementwiseUnary(c, (v) => v * beta);
  return opAdd(mm, scaledC);
}

// ---------------------------------------------------------------------------
// Einsum — special-cased for exactly the two equations this graph uses
// (a general einsum parser isn't needed for two known patterns).
// ---------------------------------------------------------------------------

/// General einsum over one or two operands (explicit labels; `...` is not
/// supported). The two transformer-hot equations keep their specialized
/// kernels; everything else runs the direct sum-of-products definition.
Tensor opEinsum(String equation, List<Tensor> inputs) {
  final eq = equation.replaceAll(' ', '');
  if (inputs.length == 2) {
    switch (eq) {
      case 'bhi,oi->bho':
        return _einsumBhiOi(inputs[0], inputs[1]);
      case 'bid,bjd->bij':
        return _einsumBidBjd(inputs[0], inputs[1]);
    }
  }
  if (eq.contains('...')) {
    throw UnsupportedError('Einsum: ellipsis not supported ("$equation")');
  }
  final arrowSplit = eq.split('->');
  final terms = arrowSplit[0].split(',');
  if (terms.length != inputs.length || inputs.length > 2) {
    throw UnsupportedError('Einsum: ${inputs.length} inputs vs '
        '${terms.length} terms in "$equation" (max 2 supported)');
  }

  // Label -> dimension size, checked for consistency across operands.
  final sizeOf = <String, int>{};
  final counts = <String, int>{};
  for (int t = 0; t < terms.length; t++) {
    if (terms[t].length != inputs[t].rank) {
      throw ArgumentError('Einsum term "${terms[t]}" vs rank '
          '${inputs[t].rank} operand');
    }
    for (int a = 0; a < terms[t].length; a++) {
      final l = terms[t][a];
      final d = inputs[t].shape[a];
      final prev = sizeOf[l];
      if (prev != null && prev != d && prev != 1 && d != 1) {
        throw ArgumentError('Einsum label "$l" size mismatch: $prev vs $d');
      }
      sizeOf[l] = math.max(prev ?? 1, d);
      counts.update(l, (v) => v + 1, ifAbsent: () => 1);
    }
  }
  final outTerm = arrowSplit.length > 1
      ? arrowSplit[1]
      : ([
          for (final l in (counts.keys.toList()..sort()))
            if (counts[l] == 1) l
        ].join());
  if (outTerm.split('').toSet().length != outTerm.length) {
    throw UnsupportedError(
        'Einsum: repeated output labels not supported ("$equation")');
  }
  final sumLabels = [
    for (final l in sizeOf.keys)
      if (!outTerm.contains(l)) l
  ]..sort();

  // Global coordinate order: output labels then contraction labels.
  final order = [...outTerm.split(''), ...sumLabels];
  final pos = {for (int i = 0; i < order.length; i++) order[i]: i};
  // Per input: stride contribution of each global coordinate (0 when the
  // operand broadcasts along that label).
  final contribs = <List<int>>[];
  for (int t = 0; t < terms.length; t++) {
    final c = List<int>.filled(order.length, 0);
    final strides = inputs[t].strides;
    for (int a = 0; a < terms[t].length; a++) {
      if (inputs[t].shape[a] != 1) c[pos[terms[t][a]]!] += strides[a];
    }
    contribs.add(c);
  }

  final outShape = [for (final l in outTerm.split('')) sizeOf[l]!];
  final outN = outShape.fold<int>(1, (a, b) => a * b);
  final sumShape = [for (final l in sumLabels) sizeOf[l]!];
  final sumN = sumShape.fold<int>(1, (a, b) => a * b);
  final out = Float32List(outN);
  final coords = List<int>.filled(order.length, 0);
  final nOut = outTerm.length;

  for (int oi = 0; oi < outN; oi++) {
    double acc = 0;
    for (int si = 0; si < sumN; si++) {
      double prod = 1;
      for (int t = 0; t < inputs.length; t++) {
        int flat = 0;
        final c = contribs[t];
        for (int p = 0; p < order.length; p++) {
          flat += coords[p] * c[p];
        }
        prod *= inputs[t].getD(flat);
      }
      acc += prod;
      for (int p = order.length - 1; p >= nOut; p--) {
        if (++coords[p] < sizeOf[order[p]]!) break;
        coords[p] = 0;
      }
    }
    out[oi] = acc;
    for (int p = nOut - 1; p >= 0; p--) {
      if (++coords[p] < outShape[p]) break;
      coords[p] = 0;
    }
  }
  // Output dtype follows the inputs (integer einsum stays integer).
  if (inputs.every((t) => !t.isFloat)) {
    return Tensor.int64(
        Int64List.fromList([for (final v in out) v.round()]), outShape);
  }
  return Tensor.float(out, outShape);
}

// A:[b,h,i], B:[o,i] -> out[b,h,o] = sum_i A[b,h,i]*B[o,i]  (a linear layer)
Tensor _einsumBhiOi(Tensor a, Tensor b) {
  // B arrives as [o, i]: both operands are already contiguous along the
  // contraction axis, so SIMD row dots beat a transpose+pack GEMM here.
  final bs = a.shape[0], h = a.shape[1], i = a.shape[2];
  final o = b.shape[0];
  assert(b.shape[1] == i);
  final out = Float32List(bs * h * o);
  final af = a.f!, bf = b.f!;
  for (int r = 0; r < bs * h; r++) {
    final aBase = r * i, outBase = r * o;
    for (int oi = 0; oi < o; oi++) {
      out[outBase + oi] = gemm.dotProduct(af, aBase, bf, oi * i, i);
    }
  }
  return Tensor.float(out, [bs, h, o]);
}

// A:[b,i,d], B:[b,j,d] -> out[b,i,j] = sum_d A[b,i,d]*B[b,j,d]  (attention scores)
Tensor _einsumBidBjd(Tensor a, Tensor b) {
  // Both operands contract along their contiguous last axis — SIMD row
  // dots, no transpose needed (attention scores).
  final bs = a.shape[0], i = a.shape[1], d = a.shape[2];
  final j = b.shape[1];
  assert(b.shape[0] == bs && b.shape[2] == d);
  final out = Float32List(bs * i * j);
  final af = a.f!, bf = b.f!;
  for (int bi = 0; bi < bs; bi++) {
    for (int ii = 0; ii < i; ii++) {
      final aBase = (bi * i + ii) * d;
      final outBase = (bi * i + ii) * j;
      final bBatch = bi * j * d;
      for (int ji = 0; ji < j; ji++) {
        out[outBase + ji] = gemm.dotProduct(af, aBase, bf, bBatch + ji * d, d);
      }
    }
  }
  return Tensor.float(out, [bs, i, j]);
}

// ---------------------------------------------------------------------------
// Extended op set (transformer embedders / rerankers): unary math, comparison
// and logical ops, selection, ranges and shape-fills. Added so the interpreter
// runs the common BERT / RoPE embedding & reranking graphs, not just Maia3.
// ---------------------------------------------------------------------------

Tensor opAbs(Tensor a) => a.isFloat
    ? _elementwiseUnary(a, (x) => x.abs())
    : _unaryInt(a, (x) => x.abs());
Tensor opNeg(Tensor a) =>
    a.isFloat ? _elementwiseUnary(a, (x) => -x) : _unaryInt(a, (x) => -x);
Tensor opSigmoid(Tensor a) =>
    _elementwiseUnary(a, (x) => 1.0 / (1.0 + math.exp(-x)));
Tensor opTanh(Tensor a) => _elementwiseUnary(a, _tanh);
Tensor opCos(Tensor a) => _elementwiseUnary(a, math.cos);
Tensor opSin(Tensor a) => _elementwiseUnary(a, math.sin);
Tensor opExp(Tensor a) => _elementwiseUnary(a, math.exp);
Tensor opLog(Tensor a) => _elementwiseUnary(a, math.log);

double _tanh(double x) {
  if (x > 20) return 1.0;
  if (x < -20) return -1.0;
  final e2 = math.exp(2 * x);
  return (e2 - 1) / (e2 + 1);
}

Tensor _unaryInt(Tensor a, int Function(int) op) {
  final n = a.length;
  final out = Int64List(n);
  for (int k = 0; k < n; k++) {
    out[k] = op(a.getI(k));
  }
  return Tensor.int64(out, a.shape);
}

// Comparison / logical ops. ONNX yields BOOL tensors; we carry them as int64
// 0/1 so downstream Where / Cast / And keep working with the two dtypes the
// tensor type supports.
Tensor _boolBinary(Tensor a, Tensor b, bool Function(double, double) p) {
  final same = _shapeEq(a.shape, b.shape);
  if (same || (b.length == 1 && a.rank >= b.rank)) {
    final n = a.length;
    final bScalar = b.length == 1;
    final out = Int64List(n);
    for (int k = 0; k < n; k++) {
      out[k] = p(a.getD(k), b.getD(bScalar ? 0 : k)) ? 1 : 0;
    }
    return Tensor.int64(out, a.shape);
  }
  if (a.length == 1 && b.rank >= a.rank) {
    final n = b.length;
    final out = Int64List(n);
    for (int k = 0; k < n; k++) {
      out[k] = p(a.getD(0), b.getD(k)) ? 1 : 0;
    }
    return Tensor.int64(out, b.shape);
  }
  final outShape = _broadcastShape(a.shape, b.shape);
  final n = outShape.fold<int>(1, (x, y) => x * y);
  final coords = List<int>.filled(outShape.length, 0);
  final out = Int64List(n);
  for (int idx = 0; idx < n; idx++) {
    out[idx] = p(a.getD(_flattenBroadcast(coords, a.shape)),
            b.getD(_flattenBroadcast(coords, b.shape)))
        ? 1
        : 0;
    for (int k = outShape.length - 1; k >= 0; k--) {
      if (++coords[k] < outShape[k]) break;
      coords[k] = 0;
    }
  }
  return Tensor.int64(out, outShape);
}

Tensor opEqual(Tensor a, Tensor b) => _boolBinary(a, b, (x, y) => x == y);
Tensor opGreater(Tensor a, Tensor b) => _boolBinary(a, b, (x, y) => x > y);
Tensor opLess(Tensor a, Tensor b) => _boolBinary(a, b, (x, y) => x < y);
Tensor opGreaterOrEqual(Tensor a, Tensor b) =>
    _boolBinary(a, b, (x, y) => x >= y);
Tensor opLessOrEqual(Tensor a, Tensor b) => _boolBinary(a, b, (x, y) => x <= y);
Tensor opAnd(Tensor a, Tensor b) =>
    _boolBinary(a, b, (x, y) => x != 0 && y != 0);
Tensor opOr(Tensor a, Tensor b) =>
    _boolBinary(a, b, (x, y) => x != 0 || y != 0);
Tensor opNot(Tensor a) => _unaryInt(a, (x) => x != 0 ? 0 : 1);

/// `IsNaN` / `IsInf` — bool (int64 0/1) mask over a float tensor. Integer
/// tensors are never NaN/Inf, so the result is all-zero.
Tensor opIsNaN(Tensor a) {
  final n = a.length;
  final out = Int64List(n);
  if (a.isFloat) {
    final af = a.f!;
    for (int k = 0; k < n; k++) {
      out[k] = af[k].isNaN ? 1 : 0;
    }
  }
  return Tensor.int64(out, a.shape);
}

Tensor opIsInf(Tensor a, {bool detectPositive = true, bool detectNegative = true}) {
  final n = a.length;
  final out = Int64List(n);
  if (a.isFloat) {
    final af = a.f!;
    for (int k = 0; k < n; k++) {
      final v = af[k];
      out[k] = (v.isInfinite &&
              ((v > 0 && detectPositive) || (v < 0 && detectNegative)))
          ? 1
          : 0;
    }
  }
  return Tensor.int64(out, a.shape);
}

Tensor opMax(List<Tensor> ins) =>
    ins.reduce((a, b) => _elementwiseBinary(a, b, math.max));
Tensor opMin(List<Tensor> ins) =>
    ins.reduce((a, b) => _elementwiseBinary(a, b, math.min));

/// Element-wise select: `cond ? a : b`, broadcasting all three inputs.
Tensor opWhere(Tensor cond, Tensor a, Tensor b) {
  final outShape =
      _broadcastShape(_broadcastShape(cond.shape, a.shape), b.shape);
  final n = outShape.fold<int>(1, (x, y) => x * y);
  final bothInt = !a.isFloat && !b.isFloat;
  final coords = List<int>.filled(outShape.length, 0);

  double pick() {
    final c = cond.getD(_flattenBroadcast(coords, cond.shape));
    return c != 0
        ? a.getD(_flattenBroadcast(coords, a.shape))
        : b.getD(_flattenBroadcast(coords, b.shape));
  }

  void advance() {
    for (int k = outShape.length - 1; k >= 0; k--) {
      if (++coords[k] < outShape[k]) return;
      coords[k] = 0;
    }
  }

  if (bothInt) {
    final out = Int64List(n);
    for (int idx = 0; idx < n; idx++) {
      out[idx] = pick().round();
      advance();
    }
    return Tensor.int64(out, outShape);
  }
  final out = Float32List(n);
  for (int idx = 0; idx < n; idx++) {
    out[idx] = pick();
    advance();
  }
  return Tensor.float(out, outShape);
}

/// `Size` — total element count as an int64 scalar.
Tensor opSize(Tensor x) => Tensor.scalarInt(x.length);

/// `RotaryEmbedding` (com.microsoft / onnx opset 23) — fused rotary position
/// embedding. [x] is `[batch, seq, hidden]` (3-D) or
/// `[batch, heads, seq, headSize]` (4-D); [cosCache]/[sinCache] are
/// `[maxPos, rotaryDim/2]`, indexed by [positionIds] `[batch, seq]`. Only the
/// first `rotaryDim` of each head is rotated (the tail passes through);
/// `interleaved` selects the GPT-J pairing, else the GPT-NeoX rotate-half.
Tensor opRotaryEmbedding(Tensor x, Tensor positionIds, Tensor cosCache,
    Tensor sinCache,
    {bool interleaved = false, int numHeads = 0, int rotaryEmbeddingDim = 0}) {
  final xf = x.f ?? x.asFloatList();
  final cos = cosCache.asFloatList(), sin = sinCache.asFloatList();
  final pos = positionIds.asIntList();
  final halfRot = cosCache.shape.last;
  final rotaryDim = rotaryEmbeddingDim > 0 ? rotaryEmbeddingDim : 2 * halfRot;

  int batch, seq, heads, headSize;
  if (x.rank == 4) {
    batch = x.shape[0];
    heads = x.shape[1];
    seq = x.shape[2];
    headSize = x.shape[3];
  } else {
    batch = x.shape[0];
    seq = x.shape[1];
    final hidden = x.shape[2];
    // num_heads=0 → infer from a full-rotation head (headSize == rotaryDim).
    headSize = numHeads > 0 ? hidden ~/ numHeads : rotaryDim;
    heads = hidden ~/ headSize;
  }
  final cosStride = halfRot; // cos/sin row length per position

  final out = Float32List(xf.length);
  // Layout: 4-D is [b,h,s,d] contiguous; 3-D is [b,s,h*d] contiguous.
  for (int b = 0; b < batch; b++) {
    for (int s = 0; s < seq; s++) {
      final p = pos[(positionIds.rank == 1 ? 0 : b) * seq + s];
      final cBase = p * cosStride;
      for (int h = 0; h < heads; h++) {
        final base = x.rank == 4
            ? ((b * heads + h) * seq + s) * headSize
            : (b * seq + s) * heads * headSize + h * headSize;
        // Rotated portion.
        if (interleaved) {
          for (int i = 0; i < rotaryDim ~/ 2; i++) {
            final c = cos[cBase + i], sn = sin[cBase + i];
            final x0 = xf[base + 2 * i], x1 = xf[base + 2 * i + 1];
            out[base + 2 * i] = x0 * c - x1 * sn;
            out[base + 2 * i + 1] = x1 * c + x0 * sn;
          }
        } else {
          final half = rotaryDim ~/ 2;
          for (int i = 0; i < half; i++) {
            final c = cos[cBase + i], sn = sin[cBase + i];
            final x0 = xf[base + i], x1 = xf[base + i + half];
            out[base + i] = x0 * c - x1 * sn;
            out[base + i + half] = x1 * c + x0 * sn;
          }
        }
        // Pass-through tail (headSize beyond rotaryDim).
        for (int i = rotaryDim; i < headSize; i++) {
          out[base + i] = xf[base + i];
        }
      }
    }
  }
  return Tensor.float(out, x.shape);
}

/// `RandomNormalLike` / `RandomNormal` — a Gaussian-noise tensor. A
/// pseudo-random draw can never match ORT's Mersenne-Twister stream across
/// implementations, so this returns the distribution *mean* (a deterministic
/// fill). That is exact for the only parity-testable case — models that scale
/// the noise to zero (e.g. VITS/Piper with `noise_scale = 0`) — and finite
/// otherwise. Genuinely stochastic sampling is not reproduced.
Tensor opRandomNormalFill(List<int> shape, double mean) =>
    Tensor.filledFloat(shape, mean);

/// `ArgMax` / `ArgMin` along [axis]; ties resolve to the first (or, with
/// [selectLastIndex], the last) occurrence.
Tensor opArgMinMax(Tensor x, int axis, bool keepdims,
    {required bool isMax, bool selectLastIndex = false}) {
  final ax = axis < 0 ? axis + x.rank : axis;
  final dim = x.shape[ax];
  int outer = 1, inner = 1;
  for (int a = 0; a < ax; a++) {
    outer *= x.shape[a];
  }
  for (int a = ax + 1; a < x.rank; a++) {
    inner *= x.shape[a];
  }
  final out = Int64List(outer * inner);
  for (int o = 0; o < outer; o++) {
    for (int i = 0; i < inner; i++) {
      final base = o * dim * inner + i;
      int bestIdx = 0;
      double best = x.getD(base);
      // NaN handling matches ORT's CPU kernel (pinned by the argmax_nan
      // fixture): plain comparisons, so a NaN never wins unless it is the
      // initial element — numpy/the ONNX reference differ here, ORT is the
      // oracle.
      for (int d = 1; d < dim; d++) {
        final v = x.getD(base + d * inner);
        final better = isMax
            ? (selectLastIndex ? v >= best : v > best)
            : (selectLastIndex ? v <= best : v < best);
        if (better) {
          best = v;
          bestIdx = d;
        }
      }
      out[o * inner + i] = bestIdx;
    }
  }
  final shape = [
    for (int a = 0; a < x.rank; a++)
      if (a != ax) x.shape[a] else if (keepdims) 1
  ];
  return Tensor.int64(out, shape);
}

/// `NonZero` — indices of nonzero elements as `[rank, count]` int64,
/// scanning in row-major order.
Tensor opNonZero(Tensor x) {
  final rank = math.max(x.rank, 1);
  final hits = <int>[];
  for (int k = 0; k < x.length; k++) {
    if (x.getD(k) != 0) hits.add(k);
  }
  final strides = x.strides;
  final out = Int64List(rank * hits.length);
  for (int j = 0; j < hits.length; j++) {
    int rem = hits[j];
    for (int a = 0; a < x.rank; a++) {
      out[a * hits.length + j] = rem ~/ strides[a];
      rem %= strides[a];
    }
  }
  return Tensor.int64(out, [rank, hits.length]);
}

/// `TopK` — the [k] largest (or smallest) values along [axis], sorted, with
/// ties broken toward the lower index. Returns `[values, indices]`.
List<Tensor> opTopK(Tensor x, int k, {int axis = -1, bool largest = true}) {
  final ax = axis < 0 ? axis + x.rank : axis;
  final dim = x.shape[ax];
  // Computed from the shape, not by division — TopK over an empty axis
  // (zero candidates after a filter) is legal and returns empty outputs.
  int outer = 1, inner = 1;
  for (int a = 0; a < ax; a++) {
    outer *= x.shape[a];
  }
  for (int a = ax + 1; a < x.rank; a++) {
    inner *= x.shape[a];
  }
  final outShape = [...x.shape]..[ax] = k;
  final values = Float32List(outer * k * inner);
  final indices = Int64List(outer * k * inner);
  final order = List<int>.generate(dim, (i) => i);
  for (int o = 0; o < outer; o++) {
    for (int i = 0; i < inner; i++) {
      final base = o * dim * inner + i;
      for (int d = 0; d < dim; d++) {
        order[d] = d;
      }
      order.sort((p, q) {
        final vp = x.getD(base + p * inner), vq = x.getD(base + q * inner);
        if (vp != vq) return largest ? vq.compareTo(vp) : vp.compareTo(vq);
        return p.compareTo(q);
      });
      final outBase = o * k * inner + i;
      for (int d = 0; d < k; d++) {
        values[outBase + d * inner] = x.getD(base + order[d] * inner);
        indices[outBase + d * inner] = order[d];
      }
    }
  }
  final vT = Tensor.float(values, outShape);
  return [
    x.isFloat ? vT : Tensor.int64(Int64List.fromList([for (final v in values) v.toInt()]), outShape),
    Tensor.int64(indices, outShape),
  ];
}

/// `NonMaxSuppression` — greedy per-(batch, class) suppression. Returns
/// `[num_selected, 3]` int64 rows of (batch, class, boxIndex), in
/// batch-major, class-major, descending-score order (matching ORT).
Tensor opNonMaxSuppression(Tensor boxes, Tensor scores,
    {int maxOutputBoxesPerClass = 0,
    double iouThreshold = 0,
    double? scoreThreshold,
    bool centerPointBox = false}) {
  final nBatch = boxes.shape[0], nBox = boxes.shape[1];
  final nClass = scores.shape[1];
  final bf = boxes.asFloatList(), sf = scores.asFloatList();
  final selected = <int>[];
  if (maxOutputBoxesPerClass > 0) {
    // Corner coords may come in either order; normalize with min/max.
    (double, double, double, double) corners(int b, int i) {
      final o = (b * nBox + i) * 4;
      if (centerPointBox) {
        final cx = bf[o], cy = bf[o + 1], w = bf[o + 2], h = bf[o + 3];
        return (cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2);
      }
      final y1 = math.min(bf[o], bf[o + 2]);
      final y2 = math.max(bf[o], bf[o + 2]);
      final x1 = math.min(bf[o + 1], bf[o + 3]);
      final x2 = math.max(bf[o + 1], bf[o + 3]);
      return (y1, x1, y2, x2);
    }

    double iou(int b, int i, int j) {
      final (iy1, ix1, iy2, ix2) = corners(b, i);
      final (jy1, jx1, jy2, jx2) = corners(b, j);
      final w = math.min(ix2, jx2) - math.max(ix1, jx1);
      final h = math.min(iy2, jy2) - math.max(iy1, jy1);
      if (w <= 0 || h <= 0) return 0;
      final inter = w * h;
      final areaI = (ix2 - ix1) * (iy2 - iy1);
      final areaJ = (jx2 - jx1) * (jy2 - jy1);
      return inter / (areaI + areaJ - inter);
    }

    final order = List<int>.generate(nBox, (i) => i);
    for (int b = 0; b < nBatch; b++) {
      for (int c = 0; c < nClass; c++) {
        final sBase = (b * nClass + c) * nBox;
        for (int i = 0; i < nBox; i++) {
          order[i] = i;
        }
        order.sort((p, q) {
          final d = sf[sBase + q].compareTo(sf[sBase + p]);
          return d != 0 ? d : p.compareTo(q);
        });
        final picked = <int>[];
        for (final cand in order) {
          if (picked.length >= maxOutputBoxesPerClass) break;
          final s = sf[sBase + cand];
          if (scoreThreshold != null && s <= scoreThreshold) break;
          var keep = true;
          for (final p in picked) {
            if (iou(b, p, cand) > iouThreshold) {
              keep = false;
              break;
            }
          }
          if (keep) {
            picked.add(cand);
            selected.addAll([b, c, cand]);
          }
        }
      }
    }
  }
  return Tensor.int64(Int64List.fromList(selected), [selected.length ~/ 3, 3]);
}

/// `Trilu` — keeps the upper (or lower) triangle of the last two dims,
/// zeroing the rest; [k] shifts the diagonal.
Tensor opTrilu(Tensor x, {bool upper = true, int k = 0}) {
  final rank = x.rank;
  final rows = x.shape[rank - 2], cols = x.shape[rank - 1];
  final batch = x.length ~/ (rows * cols);
  if (x.isFloat) {
    final out = Float32List(x.length);
    final xf = x.f!;
    for (int b = 0; b < batch; b++) {
      final base = b * rows * cols;
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          final keep = upper ? j - i >= k : j - i <= k;
          if (keep) out[base + i * cols + j] = xf[base + i * cols + j];
        }
      }
    }
    return Tensor.float(out, x.shape);
  }
  final out = Int64List(x.length);
  final xi = x.intData;
  for (int b = 0; b < batch; b++) {
    final base = b * rows * cols;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        final keep = upper ? j - i >= k : j - i <= k;
        if (keep) out[base + i * cols + j] = xi[base + i * cols + j];
      }
    }
  }
  return Tensor.int64(out, x.shape);
}

/// `ScatterND` — copy of [data] with [updates] written at [indices]
/// (index tuples over the leading dims; each update fills the remaining
/// trailing block). No reduction (last write wins, per the default).
Tensor opScatterND(Tensor data, Tensor indices, Tensor updates) {
  final idxDepth = indices.shape.last;
  final nIdx = indices.length ~/ idxDepth;
  final strides = data.strides;
  int blockLen = 1;
  for (int a = idxDepth; a < data.rank; a++) {
    blockLen *= data.shape[a];
  }
  final ii = indices.asIntList();
  if (data.isFloat) {
    final out = Float32List.fromList(data.f!);
    final uf = updates.asFloatList();
    for (int t = 0; t < nIdx; t++) {
      int off = 0;
      for (int d = 0; d < idxDepth; d++) {
        var v = ii[t * idxDepth + d];
        if (v < 0) v += data.shape[d];
        off += v * strides[d];
      }
      out.setRange(off, off + blockLen, uf, t * blockLen);
    }
    return Tensor.float(out, data.shape);
  }
  final out = Int64List.fromList(data.asIntList());
  final ui = updates.asIntList();
  for (int t = 0; t < nIdx; t++) {
    int off = 0;
    for (int d = 0; d < idxDepth; d++) {
      var v = ii[t * idxDepth + d];
      if (v < 0) v += data.shape[d];
      off += v * strides[d];
    }
    out.setRange(off, off + blockLen, Int64List.sublistView(ui, t * blockLen, (t + 1) * blockLen));
  }
  return Tensor.int64(out, data.shape);
}

/// `Tile` — repeats [x] along each axis: `out[c] = x[c % shape]`.
Tensor opTile(Tensor x, List<int> repeats) {
  final rank = x.rank;
  final outShape = [for (int a = 0; a < rank; a++) x.shape[a] * repeats[a]];
  final n = outShape.fold<int>(1, (a, b) => a * b);
  final srcStrides = x.strides;
  final coords = List<int>.filled(rank, 0);
  final isFloat = x.isFloat;
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);
  final src = isFloat ? null : x.intData;
  for (int idx = 0; idx < n; idx++) {
    int off = 0;
    for (int a = 0; a < rank; a++) {
      off += (coords[a] % x.shape[a]) * srcStrides[a];
    }
    if (isFloat) {
      outF![idx] = x.f![off];
    } else {
      outI![idx] = src![off];
    }
    for (int a = rank - 1; a >= 0; a--) {
      if (++coords[a] < outShape[a]) break;
      coords[a] = 0;
    }
  }
  return isFloat
      ? Tensor.float(outF!, outShape)
      : Tensor.int64(outI!, outShape);
}

// ---------------------------------------------------------------------------
// Quantization (QDQ format)
// ---------------------------------------------------------------------------

/// Round half to even (banker's rounding), as the ONNX quantization ops
/// require — Dart's `round()` rounds half away from zero.
int _roundEven(double v) {
  final f = v.floorToDouble();
  final frac = v - f;
  if (frac > 0.5) return f.toInt() + 1;
  if (frac < 0.5) return f.toInt();
  final i = f.toInt();
  return i.isEven ? i : i + 1;
}

/// Per-axis channel index of flat element [k]: scale/zero-point entry to use
/// when they are 1-D along [axis].
int _channelOf(int k, List<int> shape, int axis) {
  int stride = 1;
  for (int a = shape.length - 1; a > axis; a--) {
    stride *= shape[a];
  }
  return (k ~/ stride) % shape[axis];
}

/// `QuantizeLinear`: `y = saturate(roundEven(x / scale) + zeroPoint)`,
/// per-tensor or per-axis. [lo]/[hi] are the saturation bounds of the output
/// type (from the zero-point tensor's dtype: uint8 → 0..255, int8 →
/// -128..127).
Tensor opQuantizeLinear(Tensor x, Tensor scale, Tensor? zeroPoint,
    {int axis = 1, required int lo, required int hi}) {
  final n = x.length;
  // Compact output: int8 when the range is signed, uint8 otherwise.
  final signed = lo < 0;
  final outI8 = signed ? Int8List(n) : null;
  final outU8 = signed ? null : Uint8List(n);
  final perAxis = scale.length > 1;
  final ax = axis < 0 ? axis + x.rank : axis;
  final sf = scale.asFloatList();
  final zp = zeroPoint?.asIntList();
  for (int k = 0; k < n; k++) {
    final c = perAxis ? _channelOf(k, x.shape, ax) : 0;
    final q = _roundEven(x.getD(k) / sf[c]) + (zp == null ? 0 : zp[c]);
    final v = q < lo ? lo : (q > hi ? hi : q);
    if (signed) {
      outI8![k] = v;
    } else {
      outU8![k] = v;
    }
  }
  return signed
      ? Tensor.int8(outI8!, x.shape)
      : Tensor.uint8(outU8!, x.shape);
}

/// `DequantizeLinear`: `y = (x - zeroPoint) * scale`, per-tensor or per-axis.
Tensor opDequantizeLinear(Tensor x, Tensor scale, Tensor? zeroPoint,
    {int axis = 1}) {
  final n = x.length;
  final out = Float32List(n);
  final perAxis = scale.length > 1;
  final ax = axis < 0 ? axis + x.rank : axis;
  final sf = scale.asFloatList();
  final zp = zeroPoint?.asIntList();
  final xi = x.intData;
  for (int k = 0; k < n; k++) {
    final c = perAxis ? _channelOf(k, x.shape, ax) : 0;
    out[k] = (xi[k] - (zp == null ? 0 : zp[c])) * sf[c];
  }
  return Tensor.float(out, x.shape);
}

/// `DynamicQuantizeLinear`: computes uint8 scale/zero-point from the data
/// range (always spanning 0) and quantizes. Returns `[y, scale, zeroPoint]`.
List<Tensor> opDynamicQuantizeLinear(Tensor x) {
  final xf = x.asFloatList();
  double mn = 0, mx = 0; // range must include 0 per spec
  for (final v in xf) {
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  // The reference computes scale and zero point in float32 — matching that
  // exactly matters, because a scale differing in the last bit shifts every
  // quantized value (the downstream integer math then diverges everywhere).
  final f32 = Float32List(1);
  f32[0] = (mx - mn) / 255.0;
  final scale = f32[0].toDouble();
  f32[0] = scale == 0 ? 0 : -mn / scale;
  final zpF = f32[0].toDouble();
  final zp = _roundEven(zpF.clamp(0.0, 255.0));
  final out = Uint8List(xf.length);
  for (int k = 0; k < xf.length; k++) {
    f32[0] = scale == 0 ? 0 : xf[k] / scale;
    final q = scale == 0 ? zp : _roundEven(f32[0].toDouble()) + zp;
    out[k] = q < 0 ? 0 : (q > 255 ? 255 : q);
  }
  return [
    Tensor.uint8(out, x.shape),
    Tensor.scalarFloat(scale),
    Tensor.scalarInt(zp),
  ];
}

/// `Pad` — constant / reflect / edge modes over any rank. [pads] is
/// `[begin_0..begin_r, end_0..end_r]`; with [axes] (opset 18+) it lists only
/// the entries for those axes, in the same begin-then-end layout.
Tensor opPad(Tensor x, List<int> pads, {
  String mode = 'constant',
  double constantValue = 0,
  List<int>? axes,
}) {
  final rank = x.rank;
  final beg = List<int>.filled(rank, 0);
  final end = List<int>.filled(rank, 0);
  if (axes == null) {
    for (int a = 0; a < rank; a++) {
      beg[a] = pads[a];
      end[a] = pads[rank + a];
    }
  } else {
    for (int k = 0; k < axes.length; k++) {
      final a = axes[k] < 0 ? axes[k] + rank : axes[k];
      beg[a] = pads[k];
      end[a] = pads[axes.length + k];
    }
  }
  final outShape = [
    for (int a = 0; a < rank; a++) x.shape[a] + beg[a] + end[a]
  ];
  final n = outShape.fold<int>(1, (a, b) => a * b);
  final srcStrides = x.strides;
  final coords = List<int>.filled(rank, 0);
  final isFloat = x.isFloat;
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);

  for (int idx = 0; idx < n; idx++) {
    int srcOff = 0;
    bool inside = true;
    for (int a = 0; a < rank; a++) {
      int i = coords[a] - beg[a];
      final dim = x.shape[a];
      if (i < 0 || i >= dim) {
        switch (mode) {
          case 'reflect':
            // Mirror without repeating the border sample.
            while (i < 0 || i >= dim) {
              if (i < 0) i = -i;
              if (i >= dim) i = 2 * dim - 2 - i;
            }
          case 'edge':
            i = i < 0 ? 0 : dim - 1;
          default: // constant
            inside = false;
        }
      }
      if (!inside) break;
      srcOff += i * srcStrides[a];
    }
    if (isFloat) {
      outF![idx] = inside ? x.f![srcOff] : constantValue;
    } else {
      outI![idx] = inside ? x.intData[srcOff] : constantValue.toInt();
    }
    for (int a = rank - 1; a >= 0; a--) {
      if (++coords[a] < outShape[a]) break;
      coords[a] = 0;
    }
  }
  return isFloat
      ? Tensor.float(outF!, outShape)
      : Tensor.int64(outI!, outShape);
}

/// `Range(start, limit, delta)` — a 1-D sequence, int64 if the bounds are int.
Tensor opRange(Tensor start, Tensor limit, Tensor delta) {
  final s = start.getD(0), l = limit.getD(0), d = delta.getD(0);
  final n = math.max(0, ((l - s) / d).ceil());
  final bothInt = !start.isFloat && !delta.isFloat;
  if (bothInt) {
    final out = Int64List(n);
    for (int k = 0; k < n; k++) {
      out[k] = (start.getI(0) + k * delta.getI(0));
    }
    return Tensor.int64(out, [n]);
  }
  final out = Float32List(n);
  for (int k = 0; k < n; k++) {
    out[k] = s + k * d;
  }
  return Tensor.float(out, [n]);
}

/// `ConstantOfShape(shape)` filled with [value] (a 1-element tensor; default 0).
Tensor opConstantOfShape(Tensor shapeT, Tensor? value) {
  final shape = shapeT.asIntList().toList();
  final n = shape.fold<int>(1, (a, b) => a * b);
  if (value != null && !value.isFloat) {
    final v = value.getI(0);
    return Tensor.int64(Int64List(n)..fillRange(0, n, v), shape);
  }
  final v = value != null ? value.getD(0) : 0.0;
  return Tensor.float(Float32List(n)..fillRange(0, n, v), shape);
}

/// `ReduceSum` over [axes] (all axes if null), matching [opReduceMean] shape
/// semantics but summing rather than averaging.
Tensor opReduceSum(Tensor x, List<int>? axes, bool keepdims) {
  final rank = x.shape.length;
  final ax = (axes ?? List<int>.generate(rank, (k) => k))
      .map((a) => a < 0 ? a + rank : a)
      .toSet();
  final outShapeFull = [
    for (int k = 0; k < rank; k++) ax.contains(k) ? 1 : x.shape[k]
  ];
  final n = outShapeFull.fold<int>(1, (a, b) => a * b);
  final sums = Float64List(n);
  final outStridesFull = Tensor.filledFloat(outShapeFull, 0).strides;
  for (int idx = 0; idx < x.length; idx++) {
    final coords = _unflatten(idx, x.shape);
    int outFlat = 0;
    for (int k = 0; k < rank; k++) {
      outFlat += (ax.contains(k) ? 0 : coords[k]) * outStridesFull[k];
    }
    sums[outFlat] += x.getD(idx);
  }
  final out = Float32List(n);
  for (int k = 0; k < n; k++) {
    out[k] = sums[k];
  }
  if (keepdims) return Tensor.float(out, outShapeFull);
  return Tensor.float(out, [
    for (int k = 0; k < rank; k++)
      if (!ax.contains(k)) x.shape[k]
  ]);
}

/// `GatherElements` — output has the shape of [indices]; each element picks
/// `data` at that coordinate with the [axis] index replaced by the index value.
Tensor opGatherElements(Tensor data, Tensor indices, int axis) {
  final rank = data.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final dStrides = data.strides;
  final n = indices.length;
  final isFloat = data.isFloat;
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);
  for (int idx = 0; idx < n; idx++) {
    final coords = _unflatten(idx, indices.shape);
    var j = indices.getI(idx);
    if (j < 0) j += data.shape[ax];
    coords[ax] = j;
    int flat = 0;
    for (int k = 0; k < rank; k++) {
      flat += coords[k] * dStrides[k];
    }
    if (isFloat) {
      outF![idx] = data.f![flat];
    } else {
      outI![idx] = data.intData[flat];
    }
  }
  return isFloat
      ? Tensor.float(outF!, indices.shape)
      : Tensor.int64(outI!, indices.shape);
}

/// `CumSum` along [axis] (inclusive, forward by default; [exclusive] shifts the
/// sum, [reverse] accumulates from the end) — used e.g. to derive position ids
/// from an attention mask.
Tensor opCumSum(Tensor x, int axis,
    {bool exclusive = false, bool reverse = false}) {
  final rank = x.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final axisStride = x.strides[ax];
  final axisSize = x.shape[ax];
  final n = x.length;
  final isFloat = x.isFloat;
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);
  // Float accumulation rounds to float32 at every step, matching ORT —
  // long cumulative sums (e.g. vocoder phase over tens of thousands of
  // samples) otherwise drift from the oracle and chaos-amplify through
  // sin() downstream.
  final f32 = Float32List(1);
  for (int i = 0; i < n; i++) {
    if ((i ~/ axisStride) % axisSize != 0) continue; // only line starts
    var accF = 0.0;
    var accI = 0;
    for (int s = 0; s < axisSize; s++) {
      final k = reverse ? axisSize - 1 - s : s;
      final idx = i + k * axisStride;
      final v = x.getD(idx);
      if (exclusive) {
        if (isFloat) {
          outF![idx] = accF;
        } else {
          outI![idx] = accI;
        }
      }
      f32[0] = accF + v;
      accF = f32[0];
      accI += v.round();
      if (!exclusive) {
        if (isFloat) {
          outF![idx] = accF;
        } else {
          outI![idx] = accI;
        }
      }
    }
  }
  return isFloat ? Tensor.float(outF!, x.shape) : Tensor.int64(outI!, x.shape);
}
