/// Implementations of the standard ONNX operators (per the public ONNX
/// operator specification, https://onnx.ai/onnx/operators/) used by
/// transformer embedding / reranking graphs. Mechanical execution only — each
/// op is run the same way any ONNX runtime would, with no knowledge of why the
/// graph is shaped the way it is.
library;

import 'dart:math' as math;
import 'dart:typed_data';

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

  // General broadcasting path (reuses a single mutable coordinate buffer
  // instead of allocating a fresh List<int> per output element).
  final outShape = _broadcastShape(a.shape, b.shape);
  final n = outShape.fold<int>(1, (x, y) => x * y);
  final coords = List<int>.filled(outShape.length, 0);

  double runOne() {
    final av = a.getD(_flattenBroadcast(coords, a.shape));
    final bv = b.getD(_flattenBroadcast(coords, b.shape));
    return op(av, bv);
  }

  void advance() {
    for (int k = outShape.length - 1; k >= 0; k--) {
      coords[k]++;
      if (coords[k] < outShape[k]) return;
      coords[k] = 0;
    }
  }

  if (bothInt) {
    final out = Int64List(n);
    for (int idx = 0; idx < n; idx++) {
      out[idx] = runOne().round();
      advance();
    }
    return Tensor.int64(out, outShape);
  }
  final out = Float32List(n);
  for (int idx = 0; idx < n; idx++) {
    out[idx] = runOne();
    advance();
  }
  return Tensor.float(out, outShape);
}

// ---------------------------------------------------------------------------
// Elementwise ops
// ---------------------------------------------------------------------------

Tensor opAdd(Tensor a, Tensor b) => _elementwiseBinary(a, b, (x, y) => x + y);
Tensor opSub(Tensor a, Tensor b) => _elementwiseBinary(a, b, (x, y) => x - y);
Tensor opMul(Tensor a, Tensor b) => _elementwiseBinary(a, b, (x, y) => x * y);
Tensor opDiv(Tensor a, Tensor b) => _elementwiseBinary(a, b, (x, y) => x / y);
Tensor opPow(Tensor a, Tensor b) =>
    _elementwiseBinary(a, b, (x, y) => math.pow(x, y).toDouble());

Tensor _elementwiseUnary(Tensor a, double Function(double) op) {
  final out = Float32List(a.length);
  for (int k = 0; k < a.length; k++) {
    out[k] = op(a.getD(k));
  }
  return Tensor.float(out, a.shape);
}

Tensor opSqrt(Tensor a) => _elementwiseUnary(a, (x) => math.sqrt(x));
Tensor opReciprocal(Tensor a) => _elementwiseUnary(a, (x) => 1.0 / x);
Tensor opRelu(Tensor a) => _elementwiseUnary(a, (x) => x < 0 ? 0.0 : x);

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

Tensor opErf(Tensor a) => _elementwiseUnary(a, _erf);

Tensor opClip(Tensor x, Tensor? min, Tensor? max) {
  final lo = min != null ? min.getD(0) : double.negativeInfinity;
  final hi = max != null ? max.getD(0) : double.infinity;
  return _elementwiseUnary(x, (v) => v.clamp(lo, hi));
}

// ---------------------------------------------------------------------------
// Cast
// ---------------------------------------------------------------------------

Tensor opCast(Tensor x, int to) {
  // ONNX TensorProto.DataType: 1=FLOAT, 6=INT32, 7=INT64, 9=BOOL, 11=DOUBLE.
  // We carry both int32 and int64 as our int64 tensor, and bool as int64 0/1.
  if (to == 9) {
    final out = Int64List(x.length);
    for (int k = 0; k < x.length; k++) {
      out[k] = x.getD(k) != 0 ? 1 : 0;
    }
    return Tensor.int64(out, x.shape);
  }
  if (to == 6 || to == 7) return Tensor.int64(x.asIntList(), x.shape);
  return Tensor.float(x.asFloatList(), x.shape);
}

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

Tensor opShape(Tensor x) =>
    Tensor.int64(Int64List.fromList(x.shape), [x.shape.length]);

Tensor opReshape(Tensor x, Tensor shapeT) => x.reshape(shapeT.asIntList());

Tensor opTranspose(Tensor x, List<int> perm) {
  final newShape = [for (final p in perm) x.shape[p]];
  final n = x.length;
  final oldStrides = x.strides;
  if (x.isFloat) {
    final out = Float32List(n);
    for (int idx = 0; idx < n; idx++) {
      final newCoords = _unflatten(idx, newShape);
      int oldFlat = 0;
      for (int k = 0; k < perm.length; k++) {
        oldFlat += newCoords[k] * oldStrides[perm[k]];
      }
      out[idx] = x.f![oldFlat];
    }
    return Tensor.float(out, newShape);
  } else {
    final out = Int64List(n);
    for (int idx = 0; idx < n; idx++) {
      final newCoords = _unflatten(idx, newShape);
      int oldFlat = 0;
      for (int k = 0; k < perm.length; k++) {
        oldFlat += newCoords[k] * oldStrides[perm[k]];
      }
      out[idx] = x.i![oldFlat];
    }
    return Tensor.int64(out, newShape);
  }
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
  return x.isFloat
      ? Tensor.float(x.f!, newShape)
      : Tensor.int64(x.i!, newShape);
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
  return x.isFloat
      ? Tensor.float(x.f!, newShape)
      : Tensor.int64(x.i!, newShape);
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
          outI![dstStart + k] = t.i![srcStart + k];
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
          outI![outPos + k] = data.i![srcStart + k];
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
        outI![t * innerSize + j] = data.i![dataFlatStart + j];
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
      out[idx] = x.i![_flattenBroadcast(_unflatten(idx, outShape), x.shape)];
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
      outI![idx] = x.i![srcFlat];
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

Tensor opSoftmax(Tensor x, int axis) {
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
        final idx = outer * axisSize * innerSize + a * innerSize + inner;
        final v = x.getD(idx);
        if (v > maxV) maxV = v;
      }
      double sum = 0;
      for (int a = 0; a < axisSize; a++) {
        final idx = outer * axisSize * innerSize + a * innerSize + inner;
        final e = math.exp(x.getD(idx) - maxV);
        out[idx] = e;
        sum += e;
      }
      for (int a = 0; a < axisSize; a++) {
        final idx = outer * axisSize * innerSize + a * innerSize + inner;
        out[idx] = out[idx] / sum;
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
  for (int outer = 0; outer < outerSize; outer++) {
    final base = outer * normSize;
    double mean = 0;
    for (int k = 0; k < normSize; k++) {
      mean += x.getD(base + k);
    }
    mean /= normSize;
    double variance = 0;
    for (int k = 0; k < normSize; k++) {
      final d = x.getD(base + k) - mean;
      variance += d * d;
    }
    variance /= normSize;
    final invStd = 1.0 / math.sqrt(variance + epsilon);
    for (int k = 0; k < normSize; k++) {
      final normalized = (x.getD(base + k) - mean) * invStd;
      out[base + k] = normalized * scale.getD(k) + bias.getD(k);
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
      // Direct-indexed, i-k-j loop order: for each (i,kk) we sweep the whole
      // b-row and out-row contiguously, which is far more cache-friendly
      // than the naive i-j-k order (and avoids getD's per-element dtype
      // branch) — this is what actually made inference usable (~20s -> ms).
      for (int i = 0; i < m; i++) {
        final outRow = outOff + i * n;
        final aRow = aOff + i * k;
        for (int kk = 0; kk < k; kk++) {
          final aVal = af[aRow + kk];
          if (aVal == 0) continue;
          final bRow = bOff + kk * n;
          for (int j = 0; j < n; j++) {
            out[outRow + j] += aVal * bf[bRow + j];
          }
        }
      }
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

Tensor opEinsum(String equation, Tensor a, Tensor b) {
  switch (equation) {
    case 'bhi,oi->bho':
      return _einsumBhiOi(a, b);
    case 'bid,bjd->bij':
      return _einsumBidBjd(a, b);
    default:
      throw UnsupportedError('Einsum equation "$equation" not implemented');
  }
}

// A:[b,h,i], B:[o,i] -> out[b,h,o] = sum_i A[b,h,i]*B[o,i]  (a linear layer)
Tensor _einsumBhiOi(Tensor a, Tensor b) {
  final bs = a.shape[0], h = a.shape[1], i = a.shape[2];
  final o = b.shape[0];
  assert(b.shape[1] == i);
  final out = Float32List(bs * h * o);
  final af = a.f!, bf = b.f!;
  for (int bi = 0; bi < bs; bi++) {
    for (int hi = 0; hi < h; hi++) {
      final aBase = (bi * h + hi) * i;
      final outBase = (bi * h + hi) * o;
      for (int oi = 0; oi < o; oi++) {
        double sum = 0;
        final bBase = oi * i;
        for (int ii = 0; ii < i; ii++) {
          sum += af[aBase + ii] * bf[bBase + ii];
        }
        out[outBase + oi] = sum;
      }
    }
  }
  return Tensor.float(out, [bs, h, o]);
}

// A:[b,i,d], B:[b,j,d] -> out[b,i,j] = sum_d A[b,i,d]*B[b,j,d]  (attention scores)
Tensor _einsumBidBjd(Tensor a, Tensor b) {
  final bs = a.shape[0], i = a.shape[1], d = a.shape[2];
  final j = b.shape[1];
  assert(b.shape[0] == bs && b.shape[2] == d);
  final out = Float32List(bs * i * j);
  final af = a.f!, bf = b.f!;
  for (int bi = 0; bi < bs; bi++) {
    for (int ii = 0; ii < i; ii++) {
      final aBase = (bi * i + ii) * d;
      final outBase = (bi * i + ii) * j;
      for (int ji = 0; ji < j; ji++) {
        double sum = 0;
        final bBase = (bi * j + ji) * d;
        for (int di = 0; di < d; di++) {
          sum += af[aBase + di] * bf[bBase + di];
        }
        out[outBase + ji] = sum;
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
  final out = Int64List(a.length);
  for (int k = 0; k < a.length; k++) {
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
      outI![idx] = data.i![flat];
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
        accF += v;
        accI += v.round();
      } else {
        accF += v;
        accI += v.round();
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
