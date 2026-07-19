/// Convolution, pooling, normalization and resize operators (per the public
/// ONNX operator specification, https://onnx.ai/onnx/operators/) — the op
/// family used by CNN vision models. Like `onnx_ops.dart`: mechanical
/// spec execution, no model-specific assumptions.
///
/// Layout note: all ops here take NC* tensors (batch, channels, then 1–3
/// spatial dims), which is the only layout ONNX defines for them.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'gemm_kernel_scalar.dart' if (dart.library.ffi) 'gemm_kernel_simd.dart'
    as gemm;
import 'tensor.dart';

// ---------------------------------------------------------------------------
// Shared spatial-window helpers
// ---------------------------------------------------------------------------

/// Resolves `auto_pad`/`pads` into explicit begin/end pads per spatial axis
/// and the resulting output dims.
///
/// Returns (pads: [beg0..begN, end0..endN], outDims). [ceilMode] applies the
/// pooling `ceil_mode=1` output rounding (Conv always floors).
(List<int>, List<int>) _resolvePads({
  required List<int> inDims,
  required List<int> kernel,
  required List<int> strides,
  required List<int> dilations,
  required String autoPad,
  List<int>? pads,
  bool ceilMode = false,
}) {
  final nd = inDims.length;
  final p = List<int>.filled(2 * nd, 0);
  final out = List<int>.filled(nd, 0);
  for (int k = 0; k < nd; k++) {
    final window = dilations[k] * (kernel[k] - 1) + 1;
    switch (autoPad) {
      case 'SAME_UPPER':
      case 'SAME_LOWER':
        out[k] = (inDims[k] + strides[k] - 1) ~/ strides[k];
        final total =
            math.max(0, (out[k] - 1) * strides[k] + window - inDims[k]);
        final small = total ~/ 2;
        // SAME_UPPER puts the extra padding at the end, SAME_LOWER at the
        // beginning.
        p[k] = autoPad == 'SAME_UPPER' ? small : total - small;
        p[nd + k] = total - p[k];
      case 'VALID':
        out[k] = (inDims[k] - window) ~/ strides[k] + 1;
      default: // NOTSET
        p[k] = pads == null ? 0 : pads[k];
        p[nd + k] = pads == null ? 0 : pads[nd + k];
        final span = inDims[k] + p[k] + p[nd + k] - window;
        out[k] =
            (ceilMode ? (span + strides[k] - 1) ~/ strides[k] : span ~/ strides[k]) +
                1;
        if (ceilMode) {
          // Spec: the last window must start inside the input or begin-pad.
          if ((out[k] - 1) * strides[k] >= inDims[k] + p[k]) out[k]--;
        }
    }
  }
  return (p, out);
}

int _prod(List<int> v) => v.fold(1, (a, b) => a * b);

/// Round half to even (ORT's nearbyint semantics for GridSample nearest).
int _roundEvenNn(double v) {
  final f = v.floorToDouble();
  final frac = v - f;
  if (frac > 0.5) return f.toInt() + 1;
  if (frac < 0.5) return f.toInt();
  final i = f.toInt();
  return i.isEven ? i : i + 1;
}

/// Output spatial dims for a Conv with these parameters (used by the
/// executor to plan the isolate-pool band split without running the conv).
List<int> convOutputSpatial(List<int> inSp, List<int> kernel,
    List<int>? strides, List<int>? pads, List<int>? dilations,
    String autoPad) {
  final nd = inSp.length;
  final (_, outSp) = _resolvePads(
      inDims: inSp,
      kernel: kernel,
      strides: strides ?? List.filled(nd, 1),
      dilations: dilations ?? List.filled(nd, 1),
      autoPad: autoPad,
      pads: pads);
  return outSp;
}

// ---------------------------------------------------------------------------
// Conv
// ---------------------------------------------------------------------------

/// `Conv` — N-dimensional convolution (1–3 spatial dims), with strides, pads,
/// dilations, groups (incl. depthwise) and auto_pad. Naive direct loops with
/// a specialized 2D path; the im2col fast path is planned separately.
Tensor opConv(
  Tensor x,
  Tensor w,
  Tensor? bias, {
  List<int>? strides,
  List<int>? pads,
  List<int>? dilations,
  int group = 1,
  String autoPad = 'NOTSET',
  int? bandStart,
  int? bandEnd,
}) {
  final nd = x.rank - 2;
  assert(nd >= 1 && nd <= 3, 'Conv supports 1-3 spatial dims, got rank ${x.rank}');
  assert(w.rank == x.rank, 'Conv weight rank must match input rank');
  if (nd == 1) {
    // 1-D convs ride the 2-D im2col/GEMM machinery (the generic N-D path
    // walks coordinates per element — ~100x slower on vocoder-scale data).
    final y = opConv(
      x.reshape([x.shape[0], x.shape[1], 1, x.shape[2]]),
      w.reshape([w.shape[0], w.shape[1], 1, w.shape[2]]),
      bias,
      strides: [1, strides?.first ?? 1],
      pads: pads == null ? null : [0, pads[0], 0, pads[1]],
      dilations: [1, dilations?.first ?? 1],
      group: group,
      autoPad: autoPad,
    );
    return y.reshape([y.shape[0], y.shape[1], y.shape[3]]);
  }
  final n = x.shape[0], cIn = x.shape[1];
  final m = w.shape[0], cPerGroup = w.shape[1];
  assert(cIn == cPerGroup * group,
      'Conv channel mismatch: input C=$cIn, weight C/g=$cPerGroup, group=$group');
  final inSp = x.shape.sublist(2);
  final kernel = w.shape.sublist(2);
  final s = strides ?? List.filled(nd, 1);
  final d = dilations ?? List.filled(nd, 1);
  final (p, outSp) = _resolvePads(
      inDims: inSp,
      kernel: kernel,
      strides: s,
      dilations: d,
      autoPad: autoPad,
      pads: pads);

  final xf = x.asFloatList(), wf = w.asFloatList();
  final bf = bias?.asFloatList();
  final mPerGroup = m ~/ group;
  // Output-row band (2-D only): the isolate pool splits a conv by output
  // rows; each call computes rows [b0, b1) and returns that slab.
  final b0 = bandStart ?? 0;
  assert(nd == 2 || (bandStart == null && bandEnd == null),
      'Conv banding is only supported for 2-D convs');
  final out = Float32List(nd == 2
      ? n * m * ((bandEnd ?? outSp[0]) - b0) * outSp[1]
      : n * m * _prod(outSp));

  if (nd == 2) {
    final h = inSp[0], wd = inSp[1];
    final kh = kernel[0], kw = kernel[1];
    final oh = (bandEnd ?? outSp[0]) - b0, ow = outSp[1];
    final sh = s[0], sw = s[1], dh = d[0], dw = d[1], ph = p[0], pw = p[1];

    // Depthwise (1 in / 1 out channel per group): im2col+GEMM degenerates to
    // m=1 matmuls where packing dominates — the direct loop wins.
    final depthwise = cPerGroup == 1 && mPerGroup == 1;

    if (!depthwise) {
      // 1x1/stride-1/no-pad conv is exactly a GEMM over the existing layout.
      // Banded calls skip the pointwise identity: with a partial output the
      // GEMM view's column count (band*ow) no longer matches the channel
      // stride (h*w), so the direct-view trick would misindex — im2col
      // handles the band correctly.
      final pointwise = kh == 1 &&
          kw == 1 &&
          sh == 1 &&
          sw == 1 &&
          ph == 0 &&
          pw == 0 &&
          p[2] == 0 &&
          p[3] == 0 &&
          bandStart == null &&
          bandEnd == null;
      final colRows = cPerGroup * kh * kw;
      final colN = oh * ow;
      final cols = pointwise ? null : Float32List(colRows * colN);
      for (int b = 0; b < n; b++) {
        for (int g = 0; g < group; g++) {
          final xGroupBase = (b * cIn + g * cPerGroup) * h * wd;
          if (cols != null) {
            // im2col: row r=(c,ky,kx) holds x values for every output pos.
            cols.fillRange(0, cols.length, 0);
            for (int c = 0; c < cPerGroup; c++) {
              final xBase = xGroupBase + c * h * wd;
              for (int ky = 0; ky < kh; ky++) {
                for (int kx = 0; kx < kw; kx++) {
                  final row = ((c * kh + ky) * kw + kx) * colN;
                  for (int oy = 0; oy < oh; oy++) {
                    final iy = (oy + b0) * sh - ph + ky * dh;
                    if (iy < 0 || iy >= h) continue;
                    final xRow = xBase + iy * wd;
                    final colRow = row + oy * ow;
                    // Tight ox range with ix = ox*sw - pw + kx*dw in [0, wd).
                    final off = kx * dw - pw;
                    int ox0 = off < 0 ? ((-off + sw - 1) ~/ sw) : 0;
                    int ox1 = ow;
                    while (ox1 > ox0 && (ox1 - 1) * sw + off >= wd) {
                      ox1--;
                    }
                    for (int ox = ox0; ox < ox1; ox++) {
                      cols[colRow + ox] = xf[xRow + ox * sw + off];
                    }
                  }
                }
              }
            }
          }
          final outOff = (b * m + g * mPerGroup) * colN;
          // Weight rows are already [mPerGroup × colRows] contiguously.
          // Pointwise convs read the input rows matching the output band.
          gemm.matmulKernel(wf, g * mPerGroup * colRows, cols ?? xf,
              cols == null ? xGroupBase + b0 * wd : 0, out, outOff, mPerGroup,
              colRows, colN);
        }
      }
      if (bf != null) {
        for (int b = 0; b < n; b++) {
          for (int om = 0; om < m; om++) {
            final base = (b * m + om) * colN;
            final bias0 = bf[om];
            for (int k = 0; k < colN; k++) {
              out[base + k] += bias0;
            }
          }
        }
      }
      return Tensor.float(out, [n, m, oh, ow]);
    }

    // Depthwise: copy each channel once into a zero-padded buffer, then run
    // completely branchless accumulation loops (the bounds checks in the
    // naive loop cost more than the padded copy).
    final hp = h + ph + p[2], wp = wd + pw + p[3];
    final xpad = Float32List(hp * wp);
    for (int b = 0; b < n; b++) {
      for (int om = 0; om < m; om++) {
        final xBase = (b * cIn + om) * h * wd;
        xpad.fillRange(0, xpad.length, 0);
        for (int y = 0; y < h; y++) {
          xpad.setRange((y + ph) * wp + pw, (y + ph) * wp + pw + wd, xf,
              xBase + y * wd);
        }
        final acc0 = bf == null ? 0.0 : bf[om];
        final wBase = om * kh * kw;
        final outBase = (b * m + om) * oh * ow;
        for (int oy = 0; oy < oh; oy++) {
          final iyRow = (oy + b0) * sh * wp;
          final oRow = outBase + oy * ow;
          int ox = 0;
          // 4-wide unroll over output x for instruction-level parallelism.
          for (; ox + 4 <= ow; ox += 4) {
            final i0 = iyRow + ox * sw;
            double a0 = acc0, a1 = acc0, a2 = acc0, a3 = acc0;
            for (int ky = 0; ky < kh; ky++) {
              final xRow = i0 + ky * dh * wp;
              final wRow = wBase + ky * kw;
              for (int kx = 0; kx < kw; kx++) {
                final wv = wf[wRow + kx];
                final xi = xRow + kx * dw;
                a0 += xpad[xi] * wv;
                a1 += xpad[xi + sw] * wv;
                a2 += xpad[xi + 2 * sw] * wv;
                a3 += xpad[xi + 3 * sw] * wv;
              }
            }
            out[oRow + ox] = a0;
            out[oRow + ox + 1] = a1;
            out[oRow + ox + 2] = a2;
            out[oRow + ox + 3] = a3;
          }
          for (; ox < ow; ox++) {
            double acc = acc0;
            final i0 = iyRow + ox * sw;
            for (int ky = 0; ky < kh; ky++) {
              final xRow = i0 + ky * dh * wp;
              final wRow = wBase + ky * kw;
              for (int kx = 0; kx < kw; kx++) {
                acc += xpad[xRow + kx * dw] * wf[wRow + kx];
              }
            }
            out[oRow + ox] = acc;
          }
        }
      }
    }
    return Tensor.float(out, [n, m, oh, ow]);
  }

  // Generic 1D/3D path: walk output and kernel positions via coordinate
  // arrays. Slower per element but these ranks are rare and small.
  final outSpN = _prod(outSp);
  final kN = _prod(kernel);
  final inStrides = List<int>.filled(nd, 1);
  for (int k = nd - 2; k >= 0; k--) {
    inStrides[k] = inStrides[k + 1] * inSp[k + 1];
  }
  final oCoord = List<int>.filled(nd, 0);
  final kCoord = List<int>.filled(nd, 0);
  for (int b = 0; b < n; b++) {
    for (int om = 0; om < m; om++) {
      final g = om ~/ mPerGroup;
      final acc0 = bias == null ? 0.0 : bias.asFloatList()[om];
      final outBase = (b * m + om) * outSpN;
      oCoord.fillRange(0, nd, 0);
      for (int oi = 0; oi < outSpN; oi++) {
        double acc = acc0;
        for (int c = 0; c < cPerGroup; c++) {
          final xBase = (b * cIn + g * cPerGroup + c) * _prod(inSp);
          final wBase = (om * cPerGroup + c) * kN;
          kCoord.fillRange(0, nd, 0);
          for (int ki = 0; ki < kN; ki++) {
            int xOff = 0;
            bool inside = true;
            for (int a = 0; a < nd; a++) {
              final ia = oCoord[a] * s[a] - p[a] + kCoord[a] * d[a];
              if (ia < 0 || ia >= inSp[a]) {
                inside = false;
                break;
              }
              xOff += ia * inStrides[a];
            }
            if (inside) acc += xf[xBase + xOff] * wf[wBase + ki];
            for (int a = nd - 1; a >= 0; a--) {
              if (++kCoord[a] < kernel[a]) break;
              kCoord[a] = 0;
            }
          }
        }
        out[outBase + oi] = acc;
        for (int a = nd - 1; a >= 0; a--) {
          if (++oCoord[a] < outSp[a]) break;
          oCoord[a] = 0;
        }
      }
    }
  }
  return Tensor.float(out, [n, m, ...outSp]);
}

/// `ConvTranspose` — N-dimensional transposed convolution via output scatter:
/// every input element distributes through the kernel into the output.
/// Weight layout per spec: `[C_in, M/group, *kernel]`.
Tensor opConvTranspose(
  Tensor x,
  Tensor w,
  Tensor? bias, {
  List<int>? strides,
  List<int>? pads,
  List<int>? dilations,
  List<int>? outputPadding,
  List<int>? outputShape,
  int group = 1,
}) {
  final nd = x.rank - 2;
  final n = x.shape[0], cIn = x.shape[1];
  final inSp = x.shape.sublist(2);
  final kernel = w.shape.sublist(2);
  final mPerGroup = w.shape[1];
  final m = mPerGroup * group;
  final cPerGroup = cIn ~/ group;
  final s = strides ?? List.filled(nd, 1);
  final d = dilations ?? List.filled(nd, 1);
  final op = outputPadding ?? List.filled(nd, 0);
  final p = List<int>.from(pads ?? List.filled(2 * nd, 0));

  // 1-D fast path: y[m, f*s + k*d - pad] += (Wᵀ·X)[m*K + k, f] — the heavy
  // contraction over channels runs on the GEMM kernel; the scatter is plain
  // adds. (The generic path below walks per-element with inner loops; on
  // vocoder overlap-adds that is seconds per call.)
  if (nd == 1 && outputShape == null && n == 1) {
    final f = inSp[0], k = kernel[0];
    final outW = (f - 1) * s[0] + d[0] * (k - 1) + 1 + op[0] - p[0] - p[1];
    final xf = x.asFloatList(), wf = w.asFloatList();
    final out = Float32List(m * outW);
    final mk = mPerGroup * k;
    // wT[g]: [mPerGroup*k, cPerGroup] from W [C, M/g, K]
    final wT = Float32List(mk * cPerGroup);
    final prod = Float32List(mk * f);
    for (int g = 0; g < group; g++) {
      for (int c = 0; c < cPerGroup; c++) {
        final wBase = ((g * cPerGroup + c) * mPerGroup) * k;
        for (int mm = 0; mm < mPerGroup; mm++) {
          for (int kk = 0; kk < k; kk++) {
            wT[(mm * k + kk) * cPerGroup + c] = wf[wBase + mm * k + kk];
          }
        }
      }
      prod.fillRange(0, prod.length, 0);
      gemm.matmulKernel(
          wT, 0, xf, g * cPerGroup * f, prod, 0, mk, cPerGroup, f);
      for (int mm = 0; mm < mPerGroup; mm++) {
        final oBase = (g * mPerGroup + mm) * outW;
        for (int kk = 0; kk < k; kk++) {
          final pBase = (mm * k + kk) * f;
          final shift = kk * d[0] - p[0];
          for (int ff = 0; ff < f; ff++) {
            final t = ff * s[0] + shift;
            if (t >= 0 && t < outW) out[oBase + t] += prod[pBase + ff];
          }
        }
      }
    }
    if (bias != null) {
      final bf = bias.asFloatList();
      for (int mm = 0; mm < m; mm++) {
        final base = mm * outW;
        for (int t = 0; t < outW; t++) {
          out[base + t] += bf[mm];
        }
      }
    }
    return Tensor.float(out, [1, m, outW]);
  }

  // 2-D fast path: the channel contraction is one GEMM (Wᵀ·X), then a col2im
  // overlap-add scatter into the output. The generic per-element path below is
  // O(C·H·W·kH·kW·M) with cache-hostile scatter — seconds per call on the
  // Demucs / VAE decoders; this makes the heavy work a SIMD GEMM.
  if (nd == 2 && outputShape == null && n == 1) {
    final hIn = inSp[0], wIn = inSp[1], kH = kernel[0], kW = kernel[1];
    final outH = (hIn - 1) * s[0] + d[0] * (kH - 1) + 1 + op[0] - p[0] - p[2];
    final outW = (wIn - 1) * s[1] + d[1] * (kW - 1) + 1 + op[1] - p[1] - p[3];
    final xf = x.asFloatList(), wf = w.asFloatList();
    final out = Float32List(m * outH * outW);
    final hw = hIn * wIn, mkk = mPerGroup * kH * kW;
    final wT = Float32List(mkk * cPerGroup);
    final prod = Float32List(mkk * hw);
    for (int g = 0; g < group; g++) {
      // Pack Wᵀ[g]: [mPerGroup*kH*kW, cPerGroup] from W [C, M/g, kH, kW].
      for (int c = 0; c < cPerGroup; c++) {
        final wBase = ((g * cPerGroup + c) * mPerGroup) * kH * kW;
        for (int idx = 0; idx < mkk; idx++) {
          wT[idx * cPerGroup + c] = wf[wBase + idx];
        }
      }
      prod.fillRange(0, prod.length, 0);
      gemm.matmulKernel(
          wT, 0, xf, g * cPerGroup * hw, prod, 0, mkk, cPerGroup, hw);
      // col2im: overlap-add each (mm, kh, kw) plane into the output.
      for (int mm = 0; mm < mPerGroup; mm++) {
        final oBase = (g * mPerGroup + mm) * outH * outW;
        for (int kh = 0; kh < kH; kh++) {
          final ohShift = kh * d[0] - p[0];
          for (int kw = 0; kw < kW; kw++) {
            final owShift = kw * d[1] - p[1];
            final pBase = ((mm * kH + kh) * kW + kw) * hw;
            for (int ih = 0; ih < hIn; ih++) {
              final oh = ih * s[0] + ohShift;
              if (oh < 0 || oh >= outH) continue;
              final pRow = pBase + ih * wIn, oRow = oBase + oh * outW;
              for (int iw = 0; iw < wIn; iw++) {
                final ow = iw * s[1] + owShift;
                if (ow >= 0 && ow < outW) out[oRow + ow] += prod[pRow + iw];
              }
            }
          }
        }
      }
    }
    if (bias != null) {
      final bff = bias.asFloatList();
      final plane = outH * outW;
      for (int mm = 0; mm < m; mm++) {
        final base = mm * plane, bv = bff[mm];
        for (int t = 0; t < plane; t++) {
          out[base + t] += bv;
        }
      }
    }
    return Tensor.float(out, [1, m, outH, outW]);
  }

  final outSp = List<int>.filled(nd, 0);
  for (int k = 0; k < nd; k++) {
    final full = (inSp[k] - 1) * s[k] + d[k] * (kernel[k] - 1) + 1 + op[k];
    if (outputShape != null) {
      // Spec: distribute (full - requested) as begin/end pads, extra at end.
      // output_shape may list only the spatial dims or the full N,C,*spatial
      // shape (ORT accepts both) — use the last `nd` entries either way.
      outSp[k] = outputShape[outputShape.length - nd + k];
      // Spec formula for auto-generated pads (default/NOTSET case): the
      // larger half goes at the beginning.
      final total = full - outSp[k];
      p[nd + k] = total ~/ 2;
      p[k] = total - p[nd + k];
    } else {
      outSp[k] = full - p[k] - p[nd + k];
    }
  }

  final xf = x.asFloatList(), wf = w.asFloatList();
  final bf = bias?.asFloatList();
  final outSpN = _prod(outSp), inSpN = _prod(inSp), kN = _prod(kernel);
  final out = Float32List(n * m * outSpN);

  final outStrides = List<int>.filled(nd, 1);
  for (int k = nd - 2; k >= 0; k--) {
    outStrides[k] = outStrides[k + 1] * outSp[k + 1];
  }
  final iCoord = List<int>.filled(nd, 0);
  final kCoord = List<int>.filled(nd, 0);

  for (int b = 0; b < n; b++) {
    for (int c = 0; c < cIn; c++) {
      final g = c ~/ cPerGroup;
      final xBase = (b * cIn + c) * inSpN;
      iCoord.fillRange(0, nd, 0);
      for (int ii = 0; ii < inSpN; ii++) {
        final v = xf[xBase + ii];
        if (v != 0) {
          kCoord.fillRange(0, nd, 0);
          for (int ki = 0; ki < kN; ki++) {
            int oOff = 0;
            bool inside = true;
            for (int a = 0; a < nd; a++) {
              final oa = iCoord[a] * s[a] + kCoord[a] * d[a] - p[a];
              if (oa < 0 || oa >= outSp[a]) {
                inside = false;
                break;
              }
              oOff += oa * outStrides[a];
            }
            if (inside) {
              final wBase = (c * mPerGroup) * kN;
              for (int mm = 0; mm < mPerGroup; mm++) {
                out[(b * m + g * mPerGroup + mm) * outSpN + oOff] +=
                    v * wf[wBase + mm * kN + ki];
              }
            }
            for (int a = nd - 1; a >= 0; a--) {
              if (++kCoord[a] < kernel[a]) break;
              kCoord[a] = 0;
            }
          }
        }
        for (int a = nd - 1; a >= 0; a--) {
          if (++iCoord[a] < inSp[a]) break;
          iCoord[a] = 0;
        }
      }
    }
  }
  if (bf != null) {
    for (int b = 0; b < n; b++) {
      for (int om = 0; om < m; om++) {
        final base = (b * m + om) * outSpN;
        for (int k = 0; k < outSpN; k++) {
          out[base + k] += bf[om];
        }
      }
    }
  }
  return Tensor.float(out, [n, m, ...outSp]);
}

// ---------------------------------------------------------------------------
// Pooling
// ---------------------------------------------------------------------------

Tensor _pool(
  Tensor x, {
  required List<int> kernel,
  required bool isMax,
  List<int>? strides,
  List<int>? pads,
  List<int>? dilations,
  String autoPad = 'NOTSET',
  bool ceilMode = false,
  bool countIncludePad = false,
}) {
  final nd = x.rank - 2;
  final n = x.shape[0], c = x.shape[1];
  final inSp = x.shape.sublist(2);
  final s = strides ?? List.filled(nd, 1);
  final d = dilations ?? List.filled(nd, 1);
  final (p, outSp) = _resolvePads(
      inDims: inSp,
      kernel: kernel,
      strides: s,
      dilations: d,
      autoPad: autoPad,
      pads: pads,
      ceilMode: ceilMode);

  final xf = x.asFloatList();
  final outSpN = _prod(outSp), inSpN = _prod(inSp), kN = _prod(kernel);
  final out = Float32List(n * c * outSpN);
  final inStrides = List<int>.filled(nd, 1);
  for (int k = nd - 2; k >= 0; k--) {
    inStrides[k] = inStrides[k + 1] * inSp[k + 1];
  }
  final oCoord = List<int>.filled(nd, 0);
  final kCoord = List<int>.filled(nd, 0);

  for (int img = 0; img < n * c; img++) {
    final xBase = img * inSpN;
    final outBase = img * outSpN;
    oCoord.fillRange(0, nd, 0);
    for (int oi = 0; oi < outSpN; oi++) {
      double best = double.negativeInfinity;
      double sum = 0;
      int count = 0;
      kCoord.fillRange(0, nd, 0);
      for (int ki = 0; ki < kN; ki++) {
        int xOff = 0;
        bool inside = true;
        for (int a = 0; a < nd; a++) {
          final ia = oCoord[a] * s[a] - p[a] + kCoord[a] * d[a];
          if (ia < 0 || ia >= inSp[a]) {
            inside = false;
            break;
          }
          xOff += ia * inStrides[a];
        }
        if (inside) {
          final v = xf[xBase + xOff];
          if (v > best) best = v;
          sum += v;
          count++;
        }
        for (int a = nd - 1; a >= 0; a--) {
          if (++kCoord[a] < kernel[a]) break;
          kCoord[a] = 0;
        }
      }
      out[outBase + oi] =
          isMax ? best : sum / (countIncludePad ? kN : math.max(count, 1));
      for (int a = nd - 1; a >= 0; a--) {
        if (++oCoord[a] < outSp[a]) break;
        oCoord[a] = 0;
      }
    }
  }
  return Tensor.float(out, [n, c, ...outSp]);
}

Tensor opMaxPool(Tensor x,
        {required List<int> kernel,
        List<int>? strides,
        List<int>? pads,
        List<int>? dilations,
        String autoPad = 'NOTSET',
        bool ceilMode = false}) =>
    _pool(x,
        kernel: kernel,
        isMax: true,
        strides: strides,
        pads: pads,
        dilations: dilations,
        autoPad: autoPad,
        ceilMode: ceilMode);

Tensor opAveragePool(Tensor x,
        {required List<int> kernel,
        List<int>? strides,
        List<int>? pads,
        String autoPad = 'NOTSET',
        bool ceilMode = false,
        bool countIncludePad = false}) =>
    _pool(x,
        kernel: kernel,
        isMax: false,
        strides: strides,
        pads: pads,
        autoPad: autoPad,
        ceilMode: ceilMode,
        countIncludePad: countIncludePad);

/// Global pooling: reduce every spatial dim to 1 (shape [N, C, 1, ...]).
Tensor opGlobalPool(Tensor x, {required bool isMax}) {
  final n = x.shape[0], c = x.shape[1];
  final spN = _prod(x.shape.sublist(2));
  final xf = x.asFloatList();
  final out = Float32List(n * c);
  for (int img = 0; img < n * c; img++) {
    final base = img * spN;
    if (isMax) {
      double best = double.negativeInfinity;
      for (int k = 0; k < spN; k++) {
        if (xf[base + k] > best) best = xf[base + k];
      }
      out[img] = best;
    } else {
      double sum = 0;
      for (int k = 0; k < spN; k++) {
        sum += xf[base + k];
      }
      out[img] = sum / spN;
    }
  }
  return Tensor.float(out, [n, c, ...List.filled(x.rank - 2, 1)]);
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// `BatchNormalization` (inference mode): per-channel
/// `y = scale * (x - mean) / sqrt(var + eps) + bias`, folded into one
/// multiply-add per element.
Tensor opBatchNormalization(Tensor x, Tensor scale, Tensor bias, Tensor mean,
    Tensor variance, double epsilon) {
  final c = x.shape[1];
  final spN = _prod(x.shape.sublist(2));
  final n = x.shape[0];
  final sf = scale.asFloatList(),
      bf = bias.asFloatList(),
      mf = mean.asFloatList(),
      vf = variance.asFloatList();
  final a = Float32List(c), b = Float32List(c);
  for (int k = 0; k < c; k++) {
    a[k] = sf[k] / math.sqrt(vf[k] + epsilon);
    b[k] = bf[k] - mf[k] * a[k];
  }
  final xf = x.asFloatList();
  final out = Float32List(xf.length);
  for (int img = 0; img < n; img++) {
    for (int ch = 0; ch < c; ch++) {
      final base = (img * c + ch) * spN;
      final ac = a[ch], bc = b[ch];
      for (int k = 0; k < spN; k++) {
        out[base + k] = xf[base + k] * ac + bc;
      }
    }
  }
  return Tensor.float(out, x.shape);
}

/// `InstanceNormalization`: GroupNormalization with one group per channel
/// (mean/variance over each (n, c) instance's spatial dims).
Tensor opInstanceNormalization(
        Tensor x, Tensor scale, Tensor bias, double epsilon) =>
    opGroupNormalization(x, scale, bias, x.shape[1], epsilon);

/// `GroupNormalization` (opset 18+): mean/variance per (batch, group), then
/// per-channel scale and bias.
Tensor opGroupNormalization(
    Tensor x, Tensor scale, Tensor bias, int numGroups, double epsilon) {
  final n = x.shape[0], c = x.shape[1];
  final spN = _prod(x.shape.sublist(2));
  final cPerGroup = c ~/ numGroups;
  final groupLen = cPerGroup * spN;
  final sf = scale.asFloatList(), bf = bias.asFloatList();
  // Opset 18-20 defines scale/bias per GROUP; opset 21+ per channel.
  final perGroupAffine = sf.length == numGroups && numGroups != c;
  if (!perGroupAffine && sf.length != c) {
    throw ArgumentError('GroupNormalization scale length ${sf.length} '
        'matches neither num_groups=$numGroups nor channels=$c');
  }
  final xf = x.asFloatList();
  final out = Float32List(xf.length);
  for (int img = 0; img < n; img++) {
    for (int g = 0; g < numGroups; g++) {
      final base = (img * c + g * cPerGroup) * spN;
      double sum = 0;
      for (int k = 0; k < groupLen; k++) {
        sum += xf[base + k];
      }
      final mean = sum / groupLen;
      double sq = 0;
      for (int k = 0; k < groupLen; k++) {
        final d = xf[base + k] - mean;
        sq += d * d;
      }
      final inv = 1.0 / math.sqrt(sq / groupLen + epsilon);
      for (int cc = 0; cc < cPerGroup; cc++) {
        final ch = g * cPerGroup + cc;
        final affIdx = perGroupAffine ? g : ch;
        final cBase = base + cc * spN;
        final a = sf[affIdx] * inv;
        final b = bf[affIdx] - mean * a;
        for (int k = 0; k < spN; k++) {
          out[cBase + k] = xf[cBase + k] * a + b;
        }
      }
    }
  }
  return Tensor.float(out, x.shape);
}

/// `GridSample` (2-D): samples [x] `[N,C,H,W]` at normalized grid
/// coordinates `[N,Ho,Wo,2]` (x then y in [-1, 1]). Bilinear or nearest,
/// zeros / border / reflection padding.
Tensor opGridSample(Tensor x, Tensor grid,
    {String mode = 'linear',
    String paddingMode = 'zeros',
    bool alignCorners = false}) {
  if (x.rank != 4 || grid.rank != 4) {
    throw UnsupportedError('GridSample: only 2-D (rank-4) inputs supported, '
        'got input rank ${x.rank} / grid rank ${grid.rank}');
  }
  if (mode != 'linear' && mode != 'bilinear' && mode != 'nearest') {
    throw UnsupportedError('GridSample: mode "$mode" not supported');
  }
  final n = x.shape[0], c = x.shape[1], h = x.shape[2], w = x.shape[3];
  final ho = grid.shape[1], wo = grid.shape[2];
  final xf = x.asFloatList(), gf = grid.asFloatList();
  final out = Float32List(n * c * ho * wo);

  double unnormalize(double v, int size) => alignCorners
      ? (v + 1) / 2 * (size - 1)
      : ((v + 1) * size - 1) / 2;

  double reflect(double v, int size) {
    if (size == 1) return 0;
    final span = alignCorners ? 2.0 * (size - 1) : 2.0 * size;
    final low = alignCorners ? 0.0 : -0.5;
    var t = (v - low) % span;
    if (t < 0) t += span;
    if (t > span / 2) t = span - t;
    return t + low;
  }

  // Per-pixel geometry (padding transform, corner indices, weights) is
  // channel-invariant — compute it once, then loop channels with cheap
  // gathers. -1 flat offsets mark zero-padded taps.
  int flatOrPad(int iy, int ix) {
    if (paddingMode == 'border' || paddingMode == 'reflection') {
      iy = iy.clamp(0, h - 1);
      ix = ix.clamp(0, w - 1);
    } else if (iy < 0 || iy >= h || ix < 0 || ix >= w) {
      return -1; // zeros padding
    }
    return iy * w + ix;
  }

  final planeSize = h * w;
  for (int b = 0; b < n; b++) {
    for (int oy = 0; oy < ho; oy++) {
      for (int ox = 0; ox < wo; ox++) {
        final g = ((b * ho + oy) * wo + ox) * 2;
        var sx = unnormalize(gf[g], w);
        var sy = unnormalize(gf[g + 1], h);
        if (paddingMode == 'reflection') {
          sy = reflect(sy, h);
          sx = reflect(sx, w);
        }
        final outIdx = (b * c * ho + oy) * wo + ox;
        if (mode == 'nearest') {
          final off = flatOrPad(_roundEvenNn(sy), _roundEvenNn(sx));
          for (int ch = 0; ch < c; ch++) {
            final base = (b * c + ch) * planeSize;
            out[outIdx + ch * ho * wo] = off < 0 ? 0 : xf[base + off];
          }
          continue;
        }
        final y0 = sy.floor(), x0 = sx.floor();
        final fy = sy - y0, fx = sx - x0;
        final o11 = flatOrPad(y0, x0), o12 = flatOrPad(y0, x0 + 1);
        final o21 = flatOrPad(y0 + 1, x0), o22 = flatOrPad(y0 + 1, x0 + 1);
        final w11 = (1 - fy) * (1 - fx), w12 = (1 - fy) * fx;
        final w21 = fy * (1 - fx), w22 = fy * fx;
        for (int ch = 0; ch < c; ch++) {
          final base = (b * c + ch) * planeSize;
          out[outIdx + ch * ho * wo] =
              (o11 < 0 ? 0 : xf[base + o11] * w11) +
                  (o12 < 0 ? 0 : xf[base + o12] * w12) +
                  (o21 < 0 ? 0 : xf[base + o21] * w21) +
                  (o22 < 0 ? 0 : xf[base + o22] * w22);
        }
      }
    }
  }
  return Tensor.float(out, [n, c, ho, wo]);
}

/// `RoiAlign` (2-D, average or max): pools each ROI `[x1,y1,x2,y2]` (input
/// coordinate scale via [spatialScale]) into `[outH, outW]` bins with
/// bilinear sampling; `half_pixel` subtracts 0.5 after scaling.
Tensor opRoiAlign(Tensor x, Tensor rois, Tensor batchIndices,
    {required int outH,
    required int outW,
    double spatialScale = 1.0,
    int samplingRatio = 0,
    bool isMax = false,
    bool halfPixel = true}) {
  final c = x.shape[1], h = x.shape[2], w = x.shape[3];
  final nRoi = rois.shape[0];
  final xf = x.asFloatList(), rf = rois.asFloatList();
  final bi = batchIndices.asIntList();
  final out = Float32List(nRoi * c * outH * outW);
  final offset = halfPixel ? 0.5 : 0.0;

  // Max mode is NOT max-of-interpolated-values: the ONNX reference (and
  // ORT) take the max over the four weighted corner products per sample.
  double bilinear(int b, int ch, double sy, double sx, bool wantMax) {
    if (sy < -1 || sy > h || sx < -1 || sx > w) return 0;
    sy = sy.clamp(0.0, h - 1.0);
    sx = sx.clamp(0.0, w - 1.0);
    final y0 = sy.floor(), x0 = sx.floor();
    final y1 = math.min(y0 + 1, h - 1), x1 = math.min(x0 + 1, w - 1);
    final fy = sy - y0, fx = sx - x0;
    final base = (b * c + ch) * h * w;
    final p11 = xf[base + y0 * w + x0] * (1 - fy) * (1 - fx);
    final p12 = xf[base + y0 * w + x1] * (1 - fy) * fx;
    final p21 = xf[base + y1 * w + x0] * fy * (1 - fx);
    final p22 = xf[base + y1 * w + x1] * fy * fx;
    return wantMax
        ? math.max(math.max(p11, p12), math.max(p21, p22))
        : p11 + p12 + p21 + p22;
  }

  for (int r = 0; r < nRoi; r++) {
    final b = bi[r];
    final x1 = rf[r * 4] * spatialScale - offset;
    final y1 = rf[r * 4 + 1] * spatialScale - offset;
    final x2 = rf[r * 4 + 2] * spatialScale - offset;
    final y2 = rf[r * 4 + 3] * spatialScale - offset;
    var roiH = y2 - y1, roiW = x2 - x1;
    if (!halfPixel) {
      roiH = math.max(roiH, 1);
      roiW = math.max(roiW, 1);
    }
    final binH = roiH / outH, binW = roiW / outW;
    final gridH = samplingRatio > 0 ? samplingRatio : (roiH / outH).ceil();
    final gridW = samplingRatio > 0 ? samplingRatio : (roiW / outW).ceil();
    final gH = math.max(gridH, 1), gW = math.max(gridW, 1);
    for (int ch = 0; ch < c; ch++) {
      for (int oy = 0; oy < outH; oy++) {
        for (int ox = 0; ox < outW; ox++) {
          double acc = isMax ? double.negativeInfinity : 0;
          for (int iy = 0; iy < gH; iy++) {
            final sy = y1 + oy * binH + (iy + 0.5) * binH / gH;
            for (int ix = 0; ix < gW; ix++) {
              final sx = x1 + ox * binW + (ix + 0.5) * binW / gW;
              final v = bilinear(b, ch, sy, sx, isMax);
              if (isMax) {
                acc = math.max(acc, v);
              } else {
                acc += v;
              }
            }
          }
          out[((r * c + ch) * outH + oy) * outW + ox] =
              isMax ? acc : acc / (gH * gW);
        }
      }
    }
  }
  return Tensor.float(out, [nRoi, c, outH, outW]);
}

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------

double _sourceCoord(int outIdx, double scale, int inDim, int outDim,
    String coordMode) {
  switch (coordMode) {
    case 'align_corners':
      return outDim == 1 ? 0 : outIdx * (inDim - 1) / (outDim - 1);
    case 'asymmetric':
      return outIdx / scale;
    case 'pytorch_half_pixel':
      return outDim > 1 ? (outIdx + 0.5) / scale - 0.5 : 0;
    default: // half_pixel
      return (outIdx + 0.5) / scale - 0.5;
  }
}

/// `Resize` for NCHW tensors, `nearest` and `linear` modes, resizing the
/// spatial dims (N/C scales must be 1). Supports the common
/// coordinate_transformation_modes and nearest_modes.
Tensor opResize(Tensor x,
    {List<double>? scales,
    List<int>? sizes,
    String mode = 'nearest',
    String coordMode = 'half_pixel',
    String nearestMode = 'round_prefer_floor'}) {
  if (x.rank == 3) {
    // 1-D temporal resize (NCW): run as NCHW with a singleton height.
    final y = opResize(
      x.reshape([x.shape[0], x.shape[1], 1, x.shape[2]]),
      scales: scales == null
          ? null
          : [scales[0], scales[1], 1.0, scales[2]],
      sizes: sizes == null ? null : [sizes[0], sizes[1], 1, sizes[2]],
      mode: mode,
      coordMode: coordMode,
      nearestMode: nearestMode,
    );
    return y.reshape([y.shape[0], y.shape[1], y.shape[3]]);
  }
  assert(x.rank == 4, 'Resize implemented for 3-D (NCW) / 4-D NCHW tensors');
  final n = x.shape[0], c = x.shape[1], h = x.shape[2], w = x.shape[3];
  int outH, outW;
  double scaleH, scaleW;
  if (sizes != null) {
    assert(sizes.length == 4 && sizes[0] == n && sizes[1] == c,
        'Resize: only spatial dims may change');
    outH = sizes[2];
    outW = sizes[3];
    scaleH = outH / h;
    scaleW = outW / w;
  } else {
    assert(scales != null && scales.length == 4,
        'Resize needs scales or sizes');
    assert(scales![0] == 1 && scales[1] == 1,
        'Resize: only spatial dims may change');
    scaleH = scales![2];
    scaleW = scales[3];
    outH = (h * scaleH).floor();
    outW = (w * scaleW).floor();
  }

  int nearest(double v) {
    switch (nearestMode) {
      case 'floor':
        return v.floor();
      case 'ceil':
        return v.ceil();
      case 'round_prefer_ceil':
        return (v - v.floor() == 0.5) ? v.ceil() : v.round();
      default: // round_prefer_floor
        return (v - v.floor() == 0.5) ? v.floor() : v.round();
    }
  }

  final xf = x.asFloatList();
  final out = Float32List(n * c * outH * outW);
  for (int img = 0; img < n * c; img++) {
    final base = img * h * w;
    final outBase = img * outH * outW;
    for (int oy = 0; oy < outH; oy++) {
      final sy = _sourceCoord(oy, scaleH, h, outH, coordMode);
      for (int ox = 0; ox < outW; ox++) {
        final sx = _sourceCoord(ox, scaleW, w, outW, coordMode);
        double v;
        if (mode == 'nearest') {
          final iy = nearest(sy).clamp(0, h - 1);
          final ix = nearest(sx).clamp(0, w - 1);
          v = xf[base + iy * w + ix];
        } else {
          // Bilinear. Indices derive from the unclamped floor and are then
          // clamped individually — clamping before deriving the +1 neighbor
          // would shift border samples inward.
          final fy0 = sy.floor(), fx0 = sx.floor();
          final y0 = fy0.clamp(0, h - 1), x0 = fx0.clamp(0, w - 1);
          final y1 = (fy0 + 1).clamp(0, h - 1), x1 = (fx0 + 1).clamp(0, w - 1);
          final fy = sy - fy0, fx = sx - fx0;
          final top = xf[base + y0 * w + x0] * (1 - fx) +
              xf[base + y0 * w + x1] * fx;
          final bot = xf[base + y1 * w + x0] * (1 - fx) +
              xf[base + y1 * w + x1] * fx;
          v = top * (1 - fy) + bot * fy;
        }
        out[outBase + oy * outW + ox] = v;
      }
    }
  }
  return Tensor.float(out, [n, c, outH, outW]);
}

/// `Flatten`: collapse to 2-D [prod(dims[:axis]), prod(dims[axis:])].
Tensor opFlatten(Tensor x, int axis) {
  final ax = axis < 0 ? axis + x.rank : axis;
  final head = _prod(x.shape.sublist(0, ax));
  return x.reshape([head, x.length ~/ math.max(head, 1)]);
}
