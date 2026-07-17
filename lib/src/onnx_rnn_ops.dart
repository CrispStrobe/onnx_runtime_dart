/// Recurrent operators — `LSTM`, `GRU`, `RNN` — per the public ONNX operator
/// specification (https://onnx.ai/onnx/operators/). Layout 0 only (the
/// default): X is `[seq, batch, input]`, Y is `[seq, dirs, batch, hidden]`.
///
/// Supported: forward / reverse / bidirectional, biases, initial_h/initial_c,
/// sequence_lens (per-batch valid lengths; steps beyond a sequence's length
/// leave its state frozen and its Y rows zero), LSTM peepholes, GRU
/// linear_before_reset. Only the spec-default activations (Sigmoid/Tanh) are
/// supported; the `clip` attribute is not.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'tensor.dart';

double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));
double _tanh(double x) {
  if (x > 20) return 1;
  if (x < -20) return -1;
  final e2 = math.exp(2 * x);
  return (e2 - 1) / (e2 + 1);
}

/// Shared per-direction scaffolding: X·Wᵀ precomputed for all timesteps, and
/// the per-step processing order (reverse walks each sequence back to front).
class _RnnDir {
  final int seq, batch, input, hidden, gates;
  final Float32List xw; // [seq*batch, gates*hidden]
  final Float32List r; // [gates*hidden, hidden] for this direction
  final Float32List? wb; // W bias [gates*hidden]
  final Float32List? rb; // R bias [gates*hidden]
  final bool reverse;
  final Int64List? seqLens;

  _RnnDir({
    required Float32List x,
    required Float32List w, // [gates*hidden, input], this direction
    required this.r,
    required this.wb,
    required this.rb,
    required this.seq,
    required this.batch,
    required this.input,
    required this.hidden,
    required this.gates,
    required this.reverse,
    required this.seqLens,
  }) : xw = Float32List(seq * batch * gates * hidden) {
    final gh = gates * hidden;
    for (int t = 0; t < seq * batch; t++) {
      final xRow = t * input;
      final outRow = t * gh;
      for (int j = 0; j < gh; j++) {
        double s = 0;
        final wRow = j * input;
        for (int k = 0; k < input; k++) {
          s += x[xRow + k] * w[wRow + k];
        }
        xw[outRow + j] = s;
      }
    }
  }

  int lenOf(int b) => seqLens == null ? seq : seqLens![b];

  /// Actual time index for processing step [s] of batch [b], or -1 when the
  /// sequence is already exhausted at this step.
  int timeIndex(int s, int b) {
    final len = lenOf(b);
    if (s >= len) return -1;
    return reverse ? len - 1 - s : s;
  }

  /// pre[j] = X_t·Wᵀ[j] + H·Rᵀ[j] + Wb[j] + Rb[j] for one batch row.
  void gatePre(int t, int b, Float32List h, int hOff, Float32List pre) {
    final gh = gates * hidden;
    final xwRow = (t * batch + b) * gh;
    for (int j = 0; j < gh; j++) {
      double s = xw[xwRow + j] + (wb?[j] ?? 0) + (rb?[j] ?? 0);
      final rRow = j * hidden;
      for (int k = 0; k < hidden; k++) {
        s += h[hOff + k] * r[rRow + k];
      }
      pre[j] = s;
    }
  }
}

int _dirCount(String direction) => direction == 'bidirectional' ? 2 : 1;

void _checkDefaultActivations(List<String>? acts, String direction, String op,
    List<String> defaults) {
  if (acts == null) return;
  final want = [
    ...defaults,
    if (direction == 'bidirectional') ...defaults,
  ];
  if (acts.length != want.length ||
      !Iterable.generate(acts.length)
          .every((k) => acts[k].toLowerCase() == want[k].toLowerCase())) {
    throw UnsupportedError(
        '$op: only the default activations $defaults are supported, '
        'got $acts');
  }
}

Float32List? _dirSlice(Tensor? t, int dir, int len) => t == null
    ? null
    : Float32List.sublistView(t.asFloatList(), dir * len, (dir + 1) * len);

/// `LSTM`. Returns `[Y, Y_h, Y_c]`.
List<Tensor> opLSTM(
  Tensor x,
  Tensor w,
  Tensor rr, {
  Tensor? b,
  Tensor? sequenceLens,
  Tensor? initialH,
  Tensor? initialC,
  Tensor? peepholes,
  required int hiddenSize,
  String direction = 'forward',
  List<String>? activations,
  double? clip,
}) {
  if (clip != null && clip != 0) {
    throw UnsupportedError('LSTM: clip attribute not supported');
  }
  _checkDefaultActivations(
      activations, direction, 'LSTM', ['Sigmoid', 'Tanh', 'Tanh']);
  final seq = x.shape[0], batch = x.shape[1], input = x.shape[2];
  final h = hiddenSize;
  final dirs = _dirCount(direction);
  final xf = x.asFloatList(), wf = w.asFloatList(), rf = rr.asFloatList();
  final lens = sequenceLens?.asIntList();

  final y = Float32List(seq * dirs * batch * h);
  final yh = Float32List(dirs * batch * h);
  final yc = Float32List(dirs * batch * h);

  for (int dir = 0; dir < dirs; dir++) {
    final bias = _dirSlice(b, dir, 8 * h);
    final d = _RnnDir(
      x: xf,
      w: Float32List.sublistView(wf, dir * 4 * h * input,
          (dir + 1) * 4 * h * input),
      r: Float32List.sublistView(rf, dir * 4 * h * h, (dir + 1) * 4 * h * h),
      wb: bias == null ? null : Float32List.sublistView(bias, 0, 4 * h),
      rb: bias == null ? null : Float32List.sublistView(bias, 4 * h, 8 * h),
      seq: seq,
      batch: batch,
      input: input,
      hidden: h,
      gates: 4,
      reverse: direction == 'reverse' || dir == 1,
      seqLens: lens,
    );
    final p = _dirSlice(peepholes, dir, 3 * h); // [pi, po, pf]

    final hState = Float32List(batch * h);
    final cState = Float32List(batch * h);
    if (initialH != null) {
      hState.setAll(0,
          Float32List.sublistView(initialH.asFloatList(), dir * batch * h,
              (dir + 1) * batch * h));
    }
    if (initialC != null) {
      cState.setAll(0,
          Float32List.sublistView(initialC.asFloatList(), dir * batch * h,
              (dir + 1) * batch * h));
    }

    final pre = Float32List(4 * h);
    for (int s = 0; s < seq; s++) {
      for (int bb = 0; bb < batch; bb++) {
        final t = d.timeIndex(s, bb);
        if (t < 0) continue;
        final hOff = bb * h;
        d.gatePre(t, bb, hState, hOff, pre);
        // Gate blocks in spec order: i, o, f, c.
        for (int j = 0; j < h; j++) {
          final cPrev = cState[hOff + j];
          final it =
              _sigmoid(pre[j] + (p == null ? 0 : p[j] * cPrev));
          final ft =
              _sigmoid(pre[2 * h + j] + (p == null ? 0 : p[2 * h + j] * cPrev));
          final ct = _tanh(pre[3 * h + j]);
          final c = ft * cPrev + it * ct;
          final ot =
              _sigmoid(pre[h + j] + (p == null ? 0 : p[h + j] * c));
          final hv = ot * _tanh(c);
          cState[hOff + j] = c;
          hState[hOff + j] = hv;
          y[((t * dirs + dir) * batch + bb) * h + j] = hv;
        }
      }
    }
    yh.setRange(dir * batch * h, (dir + 1) * batch * h, hState);
    yc.setRange(dir * batch * h, (dir + 1) * batch * h, cState);
  }
  return [
    Tensor.float(y, [seq, dirs, batch, h]),
    Tensor.float(yh, [dirs, batch, h]),
    Tensor.float(yc, [dirs, batch, h]),
  ];
}

/// `GRU`. Returns `[Y, Y_h]`.
List<Tensor> opGRU(
  Tensor x,
  Tensor w,
  Tensor rr, {
  Tensor? b,
  Tensor? sequenceLens,
  Tensor? initialH,
  required int hiddenSize,
  String direction = 'forward',
  List<String>? activations,
  bool linearBeforeReset = false,
  double? clip,
}) {
  if (clip != null && clip != 0) {
    throw UnsupportedError('GRU: clip attribute not supported');
  }
  _checkDefaultActivations(activations, direction, 'GRU', ['Sigmoid', 'Tanh']);
  final seq = x.shape[0], batch = x.shape[1], input = x.shape[2];
  final h = hiddenSize;
  final dirs = _dirCount(direction);
  final xf = x.asFloatList(), wf = w.asFloatList(), rf = rr.asFloatList();
  final lens = sequenceLens?.asIntList();

  final y = Float32List(seq * dirs * batch * h);
  final yh = Float32List(dirs * batch * h);

  for (int dir = 0; dir < dirs; dir++) {
    final bias = _dirSlice(b, dir, 6 * h);
    final wbAll = bias == null ? null : Float32List.sublistView(bias, 0, 3 * h);
    final rbAll =
        bias == null ? null : Float32List.sublistView(bias, 3 * h, 6 * h);
    final rDir =
        Float32List.sublistView(rf, dir * 3 * h * h, (dir + 1) * 3 * h * h);
    final d = _RnnDir(
      x: xf,
      w: Float32List.sublistView(wf, dir * 3 * h * input,
          (dir + 1) * 3 * h * input),
      r: rDir,
      wb: wbAll,
      // The h-gate's R contribution differs between the two
      // linear_before_reset semantics — handled below, so gatePre only adds
      // the z/r rows' Rb here via a masked copy.
      rb: null,
      seq: seq,
      batch: batch,
      input: input,
      hidden: h,
      gates: 3,
      reverse: direction == 'reverse' || dir == 1,
      seqLens: lens,
    );

    final hState = Float32List(batch * h);
    if (initialH != null) {
      hState.setAll(0,
          Float32List.sublistView(initialH.asFloatList(), dir * batch * h,
              (dir + 1) * batch * h));
    }

    final xwRowBuf = Float32List(3 * h);
    final zBuf = Float32List(h);
    final rBuf = Float32List(h);
    final rH = Float32List(h); // reset-gated hidden ((r⊙H) for lbr=0)
    final hNew = Float32List(h);
    for (int s = 0; s < seq; s++) {
      for (int bb = 0; bb < batch; bb++) {
        final t = d.timeIndex(s, bb);
        if (t < 0) continue;
        final hOff = bb * h;
        final xwRow = (t * batch + bb) * 3 * h;
        for (int j = 0; j < 3 * h; j++) {
          xwRowBuf[j] = d.xw[xwRow + j] + (wbAll?[j] ?? 0);
        }
        // z and r gates: pre = xw + Wb + H·Rᵀ + Rb. Gate order: z, r, h.
        for (int j = 0; j < 2 * h; j++) {
          double sum = xwRowBuf[j] + (rbAll?[j] ?? 0);
          final rRow = j * h;
          for (int k = 0; k < h; k++) {
            sum += hState[hOff + k] * rDir[rRow + k];
          }
          if (j < h) {
            zBuf[j] = _sigmoid(sum);
          } else {
            rBuf[j - h] = _sigmoid(sum);
          }
        }
        if (!linearBeforeReset) {
          for (int k = 0; k < h; k++) {
            rH[k] = rBuf[k] * hState[hOff + k];
          }
        }
        for (int j = 0; j < h; j++) {
          final rRow = (2 * h + j) * h;
          double hPre;
          if (linearBeforeReset) {
            // h̃ = g(Xt·Whᵀ + Wbh + r ⊙ (H·Rhᵀ + Rbh))
            double sum = rbAll?[2 * h + j] ?? 0;
            for (int k = 0; k < h; k++) {
              sum += hState[hOff + k] * rDir[rRow + k];
            }
            hPre = xwRowBuf[2 * h + j] + rBuf[j] * sum;
          } else {
            // h̃ = g(Xt·Whᵀ + Wbh + (r ⊙ H)·Rhᵀ + Rbh)
            double sum = rbAll?[2 * h + j] ?? 0;
            for (int k = 0; k < h; k++) {
              sum += rH[k] * rDir[rRow + k];
            }
            hPre = xwRowBuf[2 * h + j] + sum;
          }
          hNew[j] =
              (1 - zBuf[j]) * _tanh(hPre) + zBuf[j] * hState[hOff + j];
        }
        for (int j = 0; j < h; j++) {
          hState[hOff + j] = hNew[j];
          y[((t * dirs + dir) * batch + bb) * h + j] = hNew[j];
        }
      }
    }
    yh.setRange(dir * batch * h, (dir + 1) * batch * h, hState);
  }
  return [
    Tensor.float(y, [seq, dirs, batch, h]),
    Tensor.float(yh, [dirs, batch, h]),
  ];
}

/// `RNN` (vanilla, Tanh activation). Returns `[Y, Y_h]`.
List<Tensor> opRNN(
  Tensor x,
  Tensor w,
  Tensor rr, {
  Tensor? b,
  Tensor? sequenceLens,
  Tensor? initialH,
  required int hiddenSize,
  String direction = 'forward',
  List<String>? activations,
  double? clip,
}) {
  if (clip != null && clip != 0) {
    throw UnsupportedError('RNN: clip attribute not supported');
  }
  _checkDefaultActivations(activations, direction, 'RNN', ['Tanh']);
  final seq = x.shape[0], batch = x.shape[1], input = x.shape[2];
  final h = hiddenSize;
  final dirs = _dirCount(direction);
  final xf = x.asFloatList(), wf = w.asFloatList(), rf = rr.asFloatList();
  final lens = sequenceLens?.asIntList();

  final y = Float32List(seq * dirs * batch * h);
  final yh = Float32List(dirs * batch * h);

  for (int dir = 0; dir < dirs; dir++) {
    final bias = _dirSlice(b, dir, 2 * h);
    final d = _RnnDir(
      x: xf,
      w: Float32List.sublistView(wf, dir * h * input, (dir + 1) * h * input),
      r: Float32List.sublistView(rf, dir * h * h, (dir + 1) * h * h),
      wb: bias == null ? null : Float32List.sublistView(bias, 0, h),
      rb: bias == null ? null : Float32List.sublistView(bias, h, 2 * h),
      seq: seq,
      batch: batch,
      input: input,
      hidden: h,
      gates: 1,
      reverse: direction == 'reverse' || dir == 1,
      seqLens: lens,
    );

    final hState = Float32List(batch * h);
    if (initialH != null) {
      hState.setAll(0,
          Float32List.sublistView(initialH.asFloatList(), dir * batch * h,
              (dir + 1) * batch * h));
    }

    final pre = Float32List(h);
    for (int s = 0; s < seq; s++) {
      for (int bb = 0; bb < batch; bb++) {
        final t = d.timeIndex(s, bb);
        if (t < 0) continue;
        final hOff = bb * h;
        d.gatePre(t, bb, hState, hOff, pre);
        for (int j = 0; j < h; j++) {
          final hv = _tanh(pre[j]);
          hState[hOff + j] = hv;
          y[((t * dirs + dir) * batch + bb) * h + j] = hv;
        }
      }
    }
    yh.setRange(dir * batch * h, (dir + 1) * batch * h, hState);
  }
  return [
    Tensor.float(y, [seq, dirs, batch, h]),
    Tensor.float(yh, [dirs, batch, h]),
  ];
}
