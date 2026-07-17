// Pure-Dart mel-spectrogram front-end matching librosa's
// `melspectrogram` / `power_to_db` defaults (librosa >= 0.10), as used by the
// AECMOS reference pipeline (microsoft/AEC-Challenge, aecmos.py):
//
//   - STFT with `center=true` and zero ("constant") padding of `nFft ~/ 2`
//     samples on both sides (librosa >= 0.10 pads with zeros, not reflect),
//     periodic Hann window of length `nFft`, power spectrogram `|STFT|^2`;
//   - Slaney-scale mel filterbank (`htk=false`, `norm='slaney'`, `fmin=0`,
//     `fmax=sr/2`);
//   - `power_to_db` with `ref=max`, `amin=1e-10`, `top_db=80`.
//
// AECMOS uses an odd `nFft` (dftSize + 1 = 513 or 1537), so instead of a
// radix-2 FFT the bins are computed with a direct DFT over precomputed
// single-period twiddle tables — ~1250 frames x 513 points at 16 kHz stays
// around a second. The pipeline stages are exposed on [MelFrontEnd] so tests
// can compare each one against the Python reference in isolation.
import 'dart:math' as math;
import 'dart:typed_data';

/// Number of STFT frames librosa produces for [signalLength] samples with
/// `center=true`: `1 + (signalLength + 2*(nFft ~/ 2) - nFft) ~/ hopLength`.
int melFrameCount(int signalLength, int nFft, int hopLength) =>
    1 + (signalLength + 2 * (nFft ~/ 2) - nFft) ~/ hopLength;

/// Power mel spectrogram of [signal], row-major `[frames x nMels]` with
/// `frames == melFrameCount(signal.length, nFft, hopLength)` — i.e. the
/// transpose of librosa's `[n_mels, frames]` layout.
Float32List melSpectrogram(Float32List signal,
        {required int sr,
        required int nFft,
        required int hopLength,
        int nMels = 160}) =>
    MelFrontEnd(sr: sr, nFft: nFft, hopLength: hopLength, nMels: nMels)
        .melSpectrogram(signal);

/// Normalized AECMOS mel features for [signal]: [melSpectrogram] followed by
/// [aecmosNormalize], `[frames x nMels]` row-major.
Float32List aecmosMelFeatures(Float32List signal,
    {required int sr,
    required int nFft,
    required int hopLength,
    int nMels = 160}) {
  final spec = melSpectrogram(signal,
      sr: sr, nFft: nFft, hopLength: hopLength, nMels: nMels);
  aecmosNormalize(spec);
  return spec;
}

/// `power_to_db` applied in place: `10*log10(max(spec, amin))` relative to the
/// global maximum (`ref=max`), floored at `top_db` below the peak. Returns the
/// reference (maximum) power used.
double powerToDb(Float32List spec, {double amin = 1e-10, double topDb = 80.0}) {
  var ref = amin;
  for (var i = 0; i < spec.length; i++) {
    if (spec[i] > ref) ref = spec[i];
  }
  final refDb = 10.0 * _log10(ref);
  var maxDb = double.negativeInfinity;
  for (var i = 0; i < spec.length; i++) {
    final v = spec[i];
    spec[i] = 10.0 * _log10(v > amin ? v : amin) - refDb;
    if (spec[i] > maxDb) maxDb = spec[i];
  }
  final floor = maxDb - topDb;
  for (var i = 0; i < spec.length; i++) {
    if (spec[i] < floor) spec[i] = floor;
  }
  return ref;
}

/// AECMOS feature normalization applied in place to a power mel spectrogram:
/// `(power_to_db(spec, ref=max) + 40) / 40`.
void aecmosNormalize(Float32List melSpec) {
  powerToDb(melSpec);
  for (var i = 0; i < melSpec.length; i++) {
    melSpec[i] = (melSpec[i] + 40.0) / 40.0;
  }
}

double _log10(double x) => math.log(x) / math.ln10;

/// The STFT + mel pipeline with each stage callable on its own.
class MelFrontEnd {
  final int sr;
  final int nFft;
  final int hopLength;
  final int nMels;
  final int nBins;

  /// Periodic Hann window of length [nFft]
  /// (`scipy.signal.get_window('hann', nFft, fftbins=True)`).
  final Float64List window;

  final Float64List _cosTab;
  final Float64List _sinTab;
  final List<Float64List> _melWeights; // dense strip per mel band
  final Int32List _melStart;
  final Int32List _melEnd;

  MelFrontEnd(
      {required this.sr,
      required this.nFft,
      required this.hopLength,
      this.nMels = 160})
      : nBins = nFft ~/ 2 + 1,
        window = _hannPeriodic(nFft),
        _cosTab = Float64List(nFft),
        _sinTab = Float64List(nFft),
        _melWeights = <Float64List>[],
        _melStart = Int32List(nMels),
        _melEnd = Int32List(nMels) {
    for (var m = 0; m < nFft; m++) {
      _cosTab[m] = math.cos(2 * math.pi * m / nFft);
      _sinTab[m] = math.sin(2 * math.pi * m / nFft);
    }
    _buildMelFilterbank();
  }

  /// [signal] centered in `nFft ~/ 2` zeros on each side, as librosa's
  /// `stft(center=True, pad_mode='constant')` pads it.
  Float64List padCentered(Float32List signal) {
    final pad = nFft ~/ 2;
    final padded = Float64List(signal.length + 2 * pad);
    for (var i = 0; i < signal.length; i++) {
      padded[pad + i] = signal[i];
    }
    return padded;
  }

  /// `|DFT|^2` of frame [frameIndex] of [padded] ([nBins] real-input bins).
  Float64List framePower(Float64List padded, int frameIndex) {
    final buf = Float64List(nFft);
    final start = frameIndex * hopLength;
    for (var n = 0; n < nFft; n++) {
      buf[n] = padded[start + n] * window[n];
    }
    final power = Float64List(nBins);
    _framePowerInto(buf, power);
    return power;
  }

  /// Applies the mel filterbank to one [nBins]-long power spectrum.
  Float64List applyMel(Float64List power) {
    final out = Float64List(nMels);
    for (var m = 0; m < nMels; m++) {
      final lo = _melStart[m], hi = _melEnd[m];
      final w = _melWeights[m];
      var acc = 0.0;
      for (var k = lo; k < hi; k++) {
        acc += w[k - lo] * power[k];
      }
      out[m] = acc;
    }
    return out;
  }

  /// Row [m] of the mel filterbank as a dense [nBins]-long vector.
  Float64List melFilterRow(int m) {
    final row = Float64List(nBins);
    final w = _melWeights[m];
    for (var k = _melStart[m]; k < _melEnd[m]; k++) {
      row[k] = w[k - _melStart[m]];
    }
    return row;
  }

  /// Sum of each mel filterbank row (a quick Slaney-normalization check).
  Float64List melFilterRowSums() {
    final sums = Float64List(nMels);
    for (var m = 0; m < nMels; m++) {
      var acc = 0.0;
      for (final w in _melWeights[m]) {
        acc += w;
      }
      sums[m] = acc;
    }
    return sums;
  }

  /// Power mel spectrogram, `[frames x nMels]` row-major.
  Float32List melSpectrogram(Float32List signal) {
    final frames = melFrameCount(signal.length, nFft, hopLength);
    final padded = padCentered(signal);
    final out = Float32List(frames * nMels);
    final buf = Float64List(nFft);
    final power = Float64List(nBins);
    for (var f = 0; f < frames; f++) {
      final start = f * hopLength;
      for (var n = 0; n < nFft; n++) {
        buf[n] = padded[start + n] * window[n];
      }
      _framePowerInto(buf, power);
      final row = f * nMels;
      for (var m = 0; m < nMels; m++) {
        final lo = _melStart[m], hi = _melEnd[m];
        final w = _melWeights[m];
        var acc = 0.0;
        for (var k = lo; k < hi; k++) {
          acc += w[k - lo] * power[k];
        }
        out[row + m] = acc;
      }
    }
    return out;
  }

  void _framePowerInto(Float64List buf, Float64List power) {
    for (var k = 0; k < nBins; k++) {
      var re = 0.0, im = 0.0, m = 0;
      for (var n = 0; n < nFft; n++) {
        final v = buf[n];
        re += v * _cosTab[m];
        im += v * _sinTab[m];
        m += k;
        if (m >= nFft) m -= nFft;
      }
      power[k] = re * re + im * im;
    }
  }

  static Float64List _hannPeriodic(int n) {
    final w = Float64List(n);
    for (var i = 0; i < n; i++) {
      w[i] = 0.5 - 0.5 * math.cos(2 * math.pi * i / n);
    }
    return w;
  }

  /// Hz -> mel on the Slaney scale (librosa `hz_to_mel(htk=False)`).
  static double _hzToMel(double hz) {
    const fSp = 200.0 / 3.0;
    const minLogHz = 1000.0;
    const minLogMel = minLogHz / fSp;
    final logStep = math.log(6.4) / 27.0;
    return hz < minLogHz
        ? hz / fSp
        : minLogMel + math.log(hz / minLogHz) / logStep;
  }

  static double _melToHz(double mel) {
    const fSp = 200.0 / 3.0;
    const minLogHz = 1000.0;
    const minLogMel = minLogHz / fSp;
    final logStep = math.log(6.4) / 27.0;
    return mel < minLogMel
        ? mel * fSp
        : minLogHz * math.exp(logStep * (mel - minLogMel));
  }

  /// Slaney-normalized triangular filterbank, matching
  /// `librosa.filters.mel(sr, n_fft, n_mels)` defaults (`fmin=0`,
  /// `fmax=sr/2`, `htk=False`, `norm='slaney'`).
  void _buildMelFilterbank() {
    final melMin = _hzToMel(0.0);
    final melMax = _hzToMel(sr / 2.0);
    final melF = Float64List(nMels + 2);
    for (var i = 0; i < nMels + 2; i++) {
      melF[i] = _melToHz(melMin + (melMax - melMin) * i / (nMels + 1));
    }
    final fftFreqs = Float64List(nBins);
    for (var k = 0; k < nBins; k++) {
      fftFreqs[k] = sr * k / nFft;
    }
    for (var m = 0; m < nMels; m++) {
      final lowerDiff = melF[m + 1] - melF[m];
      final upperDiff = melF[m + 2] - melF[m + 1];
      final enorm = 2.0 / (melF[m + 2] - melF[m]);
      var lo = -1, hi = 0;
      final dense = Float64List(nBins);
      for (var k = 0; k < nBins; k++) {
        final lower = (fftFreqs[k] - melF[m]) / lowerDiff;
        final upper = (melF[m + 2] - fftFreqs[k]) / upperDiff;
        final w = math.min(lower, upper);
        if (w > 0) {
          dense[k] = w * enorm;
          if (lo < 0) lo = k;
          hi = k + 1;
        }
      }
      if (lo < 0) {
        lo = 0;
        hi = 0;
      }
      _melStart[m] = lo;
      _melEnd[m] = hi;
      _melWeights.add(Float64List.sublistView(dense, lo, hi));
    }
  }
}
