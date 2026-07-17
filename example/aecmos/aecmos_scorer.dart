// AECMOS scorer on the pure-Dart ONNX runtime — a faithful port of
// `AECMOSEstimator` from microsoft/AEC-Challenge (AECMOS_local/aecmos.py):
// mel features per signal, scenario-marker frames appended in feature space,
// stacked to a [1, 3, frames, 160] tensor, GRU hidden state zeroed.
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';

import 'mel_spectrogram.dart';

/// Wraps an AECMOS ONNX model and scores (lpb, mic, enh) signal triples.
///
/// Model parameters (sampling rate, DFT size, GRU hidden shape, whether the
/// model expects scenario markers) are inferred from the model *filename*,
/// keyed on the run IDs the reference implementation recognizes
/// (1663915512, 1663829550: 16 kHz; 1668423760: 48 kHz).
class AecmosScorer {
  final OnnxModel _model;
  final int samplingRate;
  final int dftSize;
  final List<int> hiddenSize;
  final bool needScenarioMarker;

  /// Maximum scored length in seconds; longer inputs are truncated.
  static const maxLenSeconds = 20;

  AecmosScorer._(this._model, this.samplingRate, this.dftSize, this.hiddenSize,
      this.needScenarioMarker);

  factory AecmosScorer(String modelPath) {
    final int sr, dft;
    final List<int> hidden;
    final bool marker;
    if (modelPath.contains('1663915512')) {
      (sr, dft, hidden, marker) = (16000, 512, const [4, 1, 64], true);
    } else if (modelPath.contains('1663829550')) {
      (sr, dft, hidden, marker) = (16000, 512, const [4, 1, 64], false);
    } else if (modelPath.contains('1668423760')) {
      (sr, dft, hidden, marker) = (48000, 1536, const [4, 1, 96], true);
    } else {
      throw ArgumentError.value(modelPath, 'modelPath',
          'not a recognized AECMOS model (expected run id 1663915512, '
              '1663829550 or 1668423760 in the filename)');
    }
    return AecmosScorer._(loadOnnxModel(modelPath), sr, dft, hidden, marker);
  }

  /// Normalized mel features for one signal, `[frames x 160]` row-major.
  Float32List transform(Float32List signal) => aecmosMelFeatures(signal,
      sr: samplingRate, nFft: dftSize + 1, hopLength: dftSize ~/ 2);

  /// Builds the `[1, 3, frames, 160]` model input for one clip: per-signal
  /// normalized mel features with the scenario-marker frames appended in
  /// feature space exactly as aecmos.py does — 20 frames of a per-channel
  /// constant (lpb: `1 - ne_st`, mic: `1 - fe_st`, enh: `1`) followed by 20
  /// zero frames. Signals are truncated to their common length, then to
  /// [maxLenSeconds].
  ({Float32List feats, int frames}) buildFeatures(
      String talkType, Float32List lpb, Float32List mic, Float32List enh) {
    var len = math.min(lpb.length, math.min(mic.length, enh.length));
    len = math.min(len, maxLenSeconds * samplingRate);
    final lpbF = transform(Float32List.sublistView(lpb, 0, len));
    final micF = transform(Float32List.sublistView(mic, 0, len));
    final enhF = transform(Float32List.sublistView(enh, 0, len));

    final melFrames = lpbF.length ~/ 160;
    var frames = melFrames;
    if (needScenarioMarker) {
      if (talkType != 'nst' && talkType != 'st' && talkType != 'dt') {
        throw ArgumentError.value(
            talkType, 'talkType', "must be 'st', 'nst' or 'dt'");
      }
      frames += 40;
    }

    final feats = Float32List(3 * frames * 160);
    feats.setRange(0, lpbF.length, lpbF);
    feats.setRange(frames * 160, frames * 160 + micF.length, micF);
    feats.setRange(2 * frames * 160, 2 * frames * 160 + enhF.length, enhF);
    if (needScenarioMarker) {
      final neSt = talkType == 'nst' ? 1 : 0;
      final feSt = talkType == 'st' ? 1 : 0;
      _appendMarker(feats, 0, melFrames, frames, 1.0 - neSt); // lpb
      _appendMarker(feats, 1, melFrames, frames, 1.0 - feSt); // mic
      _appendMarker(feats, 2, melFrames, frames, 1.0); // enh
    }
    return (feats: feats, frames: frames);
  }

  /// Scores one clip. [talkType] is `'st'` (far-end single talk), `'nst'`
  /// (near-end single talk) or `'dt'` (double talk); it selects the scenario
  /// marker for models that need one (see [buildFeatures]).
  ///
  /// Returns the two model outputs in aecmos.py's order: the echo MOS and the
  /// "other" degradation MOS.
  ({double echoMos, double otherMos}) score(
      String talkType, Float32List lpb, Float32List mic, Float32List enh) {
    final (:feats, :frames) = buildFeatures(talkType, lpb, mic, enh);
    final out = _model.run({
      'input': Tensor.float(feats, [1, 3, frames, 160]),
      'h0': Tensor.float(
          Float32List(hiddenSize[0] * hiddenSize[1] * hiddenSize[2]),
          hiddenSize),
    }, [
      'output'
    ])['output']!
        .asFloatList();
    return (echoMos: out[0].toDouble(), otherMos: out[1].toDouble());
  }

  static void _appendMarker(
      Float32List feats, int channel, int melFrames, int frames, double value) {
    final base = channel * frames * 160 + melFrames * 160;
    feats.fillRange(base, base + 20 * 160, value);
    // The trailing 20 zero-frames are already zero-initialized.
  }
}
