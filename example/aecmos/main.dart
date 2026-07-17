// AECMOS scoring CLI on the pure-Dart ONNX runtime.
//
//   dart run example/aecmos/main.dart <model.onnx> <lpb.raw> <mic.raw> <enh.raw> <st|nst|dt>
//
// The .raw files are headerless 16-bit little-endian PCM, mono, at the model's
// sampling rate (16 kHz for the 1663915512/1663829550 models, 48 kHz for
// 1668423760); samples are scaled to [-1, 1) by 1/32768 as librosa.load does.
import 'dart:io';
import 'dart:typed_data';

import 'aecmos_scorer.dart';

Float32List _readPcm16(String path) {
  final bytes = File(path).readAsBytesSync();
  final samples = Int16List.view(
      bytes.buffer, bytes.offsetInBytes, bytes.lengthInBytes ~/ 2);
  final out = Float32List(samples.length);
  for (var i = 0; i < samples.length; i++) {
    out[i] = samples[i] / 32768.0;
  }
  return out;
}

void main(List<String> args) {
  if (args.length != 5) {
    stderr.writeln('usage: dart run example/aecmos/main.dart '
        '<model.onnx> <lpb.raw> <mic.raw> <enh.raw> <st|nst|dt>');
    exitCode = 64;
    return;
  }
  final scorer = AecmosScorer(args[0]);
  final lpb = _readPcm16(args[1]);
  final mic = _readPcm16(args[2]);
  final enh = _readPcm16(args[3]);
  final scores = scorer.score(args[4], lpb, mic, enh);
  print('The AECMOS echo score is ${scores.echoMos}, and (other) degradation '
      'score is ${scores.otherMos}.');
}
