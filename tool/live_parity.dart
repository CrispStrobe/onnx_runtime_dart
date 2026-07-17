/// Replays a live-parity case produced by `tool/live_parity.py` through the
/// Dart runtime and reports cosine similarity + max abs diff vs native ORT.
///
///   dart run tool/live_parity.dart model.onnx case.json
///
/// Exits non-zero if cosine < 0.999999 or max|Δ| > 1e-3 on any output.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';

Tensor tensorFromJson(Map<String, dynamic> j) {
  final shape = (j['shape'] as List).cast<int>();
  final data = j['data'] as List;
  if (j['dtype'] == 'int64') {
    return Tensor.int64(
        Int64List.fromList(data.map((v) => (v as num).toInt()).toList()),
        shape);
  }
  return Tensor.float(
      Float32List.fromList(data.map((v) => (v as num).toDouble()).toList()),
      shape);
}

void main(List<String> args) {
  final model = loadOnnxModel(args[0]);
  final j = jsonDecode(File(args[1]).readAsStringSync()) as Map<String, dynamic>;
  final inputs = (j['inputs'] as Map<String, dynamic>)
      .map((k, v) => MapEntry(k, tensorFromJson(v as Map<String, dynamic>)));
  final expected = (j['expected'] as Map<String, dynamic>)
      .map((k, v) => MapEntry(k, tensorFromJson(v as Map<String, dynamic>)));

  final sw = Stopwatch()..start();
  final got = model.run(inputs, expected.keys.toList());
  sw.stop();

  bool ok = true;
  expected.forEach((name, want) {
    final have = got[name]!;
    if (have.shape.join(',') != want.shape.join(',')) {
      print('$name: SHAPE MISMATCH dart=${have.shape} ort=${want.shape}');
      ok = false;
      return;
    }
    final w = want.asFloatList(), h = have.asFloatList();
    double maxAbs = 0, dot = 0, nw = 0, nh = 0;
    for (int k = 0; k < w.length; k++) {
      final d = (w[k] - h[k]).abs();
      if (d > maxAbs) maxAbs = d;
      dot += w[k] * h[k];
      nw += w[k] * w[k];
      nh += h[k] * h[k];
    }
    final cos = nw > 0 && nh > 0 ? dot / (math.sqrt(nw) * math.sqrt(nh)) : 1.0;
    print('$name: shape=${have.shape} cosine=${cos.toStringAsFixed(9)} '
        'max|Δ|=${maxAbs.toStringAsExponential(2)} (${sw.elapsedMilliseconds}ms)');
    if (cos < 0.999999 || maxAbs > 1e-3) ok = false;
  });
  if (!ok) exit(1);
}
