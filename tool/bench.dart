/// Wall-clock + per-op benchmark for a real model.
///
///   dart run tool/bench.dart model.onnx [--seq N] [--iters N] [--workers N]
///
/// With --workers the model runs through the isolate pool (`runAsync`).
library;

/// Inputs are synthesized from the graph's input signatures: int64 inputs get
/// BERT-style token ids / masks ([1, seq]), float inputs get deterministic
/// pseudo-random data with unknown dims defaulted (batch→1, spatial→--seq).
/// Prints min/mean wall time and the per-op-type time breakdown.
import 'dart:io';
import 'dart:typed_data';

import 'package:onnx_runtime_dart/onnx_proto.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';

Future<void> main(List<String> args) async {
  final modelPath = args.firstWhere((a) => !a.startsWith('--'));
  int flag(String name, int dflt) {
    final k = args.indexOf('--$name');
    return k >= 0 && k + 1 < args.length ? int.parse(args[k + 1]) : dflt;
  }

  final seq = flag('seq', 16);
  final iters = flag('iters', 5);
  final workers = flag('workers', 0);

  final meta = ModelProto.fromBuffer(
      Uint8List.fromList(File(modelPath).readAsBytesSync()));
  final initNames = meta.graph.initializer.map((t) => t.name).toSet();
  final model = loadOnnxModel(modelPath);
  final outName = meta.graph.output.first.name;

  final feed = <String, Tensor>{};
  for (final vi in meta.graph.input) {
    if (initNames.contains(vi.name)) continue; // weights listed as inputs
    final tt = vi.type.tensorType;
    final dims = <int>[];
    for (final d in tt.shape.dim) {
      final v = d.dimValue.toInt();
      // Unknown dims: batch-like → 1, sequence/spatial-like → seq.
      dims.add(v > 0 ? v : (dims.isEmpty ? 1 : seq));
    }
    final n = dims.fold<int>(1, (a, b) => a * b);
    // elem_type 7=int64, 6=int32, 9=bool → integer feed; else float.
    final et = tt.elemType;
    if (et == 7 || et == 6 || et == 9) {
      final v = Int64List(n);
      if (vi.name.contains('mask')) {
        v.fillRange(0, n, 1);
      } else if (vi.name.contains('type') || vi.name.contains('position')) {
        for (int k = 0; k < n; k++) {
          v[k] = vi.name.contains('position') ? k % seq : 0;
        }
      } else {
        // Token ids: deterministic small vocab ids, [CLS]/[SEP]-ish framing.
        for (int k = 0; k < n; k++) {
          v[k] = 1000 + (k * 37) % 999;
        }
        v[0] = 101;
        v[n - 1] = 102;
      }
      feed[vi.name] = Tensor.int64(v, dims);
    } else {
      final v = Float32List(n);
      for (int k = 0; k < n; k++) {
        v[k] = ((k * 2654435761) & 0xffff) / 32768.0 - 1.0; // deterministic
      }
      feed[vi.name] = Tensor.float(v, dims);
    }
    stderr.writeln('input ${vi.name}: ${feed[vi.name]}');
  }

  if (workers > 0) {
    await model.parallelize(
        workers: workers, poolConv: args.contains('--poolconv'));
  }
  Future<Map<String, Tensor>> once({ExecutionProfile? profile}) =>
      workers > 0
          ? model.runAsync(feed, [outName], profile: profile)
          : Future.value(model.run(feed, [outName], profile: profile));

  // Warmup (JIT + any lazy decode), then timed iterations.
  await once();
  final times = <int>[];
  final profile = ExecutionProfile();
  for (int k = 0; k < iters; k++) {
    final sw = Stopwatch()..start();
    await once(profile: profile);
    sw.stop();
    times.add(sw.elapsedMicroseconds);
  }
  times.sort();
  final mean = times.reduce((a, b) => a + b) / times.length;
  final out = (await once())[outName]!;
  model.dispose();
  stderr.writeln('output $outName: ${out.shape} '
      'first4=${out.asFloatList().take(4).map((v) => v.toStringAsFixed(5)).toList()}');
  print('wall: min=${(times.first / 1000).toStringAsFixed(1)}ms '
      'mean=${(mean / 1000).toStringAsFixed(1)}ms over ${times.length} iters '
      '(seq=$seq${workers > 0 ? ', workers=$workers' : ''})');
  print(profile.report());
}
