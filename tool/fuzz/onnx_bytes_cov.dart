/// Coverage-guided reader-robustness fuzzing of `OnnxModel.fromBytes`, evolving
/// a corpus toward the deep parse paths that blind mutation can't reach: tensor
/// loading (`tensorFromProto` raw-byte reads), constant folding, and fusion —
/// all behind a valid-protobuf precondition. Reads coverage from the VM
/// service, so it MUST run with it enabled:
///
///   dart run --enable-vm-service=0 --no-pause-isolates-on-exit \
///     tool/fuzz/onnx_bytes_cov.dart [targetLibSuffix]
///
/// Optional arg picks the library to score coverage on (default: the proto
/// loader). The evolved corpus and any minimized crashes persist under
/// `.corpus/` and `.crashes/` so re-runs continue from the coverage reached.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:covfuzz/covfuzz.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';

/// Seeds carry diverse weight tensors (quantized int4, int8/uint8, float,
/// float64) so mutations can reach the many `tensorFromProto` branches.
const _seedPaths = [
  'test/fixtures/matmulnbits_basic/model.onnx', // int4 block-quant weights
  'test/fixtures/quantizelinear_int8/model.onnx', // int8/uint8 tensors
  'test/fixtures/prelu_channel_slope/model.onnx', // float weights
  'test/fixtures/stft_no_window/model.onnx', // float64 weights
  'test/fixtures/gqa_kvcache/model.onnx',
];

Future<void> main(List<String> args) async {
  final targetSuffix = args.isNotEmpty ? args[0] : 'onnx_proto_loader.dart';
  final seeds = <Uint8List>[
    for (final p in _seedPaths)
      if (File(p).existsSync()) File(p).readAsBytesSync(),
  ];
  if (seeds.isEmpty) {
    stderr.writeln('No seed models found — run from the package root.');
    exit(2);
  }

  final report = await covFuzz<Uint8List>(
    seeds: seeds,
    entry: (b) => OnnxModel.fromBytes(b),
    mutate: mutateBytes,
    targetLib: 'package:onnx_runtime_dart/src/$targetSuffix',
    isClean: (e) => e is FormatException || e is UnsupportedError,
    iterations: 40000,
    budgetMs: 180000,
    corpusDir: '.corpus/onnx_$targetSuffix',
    crashDir: '.crashes/onnx_$targetSuffix',
    log: true,
  );
  exit(report.report());
}
