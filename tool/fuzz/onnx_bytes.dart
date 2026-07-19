/// Blind reader-robustness fuzzing of the ONNX model parser
/// (`OnnxModel.fromBytes`), which reads untrusted protobuf bytes. Contract: on
/// any input it must parse or reject with a documented exception — never leak a
/// RangeError/StateError/TypeError, OOM, or take multiple seconds.
///
///   dart run tool/fuzz/onnx_bytes.dart
///
/// Seeds are small valid `.onnx` fixtures so mutations keep enough protobuf
/// structure to reach deep parse paths (tensor loading, constant folding,
/// fusion). Clean rejects: FormatException (malformed) and UnsupportedError
/// (unsupported op/dtype — an intentional, documented reject). Anything else is
/// an escape; the harness minimizes it to a small reproducer.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:covfuzz/covfuzz.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';

/// Small valid ONNX fixtures used as mutation seeds.
const _seedPaths = [
  'test/fixtures/atan/model.onnx',
  'test/fixtures/size_op/model.onnx',
  'test/fixtures/gqa_kvcache/model.onnx',
  'test/fixtures/trilu_upper/model.onnx',
];

bool _isCleanReject(Object e) =>
    e is FormatException || e is UnsupportedError;

/// CI can shorten a run: `FUZZ_BUDGET_MS` / `FUZZ_ITERS`.
int _env(String k, int def) =>
    int.tryParse(Platform.environment[k] ?? '') ?? def;

void main() {
  final seeds = <Uint8List>[
    for (final p in _seedPaths)
      if (File(p).existsSync()) File(p).readAsBytesSync(),
  ];
  if (seeds.isEmpty) {
    stderr.writeln('No seed models found — run from the package root.');
    exit(2);
  }

  final report = fuzz<Uint8List>(
    seeds: seeds,
    // fuse:true exercises the fusion pass too; the parse is the whole load.
    entry: (b) => OnnxModel.fromBytes(b),
    mutate: mutateBytes,
    isClean: _isCleanReject,
    iterations: _env('FUZZ_ITERS', 300000),
    budgetMs: _env('FUZZ_BUDGET_MS', 120000),
    // Structural cases mutation rarely produces on its own. (A 4 MB all-zero
    // bulk was tried and parses in ~35 ms — no size-driven bomb; kept at 1 MB
    // so a transient JIT/GC spike on the largest input can't trip the SLOW
    // signal.)
    stressors: [
      Uint8List(0),
      Uint8List.fromList([0x08]), // truncated varint field
      Uint8List.fromList([0x08, ...List.filled(20, 0xFF), 0x7F]), // huge varint
      Uint8List(1024 * 1024), // all-zero bulk
    ],
  );
  report.report();
  // Fail CI only on a real contract violation or a genuine multi-second bomb;
  // covfuzz's 200 ms SLOW threshold is too twitchy for a JIT'd VM (transient GC
  // spikes) and, for these parsers, was confirmed to be proportional, not a
  // bomb (see the harness comments / tool/fuzz/README.md).
  exit(_verdict(report.escapes.isNotEmpty, report.maxSingleMs));
}

int _verdict(bool escaped, int maxSingleMs) =>
    escaped ? 1 : (maxSingleMs > 3000 ? 2 : 0);
