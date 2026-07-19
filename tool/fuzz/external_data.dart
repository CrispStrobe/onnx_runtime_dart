/// Blind robustness fuzzing of the external-data reference guard
/// (`checkExternalRef`), which validates a model-declared companion-file
/// `location`/`offset`/`length` before any disk access. Contract: it either
/// accepts a safe in-bounds relative reference or rejects with FormatException
/// — never leaks another error, and never lets a `..`/absolute path through.
///
///   dart run tool/fuzz/external_data.dart
library;

import 'dart:io';

import 'package:covfuzz/covfuzz.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';

int _env(String k, int def) =>
    int.tryParse(Platform.environment[k] ?? '') ?? def;

const _seeds = <String>[
  'weights.bin',
  'model.onnx_data',
  'sub/dir/weights.bin',
  '../../etc/passwd', // traversal
  '/abs/path',
  r'C:\win',
  '', // empty
];

void main() {
  final report = fuzz<String>(
    seeds: _seeds,
    entry: (loc) {
      // Vary offset/length/fileLen from the mutated location so the numeric
      // bounds logic is exercised alongside the path check.
      final n = loc.length;
      for (final (off, len, fl) in [
        (0, n, 1000),
        (n, n * 3, 500),
        (-n, n, 1 << 20),
        (n * 7, 1 << 40, 4096),
      ]) {
        try {
          checkExternalRef(loc, off, len, fl);
        } on FormatException {
          // documented clean reject
        }
      }
    },
    mutate: (s, rng) => mutateString(s, rng, maxOps: 10),
    isClean: (e) => e is FormatException,
    iterations: _env('FUZZ_ITERS', 300000),
    budgetMs: _env('FUZZ_BUDGET_MS', 30000),
    stressors: ['..', '../', 'a/../b', 'x/'.padRight(4096, 'x')],
  );
  report.report();
  exit(report.escapes.isNotEmpty ? 1 : (report.maxSingleMs > 3000 ? 2 : 0));
}
